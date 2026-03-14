"""
orchestrator_client.py — WebSocket + HTTP client for the self-heal-agent.

Registers with the orchestrator, handles capability task_requests (analyze_system,
list_improvements, implement_improvement), and runs a periodic background
analysis loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import websockets
import websockets.exceptions

from improvement_store import ImprovementStore
from analyzer import SystemAnalyzer
from executor import ImprovementExecutor

logger = logging.getLogger(__name__)

# ── Stable agent identity ──────────────────────────────────────────────────────

_AGENT_ID_FILE = Path(".agent_id")


def _stable_agent_id() -> str:
    if _AGENT_ID_FILE.exists():
        return _AGENT_ID_FILE.read_text().strip()
    new_id = str(uuid.uuid4())
    _AGENT_ID_FILE.write_text(new_id)
    logger.info("Generated new stable agent ID: %s", new_id)
    return new_id


# ── Constants ─────────────────────────────────────────────────────────────────

AGENT_NAME = "self-heal-agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = (
    "Sidecar agent that analyses system health and proposes/implements improvements. "
    "Periodically collects data from all orchestrator APIs, calls an LLM to identify "
    "issues, and presents proposals for human approval or auto-implements low-severity fixes."
)

HEARTBEAT_INTERVAL_S: int = 15
MAX_BACKOFF_S: int = 60
DRAIN_TIMEOUT_S: int = 30

# Default project root: parent of agent dir
_DEFAULT_PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

REGISTRATION_PAYLOAD: dict = {
    "name": AGENT_NAME,
    "description": AGENT_DESCRIPTION,
    "version": AGENT_VERSION,
    "tags": ["sidecar", "system", "self-heal"],
    "capabilities": [
        {
            "name": "analyze_system",
            "description": (
                "Trigger an immediate system health analysis. Collects data from all "
                "orchestrator APIs and log files, then uses an LLM to generate improvement proposals."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict analysis to these agent names (empty = all agents)",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["bug_fix", "performance", "security", "feature", "config"]},
                        "description": "Restrict proposals to these categories (empty = all categories)",
                    },
                    "workflow_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict workflow analysis to these specific task IDs (empty = all recent)",
                    },
                },
            },
            "cost": {"type": "llm", "estimated_cost_usd": 0.05},
        },
        {
            "name": "list_improvements",
            "description": (
                "List improvement proposals from the local store. "
                "Optionally filter by status: pending|approved|rejected|executing|completed|failed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status (empty = all)",
                        "default": "",
                    },
                },
            },
            "cost": {"type": "free", "estimated_cost_usd": 0.0},
        },
        {
            "name": "implement_improvement",
            "description": (
                "Execute an approved improvement using an agentic LLM loop with file tools. "
                "The improvement must be in 'approved' status."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "improvement_id": {
                        "type": "string",
                        "description": "UUID of the improvement to implement",
                    },
                },
                "required": ["improvement_id"],
            },
            "cost": {"type": "llm", "estimated_cost_usd": 0.10},
        },
    ],
    "required_settings": [
        {
            "key": "self_heal_analysis_interval_min",
            "label": "Analysis Interval (minutes)",
            "description": "How often to run automatic system analysis. Set to 0 to disable. Default: 60.",
            "type": "integer",
            "required": False,
            "default": "60",
        },
        {
            "key": "self_heal_max_proposals",
            "label": "Max Proposals per Analysis",
            "description": "Maximum number of improvement proposals to generate per analysis run. Default: 5.",
            "type": "integer",
            "required": False,
            "default": "5",
        },
        {
            "key": "self_heal_auto_approve_low",
            "label": "Auto-Approve Low Severity",
            "description": "If 'true', automatically approve and implement improvements with severity=low. Default: false.",
            "type": "string",
            "required": False,
            "default": "false",
        },
        {
            "key": "self_heal_run_command_enabled",
            "label": "Enable run_command Tool",
            "description": "If 'true', the executor may run allowlisted shell commands (git diff, etc.). Default: false.",
            "type": "string",
            "required": False,
            "default": "false",
        },
        {
            "key": "self_heal_project_root",
            "label": "Project Root Path",
            "description": "Absolute path to the project root for file operations. Defaults to parent of agent directory.",
            "type": "string",
            "required": False,
            "default": _DEFAULT_PROJECT_ROOT,
        },
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _envelope(
    sender_id: str,
    msg_type: str,
    payload: dict,
    recipient_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    msg_id: Optional[str] = None,
) -> str:
    return json.dumps({
        "id":             msg_id or str(uuid.uuid4()),
        "type":           msg_type,
        "sender_id":      sender_id,
        "recipient_id":   recipient_id,
        "payload":        payload,
        "timestamp":      _now_iso(),
        "correlation_id": correlation_id,
    })


# ── Main client ───────────────────────────────────────────────────────────────

class OrchestratorClient:
    """Registers the self-heal-agent, handles capabilities, and runs periodic analysis."""

    def __init__(self, orchestrator_url: str = "http://localhost:8000", log_level: str = "INFO") -> None:
        self._base = orchestrator_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=60)
        self._proxy_url = f"{self._base}/api/v1/llm/complete"

        self._agent_id: str = _stable_agent_id()
        self._ws_url: str = ""
        self._common_settings: dict[str, Any] = {}

        self._status: str = "starting"
        self._active_tasks: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._total_duration_ms: float = 0.0
        self._start_time: float = time.monotonic()

        self._shutting_down: bool = False
        self._current_ws: Any = None

        # Settings (defaults)
        self._analysis_interval_min: int = 60
        self._max_proposals: int = 5
        self._auto_approve_low: bool = False
        self._run_command_enabled: bool = False
        self._project_root: Path = Path(_DEFAULT_PROJECT_ROOT)

        # Subsystems (initialized after registration with real agent_id)
        self._store: ImprovementStore = ImprovementStore(Path("data/improvements.db"))
        self._analyzer: Optional[SystemAnalyzer] = None
        self._executor: Optional[ImprovementExecutor] = None

        # Background analysis task
        self._analysis_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._graceful_shutdown()))

        # Initialize store immediately (independent of registration)
        await asyncio.to_thread(self._store.init)

        await self._register()
        await self._connect_loop()

    # ── Registration ──────────────────────────────────────────────────────

    async def _register(self) -> None:
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": self._agent_id}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._agent_id = data["agent_id"]
        self._ws_url = data["ws_url"]

        # Merge common and per-agent settings
        merged = {
            **data.get("common_settings", {}),
            **data.get("agent_settings", {}),
        }
        self._common_settings.update(merged)
        self._apply_settings(merged)

        # Initialize subsystems with real agent_id
        self._analyzer = SystemAnalyzer(
            orchestrator_url=self._base,
            project_dir=self._project_root,
            agent_id=self._agent_id,
            proxy_url=self._proxy_url,
        )
        self._executor = ImprovementExecutor(
            project_dir=self._project_root,
            proxy_url=self._proxy_url,
            agent_id=self._agent_id,
            run_command_enabled=self._run_command_enabled,
        )

        logger.info("Registered — agent_id=%s  ws=%s", self._agent_id, self._ws_url)

    def _apply_settings(self, settings: dict) -> None:
        """Apply settings dict to local config."""
        if "self_heal_analysis_interval_min" in settings:
            try:
                self._analysis_interval_min = int(settings["self_heal_analysis_interval_min"])
            except (ValueError, TypeError):
                pass

        if "self_heal_max_proposals" in settings:
            try:
                self._max_proposals = int(settings["self_heal_max_proposals"])
            except (ValueError, TypeError):
                pass

        if "self_heal_auto_approve_low" in settings:
            self._auto_approve_low = str(settings["self_heal_auto_approve_low"]).lower() == "true"

        if "self_heal_run_command_enabled" in settings:
            self._run_command_enabled = str(settings["self_heal_run_command_enabled"]).lower() == "true"
            if self._executor:
                self._executor._run_command_enabled = self._run_command_enabled

        if "self_heal_project_root" in settings:
            val = str(settings["self_heal_project_root"]).strip()
            if val:
                candidate = Path(val).expanduser().resolve()
                if candidate.exists():
                    self._project_root = candidate
                    if self._analyzer:
                        self._analyzer._project_dir = candidate
                    if self._executor:
                        self._executor._project_dir = candidate
                        self._executor._project_dir = self._executor._project_dir.resolve()

    # ── WebSocket loop ────────────────────────────────────────────────────

    async def _connect_loop(self) -> None:
        backoff = 1.0
        while not self._shutting_down:
            try:
                logger.info("Connecting to %s …", self._ws_url)
                async with websockets.connect(self._ws_url) as ws:
                    backoff = 1.0
                    await self._run_session(ws)

            except websockets.exceptions.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4004:
                    logger.warning("Unknown agent_id (4004) — re-registering …")
                    try:
                        await self._register()
                    except Exception as reg_exc:
                        logger.error("Re-registration failed: %s", reg_exc)
                elif code == 4003:
                    logger.info("Agent disabled (4003) — will retry")
                    backoff = max(backoff, 10.0)
                elif self._shutting_down:
                    break
                else:
                    logger.warning("WS closed (code=%s) — retry in %.0fs", code, backoff)

            except (OSError, Exception) as exc:
                if self._shutting_down:
                    break
                logger.warning("WS error (%s) — retry in %.0fs", exc, backoff)

            if not self._shutting_down:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_S)

    async def _run_session(self, ws) -> None:
        self._current_ws = ws
        self._status = "available"
        logger.info("WebSocket session active")

        # Start background analysis loop
        if self._analysis_task is None or self._analysis_task.done():
            self._analysis_task = asyncio.create_task(
                self._analysis_loop(ws), name="self-heal-analysis-loop"
            )

        try:
            await asyncio.gather(
                self._heartbeat_loop(ws),
                self._recv_loop(ws),
            )
        finally:
            if self._analysis_task and not self._analysis_task.done():
                self._analysis_task.cancel()
                try:
                    await self._analysis_task
                except asyncio.CancelledError:
                    pass
            self._analysis_task = None
            self._current_ws = None
            self._status = "offline"

    # ── Background analysis loop ───────────────────────────────────────────

    async def _analysis_loop(self, ws) -> None:
        """Run periodic system analysis based on configured interval."""
        logger.info(
            "Analysis loop started (interval=%d min)", self._analysis_interval_min
        )
        while not self._shutting_down:
            interval_min = self._analysis_interval_min
            if interval_min <= 0:
                logger.debug("Analysis disabled (interval=0), sleeping 60s before re-check")
                await asyncio.sleep(60)
                continue

            # Wait for interval, then run
            await asyncio.sleep(interval_min * 60)

            if self._shutting_down:
                break

            logger.info("Running scheduled system analysis …")
            try:
                await self._run_analysis(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Scheduled analysis failed: %s", exc, exc_info=True)

    async def _run_analysis(
        self,
        ws,
        agents_filter: Optional[list[str]] = None,
        categories_filter: Optional[list[str]] = None,
        workflow_ids_filter: Optional[list[str]] = None,
    ) -> int:
        """Collect data, analyze, store proposals, send WS messages. Returns count of new proposals."""
        if self._analyzer is None:
            return 0

        try:
            # Fetch existing open issues before collecting data so the LLM
            # receives them as a constraint and won't regenerate duplicates.
            existing_issues = await asyncio.to_thread(self._store.get_open_summaries)

            data = await self._analyzer.collect_data(
                agents_filter=agents_filter,
                workflow_ids_filter=workflow_ids_filter,
            )
            proposals = await self._analyzer.analyze(
                data,
                max_proposals=self._max_proposals,
                categories_filter=categories_filter,
                agents_filter=agents_filter,
                existing_issues=existing_issues,
            )
        except Exception as exc:
            logger.error("Analysis error: %s", exc, exc_info=True)
            return 0

        if not proposals:
            logger.info("Analysis produced no new proposals")
            return 0

        # Filter out proposals with similar titles to existing non-rejected/completed ones
        new_proposals: list[dict] = []
        for proposal in proposals:
            has_sim = await asyncio.to_thread(self._store.has_similar, proposal["title"])
            if has_sim:
                logger.debug("Skipping duplicate proposal: %s", proposal["title"])
                continue
            stored = await asyncio.to_thread(self._store.insert, proposal)
            new_proposals.append(stored)
            logger.info("Stored new proposal [%s] %s", proposal["severity"], proposal["title"])

        if not new_proposals:
            logger.info("All analysis proposals were duplicates; nothing stored")
            return 0

        # Send improvement_proposed WS message
        analysis_id = new_proposals[0].get("analysis_id", str(uuid.uuid4()))
        await self._ws_send(ws, self._msg(
            "improvement_proposed",
            {
                "proposals": new_proposals,
                "analysis_id": analysis_id,
                "count": len(new_proposals),
            },
        ))

        # Auto-approve low severity if enabled
        if self._auto_approve_low:
            for proposal in new_proposals:
                if proposal.get("severity") == "low" and proposal.get("status") == "pending":
                    await asyncio.to_thread(
                        self._store.update_status, proposal["id"], "approved",
                        "Auto-approved (low severity)"
                    )
                    logger.info("Auto-approved low-severity proposal: %s", proposal["title"])
                    asyncio.create_task(self._implement(ws, proposal["id"]))

        return len(new_proposals)

    # ── Heartbeat ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await self._ws_send(ws, self._msg(
                "heartbeat",
                {
                    "status": self._status,
                    "current_load": min(self._active_tasks / 3.0, 1.0),
                    "active_tasks": self._active_tasks,
                    "metrics": self._metrics(),
                },
            ))
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── Receive loop ──────────────────────────────────────────────────────

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON frame ignored")
                continue
            mtype = msg.get("type", "?")
            logger.info("← [%s] from=%s", mtype, msg.get("sender_id", "?"))
            await self._dispatch(ws, msg)

    async def _dispatch(self, ws, msg: dict) -> None:
        mtype = msg.get("type", "")
        payload = msg.get("payload", {})

        if mtype == "task_request":
            asyncio.create_task(self._handle_task(ws, msg))

        elif mtype == "settings_push":
            pushed = payload.get("settings", {})
            self._common_settings.update(pushed)
            self._apply_settings(pushed)
            logger.info("Settings updated via push: %s", list(pushed.keys()))

        elif mtype in ("agent_registered", "agent_offline"):
            logger.debug("Peer event [%s]: %s", mtype, payload.get("agent_id"))

        elif mtype == "error":
            logger.error(
                "Orchestrator error [%s]: %s",
                payload.get("code"), payload.get("detail"),
            )

        elif mtype in ("improvement_proposed", "improvement_status"):
            # We send these, not receive them — ignore
            pass

        else:
            logger.debug("Unhandled message type: %r", mtype)

    # ── Task handling ─────────────────────────────────────────────────────

    async def _handle_task(self, ws, msg: dict) -> None:
        req_id = msg.get("id")
        sender_id = msg.get("sender_id")
        payload = msg.get("payload", {})
        capability = payload.get("capability", "")
        input_data = payload.get("input_data", {})

        self._active_tasks += 1
        self._status = "busy"
        t0 = time.monotonic()

        try:
            output, error = await self._dispatch_capability(ws, capability, input_data)
            duration_ms = (time.monotonic() - t0) * 1000

            if error:
                self._tasks_failed += 1
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": False, "error": error, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))
            else:
                self._tasks_completed += 1
                self._total_duration_ms += duration_ms
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": True, "output_data": output, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._tasks_failed += 1
            logger.exception("Unhandled error in capability %r", capability)
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": str(exc), "duration_ms": round(duration_ms, 1)},
                recipient_id=sender_id,
                correlation_id=req_id,
            ))

        finally:
            self._active_tasks = max(0, self._active_tasks - 1)
            self._status = "draining" if self._shutting_down else (
                "busy" if self._active_tasks else "available"
            )
            await self._send_status_update(ws)

    async def _dispatch_capability(
        self, ws, capability: str, input_data: dict
    ) -> tuple[Optional[dict], Optional[str]]:
        if capability == "analyze_system":
            return await self._cap_analyze_system(ws, input_data)

        elif capability == "list_improvements":
            return await self._cap_list_improvements(input_data)

        elif capability == "implement_improvement":
            return await self._cap_implement_improvement(ws, input_data)

        else:
            return None, f"Unknown capability: {capability!r}"

    # ── Capability: analyze_system ─────────────────────────────────────────

    async def _cap_analyze_system(self, ws, input_data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Run an on-demand system analysis with optional filters."""
        if self._analyzer is None:
            return None, "Analyzer not initialized"

        agents_filter     = input_data.get("agents") or None
        categories_filter = input_data.get("categories") or None
        workflow_ids      = input_data.get("workflow_ids") or None

        # Normalise: accept comma-separated strings OR lists
        def _to_list(v) -> Optional[list[str]]:
            if not v:
                return None
            if isinstance(v, str):
                return [x.strip() for x in v.split(",") if x.strip()]
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return None

        agents_filter     = _to_list(agents_filter)
        categories_filter = _to_list(categories_filter)
        workflow_ids      = _to_list(workflow_ids)

        try:
            count = await self._run_analysis(
                ws,
                agents_filter=agents_filter,
                categories_filter=categories_filter,
                workflow_ids_filter=workflow_ids,
            )
            filter_desc = []
            if agents_filter:
                filter_desc.append(f"agents={agents_filter}")
            if categories_filter:
                filter_desc.append(f"categories={categories_filter}")
            if workflow_ids:
                filter_desc.append(f"workflows={workflow_ids}")
            suffix = f" (filters: {', '.join(filter_desc)})" if filter_desc else ""
            return {"count": count, "message": f"Analysis complete. {count} new proposals generated.{suffix}"}, None
        except Exception as exc:
            logger.error("analyze_system capability error: %s", exc, exc_info=True)
            return None, str(exc)

    # ── Capability: list_improvements ─────────────────────────────────────

    async def _cap_list_improvements(self, input_data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Return improvements from local store."""
        status_filter = input_data.get("status", "")
        try:
            items = await asyncio.to_thread(self._store.list_all, status_filter)
            return {"improvements": items, "count": len(items)}, None
        except Exception as exc:
            logger.error("list_improvements capability error: %s", exc)
            return None, str(exc)

    # ── Capability: implement_improvement ─────────────────────────────────

    async def _cap_implement_improvement(
        self, ws, input_data: dict
    ) -> tuple[Optional[dict], Optional[str]]:
        """Execute an approved improvement."""
        improvement_id = input_data.get("improvement_id", "")
        if not improvement_id:
            return None, "improvement_id is required"

        improvement = await asyncio.to_thread(self._store.get, improvement_id)
        if improvement is None:
            return None, f"Improvement {improvement_id!r} not found"

        if improvement["status"] not in ("approved", "pending"):
            return None, (
                f"Improvement is in status {improvement['status']!r}; "
                "must be 'approved' (or 'pending' for auto-run)"
            )

        success, message = await self._implement(ws, improvement_id)
        return {
            "improvement_id": improvement_id,
            "success": success,
            "message": message,
        }, None

    async def _implement(self, ws, improvement_id: str) -> tuple[bool, str]:
        """Run the executor for a given improvement and update store + send WS event."""
        if self._executor is None:
            return False, "Executor not initialized"

        improvement = await asyncio.to_thread(self._store.get, improvement_id)
        if improvement is None:
            return False, f"Improvement {improvement_id!r} not found"

        # Mark as executing
        await asyncio.to_thread(
            self._store.update_status, improvement_id, "executing", "Execution started"
        )
        await self._ws_send(ws, self._msg(
            "improvement_status",
            {
                "improvement_id": improvement_id,
                "status": "executing",
                "message": "Execution started",
                "files_modified": [],
                "execution_log": "",
            },
        ))

        try:
            success, message = await self._executor.execute(improvement)
        except Exception as exc:
            logger.error("Executor raised exception for %s: %s", improvement_id, exc, exc_info=True)
            success, message = False, f"Executor exception: {exc}"

        final_status = "completed" if success else "failed"
        from datetime import datetime, timezone
        completed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        await asyncio.to_thread(
            self._store.update_status,
            improvement_id,
            final_status,
            message,
            [],
            "",
            completed_at,
        )

        await self._ws_send(ws, self._msg(
            "improvement_status",
            {
                "improvement_id": improvement_id,
                "status": final_status,
                "message": message,
                "files_modified": [],
                "execution_log": "",
            },
        ))

        logger.info(
            "Improvement %s → %s: %s", improvement_id, final_status, message[:200]
        )
        return success, message

    # ── Status update ─────────────────────────────────────────────────────

    async def _send_status_update(self, ws) -> None:
        await self._ws_send(ws, self._msg(
            "status_update",
            {
                "status": self._status,
                "current_load": min(self._active_tasks / 3.0, 1.0),
                "active_tasks": self._active_tasks,
                "metrics": self._metrics(),
            },
        ))

    # ── Graceful shutdown ─────────────────────────────────────────────────

    async def _graceful_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown signal received — draining …")
        self._status = "draining"

        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while self._active_tasks > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.5)

        if self._agent_id:
            try:
                await self._http.delete(f"{self._base}/api/v1/agents/{self._agent_id}")
                logger.info("Deregistered from orchestrator.")
            except Exception as exc:
                logger.warning("Deregister failed: %s", exc)

        await self._http.aclose()
        logger.info("Shutdown complete.")

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _ws_send(self, ws, msg_str: str) -> None:
        msg = json.loads(msg_str)
        mtype = msg.get("type", "?")
        noisy = mtype in ("heartbeat", "status_update")
        (logger.debug if noisy else logger.info)(
            "→ [%s] to=%s", mtype, msg.get("recipient_id") or "orchestrator"
        )
        try:
            await ws.send(msg_str)
        except websockets.exceptions.ConnectionClosed:
            raise
        except Exception as exc:
            logger.warning("WS send failed: %s", exc)

    def _msg(
        self,
        msg_type: str,
        payload: dict,
        recipient_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        return _envelope(self._agent_id, msg_type, payload, recipient_id, correlation_id)

    def _metrics(self) -> dict:
        n = self._tasks_completed + self._tasks_failed
        return {
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "avg_response_time_ms": round(self._total_duration_ms / n, 1) if n else 0.0,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "analysis_interval_min": self._analysis_interval_min,
            "auto_approve_low": self._auto_approve_low,
        }
