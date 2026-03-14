"""
analyzer.py — SystemAnalyzer: collects system-wide data from the orchestrator
REST APIs and log files, then calls the LLM proxy to generate improvement proposals.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a self-healing sidecar agent for a multi-agent AI orchestration system.
Your job is to analyse system health data and propose concrete, actionable improvements.

You will receive a JSON context containing:
- Active agents and their capabilities / health
- Recent workflow executions and their status
- LLM proxy usage statistics and recent requests
- Recent error/warning log lines from agent log files
- Current system settings
- Cortex (long-term memory) summaries

Respond ONLY with a valid JSON array. Each element must be an object with these exact fields:
{
  "title": "Short descriptive title (max 80 chars)",
  "description": "Detailed description of the problem or opportunity",
  "category": "<one of: bug_fix | performance | security | feature | config>",
  "severity": "<one of: critical | high | medium | low>",
  "rationale": "Why this improvement matters and what evidence from the data supports it",
  "affected_files": ["list", "of", "relative/file/paths"],
  "suggested_diff": "Unified diff or pseudocode showing the suggested change (may be empty string)",
  "data_sources": ["which data sources informed this proposal, e.g. 'llm_requests', 'agent_logs'"]
}

Rules:
- Return at most 10 improvements
- Order by severity descending (critical first)
- Only propose improvements that are clearly supported by the data
- Be specific — vague proposals like "improve error handling" are not useful without concrete evidence
- For config improvements, cite the specific setting key and current vs. recommended value
- For bug fixes, cite the specific error message or log line that indicates the bug
- Do not propose improvements already in progress or obviously correct
- If no improvements are warranted, return an empty array []
- Return ONLY the JSON array, no markdown, no commentary
"""


class SystemAnalyzer:
    """Collects system-wide telemetry and calls the LLM to generate improvement proposals."""

    def __init__(
        self,
        orchestrator_url: str,
        project_dir: Path,
        agent_id: str,
        proxy_url: str,
    ) -> None:
        self._base = orchestrator_url.rstrip("/")
        self._project_dir = project_dir
        self._agent_id = agent_id
        self._proxy_url = proxy_url
        self._http = httpx.AsyncClient(timeout=60)

    async def collect_data(
        self,
        agents_filter: Optional[list[str]] = None,
        workflow_ids_filter: Optional[list[str]] = None,
        since: Optional[str] = None,
    ) -> dict:
        """Gather data from all orchestrator REST APIs and log files.

        Args:
            agents_filter:       If set, restrict agent data and log excerpts to
                                 these agent names (case-insensitive).
            workflow_ids_filter: If set, restrict workflow data to these task IDs.
            since:               ISO timestamp cursor. When set, workflow and LLM
                                 request data are filtered to only include records
                                 created/started after this timestamp. Log excerpts
                                 are also filtered to lines after this time.
        """
        data: dict[str, Any] = {}

        async def _get(path: str, key: str) -> None:
            try:
                r = await self._http.get(f"{self._base}{path}")
                r.raise_for_status()
                data[key] = r.json()
            except Exception as exc:
                logger.warning("Failed to collect %s: %s", path, exc)
                data[key] = None

        # Parallel collection
        import asyncio
        await asyncio.gather(
            _get("/api/v1/agents", "agents"),
            _get("/api/v1/workflows?limit=50", "workflows"),
            _get("/api/v1/llm/stats/today", "llm_stats"),
            _get("/api/v1/llm/requests?limit=30", "llm_requests"),
            _get("/api/v1/cortex/agents", "cortex_agents"),
            _get("/api/v1/cortex/global", "global_memory"),
        )

        # Apply time cursor: drop records older than `since`
        if since:
            wf_raw = data.get("workflows") or []
            if isinstance(wf_raw, list):
                data["workflows"] = [
                    w for w in wf_raw
                    if (w.get("created_at") or w.get("started_at") or "9") >= since
                ]
            req_raw = data.get("llm_requests") or []
            # llm_requests may be a list or a dict with a "requests" key
            if isinstance(req_raw, dict):
                req_raw = req_raw.get("requests", [])
            if isinstance(req_raw, list):
                data["llm_requests"] = [
                    r for r in req_raw
                    if (r.get("created_at") or r.get("timestamp") or "9") >= since
                ]

        # Apply agent filter to agents list
        if agents_filter:
            af_lower = {a.lower() for a in agents_filter}
            agents_raw = data.get("agents") or []
            if isinstance(agents_raw, list):
                data["agents"] = [a for a in agents_raw if a.get("name", "").lower() in af_lower]

        # Apply workflow filter
        if workflow_ids_filter:
            wf_raw = data.get("workflows") or []
            if isinstance(wf_raw, list):
                data["workflows"] = [w for w in wf_raw if w.get("task_id") in workflow_ids_filter]

        # Collect settings — /api/v1/settings returns a list of {key, value, ...} objects
        try:
            r = await self._http.get(f"{self._base}/api/v1/settings")
            r.raise_for_status()
            raw_settings = r.json()
            # Normalise to dict regardless of whether the endpoint returns a list or dict
            if isinstance(raw_settings, list):
                data["settings"] = {
                    item["key"]: item.get("value") or item.get("default")
                    for item in raw_settings
                    if isinstance(item, dict) and "key" in item
                }
            elif isinstance(raw_settings, dict):
                data["settings"] = raw_settings
            else:
                data["settings"] = {}
        except Exception:
            data["settings"] = {}

        # Collect recent log excerpts, optionally filtered by agent name and since cursor
        data["agent_logs"] = self._collect_log_excerpts(agents_filter=agents_filter, since=since)

        return data

    def _collect_log_excerpts(
        self,
        agents_filter: Optional[list[str]] = None,
        since: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """Read ERROR/WARNING lines from the most recent logs directory."""
        excerpts: dict[str, list[str]] = {}
        logs_dir = self._project_dir / "logs"
        if not logs_dir.exists():
            return excerpts

        # Find the most recent agents-* subdirectory
        run_dirs = sorted(
            [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("agents-")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            return excerpts

        latest_dir = run_dirs[0]
        pattern = re.compile(r"\b(ERROR|WARNING|WARN|CRITICAL)\b", re.IGNORECASE)
        # Log lines start with a timestamp like "2026-03-14 08:05:51,679"
        # We only need the first 19 chars (YYYY-MM-DD HH:MM:SS) to compare with since.
        since_cmp = since[:19].replace("T", " ") if since else None
        af_lower = {a.lower() for a in agents_filter} if agents_filter else None

        for log_file in latest_dir.glob("*.log"):
            agent_name = log_file.stem
            if af_lower and agent_name.lower() not in af_lower:
                continue
            lines: list[str] = []
            try:
                with log_file.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        # Apply time cursor: skip lines before `since`
                        if since_cmp and len(line) >= 19:
                            line_ts = line[:19]
                            if line_ts < since_cmp:
                                continue
                        if pattern.search(line):
                            lines.append(line.rstrip())
                            if len(lines) >= 50:
                                break
            except Exception as exc:
                logger.debug("Could not read log %s: %s", log_file, exc)
                continue
            if lines:
                excerpts[agent_name] = lines

        return excerpts

    def _build_context(self, data: dict) -> str:
        """Compile collected data into a readable markdown context for the LLM."""
        parts: list[str] = []

        # Agents summary
        agents = data.get("agents") or []
        parts.append("## Active Agents")
        if agents:
            for a in agents:
                name = a.get("name", "?")
                health = a.get("health") or {}
                status = health.get("status", "unknown")
                caps = [c.get("name", "") for c in a.get("capabilities", [])]
                err = health.get("error_message", "")
                line = f"- **{name}** status={status} caps=[{', '.join(caps[:8])}]"
                if err:
                    line += f" error={err!r}"
                parts.append(line)
        else:
            parts.append("_(no agent data)_")

        # Workflows summary
        workflows = data.get("workflows") or []
        parts.append("\n## Recent Workflows (last 50)")
        if workflows:
            status_counts: dict[str, int] = {}
            failed_workflows: list[str] = []
            for wf in workflows:
                st = wf.get("status", "unknown")
                status_counts[st] = status_counts.get(st, 0) + 1
                if st in ("failed", "cancelled") and wf.get("error"):
                    title = wf.get("title", wf.get("task_id", ""))
                    failed_workflows.append(f"  - [{st}] {title}: {wf['error']}")
            parts.append(f"Status counts: {json.dumps(status_counts)}")
            if failed_workflows:
                parts.append("Failed/cancelled workflows:")
                parts.extend(failed_workflows[:10])
        else:
            parts.append("_(no workflow data)_")

        # LLM stats
        llm_stats = data.get("llm_stats") or {}
        parts.append("\n## LLM Proxy Statistics (Today)")
        if llm_stats:
            parts.append(
                f"Requests: {llm_stats.get('requests', 0)}  "
                f"Tokens: {llm_stats.get('total_tokens', 0)}  "
                f"Cost: ${llm_stats.get('cost_usd', 0):.4f}  "
                f"Blocked: {llm_stats.get('blocked', 0)}  "
                f"Errors: {llm_stats.get('errors', 0)}"
            )
        else:
            parts.append("_(no LLM stats)_")

        # Recent LLM requests — focus on errors/blocks
        llm_requests = data.get("llm_requests") or []
        if isinstance(llm_requests, dict):
            llm_requests = llm_requests.get("requests", [])
        blocked = [r for r in llm_requests if r.get("blocked")]
        errored = [r for r in llm_requests if r.get("error")]
        if blocked:
            parts.append(f"\nBlocked LLM requests ({len(blocked)}):")
            for r in blocked[:5]:
                parts.append(f"  - agent={r.get('agent_name')} reason={r.get('block_reason')}")
        if errored:
            parts.append(f"\nLLM request errors ({len(errored)}):")
            for r in errored[:5]:
                parts.append(f"  - agent={r.get('agent_name')} error={r.get('error')!r}")

        # Settings
        settings = data.get("settings") or {}
        parts.append("\n## Current System Settings")
        if settings:
            for k, v in list(settings.items())[:30]:
                # Mask secrets
                display_v = "***" if any(s in k.lower() for s in ("key", "secret", "password", "token")) else str(v)
                parts.append(f"  - {k}: {display_v}")
        else:
            parts.append("_(no settings data)_")

        # Agent log excerpts
        agent_logs = data.get("agent_logs") or {}
        parts.append("\n## Recent Error/Warning Log Lines")
        if agent_logs:
            for agent_name, lines in list(agent_logs.items())[:10]:
                parts.append(f"\n### {agent_name} ({len(lines)} lines)")
                for line in lines[:15]:
                    parts.append(f"  {line}")
        else:
            parts.append("_(no log data collected)_")

        # Cortex agents
        cortex_agents = data.get("cortex_agents") or []
        parts.append("\n## Cortex Memory Agents")
        if cortex_agents:
            for ca in cortex_agents[:10]:
                parts.append(f"  - {ca.get('name', '?')}: {ca.get('entry_count', 0)} entries")
        else:
            parts.append("_(no cortex data)_")

        return "\n".join(parts)

    async def analyze(
        self,
        data: dict,
        max_proposals: int = 5,
        categories_filter: Optional[list[str]] = None,
        agents_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """Call the LLM proxy to generate improvement proposals from collected data.

        Args:
            data:              Collected system data from collect_data().
            max_proposals:     Cap on the number of proposals to return.
            categories_filter: If set, only propose improvements in these categories
                               (bug_fix|performance|security|feature|config).
            agents_filter:     If set, only propose improvements targeting these agents.
        """
        context = self._build_context(data)

        # Build constraint clauses for the user message
        constraints: list[str] = []
        if categories_filter:
            valid = {"bug_fix", "performance", "security", "feature", "config"}
            cats = [c for c in categories_filter if c in valid]
            if cats:
                constraints.append(f"Only propose improvements in these categories: {', '.join(cats)}.")
        if agents_filter:
            constraints.append(
                f"Only propose improvements that target these agents: {', '.join(agents_filter)}. "
                "Ignore issues in other agents."
            )

        constraint_text = ("\n\nAdditional constraints:\n" + "\n".join(f"- {c}" for c in constraints)) if constraints else ""

        user_message = (
            f"Please analyse the following system health data and return up to {max_proposals} "
            f"improvement proposals as a JSON array.{constraint_text}\n\n"
            f"```\n{context[:12000]}\n```"
        )

        payload = {
            "model": "claude-haiku-4-5",
            "provider": "anthropic",
            "max_tokens": 4096,
            "temperature": 0.2,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }

        try:
            resp = await self._http.post(
                self._proxy_url,
                json=payload,
                headers={"X-Agent-Id": self._agent_id},
            )
            resp.raise_for_status()
            result = resp.json()
            content = result.get("content", "")
            if isinstance(content, list):
                # Handle Anthropic content block format
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
        except Exception as exc:
            logger.error("LLM proxy call failed during analysis: %s", exc)
            return []

        # Parse the JSON array from the LLM response
        content = content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            content = content.strip()

        try:
            proposals = json.loads(content)
            if not isinstance(proposals, list):
                logger.warning("LLM returned non-list JSON: %s", type(proposals))
                return []
        except json.JSONDecodeError as exc:
            # Try to extract JSON array from the content
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    proposals = json.loads(match.group(0))
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM analysis response: %s", exc)
                    return []
            else:
                logger.warning("No JSON array found in LLM response: %s", exc)
                return []

        # Validate and normalize each proposal
        analysis_id = str(uuid.uuid4())
        valid: list[dict] = []
        valid_categories = {"bug_fix", "performance", "security", "feature", "config"}
        valid_severities = {"critical", "high", "medium", "low"}

        for item in proposals:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            category = item.get("category", "feature")
            if category not in valid_categories:
                category = "feature"
            severity = item.get("severity", "low")
            if severity not in valid_severities:
                severity = "low"
            normalized = {
                "id": str(uuid.uuid4()),
                "analysis_id": analysis_id,
                "title": title[:200],
                "description": str(item.get("description", "")).strip(),
                "category": category,
                "severity": severity,
                "rationale": str(item.get("rationale", "")).strip(),
                "affected_files": item.get("affected_files", []) if isinstance(item.get("affected_files"), list) else [],
                "suggested_diff": str(item.get("suggested_diff", "")),
                "data_sources": item.get("data_sources", []) if isinstance(item.get("data_sources"), list) else [],
                "status": "pending",
            }
            valid.append(normalized)
            if len(valid) >= max_proposals:
                break

        logger.info("Analysis produced %d improvement proposals (analysis_id=%s)", len(valid), analysis_id)
        return valid
