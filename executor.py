"""
executor.py — ImprovementExecutor: executes approved improvements via an
agentic LLM loop with tool use (read_file, write_file, list_files, run_command, finish).

Path sandbox: all paths are resolved against project_dir; requests outside it are rejected.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 20
_READ_MAX_BYTES = 256 * 1024  # 256 KB read limit

# Commands that are safe to run in read-only mode
_ALLOWED_COMMANDS = {
    "git diff",
    "git status",
    "git log",
    "python -m py_compile",
    "ls",
    "cat",
}

_SYSTEM_PROMPT = """You are a senior software engineer implementing an improvement to a multi-agent AI orchestration system.

Project structure:
- /agent-orchestrator  — FastAPI WebSocket hub (Python)
- /browser-agent       — Playwright/Claude browser automation agent
- /filesystem-agent    — Cross-platform file/directory agent
- /task-planner-agent  — LLM-based workflow planner
- /task-executor-agent — Step-by-step workflow executor
- /code-execution-agent— Sandboxed code execution agent
- /document-agent      — PDF/image extraction agent
- /self-heal-agent     — This agent (sidecar, system health)

You have access to tools: read_file, write_file, list_files, run_command (limited), finish.

Guidelines:
- Before writing any file, read it first to understand the existing code
- Make minimal, targeted changes — do not refactor unrelated code
- Preserve existing code style, indentation, and imports
- After writing a file, verify it compiles/parses correctly if possible
- When finished, call the finish tool with a brief summary of what was done
- If you cannot safely implement the improvement, call finish with success=false and explain why
- Only modify files inside the project directory
- Never modify .env files, credential files, or database files directly
"""

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file inside the project directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (absolute or relative to project root)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (create or overwrite) a file inside the project directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Full content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory inside the project directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: project root)"},
            },
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a limited set of read-only shell commands. "
            "Allowed: git diff, git status, git log, python -m py_compile, ls, cat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "finish",
        "description": "Signal that implementation is complete (or was aborted).",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief summary of what was done"},
                "success": {"type": "boolean", "description": "True if improvement was implemented, False if aborted"},
            },
            "required": ["summary", "success"],
        },
    },
]


class PathViolation(ValueError):
    """Raised when a path escapes the project sandbox."""


class ImprovementExecutor:
    """Executes approved improvements via an agentic LLM loop with tool use."""

    def __init__(
        self,
        project_dir: Path,
        proxy_url: str,
        agent_id: str,
        run_command_enabled: bool = False,
    ) -> None:
        self._project_dir = project_dir.resolve()
        self._proxy_url = proxy_url
        self._agent_id = agent_id
        self._run_command_enabled = run_command_enabled
        self._http = httpx.AsyncClient(timeout=120)

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to project_dir, reject paths outside it."""
        p = Path(path_str)
        if not p.is_absolute():
            p = self._project_dir / p
        resolved = p.resolve()
        try:
            resolved.relative_to(self._project_dir)
        except ValueError:
            raise PathViolation(f"Path {path_str!r} is outside the project directory")
        return resolved

    def _tool_read_file(self, path: str) -> str:
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"[ERROR] File not found: {path}"
            if resolved.is_dir():
                return f"[ERROR] Path is a directory: {path}"
            size = resolved.stat().st_size
            if size > _READ_MAX_BYTES:
                return f"[ERROR] File too large ({size} bytes > {_READ_MAX_BYTES}). Read a smaller section."
            content = resolved.read_text(encoding="utf-8", errors="replace")
            return content
        except PathViolation as exc:
            return f"[SECURITY] {exc}"
        except Exception as exc:
            return f"[ERROR] {exc}"

    def _tool_write_file(self, path: str, content: str) -> str:
        try:
            resolved = self._resolve_path(path)
            # Refuse to write credential/db files
            name = resolved.name.lower()
            if any(name.endswith(ext) for ext in (".env", ".db", ".sqlite", ".sqlite3")):
                return f"[SECURITY] Writing to {name} files is not allowed"
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return f"[OK] Written {len(content)} bytes to {resolved.relative_to(self._project_dir)}"
        except PathViolation as exc:
            return f"[SECURITY] {exc}"
        except Exception as exc:
            return f"[ERROR] {exc}"

    def _tool_list_files(self, path: str = ".") -> str:
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"[ERROR] Directory not found: {path}"
            if not resolved.is_dir():
                return f"[ERROR] Not a directory: {path}"
            entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = []
            for e in entries[:100]:
                rel = e.relative_to(self._project_dir)
                if e.is_dir():
                    lines.append(f"DIR  {rel}/")
                else:
                    size = e.stat().st_size
                    lines.append(f"FILE {rel} ({size} bytes)")
            return "\n".join(lines) if lines else "(empty directory)"
        except PathViolation as exc:
            return f"[SECURITY] {exc}"
        except Exception as exc:
            return f"[ERROR] {exc}"

    def _tool_run_command(self, command: str) -> str:
        if not self._run_command_enabled:
            return "[DISABLED] run_command is disabled. Enable self_heal_run_command_enabled setting."
        # Allowlist check
        cmd_stripped = command.strip()
        allowed = any(
            cmd_stripped == allowed_cmd or cmd_stripped.startswith(allowed_cmd + " ")
            for allowed_cmd in _ALLOWED_COMMANDS
        )
        if not allowed:
            return f"[SECURITY] Command not in allowlist: {cmd_stripped!r}. Allowed: {sorted(_ALLOWED_COMMANDS)}"
        try:
            result = subprocess.run(
                shlex.split(cmd_stripped),
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._project_dir),
            )
            output = result.stdout + result.stderr
            return output[:8000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out"
        except Exception as exc:
            return f"[ERROR] {exc}"

    def _execute_tool(self, name: str, input_data: dict) -> str:
        if name == "read_file":
            return self._tool_read_file(input_data.get("path", ""))
        elif name == "write_file":
            return self._tool_write_file(
                input_data.get("path", ""),
                input_data.get("content", ""),
            )
        elif name == "list_files":
            return self._tool_list_files(input_data.get("path", "."))
        elif name == "run_command":
            return self._tool_run_command(input_data.get("command", ""))
        elif name == "finish":
            # finish is handled in the agentic loop; returning placeholder here
            return "[finish]"
        else:
            return f"[ERROR] Unknown tool: {name!r}"

    async def execute(self, improvement: dict) -> tuple[bool, str]:
        """Run the agentic loop to implement an improvement.

        Returns (success, message).
        """
        title = improvement.get("title", "")
        description = improvement.get("description", "")
        rationale = improvement.get("rationale", "")
        affected_files = improvement.get("affected_files", [])
        suggested_diff = improvement.get("suggested_diff", "")

        user_message = f"""Implement the following system improvement:

**Title:** {title}

**Description:**
{description}

**Rationale:**
{rationale}

**Affected files (hints):**
{json.dumps(affected_files, indent=2)}

**Suggested diff / pseudocode:**
{suggested_diff or '(none provided)'}

Start by listing the relevant directory and reading affected files to understand the current code.
Then make the necessary changes and verify them. Call finish() when done."""

        messages: list[dict] = [{"role": "user", "content": user_message}]
        files_modified: list[str] = []
        execution_log_lines: list[str] = []

        for iteration in range(_MAX_ITERATIONS):
            payload = {
                "model": "claude-sonnet-4-5",
                "provider": "anthropic",
                "max_tokens": 8192,
                "temperature": 0.1,
                "system": _SYSTEM_PROMPT,
                "messages": messages,
                "tools": _TOOLS,
            }

            try:
                resp = await self._http.post(
                    self._proxy_url,
                    json=payload,
                    headers={"X-Agent-Id": self._agent_id},
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                logger.error("LLM proxy error in executor iteration %d: %s", iteration, exc)
                return False, f"LLM proxy error: {exc}"

            stop_reason = result.get("stop_reason", "")
            content = result.get("content", [])

            # Normalize content to list of blocks
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = []

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": content})

            # Check for tool_use blocks
            tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]

            # Check for finish tool
            finish_block = next(
                (b for b in tool_uses if b.get("name") == "finish"), None
            )
            if finish_block:
                fin_input = finish_block.get("input", {})
                success = bool(fin_input.get("success", True))
                summary = str(fin_input.get("summary", "Implementation completed"))
                log_str = "\n".join(execution_log_lines)
                logger.info(
                    "Executor finished (success=%s iteration=%d): %s",
                    success, iteration, summary,
                )
                return success, summary

            if stop_reason == "end_turn" and not tool_uses:
                # Model stopped without calling finish — treat as success
                text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                summary = " ".join(text_parts).strip()[:500] or "Completed without explicit finish"
                return True, summary

            if not tool_uses:
                logger.warning("Executor: no tool calls and stop_reason=%r, stopping", stop_reason)
                return False, f"Unexpected stop: stop_reason={stop_reason!r}"

            # Execute tools and build tool_result blocks
            tool_results = []
            for block in tool_uses:
                tool_name = block.get("name", "")
                tool_id = block.get("id", f"tool_{iteration}")
                tool_input = block.get("input", {})

                if tool_name == "finish":
                    # Already handled above
                    continue

                logger.debug("Executor tool: %s(%s)", tool_name, list(tool_input.keys()))
                tool_output = self._execute_tool(tool_name, tool_input)

                # Track written files
                if tool_name == "write_file" and tool_output.startswith("[OK]"):
                    fpath = tool_input.get("path", "")
                    if fpath and fpath not in files_modified:
                        files_modified.append(fpath)

                execution_log_lines.append(
                    f"[iter={iteration}] {tool_name}({json.dumps(tool_input)[:200]}) → {tool_output[:300]}"
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_output,
                })

            # Add tool results as user message to continue the loop
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        # Max iterations exceeded
        logger.warning("Executor reached max iterations (%d) for: %s", _MAX_ITERATIONS, title)
        return False, f"Reached maximum iterations ({_MAX_ITERATIONS}) without completing"
