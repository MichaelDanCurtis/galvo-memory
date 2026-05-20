#!/usr/bin/env python3
"""PostToolUse hook — log tool-call state changes as graph nodes (Task 14).

Invoked by Claude Code after every tool call. Reads the tool name +
arguments + result from stdin (the standard ``HookInputBase`` payload
plus an ``extra`` dict carrying ``tool_name`` / ``tool_input`` /
``tool_response``) and decides whether to record a graph node by
delegating to the sidecar on ``http://127.0.0.1:7575``.

What we choose to log
=====================

The hook is deliberately conservative — **only high-signal events
become graph nodes**. Logging every tool call would flood the graph and
swamp the retrieval surface with noise that adds nothing for future
sessions to ground on. The tracked patterns are:

* ``Read`` / ``Edit`` / ``Write`` / ``NotebookEdit`` — the file-touching
  Claude Code tools. We upsert an ``Artifact`` node keyed off the
  ``file_path`` (or ``notebook_path``) the tool was invoked with. The
  sidecar's create endpoint accepts duplicates silently (cycle-1: a new
  node every call; cycle-2 consolidation will dedupe by path-within-
  scope). A ``TOUCHED`` edge from the current Session → Artifact is a
  cycle-2 follow-up; for cycle 1 we only persist the node so retrieval
  can still surface "what files were touched in this session".
* ``Bash`` matching ``git commit`` — parse the SHA out of git's stdout
  (the ``[<branch> <sha>]`` line) and write a ``Commit`` node with the
  parsed sha + commit message + the originating shell command as
  ``intent``. The sidecar's ``Commit`` uniqueness constraint means we
  won't accidentally double-write if the hook fires twice on the same
  SHA.
* ``Bash`` matching test patterns (``pytest`` / ``cargo test`` /
  ``pnpm test`` / ``npm test`` / ``go test``) — write a ``Test`` node
  on exit code 0 (status ``passed``) or a ``Failure`` node with
  ``failure_type='test'`` on non-zero (status ``failed``). The error
  signature is mined from the tail of stdout/stderr — last few lines
  containing ``Error`` / ``FAILED`` / ``Exception``.
* ``Bash`` matching build patterns (``cargo build`` / ``pnpm build`` /
  ``npm run build`` / ``make``) — write a ``Failure`` node with
  ``failure_type='build'`` only on non-zero exit. Successful builds are
  too frequent to be worth a node.

Every other tool call (``WebFetch``, ``WebSearch``, ``Task``, MCP tool
invocations, ``Skill``, …) gets a single line in
``~/.galvo-memory/logs/post_tool_use.log`` with the tool name only.
That's sufficient for an operator inspecting hook activity; the
sidecar isn't touched.

Failure modes
=============

The whole hook is wrapped to satisfy the **never-raise** contract:

* Malformed / missing stdin → :meth:`HookInputBase.from_stdin` returns
  ``hook_event_name == 'unknown'`` and we exit 0.
* Sidecar unreachable → :class:`SidecarClient.create` returns ``None``;
  we log a WARNING line and move on.
* Unserializable payload → ``SidecarClient`` catches the
  ``TypeError`` / ``ValueError`` internally and returns ``None``.
* Pattern parsing errors (no SHA in git output, etc.) — the per-branch
  helpers degrade silently and the hook still exits 0.

If something we did NOT anticipate raises inside ``main``, the
top-level ``try`` swallows it, logs it, and returns 0 — Claude Code's
session never sees a traceback.

Field naming gotcha (carried from Task 8)
=========================================

The sidecar's ``Failure`` model uses ``failure_type``, not ``type``,
because ``type`` collides with the neo4j-agent-memory library's own
``type`` property on the ``:Entity`` super-label. Same story for
``Constraint.constraint_type``. We honor those names in the create
payloads here.

Likewise the ``Test`` model's ``last_run_status`` enum uses
``passed`` / ``failed`` / ``skipped`` / ``error`` / ``unknown`` — the
Task-14 brief said ``"pass"`` / ``"fail"``; we use the actual model
values so the sidecar accepts the body without massaging.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hooks.claude_code.lib.logging import setup_hook_logger
from hooks.claude_code.lib.scope import detect_scope_for_hook
from hooks.claude_code.lib.sidecar_client import SidecarClient
from hooks.claude_code.lib.types import UNKNOWN_EVENT, HookInputBase

__all__ = [
    "BUILD_RE",
    "FILE_TOOLS",
    "GIT_COMMIT_RE",
    "TEST_RE",
    "main",
]


# ---------------------------------------------------------------------------
# Tool dispatch tables.
# ---------------------------------------------------------------------------

FILE_TOOLS: frozenset[str] = frozenset({"Read", "Edit", "Write", "NotebookEdit"})
"""Claude Code tools that touch a single file path → log Artifact."""

# Bash subpatterns. We anchor on word boundaries so ``git`` inside a
# subcommand (e.g. ``ls .git/``) doesn't spuriously match the commit
# detector.
GIT_COMMIT_RE: re.Pattern[str] = re.compile(r"\bgit\s+commit\b")
"""Matches an explicit ``git commit`` invocation. Doesn't try to
distinguish ``--amend`` / ``-m`` variants — they all produce a SHA we
care about."""

TEST_RE: re.Pattern[str] = re.compile(
    r"\b(pytest|cargo\s+test|pnpm\s+test|npm\s+test|go\s+test)\b"
)
"""Matches the test-runner CLIs we treat as Test/Failure write
triggers. We require a word boundary at the start so ``epytest`` isn't
accidentally a match; the lookahead allows ``-k`` / ``--workspace``
trailing args."""

BUILD_RE: re.Pattern[str] = re.compile(
    r"\b(cargo\s+build|pnpm\s+build|npm\s+run\s+build|make)\b"
)
"""Matches build commands. ``make`` matches alone because most repos
use it as a build orchestrator. We only emit ``Failure`` nodes on
non-zero exit (build successes are not worth recording)."""


# Extension → language identifier table. Used for the ``language``
# property on ``Artifact`` nodes. Anything not listed falls back to
# ``"unknown"`` rather than ``None`` because ``None`` would mean
# "non-code file" per the sidecar contract and we don't always know
# whether a path is code without reading it.
_LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cypher": "cypher",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".json": "json",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".rst": "rst",
    ".html": "html",
    ".css": "css",
    ".swift": "swift",
    ".kt": "kotlin",
    ".java": "java",
    ".ipynb": "jupyter",
}


# Regex for the git commit output line — ``[branch sha] message`` or
# ``[branch (root-commit) sha] message``. The ``sha`` is 7-40 hex chars
# (git's default short sha is 7 but some repos config more).
_GIT_SHA_RE: re.Pattern[str] = re.compile(
    r"\[[^\]]*?\b([a-f0-9]{7,40})\b[^\]]*?\]"
)


# Lines we treat as error-signature candidates when mining stdout/stderr
# for a Failure signature. Order doesn't matter — we collect last 3
# matches across the whole regex.
_ERROR_LINE_RE: re.Pattern[str] = re.compile(
    r"\b(FAILED|Error|Exception|error\[|panicked|assertion failed)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    """Read stdin, dispatch on tool name, write graph node if relevant.

    Returns:
        Exit code. Always 0 — the hook contract is "never fail the
        session". A non-zero exit from a Claude Code hook would propagate
        to the dispatcher and the user would see an error.
    """
    log = setup_hook_logger("post_tool_use")

    try:
        hook_input = HookInputBase.from_stdin()
        if hook_input.hook_event_name == UNKNOWN_EVENT:
            return 0

        extra = hook_input.extra or {}
        tool_name = extra.get("tool_name", "")
        tool_input = extra.get("tool_input", {}) or {}
        tool_response = extra.get("tool_response", {}) or {}

        # Coerce dict-likes — Claude Code sometimes wraps tool_response
        # in a different shape (string-only output for terminal-style
        # tools). We treat anything non-dict as "no response data".
        if not isinstance(tool_input, dict):
            tool_input = {}
        if not isinstance(tool_response, dict):
            tool_response = {}

        scope = detect_scope_for_hook(hook_input.cwd)
        client = SidecarClient()

        if tool_name in FILE_TOOLS:
            _log_artifact(client, log, tool_input, scope)
        elif tool_name == "Bash":
            _log_bash_outcome(client, log, tool_input, tool_response, scope)
        else:
            # Other tools: light log, no sidecar call.
            log.info("tool=%s ignored (not in tracked set)", tool_name)
    except Exception as exc:  # noqa: BLE001 — hook MUST NOT raise
        # Last-resort guard. Anything that bubbles up here is a logic
        # bug we want to know about, but we never propagate it.
        log.exception("post_tool_use hook crashed: %r", exc)

    return 0


# ---------------------------------------------------------------------------
# Per-tool handlers.
# ---------------------------------------------------------------------------


def _log_artifact(
    client: SidecarClient,
    log: Any,
    tool_input: dict[str, Any],
    scope: str,
) -> None:
    """Record an ``Artifact`` node for a file-touching tool call.

    Args:
        client: Sidecar HTTP client. Returns ``None`` on failure; we
            don't propagate it.
        log: The hook's rotating-file logger.
        tool_input: ``tool_input`` dict from the hook payload. Expected
            to contain ``file_path`` (Read / Edit / Write) or
            ``notebook_path`` (NotebookEdit).
        scope: Resolved scope string per design §D4.

    No-op if the input has no usable path field. Per the design, the
    ``last_touched`` datetime stamps when this hook fired so retrieval
    can use recency to rank files.
    """
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(path, str) or not path:
        log.info("artifact skipped (no path in tool_input)")
        return

    payload: dict[str, Any] = {
        "path": path,
        "language": _guess_language(path),
        # ``role`` is left to cycle-2 enrichment — we don't know whether
        # a touched file is a test, config, or entrypoint without
        # additional heuristics. Pass ``None`` (omitted via the sidecar's
        # serialize_for_create exclude_none).
        "role": None,
        "last_touched": datetime.now(tz=UTC).isoformat(),
        "scope": scope,
    }
    result = client.create("Artifact", payload)
    if result is None:
        log.warning("artifact create failed for path=%s", path)
    else:
        log.info("artifact recorded path=%s", path)


def _log_bash_outcome(
    client: SidecarClient,
    log: Any,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any],
    scope: str,
) -> None:
    """Dispatch a ``Bash`` tool call to the right Commit / Test / Failure
    branch based on the command and the exit code.

    Args:
        client: Sidecar HTTP client.
        log: Hook logger.
        tool_input: Must contain ``command`` (the shell string).
        tool_response: ``exit_code`` + ``output`` (or ``stdout`` /
            ``stderr``) keys. We tolerate either shape — older Claude
            Code versions surface different field names.
        scope: Resolved scope.

    Returns no value; failures degrade silently per the never-raise
    contract.
    """
    command = tool_input.get("command", "")
    if not isinstance(command, str):
        command = ""
    exit_code = _coerce_exit_code(tool_response.get("exit_code"))
    output = _coerce_output(tool_response)

    if GIT_COMMIT_RE.search(command):
        _write_commit_node(client, log, command, output, scope)
        return

    if TEST_RE.search(command):
        if exit_code == 0:
            client.create(
                "Test",
                {
                    "identifier": command[:500],
                    "last_run_status": "passed",
                    "last_run_at": datetime.now(tz=UTC).isoformat(),
                    "runner": _infer_test_runner(command),
                    "scope": scope,
                },
            )
            log.info("test pass recorded cmd=%s", command[:80])
        else:
            client.create(
                "Failure",
                {
                    "failure_type": "test",
                    "error_signature": _extract_failure_signature(output)[:500],
                    "resolved": False,
                    "full_message": output[:5000] if output else None,
                    "scope": scope,
                },
            )
            log.info("test failure recorded cmd=%s exit=%s", command[:80], exit_code)
        return

    if BUILD_RE.search(command) and exit_code != 0:
        client.create(
            "Failure",
            {
                "failure_type": "build",
                "error_signature": _extract_failure_signature(output)[:500],
                "resolved": False,
                "full_message": output[:5000] if output else None,
                "scope": scope,
            },
        )
        log.info("build failure recorded cmd=%s exit=%s", command[:80], exit_code)
        return

    # Bash call that doesn't match any tracked pattern — light log only.
    log.info("bash untracked cmd=%s exit=%s", command[:80], exit_code)


def _write_commit_node(
    client: SidecarClient,
    log: Any,
    command: str,
    output: str,
    scope: str,
) -> None:
    """Parse a SHA out of ``git commit`` stdout and write a Commit node.

    The git output for a successful commit is::

        [branch-name abc1234] commit message subject
         1 file changed, ...

    We pull the SHA from the bracketed prefix; without a SHA there's
    nothing reliable to dedupe on so we skip the write.
    """
    sha_match = _GIT_SHA_RE.search(output)
    if not sha_match:
        log.info("commit skipped (no sha parsed) cmd=%s", command[:80])
        return

    sha = sha_match.group(1)
    message = _extract_commit_message(output)
    if not message:
        # The sidecar's CommitCreate requires a non-empty message.
        # Fall back to the bracketed line itself so we at least
        # have something to embed.
        message = sha

    result = client.create(
        "Commit",
        {
            "sha": sha,
            "message": message[:500],
            "intent": command[:200],
            "scope": scope,
        },
    )
    if result is None:
        log.warning("commit create failed sha=%s", sha)
    else:
        log.info("commit recorded sha=%s", sha)


# ---------------------------------------------------------------------------
# Helpers — pure functions, no I/O. Exposed for tests via __all__-ish
# name discipline (no underscore on the regexes used by tests).
# ---------------------------------------------------------------------------


def _guess_language(path: str) -> str:
    """Map a file path's suffix to a language identifier.

    Returns ``"unknown"`` when the suffix isn't in the table. We never
    return ``None`` because ``None`` semantically means "non-code file"
    in the Artifact model and we don't have enough signal to assert
    that from inside the hook.
    """
    suffix = Path(path).suffix.lower()
    return _LANGUAGE_BY_SUFFIX.get(suffix, "unknown")


def _infer_test_runner(command: str) -> str:
    """Identify the test framework from the command string.

    Returns ``"unknown"`` if we can't tell — defensive, shouldn't
    happen if ``TEST_RE`` already matched.
    """
    if "pytest" in command:
        return "pytest"
    if "cargo test" in command:
        return "cargo test"
    if "pnpm test" in command or "npm test" in command:
        return "vitest"  # the SDK + dashboard both use vitest under npm/pnpm test
    if "go test" in command:
        return "go test"
    return "unknown"


def _coerce_exit_code(raw: Any) -> int:
    """Normalize various tool_response.exit_code shapes to an int.

    Claude Code historically passes exit codes as either ``int``,
    ``str``, or omits them entirely (treat as 0 — success). A bool
    ``True`` is treated as 0 and ``False`` as 1, matching the shell
    convention.
    """
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 0 if raw else 1
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


def _coerce_output(tool_response: dict[str, Any]) -> str:
    """Pull a single text string from the various output-key shapes.

    Tries ``output`` first (most common in the modern Claude Code
    schema), then ``stdout`` + ``stderr`` concatenated (older schema),
    then an empty string. Always returns a string — never raises on
    weird types.
    """
    direct = tool_response.get("output")
    if isinstance(direct, str):
        return direct
    parts: list[str] = []
    stdout = tool_response.get("stdout")
    if isinstance(stdout, str):
        parts.append(stdout)
    stderr = tool_response.get("stderr")
    if isinstance(stderr, str):
        parts.append(stderr)
    return "\n".join(parts)


def _extract_commit_message(output: str) -> str:
    """Pull the commit message subject from ``git commit`` stdout.

    Format:: ``[branch-name <sha>] subject line``. We split on the
    closing bracket and trim. Returns an empty string when we can't
    find a bracket line — callers fall back to the SHA in that case.
    """
    for line in output.splitlines():
        if _GIT_SHA_RE.search(line):
            after = line.split("]", 1)
            if len(after) == 2:
                return after[1].strip()
    return ""


def _extract_failure_signature(output: str) -> str:
    """Mine a compact failure signature from the tail of an output stream.

    We prefer lines containing ``Error`` / ``FAILED`` / ``Exception`` —
    those are signal-rich for dedup. If no error-like lines are present,
    fall back to the last three non-empty lines of the output (often a
    summary or assertion message).

    Returns a short string suitable for the sidecar's
    ``Failure.error_signature`` (which caps at 500 chars). Returns the
    literal ``"(empty output)"`` when there's nothing to mine, so the
    sidecar still accepts the body (min_length=1).
    """
    if not output:
        return "(empty output)"
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return "(empty output)"

    error_lines = [line.strip() for line in lines if _ERROR_LINE_RE.search(line)]
    if error_lines:
        return " | ".join(error_lines[-3:])
    return " | ".join(line.strip() for line in lines[-3:])


if __name__ == "__main__":
    sys.exit(main())
