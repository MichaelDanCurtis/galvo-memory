#!/usr/bin/env python3
"""SessionStart hook — inject top-of-mind memories into Claude Code context (Task 12).

Claude Code invokes this script at session start with a JSON payload on stdin
(see :class:`hooks.claude_code.lib.types.HookInputBase`). We:

1. Parse stdin (total — never raises).
2. Detect the project scope from the input ``cwd`` (design §D4).
3. Probe the sidecar's ``/health`` — if unreachable, emit nothing and exit 0.
4. Query the sidecar for "top-of-mind" memories scoped to the current project:

   * Recent ``Decision`` nodes (≤5)
   * Active ``Belief`` nodes (≤5)
   * Open ``Task`` nodes (≤3)
   * Open ``Failure`` nodes (≤3)

5. Format a compact markdown block (header + per-section bullets).
6. Hard-truncate to ``MAX_LINES`` (30) lines per design §10 token budget.
7. Write to **stdout** — Claude Code captures stdout and injects it into the
   model's context (NOT stderr; stderr lands in the session transcript as
   noise).

Hook contract (per Task 11's :mod:`hooks.claude_code.lib`): NEVER raises.
NEVER blocks. NEVER pollutes stderr. Every failure mode silently no-ops
and writes a WARNING line to ``~/.galvo-memory/logs/session_start.log``.

Install (one-time, per-user): symlink or copy this file under the path
Claude Code's hook loader resolves for SessionStart. See
``memory/hooks/README.md`` for the install incantation (lands in Task 18,
the docker-compose + ops task).

Token budget rationale: design §10 caps memory injection at ~30–50
"context slots"; we use a hard 30-line cap with aggressive per-section
limits (5/5/3/3). A wedged or noisy sidecar can't blow the budget because
the truncation is purely line-counted and applied after formatting.
"""

from __future__ import annotations

import sys
from typing import Any

from hooks.claude_code.lib.logging import setup_hook_logger
from hooks.claude_code.lib.scope import detect_scope_for_hook
from hooks.claude_code.lib.sidecar_client import SidecarClient
from hooks.claude_code.lib.types import UNKNOWN_EVENT, HookInputBase

__all__ = [
    "MAX_BELIEFS",
    "MAX_DECISIONS",
    "MAX_FAILURES",
    "MAX_LINES",
    "MAX_TASKS",
    "main",
]


# ---------------------------------------------------------------------------
# Token budget — design §10. Hard caps that survive a chatty sidecar.
# ---------------------------------------------------------------------------

MAX_LINES: int = 30
"""Maximum total lines of stdout. Truncated AFTER formatting."""

MAX_DECISIONS: int = 5
"""Per-section cap. Decisions are the most-frequently-useful recall surface."""

MAX_BELIEFS: int = 5
"""Per-section cap. Beliefs encode the agent's current world model."""

MAX_TASKS: int = 3
"""Per-section cap. Tasks are usually 1–2 active at once; 3 leaves headroom."""

MAX_FAILURES: int = 3
"""Per-section cap. Failures surface unresolved error signatures."""


# ---------------------------------------------------------------------------
# Query strings used to retrieve each section.
#
# The sidecar's GET /api/search/{label} endpoint (Task 8) always runs a
# semantic vector search — there's no "list-by-recency" endpoint in cycle
# 1. To approximate "give me the most recent / most-relevant decisions",
# we pass a generic descriptor string that should embed near the typical
# content of nodes in that label. This is intentionally lossy — Cycle 2
# may add a recency-first list endpoint to the sidecar. For now, an
# under-recall result is acceptable because the hook's contract is
# "graceful degradation"; an empty result just produces a shorter
# top-of-mind block.
# ---------------------------------------------------------------------------

QUERY_DECISIONS: str = "recent decision rationale"
"""Generic descriptor that embeds near Decision content."""

QUERY_BELIEFS: str = "active belief current claim"
"""Generic descriptor that embeds near Belief content."""

QUERY_TASKS: str = "open task pending work"
"""Generic descriptor that embeds near Task content."""

QUERY_FAILURES: str = "open failure unresolved error"
"""Generic descriptor that embeds near Failure content."""


def main() -> int:
    """Hook entrypoint. Returns a process exit code (0 on every path).

    Claude Code's hook loader treats a non-zero exit as a failure that
    surfaces to the user. Our contract is "never fail the session" — so
    every path returns 0, even when the sidecar is dead or stdin was
    garbage. Diagnostics go to the rotating log file.
    """
    log = setup_hook_logger("session_start")

    # Parse stdin. from_stdin() is total — see types.py docstring.
    hook_input = HookInputBase.from_stdin()
    if hook_input.hook_event_name == UNKNOWN_EVENT:
        # Malformed or empty stdin. Claude Code's protocol may
        # legitimately emit empty stdin in some configurations; we
        # treat this identically to a sidecar-down case — silently
        # emit nothing.
        log.info("malformed or empty stdin; emitting empty top-of-mind")
        return 0

    scope = detect_scope_for_hook(hook_input.cwd)
    log.info(
        "session start scope=%s session_id=%s",
        scope,
        hook_input.session_id,
    )

    client = SidecarClient()

    # Health check first. If the sidecar isn't running (which is the
    # most common dev-environment state — operator forgot to
    # ``docker compose up``), we must silently no-op. The /health probe
    # is cheap (<100ms when up; bounded by the client's 3s timeout when
    # down) and saves us from issuing four separate search requests
    # against a dead endpoint.
    health = client.health()
    if health is None:
        log.warning("sidecar unreachable; emitting empty top-of-mind")
        return 0

    # Fetch the four top-of-mind sets. Every call returns [] on failure
    # — see SidecarClient.search contract — so we never need try/except.
    decisions = client.search(
        "Decision", QUERY_DECISIONS, scope=scope, limit=MAX_DECISIONS
    )
    beliefs = client.search(
        "Belief", QUERY_BELIEFS, scope=scope, limit=MAX_BELIEFS
    )
    tasks = client.search(
        "Task", QUERY_TASKS, scope=scope, limit=MAX_TASKS
    )
    failures = client.search(
        "Failure", QUERY_FAILURES, scope=scope, limit=MAX_FAILURES
    )

    output = _format_block(scope, decisions, beliefs, tasks, failures)
    # Enforce the hard line cap. Done AFTER formatting so a chatty
    # sidecar can't blow the budget by returning huge per-node titles
    # (each title still goes onto its own line — the truncation is
    # purely line-counted, which is the design-§10 contract).
    output = "\n".join(output.splitlines()[:MAX_LINES])

    if not output:
        # Nothing to inject. Don't even emit a header — a bare header
        # would still cost the model a context slot for zero signal.
        return 0

    # Write to stdout; Claude Code's hook dispatcher captures stdout and
    # routes it into the model's context. Add a trailing newline so
    # downstream concatenation doesn't run lines together.
    sys.stdout.write(output)
    sys.stdout.write("\n")
    return 0


def _format_block(
    scope: str,
    decisions: list[dict[str, Any]],
    beliefs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    """Build a compact markdown block. Returns ``""`` if every section is empty.

    Each section is optional — if the sidecar returned no hits for a
    label, that section is omitted entirely (no empty header). When all
    four sections are empty, return the empty string so the caller can
    suppress output altogether.

    Args:
        scope: The scope string from :func:`detect_scope_for_hook`.
            Included in the header so a session running in the wrong
            project surfaces it before the model starts working.
        decisions: List of Decision node dicts from
            :meth:`SidecarClient.search`. Each dict's preferred display
            field is ``title``; we fall back to ``name`` then ``id`` so
            shape evolution doesn't crash the formatter.
        beliefs: Belief nodes. Display field: ``claim`` (per design §4
            ``NAME_PROPERTY_PER_LABEL["Belief"] == "claim"``).
        tasks: Task nodes. Display field: ``title``.
        failures: Failure nodes. Display field: ``error_signature``.
    """
    parts: list[str] = [f"# Galvo Memory — top-of-mind (scope: {scope})"]

    if decisions:
        parts.append("")
        parts.append("## Recent decisions")
        for d in decisions[:MAX_DECISIONS]:
            parts.append(f"- {_label_for(d, ('title', 'name'))}")

    if beliefs:
        parts.append("")
        parts.append("## Active beliefs")
        for b in beliefs[:MAX_BELIEFS]:
            parts.append(f"- {_label_for(b, ('claim', 'name'))}")

    if tasks:
        parts.append("")
        parts.append("## Open tasks")
        for t in tasks[:MAX_TASKS]:
            parts.append(f"- {_label_for(t, ('title', 'name'))}")

    if failures:
        parts.append("")
        parts.append("## Open failures")
        for f in failures[:MAX_FAILURES]:
            parts.append(f"- {_label_for(f, ('error_signature', 'name'))}")

    if len(parts) == 1:
        # Only the header — no signal worth injecting. Return empty so
        # main() suppresses output entirely.
        return ""

    return "\n".join(parts)


def _label_for(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Pick the first non-empty string under ``keys`` in ``node``.

    Defensive against schema drift: if a label-specific name property is
    missing (e.g. the sidecar returned a partial dict), we fall back to
    the universal ``name`` field and finally to ``"?"`` so the bullet
    point isn't blank.

    Trailing whitespace and embedded newlines are stripped — a node
    title with embedded newlines would otherwise blow the MAX_LINES
    budget by splitting onto multiple lines after :func:`str.splitlines`.
    """
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            # Collapse any whitespace + newlines into a single-line
            # representation so each node bullet occupies exactly one
            # output line.
            return " ".join(value.split())
    return "?"


if __name__ == "__main__":
    sys.exit(main())
