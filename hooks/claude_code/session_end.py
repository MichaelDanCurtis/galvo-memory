#!/usr/bin/env python3
"""SessionEnd hook — write the final :Session node + score retrieval edges (Task 15).

This is the last hook Claude Code fires per session. The job is twofold:

1. **Close out the :Session node** with ``ended_at`` + ``outcome`` +
   ``task_description``. Earlier hooks (Task 12 SessionStart, Task 14
   PostToolUse) may have already MERGEd a Session node when they wrote
   their own state; this hook idempotently re-asserts the final shape
   via :http:post:`/api/Session` (the sidecar's CRUD endpoint MERGEs by
   ``id`` so re-writing the same id is safe).
2. **Score every ``RETRIEVED_IN`` edge** the session accumulated by
   POSTing a :class:`sidecar.scoring.ScoringPayload` to
   :http:post:`/api/sessions/{id}/score`. The sidecar walks the edges and
   stamps each with a ``utility_score`` per design §5's four signals.

We derive the scoring inputs from Claude Code's JSONL transcript file
(``transcript_path`` in the hook stdin payload):

* ``assistant_outputs`` — each assistant turn's textual output, used by
  the scorer's "textual reference" signal.
* ``requeries`` — every user prompt *after* the first, used by the
  scorer's "agent re-queried for similar info" signal.
* ``task_description`` — the user's first prompt (truncated), used as
  the human-readable :Session.task_description.
* ``task_outcome`` — heuristic guess from the last assistant turn's
  content (success / failure / partial / unknown).

The transcript shape Claude Code writes is reverse-engineered from
observable behavior — we keep the parser defensive (see
:func:`_parse_transcript` for the exhaustive shape table) and treat any
malformed line as a skip-this-line, not a fail-the-hook. The shared lib's
:class:`HookInputBase.from_stdin` guarantees we never raise out of the
script, regardless of how mangled stdin is.

**Why MERGE-then-score and not score-then-MERGE?** The scorer expects
the :Session node to exist (its Cypher MATCHes by ``id``). If the earlier
hooks didn't write a Session yet (e.g. SessionStart was disabled), this
hook's create call is what makes the scoring path work. The order also
matches the natural lifecycle: close the session record first, then
compute its quality metrics.

**Critical UX constraint:** like every hook, this script MUST NOT block
the user's session shutdown. ``SidecarClient`` already caps every HTTP
call at 3 s; the transcript parse is local file I/O and tightly bounded.
Worst case: 6 s of latency at session-end (Session create + score
request, each 3 s timeout) — acceptable per the cycle-1 budget.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Importing as ``hooks.claude_code.lib.*`` works both when the package
# is installed editable (the canonical local-dev path) and when the
# hook file is invoked as a script with ``memory/`` on ``PYTHONPATH``
# (the deployed install path). The deploy machinery (Task 18-ish) is
# responsible for picking one; either way these imports resolve.
from hooks.claude_code.lib.logging import setup_hook_logger
from hooks.claude_code.lib.scope import detect_scope_for_hook
from hooks.claude_code.lib.sidecar_client import SidecarClient
from hooks.claude_code.lib.types import UNKNOWN_EVENT, HookInputBase

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Tunables — picked conservatively to bound payload size at the hook→sidecar
# boundary. Cycle 1's sidecar accepts whatever we send, but our concern is
# (a) keeping the HTTP body under a sane size (transcripts can grow into the
# hundreds of KB) and (b) limiting the scorer's work-per-session.
# ---------------------------------------------------------------------------


_MAX_ASSISTANT_OUTPUTS: int = 50
"""Trim assistant_outputs to the last N entries before posting.

50 turns is a generous cap — most coding sessions are under 30 turns.
The trim keeps the JSON body sized predictably and bounds the scorer's
substring-matching work to O(N_edges × N_outputs)."""

_MAX_REQUERIES: int = 20
"""Trim requeries to the last N before posting. Same rationale as
:data:`_MAX_ASSISTANT_OUTPUTS` — most sessions have few requeries; 20 is
a generous cap on the negative-signal corpus."""

_PER_OUTPUT_CHAR_LIMIT: int = 2000
"""Hard cap on each assistant_output entry's character count.

The scorer does substring matching against memory titles (typically
<200 chars); long assistant turns dominated by code blocks add no
matching signal beyond the first ~2 KiB. The trim keeps the JSON
serializer fast and the eventual sidecar payload bounded."""

_PER_REQUERY_CHAR_LIMIT: int = 500
"""Cap on each requery string. Queries are usually short; 500 chars is
plenty to capture the semantic content for Jaccard overlap."""

_TASK_DESC_CHAR_LIMIT: int = 500
"""Maximum characters of ``task_description`` written to the :Session
node. Matches the design's "short free-form summary" intent."""

_TASK_TITLE_CHAR_LIMIT: int = 200
"""Sidecar's :class:`SessionCreate.title` has ``max_length=200``. We
truncate locally so the sidecar's validator never has to 422."""


# Event-type discriminators we accept. Claude Code's transcript JSONL has
# evolved across versions; we keep the parser defensive by treating any
# of these short identifiers as "this line is a user message" or "this is
# an assistant message". A line that uses none of them is skipped.

_USER_EVENT_TYPES: frozenset[str] = frozenset(
    {"user", "user_message", "user_prompt"}
)
"""Event type strings we treat as user-turn markers."""

_ASSISTANT_EVENT_TYPES: frozenset[str] = frozenset(
    {"assistant", "assistant_message", "assistant_response"}
)
"""Event type strings we treat as assistant-turn markers."""


# Outcome keyword sets — keep these as module-level constants so a tuning
# PR doesn't have to touch the heuristic loop body.

_SUCCESS_KEYWORDS: tuple[str, ...] = (
    "merged",
    "shipped",
    "passed",
    "complete",
    "completed",
    "success",
    "done",
)
"""Substrings in the last assistant turn that hint at a successful task."""

_FAILURE_KEYWORDS: tuple[str, ...] = (
    "failed",
    "error",
    "aborted",
    "cancel",
    "cancelled",
    "blocked",
    "broken",
)
"""Substrings in the last assistant turn that hint at a failed task."""


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    """Hook entrypoint. Returns the desired process exit code.

    Always returns 0. Per the hook contract: a SessionEnd hook MUST NOT
    fail the session by exiting non-zero — the worst-case behavior is
    "no end-of-session signal captured", which is the same effective
    outcome as the hook not being installed at all.
    """
    log = setup_hook_logger("session_end")

    hook_input = HookInputBase.from_stdin()
    if hook_input.hook_event_name == UNKNOWN_EVENT:
        # Empty / malformed stdin — nothing actionable. Log at INFO so
        # operators tailing the log can spot the no-op without it being
        # alarming.
        log.info("stdin malformed or empty; SessionEnd no-op")
        return 0

    session_id = hook_input.session_id
    if not session_id:
        # No session id → there's no :Session node to write or score.
        # This shouldn't happen at SessionEnd in practice (Claude Code
        # always provides session_id on lifecycle events), but the
        # defensive branch keeps us total.
        log.info("no session_id in stdin; SessionEnd no-op")
        return 0

    scope = detect_scope_for_hook(hook_input.cwd)
    client = SidecarClient()

    transcript_path = hook_input.transcript_path
    assistant_outputs, requeries, task_desc, task_outcome = _parse_transcript(
        transcript_path, log
    )

    # Step 1 — write/finalize the :Session node. The sidecar's POST /api/Session
    # MERGEs by ``id``, so this is safe to call even if SessionStart already
    # wrote one earlier in the session lifecycle.
    started_at = _extract_started_at(transcript_path)
    title = task_desc[:_TASK_TITLE_CHAR_LIMIT] if task_desc else "(no task captured)"
    session_payload: dict[str, Any] = {
        "id": session_id,
        "title": title,
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "agent_tool": "claude-code",
        "outcome": task_outcome,
        "task_description": task_desc[:_TASK_DESC_CHAR_LIMIT] if task_desc else "",
        "scope": scope,
    }
    session_result = client.create("Session", session_payload)
    if session_result is None:
        # Failed write is logged but does not abort scoring. The scoring
        # endpoint will still find any pre-existing :Session node from
        # earlier hooks; if even that doesn't exist the scorer will
        # silently return an empty report.
        log.warning("Session node create failed for session_id=%s", session_id)
    else:
        log.info("Session node written id=%s scope=%s", session_id, scope)

    # Step 2 — score the session's accumulated RETRIEVED_IN edges.
    scoring_payload: dict[str, Any] = {
        "session_id": session_id,
        "assistant_outputs": assistant_outputs[-_MAX_ASSISTANT_OUTPUTS:],
        "task_outcome": task_outcome,
        "requeries": requeries[-_MAX_REQUERIES:],
    }
    score_result = client.score_session(session_id, scoring_payload)
    if score_result is None:
        log.warning("session score failed for session_id=%s", session_id)
    else:
        log.info(
            "scored session=%s edges_scored=%d edges_skipped=%d",
            session_id,
            score_result.get("edges_scored", 0),
            score_result.get("edges_skipped", 0),
        )

    return 0


# ---------------------------------------------------------------------------
# Transcript parsing — best-effort, defensive.
# ---------------------------------------------------------------------------


def _parse_transcript(
    path: str | None,
    log: logging.Logger,
) -> tuple[list[str], list[str], str, str]:
    """Read Claude Code's JSONL transcript and extract scoring inputs.

    The transcript Claude Code writes is JSONL — one JSON object per line.
    The shape has evolved across versions and may include either:

    * a top-level ``type`` discriminator
      (``"user" | "assistant" | "tool_use" | "system" | ...``) with a
      sibling ``content`` field holding the textual payload, or
    * a Anthropic-API-style envelope: ``{"role": "user" | "assistant",
      "content": [{"type": "text", "text": "..."}, ...]}`` nested inside
      a ``message`` key, or
    * a flat string content field directly on the event.

    We try all three shapes per line and skip lines we can't interpret.
    We deliberately don't pin the parser to one shape because the
    cycle-1 deliverable cares about scoring signal — under-counting a
    turn costs at most one missed signal, while raising would crash
    the hook.

    Args:
        path: The transcript file path from Claude Code's hook input.
            ``None`` or non-existent files are handled by returning the
            empty default tuple.
        log: The hook's logger, used to log a single warning if reading
            the file fails. Per-line parse errors are silent — they're
            expected when the transcript shape evolves between Claude
            Code versions.

    Returns:
        A 4-tuple ``(assistant_outputs, requeries, task_description,
        task_outcome)``. All defaults are the empty/unknown sentinels so
        the caller can build a payload even when the transcript is missing.
    """
    assistant_outputs: list[str] = []
    requeries: list[str] = []
    task_desc = ""
    task_outcome = "unknown"

    if not path:
        return assistant_outputs, requeries, task_desc, task_outcome

    transcript = Path(path)
    if not transcript.is_file():
        log.info("transcript path %s does not exist", path)
        return assistant_outputs, requeries, task_desc, task_outcome

    try:
        raw = transcript.read_text(errors="replace")
    except OSError as exc:
        log.warning("could not read transcript %s: %r", path, exc)
        return assistant_outputs, requeries, task_desc, task_outcome

    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            # Mid-file corruption (rare; usually a half-flushed final
            # line). Skip silently — adjacent good lines still contribute.
            continue
        if not isinstance(evt, dict):
            continue

        role = _event_role(evt)
        content = _event_content(evt)
        if not content:
            continue

        if role == "user":
            if not task_desc:
                # First user turn is the task statement. Note we don't
                # add the first prompt to requeries — only subsequent
                # prompts count as "agent re-queried for similar info"
                # candidates (the scorer's signal).
                task_desc = content
            else:
                requeries.append(content[:_PER_REQUERY_CHAR_LIMIT])
        elif role == "assistant":
            assistant_outputs.append(content[:_PER_OUTPUT_CHAR_LIMIT])
        # ``tool_use`` / ``system`` / unknown roles are intentionally
        # ignored — the scorer's signals only depend on user + assistant.

    # Infer the task outcome from the last assistant turn. Cycle 1 uses
    # keyword matching; cycle 2 should add an explicit-marker channel
    # (e.g. a tool the agent calls to declare completion).
    if assistant_outputs:
        last = assistant_outputs[-1].lower()
        if any(kw in last for kw in _SUCCESS_KEYWORDS):
            task_outcome = "success"
        elif any(kw in last for kw in _FAILURE_KEYWORDS):
            task_outcome = "failure"
        else:
            # We have output but no keyword match — call it partial.
            # This is a softer signal than "success" or "failure" and
            # matches the design's "task_outcome enum is 4-way".
            task_outcome = "partial"

    return assistant_outputs, requeries, task_desc, task_outcome


def _event_role(evt: dict[str, Any]) -> str | None:
    """Resolve the event's role: ``"user"`` / ``"assistant"`` / ``None``.

    Tries three discriminator locations in priority order:

    1. ``evt["type"]`` — the most common Claude Code shape.
    2. ``evt["event"]`` — a less common alternate.
    3. ``evt["message"]["role"]`` — the Anthropic-API envelope shape
       Claude Code uses on some hook payload events.

    Returns ``None`` when none of the three match a known role. The
    caller treats ``None`` as "skip this line" — no signal contribution.
    """
    # Top-level type/event discriminator
    for key in ("type", "event"):
        candidate = evt.get(key)
        if isinstance(candidate, str):
            lc = candidate.lower()
            if lc in _USER_EVENT_TYPES:
                return "user"
            if lc in _ASSISTANT_EVENT_TYPES:
                return "assistant"

    # Anthropic-envelope shape: {"message": {"role": "user", ...}}
    message = evt.get("message")
    if isinstance(message, dict):
        role = message.get("role")
        if isinstance(role, str):
            role_lc = role.lower()
            if role_lc == "user":
                return "user"
            if role_lc == "assistant":
                return "assistant"

    return None


def _event_content(evt: dict[str, Any]) -> str:
    """Extract textual content from a transcript event, returning ``""``
    when no string content can be found.

    Handles three shapes:

    * ``evt["content"]`` — a flat string. The simplest shape.
    * ``evt["text"]`` — an alternate flat-string field.
    * ``evt["message"]["content"]`` — Anthropic-envelope shape; can be
      either a string or a list of content blocks ``[{"type": "text",
      "text": "..."}, ...]``. For lists we concatenate every ``text``
      child block (skipping non-text blocks like ``tool_use``).

    Returns the joined / extracted string, stripped of surrounding
    whitespace. Empty content (``""``) signals "skip this line" to the
    caller because zero-length content contributes no scoring signal.
    """
    # Shape 1 + 2: flat string fields
    for key in ("content", "text"):
        raw = evt.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # Shape 3: Anthropic-envelope nested content
    message = evt.get("message")
    if isinstance(message, dict):
        inner = message.get("content")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
        if isinstance(inner, list):
            chunks: list[str] = []
            for block in inner:
                if not isinstance(block, dict):
                    continue
                # Block is a text block iff its 'type' is 'text' (or
                # missing — older shapes omit the discriminator); we
                # accept either a 'text' or 'content' key for the body.
                btype = block.get("type")
                if btype is not None and btype != "text":
                    # Explicitly non-text block (e.g. tool_use) — skip.
                    continue
                for sub_key in ("text", "content"):
                    sub = block.get(sub_key)
                    if isinstance(sub, str) and sub.strip():
                        chunks.append(sub.strip())
                        break
            if chunks:
                return "\n".join(chunks)

    return ""


def _extract_started_at(path: str | None) -> str:
    """Best-effort guess at session-start time.

    Cycle 1 uses the transcript file's mtime as a proxy — Claude Code
    creates the file at session start and appends throughout, so the
    inode's mtime is closer to "last write" than "first write", but for
    short sessions (the common case) the difference is small. A future
    cycle should switch to parsing the first transcript line's timestamp
    if Claude Code embeds one; for now we accept the approximation.

    Falls back to :func:`datetime.now` UTC when the transcript path is
    absent or unreadable.

    The return value is an ISO-8601 string because that's the wire
    format the sidecar's :class:`SessionCreate.started_at` accepts
    (a ``datetime`` field; Pydantic parses ISO).
    """
    if path:
        p = Path(path)
        if p.is_file():
            try:
                return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()
            except OSError:
                # Stat failed for some reason — fall through to the
                # now() fallback rather than raising.
                pass
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    sys.exit(main())
