"""Pydantic models for the hook stdin JSON Claude Code sends (Task 11).

Claude Code invokes lifecycle hooks by spawning the configured script
with the hook event details serialized as JSON on stdin. The schema is
documented loosely and has evolved across versions; we model the
conservative intersection: every hook receives at minimum the event
name, a session id, and the cwd; some events include a transcript
path or extra event-specific payload.

We do NOT model every per-event field exhaustively — when the schema
inevitably shifts, individual hook scripts can subclass
:class:`HookInputBase` and add their event-specific fields. The base
class captures the cross-event fields the shared lib needs (scope
detection consumes ``cwd``; sidecar calls need ``session_id``).

**Robustness:** :meth:`HookInputBase.from_stdin` catches everything —
JSON decode errors, malformed shapes, EOF — and returns
``HookInputBase(hook_event_name="unknown")``. The hook scripts treat
``"unknown"`` as a signal to short-circuit and exit 0 silently. Hooks
must NEVER raise; a raised exception bubbles up through Claude Code's
dispatcher and the session sees a confusing stack trace.

Schema assumptions (best-effort — Claude Code's hook protocol is still
evolving; the current shape is reverse-engineered from observable
behavior in the Galvo development environment circa 2026-05-19):

* ``hook_event_name`` — required; values like ``"SessionStart"``,
  ``"UserPromptSubmit"``, ``"PostToolUse"``, ``"SessionEnd"``.
* ``session_id`` — optional (some events may omit it during startup).
* ``cwd`` — optional; if present, an absolute path string.
* ``transcript_path`` — optional; only present for PostToolUse and
  SessionEnd in the current protocol version.
* ``extra`` — catch-all dict for event-specific payload (tool name +
  args for PostToolUse, prompt text for UserPromptSubmit, etc.).
  Unknown top-level fields land here per the model's ``extra`` config.

If the canonical schema later disagrees, we update the model — the
hook scripts will still no-op safely on unknown shapes because
``from_stdin`` is total.
"""

from __future__ import annotations

import json
import sys
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HookInputBase",
    "UNKNOWN_EVENT",
]


UNKNOWN_EVENT: str = "unknown"
"""Sentinel value :meth:`HookInputBase.from_stdin` returns on malformed input.

Hooks check ``input.hook_event_name == UNKNOWN_EVENT`` and short-circuit
without performing any side effects.
"""


class HookInputBase(BaseModel):
    """Common fields Claude Code includes in every hook invocation.

    Subclass for event-specific payloads; the base shape is what the
    shared lib needs (scope detection + sidecar session-id wiring).

    ``model_config.extra = "allow"`` so unknown top-level fields don't
    cause a validation error — Claude Code is the schema authority and
    we don't want to break the moment they add a field.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        # Be lenient with unknown top-level fields: Claude Code's
        # hook protocol may grow new fields between versions and we
        # don't want our hooks to start failing on a new field.
        extra="allow",
    )

    hook_event_name: str = Field(
        ...,
        description=(
            "Event identifier — one of 'SessionStart', "
            "'UserPromptSubmit', 'PostToolUse', 'SessionEnd', or "
            "the literal 'unknown' when stdin parsing fails."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "The Claude Code session id, used as the Session node id "
            "in the memory graph. May be absent during very-early "
            "startup events."
        ),
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "The user's working directory at the time the hook fires. "
            "Hooks pass this to scope.detect_scope_for_hook so the "
            "scope filter on retrieval queries matches the project the "
            "user is actually editing."
        ),
    )
    transcript_path: str | None = Field(
        default=None,
        description=(
            "Filesystem path to the running session's transcript file. "
            "Only present on events where the transcript exists "
            "(PostToolUse, SessionEnd). SessionEnd's scorer reads "
            "this path to extract assistant outputs."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Event-specific payload that doesn't fit a typed field. "
            "PostToolUse puts ``tool_name`` + ``tool_input`` + "
            "``tool_response`` here; UserPromptSubmit puts ``prompt``."
        ),
    )

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> "HookInputBase":
        """Lift Claude Code's flat hook payload into our ``extra`` dict.

        Real-world Claude Code dispatches hooks with event-specific keys
        (``tool_name``, ``tool_input``, ``tool_response``, ``prompt``) as
        TOP-LEVEL fields, NOT nested under ``extra``. The original model
        in cycle 1 expected the nested form, so PostToolUse and
        UserPromptSubmit silently dropped their payloads.

        This shim copies the known event-specific top-level keys into
        ``extra`` BEFORE Pydantic validates, so downstream hook code that
        reads ``hook_input.extra["tool_name"]`` continues to work
        regardless of which shape Claude Code emits. It's idempotent —
        if the caller already passed a populated ``extra`` dict, those
        values win.
        """
        if isinstance(obj, dict):
            # Don't mutate the caller's dict.
            data: dict[str, Any] = dict(obj)
            existing_extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
            absorbed = dict(existing_extra)
            for key in ("tool_name", "tool_input", "tool_response", "prompt"):
                if key in data and key not in absorbed:
                    absorbed[key] = data[key]
            if absorbed:
                data["extra"] = absorbed
            obj = data
        return super().model_validate(obj, **kwargs)

    @classmethod
    def from_stdin(cls) -> "HookInputBase":
        """Parse a hook's stdin JSON. NEVER raises.

        On any failure mode (empty stdin, non-JSON bytes, JSON that
        doesn't match the schema, EOFError, decoding error) returns
        ``HookInputBase(hook_event_name=UNKNOWN_EVENT)`` so the caller
        can branch on the sentinel and silently exit.

        The hook contract is "never fail the session" — raising here
        would propagate out the hook script's main() and Claude Code
        would surface the traceback to the user. That's a much worse
        UX than a missed memory write.
        """
        try:
            raw = sys.stdin.read()
        except Exception:  # noqa: BLE001 — stdin can fail in arbitrary ways
            return cls(hook_event_name=UNKNOWN_EVENT)
        if not raw.strip():
            return cls(hook_event_name=UNKNOWN_EVENT)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls(hook_event_name=UNKNOWN_EVENT)
        if not isinstance(data, dict):
            return cls(hook_event_name=UNKNOWN_EVENT)
        try:
            return cls.model_validate(data)
        except Exception:  # noqa: BLE001 — Pydantic raises ValidationError + subclasses
            return cls(hook_event_name=UNKNOWN_EVENT)
