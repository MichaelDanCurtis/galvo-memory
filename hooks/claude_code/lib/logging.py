"""Local rotating-file logger for Claude Code hook diagnostics (Task 11).

Hooks run synchronously inside Claude Code's lifecycle dispatcher. The
dispatcher captures the hook's stdout (for prompt injection) and stderr
(into the session transcript) — neither is appropriate for diagnostic
logging because:

* Anything we write to stdout gets injected into the model's context,
  costing tokens and confusing the model.
* Anything we write to stderr ends up in the user-visible transcript,
  which is noise the operator does not want unless something has gone
  wrong.

So all hook log output goes to a rotating file under
``~/.galvo-memory/logs/`` instead. Each hook gets its own log file
(e.g. ``session_start.log``, ``user_prompt_submit.log``) so an
operator inspecting a misbehaving hook can ``tail -f`` just the
relevant stream.

Rotation defaults (1 MB × 3 backups) keep disk usage bounded; a
chatty hook will rotate frequently but never consume more than ~4 MB.

This module is import-safe — calling :func:`setup_hook_logger` creates
the log directory on demand, so the first hook to fire in a fresh
``$HOME`` doesn't crash on the missing directory.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

__all__ = [
    "LOG_DIR",
    "LOG_FORMAT",
    "MAX_BACKUP_COUNT",
    "MAX_BYTES",
    "setup_hook_logger",
]


LOG_DIR: Path = Path.home() / ".galvo-memory" / "logs"
"""Where hook log files live. Created on first :func:`setup_hook_logger` call."""

MAX_BYTES: int = 1_000_000
"""Per-file rotation threshold (1 MB)."""

MAX_BACKUP_COUNT: int = 3
"""Number of rotated backup files kept (so ~4 MB total per hook)."""

LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
"""Compact format — ISO timestamp + level + logger name + message.

Deliberately not JSON — operators are expected to ``grep`` / ``tail``
the file, and the line-per-event format is more ergonomic than JSON
for that use case. Cycle 2 may switch to structured logging once we
have an aggregation target.
"""


def setup_hook_logger(name: str) -> logging.Logger:
    """Return a logger writing to ``~/.galvo-memory/logs/{name}.log``.

    Idempotent — calling twice with the same ``name`` returns the same
    logger and does NOT double-attach handlers. This matters because
    each hook script may import the lib module multiple times during
    test bootstrap (pytest collects modules transitively); a re-attach
    would duplicate every log line.

    Args:
        name: Short identifier for the log file (no ``.log`` extension —
            the function appends it). The hook scripts use the literal
            event name: ``"session_start"``, ``"user_prompt_submit"``,
            ``"post_tool_use"``, ``"session_end"``.

    Returns:
        A :class:`logging.Logger` with a single
        :class:`logging.handlers.RotatingFileHandler` attached, level
        set to ``INFO``, propagation disabled. The logger's name is
        prefixed with ``galvo.memory.hooks.`` so it's distinguishable
        from any other logger that happens to share the short ``name``.

    Side effects:
        Creates :data:`LOG_DIR` (and its parents) if it doesn't exist.
        File creation only — never deletes existing logs.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"galvo.memory.hooks.{name}")
    logger.setLevel(logging.INFO)

    # Idempotency guard: if we already attached a handler in this
    # process, don't re-attach. We identify "our" handler by type +
    # baseFilename so a caller who attached their own handler isn't
    # disturbed.
    expected_path = str(LOG_DIR / f"{name}.log")
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and getattr(handler, "baseFilename", None) == expected_path
        ):
            return logger

    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / f"{name}.log",
        maxBytes=MAX_BYTES,
        backupCount=MAX_BACKUP_COUNT,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)

    # Stop log records from bubbling up to the root logger. The root
    # logger's default handlers (none in a fresh Python process, but
    # arbitrary in a hooked-in test environment) might write to stderr,
    # which would defeat the "no stderr pollution" rule.
    logger.propagate = False

    return logger
