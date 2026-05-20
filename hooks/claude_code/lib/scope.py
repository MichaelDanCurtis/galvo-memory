"""Hook-friendly scope detection (Task 11).

Wraps :mod:`scope.detector` with a hook-specific entry point that takes
the cwd Claude Code provides in the hook's stdin JSON. Hooks may not
run in the same working directory as the user's shell prompt — Claude
Code's dispatcher sometimes invokes hooks with a different cwd than the
user sees (especially during SessionStart, before the user has CD'd
anywhere). The hook input's ``cwd`` field is the authoritative source.

This module is intentionally tiny — it exists so the hook scripts don't
have to know about the ``sys.path`` dance required to import the
:mod:`scope` package from outside the memory repo (hook scripts deploy
to ``~/.claude/hooks/`` and run with whatever cwd Claude Code picks,
so they can't rely on the working directory matching the package
layout).

The ``sys.path`` insertion at module top resolves the ``scope`` import
when this file is imported via its in-repo path (`memory/hooks/...`).
When deployed under `~/.claude/hooks/`, the deployment step is
responsible for either symlinking the ``scope`` package next to the
hook scripts or setting ``PYTHONPATH`` appropriately — Task 12+ owns
that deploy machinery.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "detect_scope_for_hook",
]

# The hooks package sits at ``memory/hooks/claude_code/lib/``; its
# fourth parent is ``memory/``, which contains the ``scope/`` sibling
# package. We splice ``memory/`` onto sys.path so ``import scope`` works
# regardless of where the file was invoked from.
#
# A simpler ``from ...scope.detector import detect_scope`` would only
# work when the file is imported as ``memory.hooks.claude_code.lib.scope``,
# which is not how the deployed hooks see it (they run as plain scripts).
# The sys.path approach is robust to both call patterns.
_MEMORY_DIR = Path(__file__).resolve().parents[3]
if str(_MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(_MEMORY_DIR))

# ruff: noqa: E402 — sys.path manipulation must precede the import
from scope.detector import detect_scope  # type: ignore[import-not-found]


def detect_scope_for_hook(cwd: str | None) -> str:
    """Resolve the scope string for a hook invocation.

    Args:
        cwd: The ``cwd`` value from the hook's stdin JSON. May be
            ``None`` if Claude Code didn't pass one (rare; we fall
            back to :func:`pathlib.Path.cwd` in that case).

    Returns:
        One of ``"project:<repo-id>"`` / ``"personal"`` / ``"universal"``
        per :func:`scope.detector.detect_scope`. The returned string is
        the canonical scope value used by sidecar queries and writes —
        passing it to :meth:`SidecarClient.search` as the ``scope=``
        kwarg gets the right design-§D4 filter applied server-side.

    The ``None``-fallback to :func:`Path.cwd` is mostly defensive — in
    practice, Claude Code's hook input always includes ``cwd``. Tests
    cover the explicit-cwd path; the fallback is exercised by the
    "no cwd in input" test only.
    """
    target = Path(cwd) if cwd else Path.cwd()
    return detect_scope(target)
