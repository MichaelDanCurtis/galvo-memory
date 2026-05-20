"""Galvo memory layer — Claude Code lifecycle hooks (design §8, Phase 2C).

This package collects the shared library + the four lifecycle hook scripts
that wire a running Claude Code session into the memory sidecar on
``http://127.0.0.1:7575``. The sub-package layout::

    hooks/
      claude_code/
        lib/                — shared HTTP client + types + logger + scope
        session_start.py    — Task 12 (top-of-mind injector)
        user_prompt_submit.py — Task 13 (semantic retrieval)
        post_tool_use.py    — Task 14 (artifact/commit/test logger)
        session_end.py      — Task 15 (Session node writer + scorer)

This Task-11 commit ships only the ``lib/`` package + its tests; the four
hook scripts land in Tasks 12-15.

Naming note: the sub-package is named ``claude_code`` (underscore) so the
Python import machinery can find it (``import claude_code.lib.sidecar_client``).
The corresponding install path under ``~/.claude/hooks/`` may carry a
hyphenated form (``~/.claude/hooks/claude-code/...``) — that's a filesystem
convention controlled by Claude Code's hook loader, not by us. When we
deploy the hook scripts in Task 12+, we'll either symlink or use the
loader's documented path resolution. Either way, the *importable* package
name has to be Python-legal, hence the underscore.
"""
