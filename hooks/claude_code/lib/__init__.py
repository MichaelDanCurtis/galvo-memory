"""Shared library imported by every Claude Code lifecycle hook (Task 11).

Public API:

* :class:`hooks.claude_code.lib.sidecar_client.SidecarClient` — synchronous
  HTTP client wrapping the sidecar's :7575 surface with tight timeouts and
  graceful degradation. Returns sentinel values (``None`` / ``[]``) on
  failure instead of raising — hooks MUST NOT block or fail the session.
* :class:`hooks.claude_code.lib.types.HookInputBase` — Pydantic model for
  the JSON shape Claude Code dumps to a hook's stdin. Has a
  :meth:`HookInputBase.from_stdin` factory that swallows malformed input.
* :func:`hooks.claude_code.lib.logging.setup_hook_logger` — rotating-file
  logger to ``~/.galvo-memory/logs/{name}.log``. Hooks must not pollute
  stdout/stderr (Claude Code captures both into the session transcript).
* :func:`hooks.claude_code.lib.scope.detect_scope_for_hook` — thin wrapper
  around :func:`scope.detector.detect_scope` that takes the cwd Claude Code
  passes in the hook input JSON.

Design notes (see :doc:`memory/docs/PHASE-2-PLAN.md` Task 11 for the full
rationale):

* **Synchronous HTTP, not async.** Hooks are invoked synchronously by
  Claude Code's lifecycle dispatcher (it shells out and waits for the
  process to exit). Async would complicate the integration with no
  meaningful concurrency win — each hook makes one or two HTTP calls.
* **Stdlib ``urllib``, not ``httpx``.** Hooks run in arbitrary user
  environments where the sidecar's ``[sidecar]`` extra is not necessarily
  installed. ``urllib`` is stdlib — zero deps. The ergonomics tradeoff is
  acceptable for cycle 1; cycle 2 may switch to httpx once we've decided
  whether the hooks ship as part of the sidecar wheel or as a separate
  thin package.
* **Graceful degradation is non-negotiable.** Every method on
  :class:`SidecarClient` catches network/timeout/JSON errors and returns
  a sentinel. The hook scripts treat sentinels as "sidecar unavailable —
  no-op silently". The only visible side-effect of a failure is a line
  in ``~/.galvo-memory/logs/``.
"""
