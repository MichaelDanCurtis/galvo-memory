"""Galvo memory layer — sidecar package.

Phase 2B/2C surface area: the FastAPI service on ``:7575`` that owns the
``MemoryClient``, scope partitioning, REST CRUD per node label, feedback
edge logging, and SessionEnd scoring.

Modules shipped so far:

* :mod:`cypher_helpers` (Task 5) — pure-Python composition helpers every
  downstream endpoint calls when it needs to build a scope-aware query.
* :mod:`config` (Task 6) — :class:`SidecarSettings` (Pydantic) for the
  runtime knobs the FastAPI app reads at boot.
* :mod:`app` (Task 6) — the FastAPI app + ``/health`` smoke endpoint.
* :mod:`deps` (Task 7) — FastAPI dependency providers (``MemoryDep`` +
  ``SettingsDep``) that route handlers use to receive the active client
  and settings.

Later tasks add :mod:`routers`, :mod:`feedback`, :mod:`scoring`.
"""
