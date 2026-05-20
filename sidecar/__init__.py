"""Galvo memory layer — sidecar package.

Phase 2B/2C surface area: the FastAPI service on ``:7575`` that owns the
``MemoryClient``, scope partitioning, REST CRUD per node label, feedback
edge logging, and SessionEnd scoring. Phase 2A (this task) only ships
:mod:`cypher_helpers`, the pure-Python composition helpers every downstream
sidecar endpoint will call when it needs to build a scope-aware query.

Nothing else is exported yet; later tasks add :mod:`app`, :mod:`deps`,
:mod:`routers`, :mod:`feedback`, :mod:`scoring`.
"""
