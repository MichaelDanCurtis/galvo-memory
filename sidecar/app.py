"""Galvo Memory Sidecar — FastAPI on ``:7575``.

This module owns the long-lived :class:`MemoryClient` connection and
exposes a minimal HTTP surface. Phase-2 Task 6 ships only ``/health``;
later tasks add ``/api/{label}`` CRUD (Task 8), feedback edge writers
(Task 9), and SessionEnd scoring (Task 10).

Lifespan model:

* On startup, build a :class:`MemorySettings` from :class:`SidecarSettings`,
  open a :class:`MemoryClient`, store it on ``app.state.memory``. The
  embedder downloads + warms during ``connect()`` — first boot is slow
  (~10s for the 90MB MiniLM checkpoint).
* On shutdown, ``await client.close()`` to release the Neo4j pool.
* Tests substitute the client by patching ``MemoryClient`` in this
  module's namespace; :class:`fastapi.testclient.TestClient` runs the
  lifespan inside its ``with`` block.

Acceptance gate §1 (PHASE-2-PLAN.md) requires ``curl :7575/health`` to
return 200 with ``{"neo4j": "ok", "embedder": "loaded"}``. The
implementation here calls :meth:`MemoryClient.get_stats` as the smoke
probe — if Neo4j is unreachable the stats call raises, we wrap it in a
503 with the underlying exception's repr for diagnostic value.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from neo4j_agent_memory import (  # type: ignore[import-untyped]
    EmbeddingConfig,
    EmbeddingProvider,
    MemoryClient,
    MemorySettings,
    Neo4jConfig,
)

from sidecar.config import SidecarSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


__all__ = ["app", "lifespan"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> "AsyncIterator[None]":
    """Boot the :class:`MemoryClient`, attach it to ``app.state``, tear
    down on shutdown.

    The settings object is also attached so :mod:`sidecar.deps` (Task 7)
    can surface the active config without re-reading env vars. Tests can
    monkey-patch the module-level :class:`MemoryClient` symbol; this
    function looks it up dynamically at call time (Python resolves the
    name in the function's enclosing module on each invocation).
    """
    cfg = SidecarSettings()
    settings = MemorySettings(
        neo4j=Neo4jConfig(
            uri=cfg.neo4j_uri,
            username=cfg.neo4j_user,
            password=cfg.neo4j_password,
            database=cfg.neo4j_database,
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model=cfg.embedding_model,
            dimensions=cfg.embedding_dimensions,
        ),
    )
    client = MemoryClient(settings=settings)
    await client.connect()
    app.state.memory = client
    app.state.config = cfg
    try:
        yield
    finally:
        await client.close()


app = FastAPI(
    title="Galvo Memory Sidecar",
    version="0.1.0",
    description=(
        "Phase 2 cycle 1 — owns MemoryClient + scope partitioning + "
        "feedback logging. See memory/docs/PHASE-2-PLAN.md."
    ),
    lifespan=lifespan,
)

# Task 8 — REST CRUD for the 12 ontology labels. Imported AFTER ``app``
# is constructed so the import is cheap (the router itself only depends
# on already-imported modules) and there's no chicken-and-egg between
# the router pulling ``MemoryDep`` and the app pulling the router.
from sidecar.routers import nodes as _nodes_router  # noqa: E402

app.include_router(_nodes_router.router)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Acceptance gate §1 — Neo4j connectivity + embedder load smoke.

    Calls :meth:`MemoryClient.get_stats` as a cheap probe that exercises
    both the Bolt driver and the embedder module (the library counts
    embedded entities, so the model has to be loaded to answer). A 503
    with the underlying exception's repr is preferable to a generic 500
    because the calling hook needs to log a useful diagnostic before
    falling back to no-op mode.
    """
    try:
        stats = await app.state.memory.get_stats()
    except Exception as exc:  # noqa: BLE001 — re-raised as 503 below
        raise HTTPException(
            status_code=503,
            detail=f"unhealthy: {exc!r}",
        ) from exc
    return {
        "status": "ok",
        "neo4j": "ok",
        "embedder": app.state.config.embedding_model,
        "embedding_dimensions": app.state.config.embedding_dimensions,
        "stats": stats,
    }
