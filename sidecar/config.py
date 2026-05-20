"""Sidecar runtime settings — read from environment with sane local-dev defaults.

The sidecar (FastAPI on ``:7575``) owns the :class:`MemoryClient` connection
to Neo4j plus the sentence-transformers embedder. Every knob that varies
between local development, Docker Compose, and (eventually) production lives
here so :mod:`sidecar.app` can stay declarative.

Defaults match :mod:`memory.docker/docker-compose.yml` so a developer who
``docker compose up -d``\\s the Neo4j substrate and then ``uvicorn
sidecar.app:app``\\s the sidecar gets a working stack without setting any
environment variables. Container deploys override via env vars prefixed
``GALVO_MEMORY_SIDECAR_`` — for example ``GALVO_MEMORY_SIDECAR_NEO4J_URI``
maps to :attr:`SidecarSettings.neo4j_uri`.

The defaults intentionally mirror the constants in
:mod:`ontology.label_mapping`. We don't import them at module level because
:class:`pydantic_settings.BaseSettings` evaluates field defaults at class-body
time, and the ontology module pulls in ``neo4j_agent_memory`` lazily — keeping
the values inlined here means importing :mod:`sidecar.config` never triggers
the embedder model download.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SidecarSettings(BaseSettings):
    """All runtime knobs for the FastAPI sidecar.

    Override any field via env vars prefixed ``GALVO_MEMORY_SIDECAR_``. A
    ``.env`` file in the working directory is also read (matches the
    Phase-2 plan's "no global state" rule — settings are per-process).
    Unknown env vars are silently ignored so the larger Galvo env doesn't
    bleed in.
    """

    model_config = SettingsConfigDict(
        env_prefix="GALVO_MEMORY_SIDECAR_",
        env_file=".env",
        extra="ignore",
    )

    # --- bind -------------------------------------------------------------

    host: str = "0.0.0.0"
    """Interface the uvicorn worker binds to. ``0.0.0.0`` for container
    deploys; flip to ``127.0.0.1`` if running on a shared dev host."""

    port: int = 7575
    """TCP port. Acceptance gate §1 hardcodes ``:7575`` so the hook layer
    can hit ``http://localhost:7575/health`` without env discovery."""

    # --- Neo4j ------------------------------------------------------------

    neo4j_uri: str = "bolt://localhost:7687"
    """Bolt URI. Matches the docker-compose mapping so local dev "just works"."""

    neo4j_user: str = "neo4j"
    """Neo4j user. Single-tenant in cycle 1; per-tenant in cycle 2 (Hub
    federation)."""

    neo4j_password: str = "galvo-memory-dev-2026"
    """Dev password — matches ``memory/docker/docker-compose.yml``. Container
    deploys MUST override via ``GALVO_MEMORY_SIDECAR_NEO4J_PASSWORD``."""

    neo4j_database: str = "neo4j"
    """Database name. ``neo4j`` is the Community-edition default; cycle 2
    multi-tenant may shard per project."""

    # --- embedder ---------------------------------------------------------

    embedding_model: str = "all-MiniLM-L6-v2"
    """sentence-transformers model name. D2 locked for cycle 1 — Qwen3-8B
    swap deferred. Model is downloaded on first ``MemoryClient.connect()``."""

    embedding_dimensions: int = 384
    """Vector dimensions. MUST match :data:`ontology.label_mapping.
    EXPECTED_VECTOR_DIMENSIONS` or :meth:`SchemaManager.adopt_existing_graph`
    will reject the embedder at boot."""
