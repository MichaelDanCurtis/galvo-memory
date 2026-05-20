"""Maps Galvo memory ontology labels onto neo4j-agent-memory's EntityType machinery.

Per design §4 (`memory/docs/MEMORY-LAYER-DESIGN.md`), Galvo's custom code/dev
ontology has 12 node types. The library exposes 9 entity types
(``PERSON``, ``OBJECT``, ``LOCATION``, ``EVENT``, ``ORGANIZATION``, ``CONCEPT``,
``EMOTION``, ``PREFERENCE``, ``FACT``). Adoption via
``SchemaManager.adopt_existing_graph`` layers our labels onto the library's
``:Entity`` super-label so we inherit:

* the vector index for semantic search (``entity_embedding_idx`` at 384 dims)
* the MCP toolset (``memory_store``, ``memory_search``, …)
* the duplicate-node merge logic that fires on ``(name, type)`` keys

Mapping rationale:

* ``Decision``, ``Pattern``, ``Convention``, ``Constraint``, ``Task`` →
  ``CONCEPT`` — abstract things we hold or apply
* ``Session``, ``Mistake``, ``Commit``, ``Failure`` → ``EVENT`` — things that
  happened in time
* ``Artifact``, ``Test`` → ``OBJECT`` — concrete files / resources
* ``Belief`` → ``FACT`` — the only one that explicitly aligns with the
  library's ``FACT`` entity type (temporal-validity claim per design §4)

The ``apply_ontology`` helper wraps the library's
:meth:`SchemaManager.adopt_existing_graph` so callers don't reimplement the
``Neo4jConfig`` + ``EmbeddingConfig`` boilerplate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from neo4j_agent_memory import MemorySettings  # type: ignore[import-untyped]
    from neo4j_agent_memory.schema.models import (  # type: ignore[import-untyped]
        AdoptionReport,
    )


# ---------------------------------------------------------------------------
# Canonical mapping — Task 1 deliverable.
# ---------------------------------------------------------------------------

LABEL_TO_TYPE: Final[dict[str, str]] = {
    "Decision":   "CONCEPT",
    "Pattern":    "CONCEPT",
    "Convention": "CONCEPT",
    "Constraint": "CONCEPT",
    "Task":       "CONCEPT",
    "Session":    "EVENT",
    "Mistake":    "EVENT",
    "Commit":     "EVENT",
    "Failure":    "EVENT",
    "Artifact":   "OBJECT",
    "Test":       "OBJECT",
    "Belief":     "FACT",
}

NAME_PROPERTY_PER_LABEL: Final[dict[str, str]] = {
    "Decision":   "title",          # short imperative title
    "Pattern":    "name",
    "Convention": "name",
    "Constraint": "name",
    "Task":       "title",
    "Session":    "title",
    "Mistake":    "summary",
    "Commit":     "sha",            # SHA is the canonical name
    "Failure":    "error_signature",
    "Artifact":   "path",
    "Test":       "identifier",
    "Belief":     "claim",
}

# ---------------------------------------------------------------------------
# Embedder + index constants (D2 locked: MiniLM 384-dim for cycle 1).
# ---------------------------------------------------------------------------

EXPECTED_VECTOR_DIMENSIONS: Final[int] = 384
VECTOR_INDEX_NAME: Final[str] = "entity_embedding_idx"

# Neo4j connection — matches `memory/docker/docker-compose.yml`.
NEO4J_URI: Final[str] = "bolt://localhost:7687"
NEO4J_USERNAME: Final[str] = "neo4j"
NEO4J_PASSWORD: Final[str] = "galvo-memory-dev-2026"
NEO4J_DATABASE: Final[str] = "neo4j"

# Embedder — D2 cycle 1 default.
EMBEDDER_MODEL: Final[str] = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _build_memory_settings() -> MemorySettings:
    """Build the ``MemorySettings`` used by every Task-1+ helper.

    Kept private so callers don't accidentally reach past the helper and
    diverge on connection parameters. Matches the Phase-1 spike findings
    document (`memory/docs/PHASE-1-SPIKE-FINDINGS.md`).
    """
    from neo4j_agent_memory import (  # type: ignore[import-untyped]
        EmbeddingConfig,
        EmbeddingProvider,
        MemorySettings,
        Neo4jConfig,
    )

    return MemorySettings(
        neo4j=Neo4jConfig(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model=EMBEDDER_MODEL,
            dimensions=EXPECTED_VECTOR_DIMENSIONS,
        ),
    )


async def apply_ontology(*, dry_run: bool) -> AdoptionReport:
    """Adopt the 12 Galvo labels into the library's ``:Entity`` machinery.

    Args:
        dry_run: When True, the library returns counts of what *would* be
            adopted without mutating the graph. Phase-2 Task 1 calls this
            first as a risk gate before committing.

    Returns:
        The :class:`AdoptionReport` returned by
        :meth:`SchemaManager.adopt_existing_graph`. Inspect
        ``report.by_label`` for per-label outcome counts.

    Raises:
        SchemaError: If the library rejects any of the labels in
            ``LABEL_TO_TYPE`` or ``NAME_PROPERTY_PER_LABEL`` (e.g. unsafe
            identifier characters). Hits the Task-1 decision gate — caller
            must rename or escalate to forking.
    """
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    async with MemoryClient(settings=_build_memory_settings()) as client:
        return await client.schema.adopt_existing_graph(
            LABEL_TO_TYPE,
            name_property_per_label=NAME_PROPERTY_PER_LABEL,
            dry_run=dry_run,
        )


async def drop_database() -> None:
    """Wipe every node and relationship in the Neo4j instance.

    Used by tests for a clean slate. The library's
    :meth:`SchemaManager.drop_all` removes managed schema artefacts but does
    not delete user data — for the risk-gate dry-run we want both gone.
    """
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    async with MemoryClient(settings=_build_memory_settings()) as client:
        # Delete every node + relationship (DETACH for relationships).
        await client.graph.execute_write("MATCH (n) DETACH DELETE n", {})
