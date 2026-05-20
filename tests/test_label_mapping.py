"""Task 1 — risk gate for label_to_type mapping.

Verifies that ``neo4j-agent-memory`` v0.2.1's ``SchemaManager.adopt_existing_graph``
accepts the 12-label Galvo ontology mapping defined in
``memory/ontology/label_mapping.py``.

These tests REQUIRE a live Neo4j substrate on ``bolt://localhost:7687`` —
they're marked ``@pytest.mark.integration`` so a future ``pytest -m 'not integration'``
run can skip them on machines without Docker.

Per phase-2 plan §"Decision gate at end of Task 1": if either test fails because
the library rejects the mapping (e.g. ``SchemaError`` raised on dry-run), STOP
and report the offending label.
"""

from __future__ import annotations

import pytest

from ontology.label_mapping import (
    EXPECTED_VECTOR_DIMENSIONS,
    LABEL_TO_TYPE,
    NAME_PROPERTY_PER_LABEL,
    VECTOR_INDEX_NAME,
    apply_ontology,
    drop_database,
)


# ---------------------------------------------------------------------------
# Dict-level invariants — no Neo4j required.
# ---------------------------------------------------------------------------


def test_label_to_type_has_twelve_entries() -> None:
    """Design §4 specifies exactly 12 node types."""
    assert len(LABEL_TO_TYPE) == 12, (
        f"Expected 12 labels per design §4, got {len(LABEL_TO_TYPE)}: "
        f"{sorted(LABEL_TO_TYPE)}"
    )


def test_label_to_type_values_are_library_entity_types() -> None:
    """Every value in LABEL_TO_TYPE must be a valid neo4j-agent-memory EntityType."""
    from neo4j_agent_memory import EntityType  # type: ignore[import-untyped]

    valid_types = {e.value for e in EntityType}
    for label, entity_type in LABEL_TO_TYPE.items():
        assert entity_type in valid_types, (
            f"Label {label!r} maps to invalid EntityType {entity_type!r}; "
            f"valid values are {sorted(valid_types)}"
        )


def test_name_property_per_label_covers_all_labels() -> None:
    """Every adopted label must declare its name property."""
    assert set(NAME_PROPERTY_PER_LABEL) == set(LABEL_TO_TYPE), (
        "NAME_PROPERTY_PER_LABEL keys must match LABEL_TO_TYPE keys. "
        f"Missing: {set(LABEL_TO_TYPE) - set(NAME_PROPERTY_PER_LABEL)}; "
        f"Extra: {set(NAME_PROPERTY_PER_LABEL) - set(LABEL_TO_TYPE)}"
    )


def test_no_label_collides_with_library_reserved_labels() -> None:
    """Library reserves: Entity, Message, Preference, Fact, ReasoningTrace, ReasoningStep.

    Adopting one of these as a custom label would corrupt the library's machinery.
    """
    reserved = {"Entity", "Message", "Preference", "Fact", "ReasoningTrace", "ReasoningStep"}
    collisions = set(LABEL_TO_TYPE) & reserved
    assert not collisions, (
        f"These custom labels collide with library-reserved labels: {collisions}. "
        f"Rename them before proceeding."
    )


# ---------------------------------------------------------------------------
# Live-Neo4j integration tests — risk gate.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dry_run_succeeds() -> None:
    """RISK GATE: dry-run adopt_existing_graph against live Neo4j.

    Acceptance per phase-2 plan §Task 1 step 1: the call returns an
    ``AdoptionReport`` with no raised exception. Per-label counts are zero
    because the DB is empty post ``docker compose down -v``.
    """
    await drop_database()  # ensure clean state
    report = await apply_ontology(dry_run=True)

    assert report.dry_run is True, "dry_run=True should be reflected in the report"
    assert len(report.by_label) == len(LABEL_TO_TYPE), (
        f"Expected report for all {len(LABEL_TO_TYPE)} labels, "
        f"got {len(report.by_label)}: "
        f"{[r.label for r in report.by_label]}"
    )
    # Sanity: every reported label is in our mapping.
    reported_labels = {r.label for r in report.by_label}
    assert reported_labels == set(LABEL_TO_TYPE), (
        f"Reported labels {reported_labels} != mapping labels {set(LABEL_TO_TYPE)}"
    )
    # Sanity: per-label type matches our mapping.
    for r in report.by_label:
        assert r.type == LABEL_TO_TYPE[r.label], (
            f"Label {r.label!r}: report says type={r.type!r}, "
            f"mapping says {LABEL_TO_TYPE[r.label]!r}"
        )
        assert r.name_property == NAME_PROPERTY_PER_LABEL[r.label], (
            f"Label {r.label!r}: report says name_property={r.name_property!r}, "
            f"mapping says {NAME_PROPERTY_PER_LABEL[r.label]!r}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_apply_creates_indexes() -> None:
    """Real (non-dry) adoption must leave the entity_embedding_idx vector index
    in place with the configured embedding dimension (384 for MiniLM)."""
    await drop_database()  # clean slate
    report = await apply_ontology(dry_run=False)
    assert report.dry_run is False
    assert len(report.by_label) == len(LABEL_TO_TYPE)

    # Verify the vector index exists with the right dimension.
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    from ontology.label_mapping import _build_memory_settings

    async with MemoryClient(settings=_build_memory_settings()) as client:
        # SHOW VECTOR INDEXES returns rows with name, labelsOrTypes, options.indexConfig.vector.dimensions
        rows = await client.graph.execute_read(
            "SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties, options "
            "WHERE name = $name RETURN name, labelsOrTypes, properties, options",
            {"name": VECTOR_INDEX_NAME},
        )

    assert rows, (
        f"Vector index {VECTOR_INDEX_NAME!r} not found after apply_ontology — "
        f"library did not provision its managed indexes."
    )
    row = rows[0]
    # Cypher returns options as a Map; the dimension is nested under indexConfig.
    config = row["options"]["indexConfig"]
    dim = config.get("vector.dimensions") or config.get("vector.dimension")
    assert dim == EXPECTED_VECTOR_DIMENSIONS, (
        f"Vector index dimension is {dim}, expected {EXPECTED_VECTOR_DIMENSIONS}. "
        f"This means the embedder dimensions didn't propagate — check "
        f"EmbeddingConfig.dimensions in _build_memory_settings()."
    )
