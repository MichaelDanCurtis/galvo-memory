"""Task 2 — per-label property constraints + indexes.

Verifies that :func:`ontology.apply_properties.apply_properties` provisions
the expected constraints and indexes against the live Neo4j substrate, and
that the operation is idempotent (re-running is a no-op).

These tests REQUIRE a live Neo4j on ``bolt://localhost:7687`` (the dev
substrate at ``memory/docker/docker-compose.yml``). They are marked
``@pytest.mark.integration`` for parity with ``test_label_mapping.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ontology.apply_properties import (
    PROPERTIES_CYPHER_PATH,
    _parse_statements,
    apply_properties,
)
from ontology.label_mapping import _build_memory_settings

if TYPE_CHECKING:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Constraints + indexes we expect after a successful apply.
#
# Every entry is the schema-element ``name`` that the CREATE statement
# in ``properties.cypher`` assigns. Keeping these as module constants gives
# the tests a clear ledger of intent vs. what landed in the database.
# ---------------------------------------------------------------------------

EXPECTED_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "galvo_decision_id_unique",
        "galvo_pattern_id_unique",
        "galvo_convention_id_unique",
        "galvo_constraint_id_unique",
        "galvo_task_id_unique",
        "galvo_session_id_unique",
        "galvo_mistake_id_unique",
        "galvo_commit_sha_unique",
        "galvo_failure_id_unique",
        "galvo_artifact_id_unique",
        "galvo_test_id_unique",
        "galvo_belief_id_unique",
    }
)

EXPECTED_INDEXES: frozenset[str] = frozenset(
    {
        # Universal
        "galvo_node_scope_idx",
        # Decision
        "galvo_decision_confidence_idx",
        "galvo_decision_scope_idx",
        # Pattern
        "galvo_pattern_success_rate_idx",
        "galvo_pattern_codebase_scope_idx",
        "galvo_pattern_scope_idx",
        # Convention
        "galvo_convention_source_idx",
        "galvo_convention_strength_idx",
        "galvo_convention_scope_idx",
        # Constraint
        "galvo_constraint_type_idx",
        "galvo_constraint_source_idx",
        "galvo_constraint_scope_idx",
        # Task
        "galvo_task_status_idx",
        "galvo_task_priority_idx",
        "galvo_task_scope_idx",
        # Session
        "galvo_session_started_at_idx",
        "galvo_session_ended_at_idx",
        "galvo_session_agent_tool_idx",
        "galvo_session_outcome_idx",
        "galvo_session_scope_idx",
        # Mistake
        "galvo_mistake_scope_idx",
        # Commit
        "galvo_commit_intent_idx",
        "galvo_commit_reverted_by_idx",
        "galvo_commit_scope_idx",
        # Failure
        "galvo_failure_type_idx",
        "galvo_failure_error_signature_idx",
        "galvo_failure_resolved_idx",
        "galvo_failure_scope_idx",
        # Artifact
        "galvo_artifact_path_idx",
        "galvo_artifact_language_idx",
        "galvo_artifact_last_touched_idx",
        "galvo_artifact_scope_idx",
        # Test
        "galvo_test_identifier_idx",
        "galvo_test_last_run_status_idx",
        "galvo_test_last_run_at_idx",
        "galvo_test_scope_idx",
        # Belief
        "galvo_belief_valid_to_idx",
        "galvo_belief_valid_from_idx",
        "galvo_belief_confidence_idx",
        "galvo_belief_source_session_id_idx",
        "galvo_belief_scope_idx",
    }
)


# ---------------------------------------------------------------------------
# Unit tests — no Neo4j required.
# ---------------------------------------------------------------------------


def test_properties_cypher_file_exists() -> None:
    """DDL file ships alongside ``apply_properties.py``."""
    assert PROPERTIES_CYPHER_PATH.is_file(), (
        f"properties.cypher missing at {PROPERTIES_CYPHER_PATH!s}; "
        f"Task 2 deliverable not committed."
    )


def test_parse_statements_strips_comments_and_splits_on_semicolon() -> None:
    """Parser must drop ``//`` comments + split top-level semicolons."""
    source = """
    // top-level comment
    CREATE INDEX foo_idx IF NOT EXISTS FOR (n:Foo) ON (n.bar);
    // another comment
    CREATE CONSTRAINT bar_unique IF NOT EXISTS FOR (n:Bar) REQUIRE n.id IS UNIQUE; // inline
    """
    statements = _parse_statements(source)
    assert len(statements) == 2, f"Expected 2 statements, got {len(statements)}: {statements}"
    assert "CREATE INDEX foo_idx" in statements[0]
    assert "CREATE CONSTRAINT bar_unique" in statements[1]
    # No leftover comment fragments.
    for s in statements:
        assert "//" not in s, f"Comment leaked into statement: {s!r}"


def test_parse_statements_real_file_has_expected_count() -> None:
    """The DDL file should parse into one statement per expected schema element."""
    source = PROPERTIES_CYPHER_PATH.read_text(encoding="utf-8")
    statements = _parse_statements(source)
    expected = len(EXPECTED_CONSTRAINTS) + len(EXPECTED_INDEXES)
    assert len(statements) == expected, (
        f"properties.cypher parsed into {len(statements)} statements, "
        f"expected {expected} (constraints + indexes). Drift between the "
        f"DDL file and the test ledger — update one or the other."
    )


def test_expected_constraints_cover_all_twelve_labels() -> None:
    """Sanity: 12 unique-identity constraints, one per label."""
    assert len(EXPECTED_CONSTRAINTS) == 12, (
        f"Expected exactly 12 constraints (one per label), got "
        f"{len(EXPECTED_CONSTRAINTS)}: {sorted(EXPECTED_CONSTRAINTS)}"
    )


# ---------------------------------------------------------------------------
# Integration tests — require live Neo4j.
# ---------------------------------------------------------------------------


async def _show_schema_names(kind: str) -> set[str]:
    """Return the set of schema-element ``name`` values for the given kind.

    Args:
        kind: Either ``"CONSTRAINTS"`` or ``"INDEXES"``.
    """
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    query = f"SHOW {kind} YIELD name RETURN name"
    async with MemoryClient(settings=_build_memory_settings()) as client:
        rows = await client.graph.execute_read(query, {})
    return {row["name"] for row in rows}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_constraints_applied() -> None:
    """After apply_properties(), SHOW CONSTRAINTS lists every expected name."""
    n_applied = await apply_properties()
    assert n_applied > 0, "apply_properties() returned zero — DDL file empty?"

    actual = await _show_schema_names("CONSTRAINTS")
    missing = EXPECTED_CONSTRAINTS - actual
    assert not missing, (
        f"Expected constraints missing after apply_properties(): {sorted(missing)}. "
        f"Got: {sorted(actual)}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_indexes_applied() -> None:
    """After apply_properties(), SHOW INDEXES lists every expected name."""
    await apply_properties()

    actual = await _show_schema_names("INDEXES")
    missing = EXPECTED_INDEXES - actual
    assert not missing, (
        f"Expected indexes missing after apply_properties(): {sorted(missing)}. "
        f"Got: {sorted(actual)}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_idempotent() -> None:
    """Calling apply_properties() twice produces no errors and yields the
    same schema-element count.

    All DDL statements use ``IF NOT EXISTS`` so the second invocation must
    succeed and must NOT create duplicate constraints/indexes.
    """
    # First apply — may be a no-op if test_constraints_applied ran first.
    await apply_properties()
    constraints_first = await _show_schema_names("CONSTRAINTS")
    indexes_first = await _show_schema_names("INDEXES")

    # Second apply — must not raise.
    await apply_properties()
    constraints_second = await _show_schema_names("CONSTRAINTS")
    indexes_second = await _show_schema_names("INDEXES")

    assert constraints_second == constraints_first, (
        f"Re-running apply_properties() changed constraints. "
        f"Added: {constraints_second - constraints_first}; "
        f"Removed: {constraints_first - constraints_second}"
    )
    assert indexes_second == indexes_first, (
        f"Re-running apply_properties() changed indexes. "
        f"Added: {indexes_second - indexes_first}; "
        f"Removed: {indexes_first - indexes_second}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_returns_statement_count() -> None:
    """apply_properties() returns the number of statements parsed from
    properties.cypher — useful for surfacing the "what just happened" count
    in CLI output and integration logs.
    """
    n_applied = await apply_properties()
    expected = len(EXPECTED_CONSTRAINTS) + len(EXPECTED_INDEXES)
    assert n_applied == expected, (
        f"apply_properties() returned {n_applied} statements, "
        f"expected {expected} (constraints + indexes per the test ledger)."
    )
