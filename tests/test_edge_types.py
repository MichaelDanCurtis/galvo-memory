"""Task 3 — edge type constants + RETRIEVED_IN feedback edge.

Two tiers:

* **Dict-level invariants** (no Neo4j) — assert design §4 cardinality
  (15 custom + 1 feedback = 16), naming conventions, registry membership.
* **Integration** (live Neo4j on ``bolt://localhost:7687``) — apply
  ``edges.cypher`` and assert ``SHOW INDEXES`` returns the three declared
  indexes, plus idempotency.
"""

from __future__ import annotations

import re

import pytest

from ontology.edges import (
    ALL_EDGE_INDEX_NAMES,
    ALL_EDGE_TYPES,
    CUSTOM_EDGE_TYPES,
    EDGE_APPLIES_TO,
    EDGE_BASED_ON,
    EDGE_BLOCKED_BY,
    EDGE_CAUSED,
    EDGE_CONSIDERED,
    EDGE_CONTRADICTS,
    EDGE_CORRECTED_BY,
    EDGE_LED_TO,
    EDGE_OBSERVED_IN,
    EDGE_PRODUCED,
    EDGE_RETRIEVED_IN,
    EDGE_REVERTED,
    EDGE_SUPERSEDES,
    EDGE_TOUCHED,
    EDGE_VALIDATED_BY,
    EDGE_WORKED_ON,
    INDEX_RETRIEVED_IN_RANK,
    INDEX_RETRIEVED_IN_UTILITY,
    INDEX_SUPERSEDES_VALID_FROM,
)


# ---------------------------------------------------------------------------
# Dict-level invariants — no Neo4j required.
# ---------------------------------------------------------------------------


def test_all_16_edges_defined() -> None:
    """Design §4 enumerates exactly 16 edge types (15 custom + RETRIEVED_IN)."""
    assert len(ALL_EDGE_TYPES) == 16, (
        f"Expected 16 edge types per design §4, got {len(ALL_EDGE_TYPES)}: "
        f"{ALL_EDGE_TYPES}"
    )


def test_15_custom_edges_defined() -> None:
    """The 15 custom edges enumerated in design §4 (excluding RETRIEVED_IN)."""
    assert len(CUSTOM_EDGE_TYPES) == 15, (
        f"Expected 15 custom edges per design §4, got {len(CUSTOM_EDGE_TYPES)}: "
        f"{CUSTOM_EDGE_TYPES}"
    )


def test_retrieved_in_separate_from_custom() -> None:
    """RETRIEVED_IN is the feedback edge — must NOT appear in CUSTOM_EDGE_TYPES.

    Helpers that iterate "real" domain edges (e.g. consolidation reasoning
    about Decision→LED_TO→Outcome chains) iterate CUSTOM_EDGE_TYPES. Including
    RETRIEVED_IN there would let the consolidator walk its own breadcrumbs.
    """
    assert EDGE_RETRIEVED_IN not in CUSTOM_EDGE_TYPES, (
        "RETRIEVED_IN must not appear in CUSTOM_EDGE_TYPES — it's the D5 "
        "feedback edge, not a domain edge."
    )
    assert EDGE_RETRIEVED_IN in ALL_EDGE_TYPES, (
        "RETRIEVED_IN must appear in ALL_EDGE_TYPES (it's still one of the 16)."
    )


def test_edge_names_are_shouty_snake_case() -> None:
    """Neo4j convention: relationship types are UPPER_SNAKE_CASE.

    Mixing case here would let one helper match ``LED_TO`` and another match
    ``led_to`` and they'd be different edge types in the graph.
    """
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*[A-Z0-9]$")
    for edge_type in ALL_EDGE_TYPES:
        assert pattern.match(edge_type), (
            f"Edge type {edge_type!r} is not SHOUTY_SNAKE_CASE."
        )


def test_no_duplicate_edge_types() -> None:
    """A typo in one constant could give two constants the same string value."""
    assert len(set(ALL_EDGE_TYPES)) == len(ALL_EDGE_TYPES), (
        f"Duplicate edge type strings: {ALL_EDGE_TYPES}"
    )


def test_design_edges_all_present() -> None:
    """Spot-check every individual constant resolves to the expected string.

    This is the canonical "did the design diff cleanly?" test. If design §4
    ever adds or renames an edge, this is the first test that fails.
    """
    expected = {
        EDGE_LED_TO: "LED_TO",
        EDGE_CONSIDERED: "CONSIDERED",
        EDGE_BASED_ON: "BASED_ON",
        EDGE_OBSERVED_IN: "OBSERVED_IN",
        EDGE_CONTRADICTS: "CONTRADICTS",
        EDGE_CORRECTED_BY: "CORRECTED_BY",
        EDGE_CAUSED: "CAUSED",
        EDGE_APPLIES_TO: "APPLIES_TO",
        EDGE_SUPERSEDES: "SUPERSEDES",
        EDGE_VALIDATED_BY: "VALIDATED_BY",
        EDGE_WORKED_ON: "WORKED_ON",
        EDGE_TOUCHED: "TOUCHED",
        EDGE_PRODUCED: "PRODUCED",
        EDGE_REVERTED: "REVERTED",
        EDGE_BLOCKED_BY: "BLOCKED_BY",
        EDGE_RETRIEVED_IN: "RETRIEVED_IN",
    }
    for constant, expected_value in expected.items():
        assert constant == expected_value, (
            f"Constant resolves to {constant!r}, expected {expected_value!r}"
        )


def test_index_names_registry_lists_three() -> None:
    """Three edge property indexes per ``edges.cypher``."""
    assert len(ALL_EDGE_INDEX_NAMES) == 3, (
        f"Expected 3 edge indexes, got {len(ALL_EDGE_INDEX_NAMES)}: "
        f"{ALL_EDGE_INDEX_NAMES}"
    )
    assert set(ALL_EDGE_INDEX_NAMES) == {
        INDEX_RETRIEVED_IN_UTILITY,
        INDEX_RETRIEVED_IN_RANK,
        INDEX_SUPERSEDES_VALID_FROM,
    }


# ---------------------------------------------------------------------------
# Live-Neo4j integration tests — actually apply the DDL.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_index_applied() -> None:
    """After ``apply_edge_indexes()``, ``SHOW INDEXES`` includes all three."""
    from ontology.apply_edges import apply_edge_indexes, list_edge_indexes

    statements = await apply_edge_indexes()
    assert len(statements) == 3, (
        f"Expected 3 statements from edges.cypher, got {len(statements)}: "
        f"{statements}"
    )

    rows = await list_edge_indexes()
    present = {row["name"] for row in rows}

    assert INDEX_RETRIEVED_IN_UTILITY in present, (
        f"{INDEX_RETRIEVED_IN_UTILITY!r} not present in SHOW INDEXES output. "
        f"Found: {sorted(present)}"
    )
    assert INDEX_RETRIEVED_IN_RANK in present, (
        f"{INDEX_RETRIEVED_IN_RANK!r} not present in SHOW INDEXES output. "
        f"Found: {sorted(present)}"
    )
    assert INDEX_SUPERSEDES_VALID_FROM in present, (
        f"{INDEX_SUPERSEDES_VALID_FROM!r} not present in SHOW INDEXES output. "
        f"Found: {sorted(present)}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_index_targets_correct_relationships() -> None:
    """Verify each index is on the right relationship type + property.

    Otherwise we could ship a typo'd Cypher that creates a misnamed index
    pointing at the wrong property and tests-against-name alone wouldn't
    catch it.
    """
    from ontology.apply_edges import apply_edge_indexes, list_edge_indexes

    await apply_edge_indexes()
    rows = await list_edge_indexes()
    by_name = {row["name"]: row for row in rows}

    # All three should be RELATIONSHIP indexes, not NODE.
    for name in (
        INDEX_RETRIEVED_IN_UTILITY,
        INDEX_RETRIEVED_IN_RANK,
        INDEX_SUPERSEDES_VALID_FROM,
    ):
        row = by_name[name]
        assert row["entityType"] == "RELATIONSHIP", (
            f"Index {name!r} has entityType={row['entityType']!r}, "
            f"expected 'RELATIONSHIP'."
        )

    assert by_name[INDEX_RETRIEVED_IN_UTILITY]["labelsOrTypes"] == [EDGE_RETRIEVED_IN]
    assert by_name[INDEX_RETRIEVED_IN_UTILITY]["properties"] == ["utility_score"]

    assert by_name[INDEX_RETRIEVED_IN_RANK]["labelsOrTypes"] == [EDGE_RETRIEVED_IN]
    assert by_name[INDEX_RETRIEVED_IN_RANK]["properties"] == ["retrieval_rank"]

    assert by_name[INDEX_SUPERSEDES_VALID_FROM]["labelsOrTypes"] == [EDGE_SUPERSEDES]
    assert by_name[INDEX_SUPERSEDES_VALID_FROM]["properties"] == ["valid_from"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_idempotent() -> None:
    """Second call must succeed without raising — every statement uses
    ``IF NOT EXISTS``."""
    from ontology.apply_edges import apply_edge_indexes, list_edge_indexes

    await apply_edge_indexes()
    await apply_edge_indexes()  # second call — must not raise

    rows = await list_edge_indexes()
    # Still exactly three; no duplicates introduced.
    names = [row["name"] for row in rows]
    assert sorted(names) == sorted(ALL_EDGE_INDEX_NAMES), (
        f"After two applies, SHOW INDEXES shows {sorted(names)}, "
        f"expected {sorted(ALL_EDGE_INDEX_NAMES)}."
    )
