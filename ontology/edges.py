"""Edge type constants for the Galvo memory ontology (design §4).

Design §4 enumerates 16 edge types. They split into two categories:

1. **Custom edges (15)** — domain semantics for the code/dev ontology. These are
   plain Cypher relationships with no library coupling; ``neo4j-agent-memory``
   does not model them and the library's MCP tools never touch them directly.
   The sidecar writes them through ``client.graph.execute_write(...)`` raw
   Cypher in Phase 2B.
2. **Feedback edge (1)** — ``RETRIEVED_IN``. This is the D5 cycle-1 logging
   surface; every retrieval writes one edge per hit, pointing the retrieved
   node at the current :Session. SessionEnd (Task 10) fills the
   ``utility_score`` property.

Edge type names are SHOUTY_SNAKE_CASE per Neo4j convention. They are used as
identifiers in Cypher MATCH/MERGE clauses, so any rename ripples through
every helper and the sidecar's feedback writer (Task 9). Treat the constants
as load-bearing.

References:
    * Design §4 "Edge types"
    * Design §5 "Per-retrieval logging"
    * Phase 2 plan §Task 3
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Custom edges — design §4. Authored edge order matches the design doc so a
# reader can diff the design against the code without scanning.
# ---------------------------------------------------------------------------

#: ``Decision -[LED_TO]-> Commit | Failure | Belief`` — observed outcome of a
#: choice. Polymorphic target; queries should match on the target label.
EDGE_LED_TO: Final[str] = "LED_TO"

#: ``Decision -[CONSIDERED]-> Decision`` — alternative considered but not
#: chosen. The "rejected" decision still exists as a node so the rationale
#: chain is queryable.
EDGE_CONSIDERED: Final[str] = "CONSIDERED"

#: ``Decision -[BASED_ON]-> Belief`` — the belief that justified the decision.
#: When the belief is invalidated (SUPERSEDES chain), the decision becomes
#: suspect.
EDGE_BASED_ON: Final[str] = "BASED_ON"

#: ``Pattern -[OBSERVED_IN]-> Session`` — sessions where the pattern fired,
#: used to compute Pattern.evidence_count + Pattern.success_rate.
EDGE_OBSERVED_IN: Final[str] = "OBSERVED_IN"

#: ``Pattern -[CONTRADICTS]-> Pattern`` — two patterns that cannot both be
#: true; one of them is wrong (or context-specific).
EDGE_CONTRADICTS: Final[str] = "CONTRADICTS"

#: ``Mistake -[CORRECTED_BY]-> Commit`` — the commit that fixed the mistake.
EDGE_CORRECTED_BY: Final[str] = "CORRECTED_BY"

#: ``Mistake -[CAUSED]-> Failure`` — the failure that surfaced the mistake.
EDGE_CAUSED: Final[str] = "CAUSED"

#: ``Convention -[APPLIES_TO]-> Artifact`` — explicit scoping of a convention
#: to a particular file / module.
EDGE_APPLIES_TO: Final[str] = "APPLIES_TO"

#: ``Belief -[SUPERSEDES]-> Belief`` — temporal-validity chain. New belief
#: invalidates old (the predecessor's ``valid_to`` is set when the edge is
#: created). The SUPERSEDES edge carries ``valid_from`` so belief-timeline
#: queries don't need to follow the chain end-to-end.
EDGE_SUPERSEDES: Final[str] = "SUPERSEDES"

#: ``Belief -[VALIDATED_BY]-> Test`` — empirical evidence backing the belief.
#: A green test maps to belief confidence; a failing test triggers a
#: SUPERSEDES candidate.
EDGE_VALIDATED_BY: Final[str] = "VALIDATED_BY"

#: ``Session -[WORKED_ON]-> Task`` — what the session set out to do.
EDGE_WORKED_ON: Final[str] = "WORKED_ON"

#: ``Session -[TOUCHED]-> Artifact`` — every file read/edited/written during
#: the session. The PostToolUse hook (Task 14) writes these.
EDGE_TOUCHED: Final[str] = "TOUCHED"

#: ``Session -[PRODUCED]-> Commit`` — commits authored during the session.
EDGE_PRODUCED: Final[str] = "PRODUCED"

#: ``Commit -[REVERTED]-> Commit`` — reverts another commit. Both commits
#: remain in the graph (the revert is itself a Commit node).
EDGE_REVERTED: Final[str] = "REVERTED"

#: ``Task -[BLOCKED_BY]-> Constraint`` — hard requirement keeping the task
#: from completing.
EDGE_BLOCKED_BY: Final[str] = "BLOCKED_BY"


# ---------------------------------------------------------------------------
# Feedback edge — design §5, D5 cycle 1.
# ---------------------------------------------------------------------------

#: ``* -[RETRIEVED_IN]-> Session`` — written by the sidecar's feedback logger
#: every time a search returns a hit. Properties:
#:
#: * ``retrieval_rank: int`` (0-indexed, 0 = top hit)
#: * ``retrieval_score: float`` (vector similarity from the library)
#: * ``retrieval_context: str`` (the query that produced the hit)
#: * ``created_at: datetime`` (when the edge was written)
#: * ``utility_score: float | null`` (range [-1, +1]; populated at
#:   SessionEnd by Task 10's scorer per design §5)
EDGE_RETRIEVED_IN: Final[str] = "RETRIEVED_IN"


# ---------------------------------------------------------------------------
# Registries — keep in sync with the constants above. Tests assert
# cardinalities (15 + 1 = 16) so a missed entry trips a guard.
# ---------------------------------------------------------------------------

#: All 15 custom edges. Excludes ``RETRIEVED_IN`` — that's a feedback edge
#: with library-aware semantics, not a domain edge. Consolidation in cycle 2
#: walks ``RETRIEVED_IN`` to compute utility; cycle-1 helpers iterating "real"
#: graph edges should iterate over this list.
CUSTOM_EDGE_TYPES: Final[list[str]] = [
    EDGE_LED_TO,
    EDGE_CONSIDERED,
    EDGE_BASED_ON,
    EDGE_OBSERVED_IN,
    EDGE_CONTRADICTS,
    EDGE_CORRECTED_BY,
    EDGE_CAUSED,
    EDGE_APPLIES_TO,
    EDGE_SUPERSEDES,
    EDGE_VALIDATED_BY,
    EDGE_WORKED_ON,
    EDGE_TOUCHED,
    EDGE_PRODUCED,
    EDGE_REVERTED,
    EDGE_BLOCKED_BY,
]

#: Every edge type, custom + feedback. 16 entries per design §4.
ALL_EDGE_TYPES: Final[list[str]] = [*CUSTOM_EDGE_TYPES, EDGE_RETRIEVED_IN]


# ---------------------------------------------------------------------------
# Edge property index names — referenced by ``edges.cypher`` so a rename in
# either file fails fast at test time rather than silently drifting.
# ---------------------------------------------------------------------------

#: Index on ``RETRIEVED_IN.utility_score`` — supports D5 cycle-2 consolidation
#: queries that fetch "high-utility memories" + "low-utility candidates for
#: demotion" in a single pass.
INDEX_RETRIEVED_IN_UTILITY: Final[str] = "retrieved_in_utility_idx"

#: Index on ``RETRIEVED_IN.retrieval_rank`` — supports retrieval-rank
#: distribution analysis (e.g. "what fraction of top-1 hits were useful?").
INDEX_RETRIEVED_IN_RANK: Final[str] = "retrieved_in_rank_idx"

#: Index on ``SUPERSEDES.valid_from`` — supports belief-timeline queries that
#: ask "what was true at time T?" without walking the entire chain.
INDEX_SUPERSEDES_VALID_FROM: Final[str] = "supersedes_valid_from_idx"

#: All edge property indexes created by ``edges.cypher``. Tests assert each
#: appears in ``SHOW INDEXES`` after a successful apply.
ALL_EDGE_INDEX_NAMES: Final[list[str]] = [
    INDEX_RETRIEVED_IN_UTILITY,
    INDEX_RETRIEVED_IN_RANK,
    INDEX_SUPERSEDES_VALID_FROM,
]
