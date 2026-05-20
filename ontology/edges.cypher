// memory/ontology/edges.cypher
//
// Edge property indexes for the Galvo memory ontology.
//
// The 16 edge *types* themselves are not declared in Cypher — Neo4j creates
// relationship types on first MERGE. What we declare here is the supporting
// index machinery for properties we actually filter on.
//
// All statements use `IF NOT EXISTS` so this file is idempotent and safe to
// re-apply on every sidecar boot.
//
// References:
//   * Design §4 "Edge types" (the 16 enumerated relationships)
//   * Design §5 "Per-retrieval logging" (the RETRIEVED_IN feedback edge)
//   * Phase 2 plan §Task 3 ("Index RETRIEVED_IN.utility_score (Task 5 +
//     cycle-2 consolidation need it)")
//   * Phase 2 plan §Task 10 (SessionEnd scorer that populates utility_score)

// ---------------------------------------------------------------------------
// RETRIEVED_IN — the feedback edge (D5 cycle 1).
// ---------------------------------------------------------------------------

// Cycle-2 consolidation will run "give me the lowest-utility memories so we
// can demote them" and "give me the highest-utility memories so we can
// reinforce them" queries. Without this index the consolidator does a full
// scan of every RETRIEVED_IN edge ever written.
CREATE INDEX retrieved_in_utility_idx IF NOT EXISTS
  FOR ()-[r:RETRIEVED_IN]-() ON (r.utility_score);

// Retrieval-rank analysis ("what fraction of rank-0 hits were useful?")
// is a recurring sanity-check query during cycle-1 development. Cheap to
// index; the property is a small int and the index pays for itself once we
// have more than a few hundred edges.
CREATE INDEX retrieved_in_rank_idx IF NOT EXISTS
  FOR ()-[r:RETRIEVED_IN]-() ON (r.retrieval_rank);

// ---------------------------------------------------------------------------
// SUPERSEDES — the temporal-validity chain on Belief (design §4).
// ---------------------------------------------------------------------------

// Belief-timeline queries ("what was true at time T?") filter on the
// SUPERSEDES edge's valid_from property to traverse the chain without
// walking every Belief in the graph.
CREATE INDEX supersedes_valid_from_idx IF NOT EXISTS
  FOR ()-[r:SUPERSEDES]-() ON (r.valid_from);
