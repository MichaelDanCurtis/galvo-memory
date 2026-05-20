// Galvo Memory — Per-label property constraints + indexes (Phase 2 Task 2).
//
// Layer the design §4 property shapes onto the 12 adopted labels. Every
// statement uses ``IF NOT EXISTS`` so this script is idempotent and safe to
// re-run any number of times. ``memory/ontology/apply_properties.py`` splits
// this file on ``;`` and applies each statement via
// ``client.graph.execute_write``.
//
// Conventions:
//
//   * Every Galvo node carries ``:Entity`` (Task 1 adoption layers all 12
//     custom labels onto the library's ``:Entity`` super-label). So
//     ``CREATE INDEX FOR (n:Entity) ON (n.scope)`` covers every Galvo node
//     including Beliefs, which carry both ``:Entity`` and ``:Fact``.
//   * Identity is by ``id`` (UUID) for all 12 labels, EXCEPT ``Commit`` where
//     the SHA is the canonical identity. ``Belief`` also takes ``id`` because
//     SUPERSEDES creates new beliefs with new ids — see design §4 temporal
//     validity.
//   * Indexes lean toward more rather than fewer: cheap to maintain, expand
//     the query planner's options for the hooks + sidecar lookups in
//     Phase 2C+.
//   * Datetime properties land as ``RANGE`` indexes (Neo4j default for
//     ordered properties) — used by time-range queries in the SessionStart
//     hook (Task 12).

// ---------------------------------------------------------------------------
// Universal — every Galvo node has a scope (D4, design §4 "Scope
// partitioning"). One composite-friendly RANGE index on ``:Entity`` covers
// all 12 labels.
// ---------------------------------------------------------------------------
CREATE INDEX galvo_node_scope_idx IF NOT EXISTS FOR (n:Entity) ON (n.scope);

// ---------------------------------------------------------------------------
// Decision (CONCEPT) — non-trivial choices.
// Properties: id, title, rationale, alternatives_considered, confidence,
// scope, created_at.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_decision_id_unique IF NOT EXISTS FOR (n:Decision) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_decision_confidence_idx IF NOT EXISTS FOR (n:Decision) ON (n.confidence);
CREATE INDEX galvo_decision_scope_idx IF NOT EXISTS FOR (n:Decision) ON (n.scope);

// ---------------------------------------------------------------------------
// Pattern (CONCEPT) — recurring approaches.
// Properties: id, name, description, evidence_count, success_rate,
// codebase_scope, scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_pattern_id_unique IF NOT EXISTS FOR (n:Pattern) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_pattern_success_rate_idx IF NOT EXISTS FOR (n:Pattern) ON (n.success_rate);
CREATE INDEX galvo_pattern_codebase_scope_idx IF NOT EXISTS FOR (n:Pattern) ON (n.codebase_scope);
CREATE INDEX galvo_pattern_scope_idx IF NOT EXISTS FOR (n:Pattern) ON (n.scope);

// ---------------------------------------------------------------------------
// Convention (CONCEPT) — established way of doing things in a codebase.
// Properties: id, name, description, source (inferred/explicit/from_AGENTS.md),
// strength, scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_convention_id_unique IF NOT EXISTS FOR (n:Convention) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_convention_source_idx IF NOT EXISTS FOR (n:Convention) ON (n.source);
CREATE INDEX galvo_convention_strength_idx IF NOT EXISTS FOR (n:Convention) ON (n.strength);
CREATE INDEX galvo_convention_scope_idx IF NOT EXISTS FOR (n:Convention) ON (n.scope);

// ---------------------------------------------------------------------------
// Constraint (CONCEPT) — hard requirements.
// Properties: id, name, description, type (performance/security/...), source,
// scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_constraint_id_unique IF NOT EXISTS FOR (n:Constraint) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_constraint_type_idx IF NOT EXISTS FOR (n:Constraint) ON (n.type);
CREATE INDEX galvo_constraint_source_idx IF NOT EXISTS FOR (n:Constraint) ON (n.source);
CREATE INDEX galvo_constraint_scope_idx IF NOT EXISTS FOR (n:Constraint) ON (n.scope);

// ---------------------------------------------------------------------------
// Task (CONCEPT) — what was being worked on.
// Properties: id, title, description, status, priority, scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_task_id_unique IF NOT EXISTS FOR (n:Task) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_task_status_idx IF NOT EXISTS FOR (n:Task) ON (n.status);
CREATE INDEX galvo_task_priority_idx IF NOT EXISTS FOR (n:Task) ON (n.priority);
CREATE INDEX galvo_task_scope_idx IF NOT EXISTS FOR (n:Task) ON (n.scope);

// ---------------------------------------------------------------------------
// Session (EVENT) — a unit of work.
// Properties: id, title, started_at, ended_at, agent_tool, task_description,
// outcome, scope.
// SessionStart (Task 12) sorts by started_at DESC; SessionEnd (Task 15)
// writes ended_at. Both want range indexes.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_session_id_unique IF NOT EXISTS FOR (n:Session) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_session_started_at_idx IF NOT EXISTS FOR (n:Session) ON (n.started_at);
CREATE INDEX galvo_session_ended_at_idx IF NOT EXISTS FOR (n:Session) ON (n.ended_at);
CREATE INDEX galvo_session_agent_tool_idx IF NOT EXISTS FOR (n:Session) ON (n.agent_tool);
CREATE INDEX galvo_session_outcome_idx IF NOT EXISTS FOR (n:Session) ON (n.outcome);
CREATE INDEX galvo_session_scope_idx IF NOT EXISTS FOR (n:Session) ON (n.scope);

// ---------------------------------------------------------------------------
// Mistake (EVENT) — something that went wrong.
// Properties: id, summary, description, root_cause, fix_applied,
// time_to_discover, scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_mistake_id_unique IF NOT EXISTS FOR (n:Mistake) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_mistake_scope_idx IF NOT EXISTS FOR (n:Mistake) ON (n.scope);

// ---------------------------------------------------------------------------
// Commit (EVENT) — a code change. SHA is canonical identity (design §4).
// Properties: sha, message, intent, reverted_by (nullable), scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_commit_sha_unique IF NOT EXISTS FOR (n:Commit) REQUIRE n.sha IS UNIQUE;
CREATE INDEX galvo_commit_intent_idx IF NOT EXISTS FOR (n:Commit) ON (n.intent);
CREATE INDEX galvo_commit_reverted_by_idx IF NOT EXISTS FOR (n:Commit) ON (n.reverted_by);
CREATE INDEX galvo_commit_scope_idx IF NOT EXISTS FOR (n:Commit) ON (n.scope);

// ---------------------------------------------------------------------------
// Failure (EVENT) — a specific run failure.
// Properties: id, type (test/build/lint/runtime), error_signature, resolved,
// scope. error_signature is the dedup key for "have we seen this before?".
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_failure_id_unique IF NOT EXISTS FOR (n:Failure) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_failure_type_idx IF NOT EXISTS FOR (n:Failure) ON (n.type);
CREATE INDEX galvo_failure_error_signature_idx IF NOT EXISTS FOR (n:Failure) ON (n.error_signature);
CREATE INDEX galvo_failure_resolved_idx IF NOT EXISTS FOR (n:Failure) ON (n.resolved);
CREATE INDEX galvo_failure_scope_idx IF NOT EXISTS FOR (n:Failure) ON (n.scope);

// ---------------------------------------------------------------------------
// Artifact (OBJECT) — files / modules / components.
// Properties: id, path, language, role, last_touched, scope. path is the
// natural lookup key; not unique because different scopes can legitimately
// have the same path (e.g. ``README.md``).
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_artifact_id_unique IF NOT EXISTS FOR (n:Artifact) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_artifact_path_idx IF NOT EXISTS FOR (n:Artifact) ON (n.path);
CREATE INDEX galvo_artifact_language_idx IF NOT EXISTS FOR (n:Artifact) ON (n.language);
CREATE INDEX galvo_artifact_last_touched_idx IF NOT EXISTS FOR (n:Artifact) ON (n.last_touched);
CREATE INDEX galvo_artifact_scope_idx IF NOT EXISTS FOR (n:Artifact) ON (n.scope);

// ---------------------------------------------------------------------------
// Test (OBJECT) — a test case or last-run result.
// Properties: id, identifier, last_run_status, last_run_at, scope.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_test_id_unique IF NOT EXISTS FOR (n:Test) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_test_identifier_idx IF NOT EXISTS FOR (n:Test) ON (n.identifier);
CREATE INDEX galvo_test_last_run_status_idx IF NOT EXISTS FOR (n:Test) ON (n.last_run_status);
CREATE INDEX galvo_test_last_run_at_idx IF NOT EXISTS FOR (n:Test) ON (n.last_run_at);
CREATE INDEX galvo_test_scope_idx IF NOT EXISTS FOR (n:Test) ON (n.scope);

// ---------------------------------------------------------------------------
// Belief (FACT) — the temporal-validity node type (design §4 §"Temporal
// validity").
// Properties: id, claim, confidence, valid_from, valid_to (nullable),
// source_session_id, scope. valid_to index supports the
// "currently-active beliefs" query (WHERE valid_to IS NULL).
// ---------------------------------------------------------------------------
CREATE CONSTRAINT galvo_belief_id_unique IF NOT EXISTS FOR (n:Belief) REQUIRE n.id IS UNIQUE;
CREATE INDEX galvo_belief_valid_to_idx IF NOT EXISTS FOR (n:Belief) ON (n.valid_to);
CREATE INDEX galvo_belief_valid_from_idx IF NOT EXISTS FOR (n:Belief) ON (n.valid_from);
CREATE INDEX galvo_belief_confidence_idx IF NOT EXISTS FOR (n:Belief) ON (n.confidence);
CREATE INDEX galvo_belief_source_session_id_idx IF NOT EXISTS FOR (n:Belief) ON (n.source_session_id);
CREATE INDEX galvo_belief_scope_idx IF NOT EXISTS FOR (n:Belief) ON (n.scope);
