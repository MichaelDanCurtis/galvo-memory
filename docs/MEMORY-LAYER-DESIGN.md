# Memory Layer Design — Personal Dev Use → Galvo

**Author:** Michael Curtis
**Status:** Design brief for implementation
**Last updated:** May 16, 2026

---

## 1. What We're Building

A graph-native memory layer for AI coding agents. Two shipping paths from one substrate:

**Personal use (this build):** Persistent, queryable memory across Claude Code, Codex, and VS Code, running locally on the M4 Pro. Replaces the brittle CLAUDE.md / MEMORY.md flat-file approach with a typed knowledge graph that supports semantic retrieval, multi-hop traversal, temporal validity, and project scoping.

**Galvo productization (downstream):** Same architectural foundation, productized as a customer-deployable memory service with a custom ontology for code/dev work, a SAGE-style self-evolution layer, and a dream-state consolidation service. Deployed via Aura (managed) or self-hosted Neo4j (air-gapped).

## 2. Architectural Decisions (with rationale)

These are the load-bearing choices. If you change one, revisit the rest.

**Substrate: Neo4j Community Edition (local), Neo4j Aura (Galvo managed option).**
Considered KuzuDB + LanceDB for embedded simplicity. Rejected because (a) Neo4j 2026.01+ ships in-index filtering with the SEARCH clause, which is the right primitive for "semantically similar AND scoped AND temporally valid" queries we'll run constantly; (b) Neo4j has a productized agent memory package (`neo4j-agent-memory`) that gives us the writer/retriever/MCP scaffold for free; (c) deployment friction doesn't apply to single-user local use, and Aura solves it for Galvo customers. Embedded was the right answer for distributing to non-technical users — that's not our profile.

**Library: `neo4j-agent-memory` v0.1.x as foundation, custom ontology layered on top.**
The library handles short-term (conversation), long-term (entity graph), and reasoning (decision traces) memory tiers. Its MCP server exposes 16 tools that work across Claude Code, Codex, Cursor, and VS Code Copilot out of the box. We use `client.schema.adopt_existing_graph(...)` to layer our custom code/dev ontology over the library's machinery. **Do not use the default POLE+O ontology** — it's optimized for entity tracking in intelligence/investigative use cases, not code memory.

**Risk acknowledged:** the library is v0.1.0 (started Jan 2026, active development, breaking changes expected). Pin versions explicitly. Plan to fork if a schema change ever blocks us. For Galvo, vendor at a known-good version and own the dependency.

**Integration: MCP + hooks + sidecar, all three.**
- MCP server (provided by `neo4j-agent-memory`) gives the agent proactive recall — Claude calls `memory_search` when it decides it needs context.
- Hooks (Claude Code's lifecycle hooks: SessionStart, UserPromptSubmit, PostToolUse, SessionEnd) handle injection-on-rails — surface relevant memory at session start, log decisions and tool outcomes automatically.
- Sidecar (the Neo4j daemon plus a thin FastAPI bridge on localhost:7575, modeled on the openclaw plugin) owns the data layer and runs whether or not any client is connected.

All three integration points share one database. Hooks can fail and MCP still works. MCP can be unavailable and hooks still log events. The sidecar persists regardless.

**Cross-tool strategy: one server, many adapters.**
The MCP server is tool-agnostic. Claude Code, Codex, and VS Code each get a client config that points at the same local MCP endpoint. No duplication. AGENTS.md becomes the canonical instruction file, indexed by the sidecar; tool-specific files (CLAUDE.md, etc.) are also ingested but the graph is the source of truth for retrieval.

**Reference implementation: `johnymontana/openclaw-neo4j-agent-memory-plugin`.**
Will Lyon's plugin already wires neo4j-agent-memory into a Claude-Code-like harness. Architecture: native MCP tools (memory_search, memory_get, memory_store) + auto hooks (before_prompt_build, agent_end, after_tool_call) bridged through FastAPI on :7575 to Neo4j on :7687. Study this. Fork if useful. Don't reinvent the bridge layer.

## 3. The Stack

```
┌──────────────────────────────────────────────────┐
│  Clients (any/all, same backend)                 │
│  • Claude Code      • Codex      • VS Code       │
└─────────────┬──────────────────┬─────────────────┘
              │                  │
        MCP (16 tools)    Lifecycle hooks
              │                  │
              ▼                  ▼
┌──────────────────────────────────────────────────┐
│  Sidecar: FastAPI bridge (:7575)                 │
│  • neo4j-agent-memory client                     │
│  • Custom ontology layer (Decision, Pattern,     │
│    Mistake, Convention, Belief, Artifact, …)     │
│  • Feedback logger (every retrieval scored)      │
│  • File watcher (AGENTS.md, CLAUDE.md, etc.)     │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  Neo4j Community Edition (:7687)                 │
│  • Graph: nodes + typed edges                    │
│  • Vector index (in-index filtering, 2026.01+)   │
│  • Full-text search                              │
│  • Temporal properties                           │
└──────────────────────────────────────────────────┘
                      │
                      ▼ (scheduled, offline)
┌──────────────────────────────────────────────────┐
│  Consolidation service ("dream-state")           │
│  • Edge-weight updates from feedback             │
│  • Belief invalidation propagation               │
│  • Redundancy merging                            │
│  • Low-utility pruning                           │
└──────────────────────────────────────────────────┘
```

**Embedding model:** local sentence-transformers default (no API key required). Optional upgrade to Qwen3-Embedding-8B via MLX on the M4 Pro once we know the access pattern. Do not hardcode embedding dimension — store it as a graph-level config property.

## 4. Custom Ontology (the meat)

This replaces POLE+O. Designed for code/dev memory specifically.

### Node types

- **Decision** — a non-trivial choice made during a session. Properties: rationale, alternatives_considered, confidence, scope (project / personal / universal).
- **Pattern** — a recurring approach observed across sessions. Properties: description, evidence_count, success_rate, codebase_scope.
- **Mistake** — something that went wrong. Properties: description, root_cause, fix_applied, time_to_discover.
- **Convention** — an established way of doing things in a specific codebase. Properties: description, source (inferred / explicit / from_AGENTS.md), strength.
- **Artifact** — a file, module, component, or function of significance. Properties: path, language, role, last_touched.
- **Session** — a unit of work. Properties: started_at, ended_at, agent_tool, task_description, outcome.
- **Belief** — an inferred fact about the codebase or environment. Properties: claim, confidence, valid_from, valid_to (nullable), source_session_id.
- **Test** — a test case or run result. Properties: identifier, last_run_status, last_run_at.
- **Commit** — a code change. Properties: sha, message, intent, reverted_by (nullable).
- **Constraint** — a hard requirement. Properties: description, type (performance / security / compatibility / regulatory), source.
- **Task** — what was being worked on. Properties: description, status, priority.
- **Failure** — a specific run failure. Properties: type (test / build / lint / runtime), error_signature, resolved (bool).

### Edge types

- `Decision -[LED_TO]-> Outcome (Commit | Failure | Belief)`
- `Decision -[CONSIDERED]-> Alternative (Decision)`
- `Decision -[BASED_ON]-> Belief`
- `Pattern -[OBSERVED_IN]-> Session`
- `Pattern -[CONTRADICTS]-> Pattern`
- `Mistake -[CORRECTED_BY]-> Commit`
- `Mistake -[CAUSED]-> Failure`
- `Convention -[APPLIES_TO]-> Artifact`
- `Belief -[SUPERSEDES]-> Belief` (critical for temporal invalidation)
- `Belief -[VALIDATED_BY]-> Test`
- `Session -[WORKED_ON]-> Task`
- `Session -[TOUCHED]-> Artifact`
- `Session -[PRODUCED]-> Commit`
- `Commit -[REVERTED]-> Commit`
- `Task -[BLOCKED_BY]-> Constraint`
- `* -[RETRIEVED_IN]-> Session` (feedback edge, added by the feedback logger on every retrieval)

### Scope partitioning

Every node has a `scope` property: `project:<repo-id>`, `personal` (cross-project user preferences), or `universal` (general dev knowledge). All queries filter by scope to prevent cross-project contamination. The hook layer determines current scope from the working directory at session start.

### Temporal validity

`Belief` nodes are the only nodes that explicitly expire. When new evidence contradicts a belief, the writer creates a new `Belief` node and a `SUPERSEDES` edge from new to old. The old belief's `valid_to` is set. Queries default to "valid as of now" but can request historical state. Other node types are immutable; corrections create new nodes with appropriate edges, never mutate in place.

## 5. The Feedback Loop (SAGE-style, slimmed)

This is what differentiates the system from raw `neo4j-agent-memory`. The principle: every retrieval gets a utility signal, and the consolidation service uses those signals to evolve the graph.

**Per-retrieval logging.** The hook and MCP layers both tag retrieved nodes with the requesting session_id via a `RETRIEVED_IN` edge. Properties on that edge: retrieval_rank, retrieval_score, retrieval_context (the query that produced it).

**Per-session scoring.** At SessionEnd, the hook scores each retrieved memory on:
- Was it actually referenced in the agent's output? (textual evidence)
- Did the task complete successfully? (test pass, no revert, user accepted)
- Did the agent immediately re-query for similar info (signal that the retrieval was insufficient)?

These produce a utility score [-1, +1] written to the RETRIEVED_IN edge.

**Consolidation (dream-state), runs nightly or on demand.**
- Edge weights between frequently co-retrieved-and-useful nodes get reinforced.
- Nodes with consistent negative utility get demoted (lower retrieval rank, not deleted).
- Beliefs marked stale by repeated contradiction get invalidated.
- Redundant nodes (same claim, different wording) get merged with a CANONICAL edge.

**Don't build the full SAGE Graph Foundation Model reader.** That's a research project. The above is 80% of the value at 10% of the cost.

## 6. Phase 1 — Personal Use Spike (next 48 hours)

Goal: prove the substrate works end-to-end before designing the real ontology.

1. Install Neo4j Community via Docker. Single container, port 7687.
2. Install `neo4j-agent-memory` MCP server via `uvx`. Test the MCP connection from Claude Code with the default POLE+O schema.
3. Have one real coding session using it. Don't overthink the ontology — let it write whatever it wants.
4. Inspect the resulting graph in Neo4j Browser (http://localhost:7474). See what got stored, what's useful, what's noise.
5. Decide: does the MCP-only path feel like enough, or do we definitely need the hook layer too?

**Hard rule for this phase:** anything written during the spike gets thrown away. Do not let yourself accumulate POLE+O-shaped data and then try to migrate. Drop the database before phase 2.

## 7. Phase 2 — Custom Ontology + Hooks (week 1-2)

1. Define the custom ontology in Neo4j (node labels, property constraints, indexes, vector index with filterable properties for `scope` and `valid_to`).
2. Use `client.schema.adopt_existing_graph(...)` to layer `neo4j-agent-memory` over the custom schema.
3. Fork or adapt the openclaw plugin's hook architecture for Claude Code. Hooks needed:
   - **SessionStart**: query top-of-mind memories for current project scope, inject a curated summary into context.
   - **UserPromptSubmit**: targeted semantic retrieval on the prompt, inject results.
   - **PostToolUse**: log artifact touches, test results, build failures.
   - **SessionEnd**: write Session node, score retrievals, mark task outcome.
4. Stand up the FastAPI sidecar on :7575. Health checks, graceful degradation if Neo4j is down.
5. File watcher: detect AGENTS.md, CLAUDE.md, .cursorrules, .codex/* changes and re-index.

## 8. Phase 3 — Codex + VS Code (week 3)

The MCP server is tool-agnostic, so this is mostly configuration. Add Codex's MCP config pointing at the same localhost endpoint. Add VS Code MCP-capable extensions (Cline, etc.) similarly. **Hooks won't port directly** — Codex and VS Code have weaker extensibility — so on those tools we rely primarily on MCP for retrieval and lose the auto-logging hooks. Acceptable tradeoff for personal use.

Optional: build a VS Code extension that surfaces the memory graph as a sidebar panel (visual inspection, pin/unpin nodes, manual invalidation). High value for debugging trust issues with the system.

## 9. Phase 4 — Galvo Productization (later, separate workstream)

Different shipping model. Key changes vs. personal use:

- Vendor `neo4j-agent-memory` at a pinned version. Own it.
- Custom ontology becomes the *only* ontology. Hide POLE+O entirely.
- Aura as managed backend default. Self-hosted Neo4j for air-gapped customers.
- The dream-state consolidation service is the productized differentiator. Build it as a separately-deployable component (probably Kubernetes-friendly, given the Galvo customer profile).
- Multi-tenant scope partitioning becomes mandatory and non-negotiable. Personal use can be lax; Galvo cannot.
- Add a "memory inspector" UI as a first-class product surface. Customers will not trust a black-box memory system in regulated industries.
- License audit: `neo4j-agent-memory` is Apache 2.0, Neo4j Community is GPLv3, Aura is commercial. Confirm before any customer commitments.

## 10. Open Questions / Decisions Deferred

- **Ontology coverage.** Does the 12-node-type ontology above cover real coding work, or will we hit edge cases that demand new types within a week? Validate empirically in phase 2.
- **Retrieval injection budget.** How much memory do we inject per session start before context-window pressure starts hurting? Claude Code documents ~150-200 instruction slots total, with ~50 already used by its own system prompt. Budget our injection at 30-50 slots max.
- **AGENTS.md write-back.** Do we ever push learned facts back into AGENTS.md (so they're visible to humans and survive a database loss), or is the graph the only source of truth? Lean toward explicit "promote to AGENTS.md" action, not automatic sync.
- **Embedding model swap timing.** Start with default sentence-transformers. Swap to local Qwen3-Embedding-8B (MLX) only if retrieval quality is visibly bad. Don't optimize prematurely.
- **Feedback signal calibration.** "Was retrieval useful" is a noisy signal in the personal phase (n=1 user). Will need significantly more data before consolidation can be trusted to auto-prune.
- **Project boundary detection.** How does the system know "this is a different project"? Working directory is the obvious answer but breaks for monorepos and worktrees. Provisional answer: explicit project marker file (`.galvo-mem/project.toml`) with a stable project ID.

## 11. Initial Build Steps (for Claude Code)

When you sit down with Claude Code, the ordered task list is:

1. Create the project directory. Initialize git. Create a `docs/` folder, copy this file into it as `MEMORY-LAYER-DESIGN.md`.
2. Set up `docker-compose.yml` for Neo4j Community 2026.04+ (need 2026.01+ for in-index filtering; latest is safer).
3. Verify `neo4j-agent-memory` works against the local Neo4j: `uvx "neo4j-agent-memory[mcp]" mcp serve --password <pw>`.
4. Add the MCP server to Claude Code config. Test with a trivial memory_store + memory_search round-trip.
5. (Decision gate) Confirm phase 1 spike approach before going further.
6. Begin phase 2: ontology DDL (Cypher), sidecar scaffolding (FastAPI), hook scripts.

## 12. References

- `neo4j-labs/agent-memory` — foundation library, v0.1.x, Apache 2.0
- `johnymontana/openclaw-neo4j-agent-memory-plugin` — reference Claude-Code-shaped integration
- `neo4j-graphrag-python` v1.16 — official Neo4j GraphRAG package (use for retriever patterns)
- SAGE paper (arXiv 2605.12061) — source of the self-evolution feedback loop idea; do not implement directly, steal the principle
- AGENTS.md convention — canonical cross-tool instruction file

## 13. What We Are Explicitly NOT Doing

Important to write down because each of these will be tempting:

- Not implementing SAGE's Graph Foundation Model reader from scratch.
- Not using POLE+O ontology (despite it being the library default).
- Not synchronizing AGENTS.md / CLAUDE.md / MEMORY.md bidirectionally with the graph — graph is canonical, files are inputs only, with an explicit promote action.
- Not building a custom embedded graph DB (KuzuDB was tempting; rejected).
- Not building Bayesian inference into Galvo's memory retrieval layer (despite it being a natural reach given DMTA work; not the right tool for relevance/retrieval problems).
- Not shipping to Galvo customers from this codebase directly. Galvo is a parallel productization, not an upgrade path from personal use.

---

*End of design brief.*
