# Phase 1 Spike — Findings

**Status:** Substrate proven end-to-end. Decision gate (design §11 step 5) raised.
**Date:** 2026-05-17

## What works

- **Neo4j Community 2026.04** image up + healthy on `bolt://localhost:7687`, browser on `http://localhost:7474`. Docker compose at `memory/docker/docker-compose.yml`.
- **`neo4j-agent-memory` v0.2.1** installs cleanly via `uvx --from "neo4j-agent-memory[mcp,sentence-transformers]" ...`.
- **MCP server CLI** matches the design's expectation:
  ```
  neo4j-agent-memory mcp serve --password <pw> [--uri ...] [--profile core|extended] \
    [--transport stdio|sse|http] [--session-strategy per_conversation|per_day|persistent]
  ```
- **Round-trip succeeds**: prose written via `client.long_term.add(...)` shows up in `client.long_term.search(...)` hits via vector similarity.
- **Three memory tiers exist** as design says: `client.short_term`, `client.long_term`, `client.reasoning`. The `reasoning` tier has `start_trace`, `add_step`, `record_tool_call` — perfectly matches our SessionStart / PostToolUse / SessionEnd hook story.

## Design-vs-reality gaps (real findings)

### 1. `neo4j-agent-memory` is v0.2.1, not v0.1.x

Design §2 says "library is v0.1.0 (started Jan 2026, active development, breaking changes expected)." Actual v0.2.1 on PyPI as of 2026-05-17. Minor-version bump implies breaking changes already happened. **Action:** pin version explicitly in our config; revisit the "fork if blocked" plan with v0.2.x as the baseline.

### 2. Default embedder is OpenAI, not local sentence-transformers

Design §3 says "Embedding model: local sentence-transformers default (no API key required)." Library default is OpenAI — needs `OPENAI_API_KEY` and the `[openai]` extra. **Mitigation:** explicit `EmbeddingConfig(provider=EmbeddingProvider.SENTENCE_TRANSFORMERS, model='all-MiniLM-L6-v2', dimensions=384)` + install the `[sentence-transformers]` extra. Tested and works locally.

### 3. Vector-index dimension is baked at first bootstrap

The first `client.long_term.add()` provisions the `entity_embedding_idx` index with the embedder's dimension count. If you ever change embedder dims (e.g. swap MiniLM 384 → Qwen3-Embedding 4096), you must drop the index (or wipe the DB). **Implication:** the design's "swap embedder later" path is non-trivial — needs a migration recipe. Add to Phase 2 ontology DDL: explicit `CREATE VECTOR INDEX ... DIMENSIONS $dim` with dim from a graph-level config property (design §3 actually called this out as a `do_not_hardcode_embedding_dimension` rule — good foresight).

### 4. Library API is fully async (good — keep it)

Design implied sync API in places (`client.schema.bootstrap()`, `client.memory.store(...)`). Reality: `async with MemoryClient(settings=...) as client: await client.long_term.add(...)`. This is correct for our use case (sidecar = FastAPI async; MCP = async stdio) but means **the design's reference code in §11 needs an async rewrite** before Phase 2.

### 5. Default extraction lands prose as one Entity

Without the `[extraction]` extra (GLiNER or LLM-based extractor), `client.long_term.add(prose)` stores the entire prose string as a single Entity's `name` field. To get the design's auto-extracted entities → relationships graph, install `[extraction]` (heavyweight: spacy + gliner deps). **Decision pending:** for personal-use spike, do we want LLM-side extraction (Claude does it via MCP tool calls) or library-side extraction (GLiNER local NER)? Likely the former — we have an LLM right there.

### 6. Library uses deprecated `db.index.vector.queryNodes` not the new `SEARCH` clause

Library is on the old Cypher procedure-based vector query, not the 2026.01+ `SEARCH` clause that the design §2 rationale specifically cited as the reason we picked Neo4j. **No action needed** — the procedure still works, deprecation is non-blocking. But it means the in-index filtering benefit isn't being captured today; the library will need an update. If we fork, prioritize this.

## What I have NOT done yet

- **Wired the MCP server into Claude Code's config.** Needs Michael to add the snippet (below) and restart the Claude Code session — I can't modify my own session's `.claude.json` mid-execution. Once wired, I can call `memory_store` / `memory_search` directly via MCP tool calls.
- **Run a real coding session against it.** The design's spike-success criterion ("see what got stored, what's useful, what's noise") requires actual use across one full session. Pending Michael's MCP wiring + a session.
- **Touched the ontology.** The library's default POLE+O is what the spike will write into; per the design's hard rule, all that data gets dropped before Phase 2.

## Suggested MCP config snippet for Michael

Add to your Claude Code MCP config (`~/.claude.json` under `mcpServers`, or via `claude mcp add`):

```json
{
  "mcpServers": {
    "galvo-memory": {
      "command": "uvx",
      "args": [
        "--from", "neo4j-agent-memory[mcp,sentence-transformers]",
        "neo4j-agent-memory", "mcp", "serve",
        "--transport", "stdio",
        "--uri", "bolt://localhost:7687",
        "--user", "neo4j",
        "--password", "galvo-memory-dev-2026",
        "--database", "neo4j",
        "--profile", "extended",
        "--session-strategy", "per_day",
        "--user-id", "michael"
      ],
      "env": {
        "NEO4J_AGENT_MEMORY__EMBEDDING__PROVIDER": "sentence_transformers",
        "NEO4J_AGENT_MEMORY__EMBEDDING__MODEL": "all-MiniLM-L6-v2",
        "NEO4J_AGENT_MEMORY__EMBEDDING__DIMENSIONS": "384"
      }
    }
  }
}
```

Or equivalent:
```bash
claude mcp add galvo-memory \
  -- uvx --from "neo4j-agent-memory[mcp,sentence-transformers]" \
     neo4j-agent-memory mcp serve --transport stdio \
     --uri bolt://localhost:7687 --user neo4j --password galvo-memory-dev-2026 \
     --database neo4j --profile extended --session-strategy per_day --user-id michael
```

After adding + restarting Claude Code, `/mcp` should list `galvo-memory` and the extended (16-tool) toolset should be available.

## Decision gate (design §11 step 5) — LOCKED 2026-05-17

Five decisions confirmed after walking through tradeoffs:

| | Decision | Locked value |
|---|---|---|
| **D1** | Integration channels | **C — MCP + Claude Code hooks + FastAPI sidecar** (design default §2). Three-channel: MCP for proactive recall, hooks for SessionStart/UserPromptSubmit/PostToolUse/SessionEnd auto-logging, sidecar daemon owning the data layer. |
| **D2** | Embedder | **A — sentence-transformers all-MiniLM-L6-v2 (384-dim) for cycle 1; deliberate Qwen3-Embedding-8B swap in cycle 2.** Conservative default; throw away spike DB at Phase 2 boundary per §6 hard rule; swap embedder once the system has been used enough to know if retrieval quality is the actual bottleneck. |
| **D3** | Extraction | **B — LLM-side via MCP tools.** Claude (or Codex/VS Code) decides what's worth extracting and calls memory_store with structured entities. No library-side NER (no GLiNER/spacy install). Reasoning: the LLM is the smartest entity-recognizer in the loop. |
| **D4** | Scope partitioning | **A — `project:<repo-id>` + `personal` + `universal`** (design default §4). Three-tier scope on every node. Repo-id derived from `.galvo-mem/project.toml` marker file per design §10 (stable across worktrees + monorepos). |
| **D5** | Feedback loop | **B — Logging + per-session scoring in cycle 1; nightly consolidation in cycle 2.** Cycle 1: every retrieval gets RETRIEVED_IN edge + utility score [-1,+1] computed at SessionEnd from textual-reference / task-outcome / re-query-immediately signals. Cycle 2: tune consolidation algorithm against ~30 sessions of real scoring data. Same end state, lower risk of mis-tuned consolidation harming retrieval before signal trust. |

### Why D2 and D5 were de-escalated from initial picks

Michael's first picks were the maximalist options on D2 (Qwen3 cycle 1) and D5 (full SAGE cycle 1). After surfacing the scope implications:

- **D2 → MiniLM:** Qwen3 requires ~1-2 days of custom MLX embedder engineering (the library's `CUSTOM` provider hook + tokenizer + inference) plus ~4GB model download, all before knowing if MiniLM's retrieval quality is the bottleneck. Design §3 itself positioned Qwen3 as a later upgrade ("once we know the access pattern"). The throw-away-spike-DB rule means the embedder swap is a clean operation later.
- **D5 → logging + scoring only:** the consolidation service is a 1-2 week subsystem. Design §10 already flagged that consolidation tuning on n=1 user data risks pruning the wrong things. Better to collect ~30 sessions of real scoring data before locking the algorithm.

Both downgrades are explicitly reversible — they're temporal sequencing decisions, not capability cuts.

## Phase 2 readiness checklist (what's blocked on Phase 1 close)

Two gates remaining before Phase 2 begins:

1. **Michael wires the MCP server into Claude Code** (snippet above) and restarts. After restart, `/mcp` should list `galvo-memory` with 16 tools available.
2. **One real coding session against the spike** — the design §6 spike-success criterion: "see what got stored, what's useful, what's noise." After the session, hit the proper gate and write the Phase 2 plan from real observations.

When the gate passes (i.e. after session 1), Phase 2 work:

- [ ] Drop spike DB: `cd memory/docker && docker compose down -v` (design §6 hard rule)
- [ ] Define custom ontology DDL in `memory/ontology/` (12 node types per design §4)
- [ ] Verify `client.schema.adopt_existing_graph(...)` exists in v0.2.1 (library has `SchemaManager` class — needs probing) and layer custom ontology over it
- [ ] Stand up FastAPI sidecar on `:7575` in `memory/sidecar/`
- [ ] Build Claude Code hooks in `memory/hooks/claude-code/`: SessionStart, UserPromptSubmit, PostToolUse, SessionEnd
- [ ] File watcher for AGENTS.md / CLAUDE.md / .cursorrules / .codex/* changes
- [ ] Implement scope partitioning convention (D4 — every node gets `scope` property; hooks read working-directory project marker file)
- [ ] Implement feedback-loop logging + scoring (D5 cycle 1): RETRIEVED_IN edge writer; SessionEnd scorer

Cycle 2 (separate from this plan, do not bundle):

- [ ] Wire MLX-Qwen3 as `CUSTOM` embedder, drop DB, re-bootstrap with 4096-dim index (D2 swap)
- [ ] Implement consolidation service (D5 cycle 2 — edge-weight updates, belief invalidation, redundancy merging, low-utility pruning)
