# Galvo Memory

Graph-native memory layer for AI coding agents. Two shipping paths from one substrate:

- **Personal-use today** — replaces flat-file `MEMORY.md` / `CLAUDE.md` / `AGENTS.md` with a typed knowledge graph queryable across Claude Code, Codex, and VS Code.
- **Galvo productization** — same substrate, customer-deployable as a memory service for Lucidity's agent platform.

**Canonical design:** [`docs/MEMORY-LAYER-DESIGN.md`](docs/MEMORY-LAYER-DESIGN.md). Read that first.

## Layout

```
memory/
├── docker/        # docker-compose.yml for Neo4j Community + ancillary services
├── sidecar/       # FastAPI bridge (localhost:7575) — owns the data layer, runs whether or not a client is connected
├── ontology/      # Cypher DDL for the custom code/dev ontology (Decision, Pattern, Mistake, Convention, Belief, …)
├── hooks/         # Per-tool lifecycle hooks (Claude Code first, Codex + VS Code follow)
└── docs/          # Design brief + ADRs
```

## Phase 1 — Spike (in progress)

Goal: prove the substrate works end-to-end before designing the real ontology. See design §6.

Run the spike:

```bash
cd memory/docker
docker compose up -d        # Neo4j Community on :7687, browser on :7474
uvx "neo4j-agent-memory[mcp]" mcp serve --password <pw>
```

Wire the MCP server into Claude Code (see design §11 step 4). Have one real session. Inspect the graph at http://localhost:7474. Decide.

**Hard rule:** drop the database before Phase 2. Anything written during the spike is throwaway.

## Architecture

```
Clients (Claude Code, Codex, VS Code)
    ↕ MCP (16 tools) + lifecycle hooks
Sidecar (FastAPI :7575)
    ↕ neo4j-agent-memory client + custom ontology
Neo4j Community (:7687) — graph + vector index + full-text + temporal
    ↓ (scheduled, offline)
Consolidation service ("dream-state")
```

Full architecture in design §3.

## Status

Phase 1 spike active. See [project handoff](../../memory-layer-progress.md) (when created) for current cycle state.
