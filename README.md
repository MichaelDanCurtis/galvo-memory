# Galvo Memory

Graph-native memory layer for AI coding agents. Two shipping paths from one substrate:

- **Personal-use today** — replaces flat-file `MEMORY.md` / `CLAUDE.md` / `AGENTS.md` with a typed knowledge graph queryable across Claude Code, Codex, and VS Code.
- **Galvo productization** — same substrate, customer-deployable as a memory service for Lucidity's agent platform.

**Canonical design:** [`docs/MEMORY-LAYER-DESIGN.md`](docs/MEMORY-LAYER-DESIGN.md). Read that first.
**Phase 2 plan + acceptance gates:** [`docs/PHASE-2-PLAN.md`](docs/PHASE-2-PLAN.md).

## Status

**Phase 2 cycle 1 — feature-complete.** Six acceptance gates verified by
[`examples/sidecar_demo.sh`](examples/sidecar_demo.sh):

- 12-label custom ontology (`Decision`, `Pattern`, `Convention`, `Constraint`, `Task`, `Session`, `Mistake`, `Commit`, `Failure`, `Artifact`, `Test`, `Belief`) layered on `neo4j-agent-memory==0.2.1`.
- FastAPI sidecar on `:7575` owning the `MemoryClient` + the MiniLM-L6-v2 384-dim embedder warmup.
- 3-tier scope partitioning (`project:<id>` / `personal` / `universal`) with in-cypher filtering.
- REST CRUD over all 12 labels + semantic search via the library's vector index.
- 4 Claude Code lifecycle hooks (`SessionStart`, `UserPromptSubmit`, `PostToolUse`, `SessionEnd`).
- `AGENTS.md` / `CLAUDE.md` file watcher → Convention writer.
- `python -m memory.cli promote` — pin a graph node back into a markdown file.
- D5 SessionEnd utility scorer — 4-signal feedback writer on `RETRIEVED_IN` edges.

**Cycle 2 parking lot** (explicit deferrals — out of scope here):

- Qwen3-Embedding swap (currently MiniLM-L6-v2 / 384-dim; swap requires vector-index migration recipe).
- Consolidation service ("dream-state" rollups, Belief supersession orchestration).
- Codex / VS Code hook adapters (cycle 1 only ships Claude Code).
- Embedding-cosine re-query detection in the scorer (cycle 1 uses Jaccard word-overlap).
- `GET /api/node/{id}` one-shot label+props lookup (cycle 1 walks per-label routes; acceptable on loopback).

## Layout

```
memory/
├── docker/             # docker-compose.yml — Neo4j Community + sidecar
├── sidecar/            # FastAPI bridge on :7575 — REST CRUD + scoring + health
├── ontology/           # Custom 12-label mapping + adoption driver
├── scope/              # Project marker (.galvo-mem/project.toml) + scope detector
├── hooks/claude_code/  # 4 lifecycle hooks (SessionStart / UserPromptSubmit / PostToolUse / SessionEnd)
├── watcher/            # AGENTS.md / CLAUDE.md file watcher → Convention writer
├── cli/                # `python -m memory.cli promote ...`
├── examples/           # End-to-end demo + operator README
├── tests/              # 330 unit + integration tests (uv run --extra dev --extra sidecar --extra watcher pytest)
└── docs/               # Design brief + Phase 1 spike findings + Phase 2 plan
```

## Quick start

```bash
# 1. Bring up Neo4j + sidecar.
cd memory/docker
docker compose up -d

# 2. Verify both healthy (first boot warms the MiniLM embedder; ~10-30s).
curl -fsS http://localhost:7575/health
# Expect: {"status":"ok","neo4j":"ok","embedder":"all-MiniLM-L6-v2",...}

# 3. Run the end-to-end smoke — 6 acceptance gates.
bash memory/examples/sidecar_demo.sh
# Exit code 0 = all gates passed.
```

If you'd rather have the script own the stack lifecycle:

```bash
GALVO_MEMORY_DEMO_STARTUP=1 bash memory/examples/sidecar_demo.sh
```

## Scope: mark a project

Memory writes are tagged with a stable `scope` so queries from one project don't surface another's notes. Mark a project once at the repo root:

```bash
mkdir .galvo-mem
cat > .galvo-mem/project.toml <<'EOF'
[project]
id = "myproj"
name = "My Project"

[scope]
default = "project:myproj"
EOF
```

The marker file survives renames, moves, and worktrees. Sessions opened anywhere under that tree (including subdirectories and `git worktree add` paths) inherit `scope=project:myproj`. Without a marker, the scope detector falls back to `personal` for the current user.

Source: [`scope/marker.py`](scope/marker.py), [`scope/detector.py`](scope/detector.py).

## Hook installation (Claude Code)

The four lifecycle hooks live under [`hooks/claude_code/`](hooks/claude_code/). Each is an executable Python entrypoint that reads JSON from stdin (Claude Code's hook protocol) and talks to the sidecar over loopback.

1. Install the memory package in editable mode so the hooks can import their shared lib:

   ```bash
   cd memory
   pip install -e ".[sidecar]"
   ```

2. Wire the hooks into Claude Code's `~/.claude/settings.json`. Use absolute paths to the worktree:

   ```jsonc
   {
     "hooks": {
       "SessionStart":      ["python3 /abs/path/to/memory/hooks/claude_code/session_start.py"],
       "UserPromptSubmit":  ["python3 /abs/path/to/memory/hooks/claude_code/user_prompt_submit.py"],
       "PostToolUse":       ["python3 /abs/path/to/memory/hooks/claude_code/post_tool_use.py"],
       "SessionEnd":        ["python3 /abs/path/to/memory/hooks/claude_code/session_end.py"]
     }
   }
   ```

The hook contract is "NEVER raises, NEVER blocks, NEVER pollutes stderr" — a wedged sidecar silently no-ops with a WARNING line to `~/.galvo-memory/logs/<hook>.log`.

Codex / VS Code adapters: cycle 2.

## MCP wiring (optional — direct graph access from inside Claude Code)

If you want `memory_store` / `memory_search` MCP tools inside Claude Code (parallel to the hooks, which run automatically), add `galvo-memory` to your MCP config:

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
        "--session-strategy", "per_day"
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

Or:

```bash
claude mcp add galvo-memory \
  -- uvx --from "neo4j-agent-memory[mcp,sentence-transformers]" \
     neo4j-agent-memory mcp serve --transport stdio \
     --uri bolt://localhost:7687 --user neo4j --password galvo-memory-dev-2026 \
     --database neo4j --profile extended --session-strategy per_day
```

The MCP server and the sidecar share the same Neo4j; pick whichever surface fits the workflow.

## Promote CLI — pin a graph node back into markdown

`AGENTS.md` and `CLAUDE.md` remain the readable lingua franca for teammates without graph access. The `promote` subcommand reads a node via the sidecar and appends a formatted projection to a target file:

```bash
# Append the Convention node `conv_abc123` under "## Conventions" in AGENTS.md.
python -m memory.cli promote conv_abc123 --to AGENTS.md --section "## Conventions"

# Preview the diff without writing.
python -m memory.cli promote conv_abc123 --to AGENTS.md --dry-run
```

The graph remains canonical; promoted files are read at SessionStart but not written back to the graph automatically (the watcher handles the reverse direction for explicit edits).

Source: [`cli/promote.py`](cli/promote.py).

## File watcher — Convention writer

The watcher tails `AGENTS.md` / `CLAUDE.md` / `.galvo-mem/*.md` and writes `Convention` nodes for each rule-shaped paragraph. Runs in-process:

```bash
cd memory
pip install -e ".[watcher]"          # adds watchdog
python -m memory.watcher              # foreground daemon; Ctrl-C to stop
```

The watcher is opt-in because some teams prefer one-way (graph → markdown via `promote`) over two-way (markdown ↔ graph).

Source: [`watcher/daemon.py`](watcher/daemon.py), [`watcher/parsers.py`](watcher/parsers.py).

## Architecture

```
Clients (Claude Code today; Codex + VS Code cycle 2)
    | MCP tools (16) + 4 lifecycle hooks
Sidecar (FastAPI :7575)
    | neo4j-agent-memory 0.2.1 + custom 12-label ontology
Neo4j Community 2026.04 (:7687)
    - graph + vector index (MiniLM 384-dim) + full-text + temporal
    ↓ (cycle 2, deferred)
Consolidation service ("dream-state" rollups)
```

Full architecture in [design §3](docs/MEMORY-LAYER-DESIGN.md).

## Tests

```bash
cd memory
uv run --extra dev --extra sidecar --extra watcher \
    pytest tests/ -m "not integration" -q
# 330 unit tests; integration tests need a live Neo4j volume (see HACKING.md).
```
