# Galvo Memory — Local Development

## Prerequisites

- Docker (for the Neo4j substrate)
- `uv` / `uvx` (Python tooling)

## Boot the substrate

```bash
cd memory/docker
docker compose up -d
# wait ~15s for the JVM + plugin install; container reports `healthy` when ready
```

Neo4j Browser at `http://localhost:7474` (user `neo4j`, password `galvo-memory-dev-2026`).
Bolt protocol on `bolt://localhost:7687`.

## Run tests

Tests live in `memory/tests/`. They are split into two tiers:

* **Dict-level invariants** — no Neo4j required, run anywhere.
* **Integration tests** (marked `@pytest.mark.integration`) — require live Neo4j
  on `bolt://localhost:7687`.

### Recommended: uvx one-shot

The repo deliberately avoids a top-level `uv sync` so subagents can run tests in
sealed environments without touching the workspace. Use `uvx` to install the
library + pytest in an ephemeral venv:

```bash
# from memory/
uvx --from "neo4j-agent-memory[mcp,sentence-transformers]" \
    --with pytest \
    --with pytest-asyncio \
    python -m pytest tests/ -v
```

To skip Neo4j-dependent tests:

```bash
uvx --from "neo4j-agent-memory[mcp,sentence-transformers]" \
    --with pytest \
    --with pytest-asyncio \
    python -m pytest tests/ -v -m "not integration"
```

### Alternative: `uv sync` in memory/

If you'd rather have a persistent venv:

```bash
cd memory
uv venv
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

## Drop the DB

Per design §6 hard rule, the Phase 2 work starts from a clean DB. Re-run between
risky changes:

```bash
cd memory/docker
docker compose down -v   # removes the named volume
docker compose up -d
```

## Lint

```bash
uvx ruff check ontology/ scope/ sidecar/ tests/
```
