# Galvo Memory Sidecar — FastAPI on :7575
#
# Build context: the `memory/` directory (one level up from this file).
# This is what `docker compose build` will use after Task 18 wires the
# service into `memory/docker/docker-compose.yml`.
#
# Layer cache strategy:
#   1. install uv (rarely changes)
#   2. install Python deps from neo4j-agent-memory (slow — wheels for
#      torch + sentence-transformers; pinning prevents recompile)
#   3. copy in our thin sidecar/ontology/scope packages (changes often)
#
# Image size is dominated by the torch + sentence-transformers wheels
# (~1.5GB). Cycle 2's CPU-only Qwen3 swap is roughly the same footprint.

FROM python:3.12-slim AS base

WORKDIR /app

# uv gives us fast resolution + install with a single binary.
RUN pip install --no-cache-dir uv

# Copy in the memory tree. The build context is `memory/` so paths are
# relative to that directory (matches Phase 2 Task 18's docker-compose
# `build.context: ./memory` mapping).
COPY pyproject.toml ./
COPY sidecar/ ./sidecar/
COPY ontology/ ./ontology/
COPY scope/ ./scope/

# Pin the same neo4j-agent-memory version the repo locks (Phase 1 finding
# §1 — minor-version bumps are breaking). The `sentence-transformers`
# extra pulls in the embedder. Sidecar deps are listed inline rather than
# via `.[sidecar]` so the image build doesn't pay the cost of the full
# wheel + hatchling for our thin shim.
RUN uv pip install --system --no-cache-dir \
    "neo4j-agent-memory[sentence-transformers]==0.2.1" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic-settings>=2.5.0"

EXPOSE 7575

# uvicorn's lifespan integration runs `lifespan` from `sidecar.app:app`,
# which boots MemoryClient + warms the embedder before serving requests.
# First boot in a fresh container is ~10s (model download) — Compose
# healthchecks should allow for this.
CMD ["uvicorn", "sidecar.app:app", "--host", "0.0.0.0", "--port", "7575"]
