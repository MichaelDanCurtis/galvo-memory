# Memory Layer — Phase 2 Implementation Plan

> **For agentic workers:** Execute via `superpowers:subagent-driven-development` per the Eval Harness (sub-project F) precedent. Each task is self-contained for dispatch as a fresh subagent in its own worktree off `memory-layer`.

**Goal:** Stand up the custom-ontology + sidecar + hooks layer so a fresh Claude Code session writes to and reads from a typed knowledge graph instead of flat-file MEMORY.md.

**Architecture:** Layer the 12-node-type code/dev ontology over `neo4j-agent-memory` v0.2.1 via `SchemaManager.adopt_existing_graph`. Stand up a FastAPI sidecar on `:7575` that owns the MemoryClient + scope partitioning + feedback logging. Wire 4 Claude Code lifecycle hooks (SessionStart / UserPromptSubmit / PostToolUse / SessionEnd) through HTTP to the sidecar. File watcher detects AGENTS.md / CLAUDE.md changes and re-indexes.

**Tech stack:** Python 3.12 (Pydantic v2, FastAPI, neo4j-agent-memory 0.2.1, sentence-transformers MiniLM-L6-v2 384-dim), Neo4j Community 2026.04 via Docker, Claude Code lifecycle hooks (bash + python).

**Sources of truth:**
- Design: `memory/docs/MEMORY-LAYER-DESIGN.md` (Michael's brief)
- Phase 1 findings + locked decisions: `memory/docs/PHASE-1-SPIKE-FINDINGS.md` (D1-D5)

---

## Locked decisions (recap)

| ID | Decision | Cycle 1 value |
|---|---|---|
| D1 | Integration channels | MCP + Claude Code hooks + FastAPI sidecar (all three) |
| D2 | Embedder | sentence-transformers MiniLM-L6-v2 (384-dim); Qwen3 swap deferred to cycle 2 |
| D3 | Entity extraction | LLM-side via MCP tools; no library-side NER |
| D4 | Scope partitioning | 3-tier: `project:<id>` / `personal` / `universal` |
| D5 | Feedback loop | Logging + per-session scoring; consolidation deferred to cycle 2 |

---

## Acceptance gates (cycle 1 close)

1. **Sidecar health** — `curl http://localhost:7575/health` returns 200 with `{"neo4j": "ok", "embedder": "loaded"}`.
2. **Custom ontology applied** — Neo4j Browser shows the 12 custom labels (Decision, Pattern, Mistake, Convention, Artifact, Session, Belief, Test, Commit, Constraint, Task, Failure) with appropriate property schemas, plus the library's `Entity`/`Fact`/`Preference` labels each entity is also tagged with (multi-label nodes per `adopt_existing_graph` design).
3. **End-to-end hook flow** — a fresh Claude Code session in `/Volumes/.../Galvo`:
   - SessionStart injects ≤50 lines of curated context from graph
   - UserPromptSubmit retrieves ≤5 relevant nodes per prompt
   - PostToolUse logs at least 1 artifact touch per real tool call
   - SessionEnd writes a `Session` node + scores all `RETRIEVED_IN` edges from this session
4. **Scope partitioning** — writing a `Decision` while `cwd` is Galvo tags it `scope=project:galvo`. Writing while `cwd` is a different project tags it differently. Queries default to scope-filtered.
5. **Feedback signal lands** — `MATCH (m)-[r:RETRIEVED_IN]->(s:Session) WHERE r.utility_score IS NOT NULL RETURN count(*)` returns >0 after one real session ends.
6. **Demo + smoke** — `bash memory/examples/sidecar_demo.sh` runs end-to-end (start sidecar → write 5 nodes → query → score → shutdown) and exits 0.

---

## Phase structure

| Phase | Tasks | Parallelism | Output |
|---|---|---|---|
| 2A — Foundation | 1-5 | 1 serial (Task 1 is a risk gate) | Custom ontology + scope partitioning |
| 2B — Sidecar | 6-10 | Tasks 6-7 sequential; 8-10 parallel after | FastAPI on :7575 |
| 2C — Hooks | 11-15 | Task 11 sequential; 12-15 4-parallel | 4 Claude Code lifecycle hooks |
| 2D — Integration | 16-20 | 16-18 parallel; 19-20 sequential | File watcher + demo + acceptance |

Estimated wall-clock: 1-2 weeks Agent Swarm pace.

---

## Phase 2A — Foundation (Tasks 1-5)

### Task 1 — Probe & lock `adopt_existing_graph` label mapping ⚠ RISK GATE

**Files:**
- Create: `memory/ontology/label_mapping.py` — Python module that defines the dict + invokes adoption with `dry_run=True` first
- Create: `memory/ontology/__init__.py`
- Create: `memory/tests/test_label_mapping.py`

**Goal:** Verify the library can actually adopt our 12 labels as proposed; produce the canonical `LABEL_TO_TYPE` dict.

**Proposed mapping (subject to spike findings):**

```python
# memory/ontology/label_mapping.py
"""Maps Galvo memory ontology labels → neo4j-agent-memory EntityType.

Per design §4, we have 12 node types. The library has 9 EntityType values
(PERSON, OBJECT, LOCATION, EVENT, ORGANIZATION, CONCEPT, EMOTION,
PREFERENCE, FACT). Adoption layers our labels onto the library's machinery
so we inherit embedding vector index + the MCP tools.

Mapping rationale:
- Decision, Pattern, Convention, Constraint, Task: CONCEPT (abstract things)
- Session, Mistake, Commit, Failure: EVENT (things that happened in time)
- Artifact, Test: OBJECT (concrete files/resources)
- Belief: FACT (the only one that explicitly maps to the library's FACT)
"""
from typing import Final

LABEL_TO_TYPE: Final[dict[str, str]] = {
    "Decision":   "CONCEPT",
    "Pattern":    "CONCEPT",
    "Convention": "CONCEPT",
    "Constraint": "CONCEPT",
    "Task":       "CONCEPT",
    "Session":    "EVENT",
    "Mistake":    "EVENT",
    "Commit":     "EVENT",
    "Failure":    "EVENT",
    "Artifact":   "OBJECT",
    "Test":       "OBJECT",
    "Belief":     "FACT",
}

NAME_PROPERTY_PER_LABEL: Final[dict[str, str]] = {
    "Decision":   "title",         # short imperative title
    "Pattern":    "name",
    "Convention": "name",
    "Constraint": "name",
    "Task":       "title",
    "Session":    "title",
    "Mistake":    "summary",
    "Commit":     "sha",           # SHA is the canonical name
    "Failure":    "error_signature",
    "Artifact":   "path",
    "Test":       "identifier",
    "Belief":     "claim",
}
```

**TDD steps:**

1. Write failing test `test_dry_run_succeeds_against_live_neo4j` — invokes `SchemaManager.adopt_existing_graph(LABEL_TO_TYPE, name_property_per_label=NAME_PROPERTY_PER_LABEL, dry_run=True)` and asserts the returned `AdoptionReport` has no errors per label.
2. Run test → FAIL (module doesn't exist).
3. Implement label_mapping.py + a thin helper `apply_ontology(dry_run: bool) -> AdoptionReport`.
4. Run test → expect PASS. If fails, the proposed mapping has a conflict; iterate.
5. Add a second test `test_real_apply_creates_indexes` that calls `apply_ontology(dry_run=False)` against a fresh DB and asserts the `entity_embedding_idx` index has the 384 dimension.
6. Commit: `memory/ontology: label_to_type mapping + dry-run gate`.

**Decision gate at end of Task 1:** if `adopt_existing_graph` rejects the mapping (e.g. one of our labels collides with a library-reserved label), STOP and report. We may need to rename a label, or escalate to forking the library.

---

### Task 2 — Custom property schema (per-label constraints + indexes)

**Files:**
- Create: `memory/ontology/properties.cypher` — DDL for property constraints + indexes per label
- Create: `memory/ontology/apply_properties.py` — applies the Cypher
- Create: `memory/tests/test_property_schema.py`

**Goal:** Layer our design §4 properties onto the adopted labels. Each label gets:
- Property uniqueness constraints (e.g. `Commit.sha` unique)
- Filterable property indexes (e.g. `*.scope` indexed for fast scope-filtering)
- Datetime property indexes where needed (e.g. `Session.started_at`)

**Skeleton Cypher (illustrative):**

```cypher
// Universal: every node has a scope (D4) and a created_at
CREATE INDEX node_scope_idx IF NOT EXISTS FOR (n:Entity) ON (n.scope);

// Decision
CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (n:Decision) REQUIRE n.id IS UNIQUE;
CREATE INDEX decision_confidence_idx IF NOT EXISTS FOR (n:Decision) ON (n.confidence);

// Belief — the temporal-validity node type per design §4
CREATE CONSTRAINT belief_id_unique IF NOT EXISTS FOR (n:Belief) REQUIRE n.id IS UNIQUE;
CREATE INDEX belief_valid_to_idx IF NOT EXISTS FOR (n:Belief) ON (n.valid_to);

// Commit
CREATE CONSTRAINT commit_sha_unique IF NOT EXISTS FOR (n:Commit) REQUIRE n.sha IS UNIQUE;

// ... rest of 12 labels
```

**Tests:** assert each constraint + index exists post-apply via `SHOW CONSTRAINTS` / `SHOW INDEXES`.

**Commit:** `memory/ontology: per-label property constraints + indexes`.

---

### Task 3 — Edge type mapping (16 design edges → library + custom)

**Files:**
- Create: `memory/ontology/edges.py` — Python constants for edge types
- Create: `memory/ontology/edges.cypher` — DDL for edge-property indexes
- Create: `memory/tests/test_edge_types.py`

**Goal:** Map the 16 edges from design §4 to canonical edge-type names. Two categories:

1. **Custom edge types** (most): `LED_TO`, `CONSIDERED`, `BASED_ON`, `OBSERVED_IN`, `CONTRADICTS`, `CORRECTED_BY`, `CAUSED`, `APPLIES_TO`, `SUPERSEDES`, `VALIDATED_BY`, `WORKED_ON`, `TOUCHED`, `PRODUCED`, `REVERTED`, `BLOCKED_BY`. These are not in the library; we create them as plain Cypher relationships with no library coupling.
2. **Feedback edge** (special): `RETRIEVED_IN` from any node to `Session`. Properties: `retrieval_rank: int`, `retrieval_score: float`, `retrieval_context: str`, `utility_score: float | null`. This is the D5 cycle-1 logging surface.

**Skeleton:**

```python
# memory/ontology/edges.py
from typing import Final

# Custom edges — design §4
EDGE_LED_TO:        Final[str] = "LED_TO"
EDGE_CONSIDERED:    Final[str] = "CONSIDERED"
EDGE_BASED_ON:      Final[str] = "BASED_ON"
EDGE_OBSERVED_IN:   Final[str] = "OBSERVED_IN"
EDGE_CONTRADICTS:   Final[str] = "CONTRADICTS"
EDGE_CORRECTED_BY:  Final[str] = "CORRECTED_BY"
EDGE_CAUSED:        Final[str] = "CAUSED"
EDGE_APPLIES_TO:    Final[str] = "APPLIES_TO"
EDGE_SUPERSEDES:    Final[str] = "SUPERSEDES"
EDGE_VALIDATED_BY:  Final[str] = "VALIDATED_BY"
EDGE_WORKED_ON:     Final[str] = "WORKED_ON"
EDGE_TOUCHED:       Final[str] = "TOUCHED"
EDGE_PRODUCED:      Final[str] = "PRODUCED"
EDGE_REVERTED:      Final[str] = "REVERTED"
EDGE_BLOCKED_BY:    Final[str] = "BLOCKED_BY"

# Feedback (D5)
EDGE_RETRIEVED_IN:  Final[str] = "RETRIEVED_IN"

ALL_EDGE_TYPES: Final[list[str]] = [
    EDGE_LED_TO, EDGE_CONSIDERED, EDGE_BASED_ON, EDGE_OBSERVED_IN,
    EDGE_CONTRADICTS, EDGE_CORRECTED_BY, EDGE_CAUSED, EDGE_APPLIES_TO,
    EDGE_SUPERSEDES, EDGE_VALIDATED_BY, EDGE_WORKED_ON, EDGE_TOUCHED,
    EDGE_PRODUCED, EDGE_REVERTED, EDGE_BLOCKED_BY, EDGE_RETRIEVED_IN,
]
```

DDL: index `RETRIEVED_IN.utility_score` (Task 5 + cycle-2 consolidation need it).

**Commit:** `memory/ontology: edge type constants + RETRIEVED_IN feedback edge`.

---

### Task 4 — Project marker file + scope-detection helper

**Files:**
- Create: `memory/scope/__init__.py`
- Create: `memory/scope/detector.py` — `detect_scope(cwd: Path) -> str`
- Create: `memory/scope/marker.py` — read/write `.galvo-mem/project.toml`
- Create: `memory/tests/test_scope.py`

**Goal:** Given a working directory, return one of:
- `project:<repo-id>` — when a `.galvo-mem/project.toml` is found (walking up from cwd)
- `personal` — fallback when no marker exists but we're in `$HOME` somewhere
- `universal` — explicit writes only (never auto-detected)

**`.galvo-mem/project.toml` format:**

```toml
# Created at first use. Stable repo identifier survives renames, moves, worktrees.
[project]
id = "galvo"                        # short stable id
name = "Galvo FACT"                 # human-readable, can change
created_at = "2026-05-17T00:00:00Z"

[scope]
default = "project:galvo"           # nodes written from this tree default here
allowed_universal_topics = ["python", "rust", "neo4j"]  # opt-in writeable
```

**API:**

```python
# memory/scope/detector.py
from pathlib import Path
from .marker import find_marker, ProjectMarker

def detect_scope(cwd: Path | None = None) -> str:
    """Walk up from cwd to find .galvo-mem/project.toml; return scope string."""
    cwd = cwd or Path.cwd()
    marker = find_marker(cwd)
    if marker is not None:
        return marker.scope.default  # e.g. "project:galvo"
    if str(cwd).startswith(str(Path.home())):
        return "personal"
    return "universal"

def is_project_scope(scope: str) -> bool:
    return scope.startswith("project:")
```

**Tests:**
- `cwd` inside Galvo → `project:galvo` (after marker present)
- `cwd = ~/Documents/notes` → `personal`
- `cwd = /tmp/random` → `universal`
- Marker walks up: `cwd = Galvo/sdk/galvo/eval` → still `project:galvo`

**Initial marker placement (one-time setup):** create `/Volumes/.../Galvo/.galvo-mem/project.toml` at Phase 2A close.

**Commit:** `memory/scope: project marker file + scope detector`.

---

### Task 5 — Scope-aware Cypher query helpers

**Files:**
- Create: `memory/sidecar/cypher_helpers.py` (placeholder dir; populated more in Phase 2B)
- Create: `memory/tests/test_cypher_helpers.py`

**Goal:** Pure Python helpers that take a base Cypher query + a scope filter + emit final Cypher with the scope WHERE clause inserted. All retrievals go through these.

**API:**

```python
# memory/sidecar/cypher_helpers.py
def scope_filter_clause(scope: str | None) -> str:
    """Returns 'WHERE n.scope = $scope' or '' for universal queries."""
    if scope is None:
        return ""  # caller explicitly wants cross-scope (rare)
    return "WHERE n.scope = $scope OR n.scope = 'universal'"

def search_with_scope(base_match: str, scope: str | None, params: dict) -> tuple[str, dict]:
    """Compose final Cypher + params dict for a scoped search."""
    where = scope_filter_clause(scope)
    final = f"{base_match} {where} RETURN n ORDER BY n.created_at DESC LIMIT $limit"
    return final, {**params, "scope": scope}
```

**Tests:** unit-test composition; no live Neo4j needed.

**Commit:** `memory/sidecar: scope-aware Cypher query helpers`.

---

## Phase 2B — Sidecar (Tasks 6-10)

### Task 6 — FastAPI sidecar skeleton on :7575

**Files:**
- Create: `memory/sidecar/app.py` — FastAPI app
- Create: `memory/sidecar/__init__.py`
- Create: `memory/sidecar/pyproject.toml` — separate from main repo's pyproject; sidecar deps
- Create: `memory/sidecar/Dockerfile` — runs the sidecar in a container
- Create: `memory/sidecar/tests/test_health.py`

**Goal:** Minimal FastAPI app on :7575 with a `/health` endpoint that checks Neo4j connectivity + embedder load status. Graceful degradation if Neo4j is down (returns 503 with diagnostic).

**Skeleton:**

```python
# memory/sidecar/app.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot: connect to Neo4j, warm embedder
    from neo4j_agent_memory import MemoryClient, MemorySettings, Neo4jConfig, EmbeddingConfig, EmbeddingProvider
    settings = MemorySettings(
        neo4j=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="galvo-memory-dev-2026",
            database="neo4j",
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model="all-MiniLM-L6-v2",
            dimensions=384,
        ),
    )
    client = MemoryClient(settings=settings)
    await client.connect()
    app.state.memory = client
    yield
    await client.close()

app = FastAPI(title="Galvo Memory Sidecar", lifespan=lifespan)

@app.get("/health")
async def health() -> dict:
    try:
        stats = await app.state.memory.get_stats()
        return {"neo4j": "ok", "embedder": "loaded", "stats": stats}
    except Exception as e:
        raise HTTPException(503, detail=str(e))
```

**Tests:** `pytest-asyncio` + `httpx.AsyncClient` against a TestClient. Mock the MemoryClient lifespan for unit tests; mark `@pytest.mark.integration` for the real-Neo4j tests.

**Commit:** `memory/sidecar: FastAPI app skeleton + /health endpoint`.

---

### Task 7 — MemoryClient async wrapper + dependency injection

**Files:**
- Create: `memory/sidecar/deps.py` — FastAPI dependencies (Depends-injected MemoryClient)
- Create: `memory/sidecar/tests/test_deps.py`

**Goal:** Reusable `get_memory_client()` dependency for all endpoints. Wires the lifespan's client into each request. Mockable for unit tests.

**Skeleton:**

```python
# memory/sidecar/deps.py
from typing import Annotated
from fastapi import Depends, Request
from neo4j_agent_memory import MemoryClient

def get_memory_client(request: Request) -> MemoryClient:
    return request.app.state.memory

MemoryDep = Annotated[MemoryClient, Depends(get_memory_client)]
```

**Commit:** `memory/sidecar: MemoryClient dependency injection`.

---

### Task 8 — REST CRUD for 12 node types

**Files:**
- Create: `memory/sidecar/routers/nodes.py` — generic CRUD router
- Create: `memory/sidecar/routers/__init__.py`
- Create: `memory/sidecar/models.py` — Pydantic request/response models per node type
- Create: `memory/sidecar/tests/test_nodes_router.py`

**Goal:** 12 endpoint groups, one per node type:
- `POST /api/{label}` — create a node
- `GET /api/{label}/{id}` — read
- `PATCH /api/{label}/{id}` — update mutable properties (NOT `Belief` — beliefs are immutable; use POST /api/Belief with SUPERSEDES edge)
- `GET /api/search/{label}?q={query}&scope={scope}&limit=10` — semantic search
- `DELETE` deliberately omitted in cycle 1 — design §4 says corrections create new nodes, never mutate

Sidecar uses the library's `client.long_term.add_entity(name, entity_type, ...)` with the right `EntityType` + label.

**Skeleton (Decision example):**

```python
# memory/sidecar/routers/nodes.py
from fastapi import APIRouter, HTTPException
from ..deps import MemoryDep
from ..models import DecisionCreate, DecisionResponse

router = APIRouter(prefix="/api")

@router.post("/Decision", response_model=DecisionResponse)
async def create_decision(payload: DecisionCreate, memory: MemoryDep) -> DecisionResponse:
    # Add as Entity with extra labels + properties
    entity = await memory.long_term.add_entity(
        name=payload.title,
        entity_type="CONCEPT",  # from LABEL_TO_TYPE
        extra_labels=["Decision"],
        properties={
            "rationale": payload.rationale,
            "confidence": payload.confidence,
            "scope": payload.scope,
            "alternatives_considered": payload.alternatives,
        },
    )
    return DecisionResponse.from_entity(entity)
```

NOTE: must verify `add_entity` actually accepts `extra_labels` + `properties` kwargs in v0.2.1. If not, drop down to a raw Cypher write via `memory.graph_client.execute_write(...)`. Task includes the probe.

**Commit:** `memory/sidecar: REST CRUD for 12 node types`.

---

### Task 9 — Feedback logging — `RETRIEVED_IN` edge writer

**Files:**
- Create: `memory/sidecar/feedback.py` — write `RETRIEVED_IN` edges
- Modify: `memory/sidecar/routers/nodes.py` — every search call writes RETRIEVED_IN edges
- Create: `memory/sidecar/tests/test_feedback.py`

**Goal:** Whenever a search returns N nodes, write N `RETRIEVED_IN` edges from those nodes to the current `Session` node. Edge properties:
- `retrieval_rank: int` (0 = top hit)
- `retrieval_score: float` (vector similarity score from library)
- `retrieval_context: str` (the query string)
- `created_at: datetime`
- `utility_score: float | null` (populated later by Task 10 SessionEnd scorer)

**Skeleton:**

```python
# memory/sidecar/feedback.py
async def log_retrieval(
    memory: MemoryClient,
    session_id: str,
    query: str,
    hits: list[Entity],
) -> None:
    for rank, hit in enumerate(hits):
        await memory.graph_client.execute_write(
            """
            MATCH (n) WHERE n.id = $node_id
            MATCH (s:Session {id: $session_id})
            MERGE (n)-[r:RETRIEVED_IN]->(s)
            SET r.retrieval_rank = $rank,
                r.retrieval_score = $score,
                r.retrieval_context = $query,
                r.created_at = datetime()
            """,
            node_id=hit.id, session_id=session_id, rank=rank,
            score=hit.score, query=query,
        )
```

**Commit:** `memory/sidecar: RETRIEVED_IN edge writer (D5 cycle 1 logging)`.

---

### Task 10 — SessionEnd scorer — utility score writer

**Files:**
- Create: `memory/sidecar/scoring.py` — utility scorer
- Create: `memory/sidecar/routers/sessions.py` — `POST /api/sessions/{id}/score` endpoint
- Create: `memory/sidecar/tests/test_scoring.py`

**Goal:** At SessionEnd, walk all `RETRIEVED_IN` edges for the session and compute utility score per design §5:

```
utility_score [-1, +1] = sum of:
  + 0.5 if memory's content appears in any assistant output during session (textual evidence)
  + 0.3 if task completed successfully (no revert, no error in final turns)
  - 0.4 if agent immediately re-queried for similar info (insufficient retrieval signal)
  - 0.2 if memory was ranked top-3 but never referenced
```

Endpoint signature:

```python
@router.post("/sessions/{session_id}/score")
async def score_session(
    session_id: str,
    payload: ScoringPayload,
    memory: MemoryDep,
) -> ScoringReport:
    """Walk RETRIEVED_IN edges; compute utility; write back."""
```

`ScoringPayload` carries the session's full transcript + tool outputs (passed from SessionEnd hook).

**Commit:** `memory/sidecar: SessionEnd utility scorer (D5 cycle 1 signal)`.

---

## Phase 2C — Claude Code Hooks (Tasks 11-15)

### Task 11 — Hook framework + HTTP client

**Files:**
- Create: `memory/hooks/claude-code/lib/sidecar_client.py` — sidecar HTTP wrapper
- Create: `memory/hooks/claude-code/lib/__init__.py`
- Create: `memory/hooks/claude-code/lib/types.py` — Pydantic for hook message format
- Create: `memory/hooks/claude-code/tests/test_sidecar_client.py`

**Goal:** Shared library all 4 hooks import. Wraps `POST /api/{label}` + `GET /api/search/{label}`. Graceful degradation when sidecar is down (logs warning, no-ops — the hook MUST NOT block the session).

**Critical constraint:** hooks run in the user's shell with no internet access guarantee, so HTTP timeouts must be tight (≤3s) and failures must be silent except to a local log file.

**Commit:** `memory/hooks: sidecar HTTP client + graceful degradation`.

---

### Task 12 — SessionStart hook

**Files:**
- Create: `memory/hooks/claude-code/session_start.py`
- Modify: `~/.claude/hooks.json` (instructions for user; not committed to repo)

**Goal:** At session start, query "top-of-mind" memories for the current scope + inject a curated summary into context.

**Algorithm:**

1. Detect scope via `memory.scope.detect_scope(cwd)`.
2. Query sidecar for top-K nodes per scope:
   - Most-recently-created Decisions (≤5)
   - Active (non-expired) Beliefs (≤5)
   - Open Tasks (≤3)
   - Open Failures (≤3)
3. Format as a compact markdown block (≤30 lines).
4. Print to stdout (Claude Code injects into context).

Output skeleton:

```markdown
# Galvo Memory — top-of-mind (scope: project:galvo)

## Recent decisions
- 2026-05-17: Chose Neo4j Community over KuzuDB (rationale: in-index filtering)
- 2026-05-16: F sub-project shipped at a2e3959 (5 acceptance gates clean)
...

## Active beliefs
- ScoreRow shape is nested ({score: Score}), not flat
- pass_/pass alias requires # type: ignore[call-arg] OR _make_score helper
...

## Open tasks
- Memory Layer Phase 2 (this!)
- D — Hub Dashboard brainstorm pending
...
```

**Token budget:** ≤50 lines per design §10. Hard truncate.

**Commit:** `memory/hooks: SessionStart top-of-mind injector`.

---

### Task 13 — UserPromptSubmit hook

**Files:**
- Create: `memory/hooks/claude-code/user_prompt_submit.py`

**Goal:** On every user prompt, do a targeted semantic search against the graph + inject the top 3-5 hits. Quick (≤500ms) — semantic search via embedding similarity.

**Algorithm:**

1. Extract the user's prompt text.
2. `POST /api/search/Entity` with `q={prompt_text[:500]}` (truncate to embedder context length).
3. Filter results by scope + by recency (last 30 days weighted higher).
4. Format ≤5 hits as a markdown context block.
5. Output goes to stdout (injected before the prompt reaches the model).

**Commit:** `memory/hooks: UserPromptSubmit semantic-retrieval injector`.

---

### Task 14 — PostToolUse hook

**Files:**
- Create: `memory/hooks/claude-code/post_tool_use.py`

**Goal:** After every tool call, log relevant state changes as graph nodes/edges:
- `Read` / `Edit` / `Write` → write `Artifact` node (or update `last_touched`) + `TOUCHED` edge from current `Session`
- `Bash` calls matching git patterns → write `Commit` node
- Test/build commands → write `Test` or `Failure` node based on exit code
- Other tools → log lightly (tool name + timestamp in session log)

**Detection logic:** parse the tool name + args from the hook's input JSON. Don't over-fire; conservative is better than noisy.

**Commit:** `memory/hooks: PostToolUse artifact + commit + test logger`.

---

### Task 15 — SessionEnd hook

**Files:**
- Create: `memory/hooks/claude-code/session_end.py`

**Goal:** At session end:
1. Write final `Session` node properties (ended_at, outcome, task_description summary).
2. Submit transcript + tool outputs to `POST /api/sessions/{id}/score` (Task 10 endpoint).
3. Apply utility scores to all `RETRIEVED_IN` edges from this session.

**Commit:** `memory/hooks: SessionEnd session-node writer + retrieval scorer`.

---

## Phase 2D — Integration (Tasks 16-20)

### Task 16 — File watcher daemon

**Files:**
- Create: `memory/watcher/__init__.py`
- Create: `memory/watcher/daemon.py` — watchdog-based daemon
- Create: `memory/watcher/parsers.py` — AGENTS.md / CLAUDE.md / .cursorrules / .codex/* parsers

**Goal:** Daemon process watching for changes to instruction files in any active project. On change:
1. Parse the file's contents.
2. Diff against current ingested representation.
3. Write new `Convention` nodes (source = "from_AGENTS.md" etc.) and SUPERSEDE old ones.

Runs as part of the sidecar lifespan OR as a separate `memory-watcher` Python process.

**Commit:** `memory/watcher: AGENTS.md/CLAUDE.md file watcher + Convention writer`.

---

### Task 17 — Promote-action CLI (graph → AGENTS.md)

**Files:**
- Create: `memory/cli/promote.py` — `python -m memory.cli promote <node_id> --to AGENTS.md`

**Goal:** Explicit operator action — pick a graph node and write it back to AGENTS.md / CLAUDE.md / a feedback_*.md file. Per design §10: "graph is canonical, files are inputs only, with explicit promote action."

**Commit:** `memory/cli: promote command (graph node → markdown file)`.

---

### Task 18 — docker-compose: sidecar service

**Files:**
- Modify: `memory/docker/docker-compose.yml` — add sidecar service alongside Neo4j

**Goal:** `docker compose up -d` starts BOTH Neo4j + sidecar. Sidecar depends_on Neo4j healthcheck.

Container config:
- Builds from `memory/sidecar/Dockerfile`
- Ports: 7575
- Environment: NEO4J_URI=bolt://neo4j:7687, etc.
- Healthcheck: `curl http://localhost:7575/health`

**Commit:** `memory/docker: add sidecar service to docker-compose`.

---

### Task 19 — Demo script + README

**Files:**
- Create: `memory/examples/sidecar_demo.sh`
- Modify: `memory/README.md` — update with cycle-1 status + run instructions

**Goal:** `bash memory/examples/sidecar_demo.sh` does an end-to-end smoke:
1. `docker compose up -d` (Neo4j + sidecar)
2. `curl :7575/health` — expect 200
3. POST a Decision, Belief, Session via REST
4. GET search results, verify embeddings work
5. POST a session-score request, verify RETRIEVED_IN edges get utility_score
6. Exit 0

**Commit:** `memory/examples: end-to-end sidecar demo`.

---

### Task 20 — Acceptance gate sweep + merge

Run the 6 acceptance gates. Iterate on failures inline. Then:

1. Final code review (dispatch reviewer subagent if executing via subagent-driven-development).
2. Merge `memory-layer` to `main` from the main checkout via `git merge --no-ff`.
3. Push, clean up worktree + branch.
4. Update `~/.claude/projects/.../memory/MEMORY.md` to point at the new graph (or just document the transition).
5. Write `project_memory_layer_progress.md` SHIPPED note.

**Commit:** the merge commit is the final artifact.

---

## Cycle 2 parking lot (explicit deferrals)

Items not in cycle 1 per locked D2/D5 + scope discipline:

- **Qwen3-Embedding-8B via MLX** (D2 cycle-2 upgrade) — wire as `CUSTOM` embedder, drop DB, re-bootstrap with 4096-dim index
- **Consolidation service ("dream-state")** (D5 cycle-2) — edge-weight updates, belief invalidation, redundancy merging, low-utility pruning
- **Codex + VS Code adapters** (design §8, Phase 3) — MCP-only for non-Claude clients (no hooks)
- **VS Code sidebar memory inspector** — visual graph browsing, manual invalidation
- **Hub federation publication of scores** (cross-pillar integration with sub-F)
- **Graph-Foundation-Model reader** (full SAGE) — explicit non-goal per design §13

---

## Self-Review

### Spec coverage

- D1 (full stack): Tasks 6-10 (sidecar) + 11-15 (hooks) + the existing MCP server wired in Phase 1.
- D2 (MiniLM): Task 6 lifespan hardcodes the embedder config; cycle-2 swap is a config change.
- D3 (LLM-side extraction): no library NER deps; Tasks 12-13 hooks DO call the library's MCP tools which let the LLM decide what to extract.
- D4 (3-tier scope): Tasks 4-5 (detection + Cypher helpers) + property schema in Task 2.
- D5 cycle 1 (logging + scoring): Tasks 9-10 + 15.

### Type consistency

- `EntityType` values from `neo4j_agent_memory` package root, NOT `schema.models` (probed in Phase 1).
- `LABEL_TO_TYPE` dict canonical in `memory/ontology/label_mapping.py`; imported everywhere.
- `RETRIEVED_IN` edge schema (rank/score/context/utility_score) consistent across Tasks 9 + 10.
- `Session` node id format: `session_<uuid>` per library convention (verify in Task 11).

### Known unknowns (flagged as task-internal risks)

- Task 8: does `add_entity(extra_labels=, properties=)` exist in v0.2.1? Drop-down to raw Cypher if not.
- Task 12-13: hook-injection token budget — design §10 says ~30-50 slots; need to measure post-implementation.
- Task 16: file watcher daemon process lifecycle — is the sidecar the right host, or a separate `memory-watcher` process? Decide in Task 18.

---

## Execution Handoff

Plan saved to `memory/docs/PHASE-2-PLAN.md`. Recommended execution:

**Subagent-Driven (recommended)** — Agent Swarm pattern per sub-F precedent. Dispatch waves:
- Wave A: Task 1 (risk gate) — single agent, serial
- Wave B: Tasks 2, 3, 4 — 3 parallel agents in worktrees
- Wave C: Task 5 — single agent (depends on Task 4)
- Wave D: Tasks 6, 7 — sequential
- Wave E: Tasks 8, 9, 10 — 3 parallel agents
- Wave F: Task 11 — single agent
- Wave G: Tasks 12, 13, 14, 15 — 4 parallel agents
- Wave H: Tasks 16, 17, 18 — 3 parallel agents
- Wave I: Tasks 19, 20 — sequential (acceptance + merge)

Total: 9 waves, ~20 tasks, peak 4-way parallelism.
