# Galvo Memory Examples

Operator-facing smokes and walkthroughs.

## Contents

| File | Purpose |
|---|---|
| `sidecar_demo.sh` | End-to-end smoke that exercises all 6 cycle-1 acceptance gates. Drives the sidecar via curl. |
| `__init__.py` | Reserves this directory as a Python package for future Python-native demos. |

## Running `sidecar_demo.sh`

**Prerequisites:** bash, curl, python3. Neo4j + sidecar running on default ports.

### Quick start (stack already up)

```bash
cd memory/docker && docker compose up -d
# Wait ~10–30s for the embedder to warm on first boot; check with:
curl -fsS http://localhost:7575/health

bash memory/examples/sidecar_demo.sh
```

Exit code `0` means all 6 gates passed. Non-zero means the first failing gate aborted; scroll up for the `[demo] FAIL: ...` line.

### Bring the stack up + tear it down inside the script

```bash
GALVO_MEMORY_DEMO_STARTUP=1 bash memory/examples/sidecar_demo.sh
```

Useful for CI and Task 20's automated sweep. The script waits up to 90s for the sidecar to report healthy, then runs the gates, then `docker compose down`s.

### Point at a non-default sidecar

```bash
GALVO_MEMORY_SIDECAR_URL=http://localhost:7600 bash memory/examples/sidecar_demo.sh
```

## What the 6 gates verify

Authoritative descriptions: [`../docs/PHASE-2-PLAN.md`](../docs/PHASE-2-PLAN.md) §"Acceptance gates (cycle 1 close)".

1. **`/health` is 200** — Neo4j connection + embedder both report OK.
2. **Custom ontology applied** — `POST /api/Decision` succeeds (label routing + 12-label Pydantic validation both work).
3. **CRUD round-trip** — Belief + Convention POSTs land, and `GET /api/search/Decision` returns the planted node.
4. **Scope partitioning** — a `scope=personal` Decision does NOT leak into a `scope=project:demo` search.
5. **Feedback signal lands** — `POST /api/sessions/{id}/score` returns a `ScoringReport` (utility-score writer reachable).
6. **Clean exit** — script completes with exit code 0.

## When a gate fails

The script exits non-zero on the *first* failing gate with a `[demo] FAIL: <reason> | body=<curl response>` line. Map the gate number back to its plan section to find the unit under test:

- Gate 1 → `memory/sidecar/app.py` (lifespan + `/health`)
- Gate 2 → `memory/ontology/label_mapping.py` + `memory/sidecar/routers/nodes.py`
- Gate 3 → `memory/sidecar/cypher_helpers.py::build_scoped_search_with_embedding`
- Gate 4 → same helper, scope-filter branch
- Gate 5 → `memory/sidecar/scoring.py` + `memory/sidecar/routers/sessions.py`
- Gate 6 → the script itself; usually a Gate 1-5 failure short-circuits here

Re-run with `GALVO_MEMORY_DEMO_STARTUP=1` if you suspect the failure is stack-state-related (e.g. embedder didn't warm in time).
