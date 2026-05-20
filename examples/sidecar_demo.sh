#!/usr/bin/env bash
# Galvo memory cycle-1 end-to-end smoke.
#
# Default: assumes Neo4j + sidecar are already running (you ran
#   `cd memory/docker && docker compose up -d`
# and `curl :7575/health` is green).
#
# Set GALVO_MEMORY_DEMO_STARTUP=1 to also bring the stack up + tear it
# down inside this script. Useful for CI / Task 20's automated sweep.
#
# Set GALVO_MEMORY_SIDECAR_URL=... to point at a non-default sidecar
# (e.g. http://localhost:7600 when running two stacks side by side).
#
# What this verifies — the 6 acceptance gates from PHASE-2-PLAN.md §"Acceptance
# gates (cycle 1 close)":
#
#   1. Sidecar /health  — 200 with neo4j=ok + embedder field present.
#   2. Custom ontology  — POST a Decision succeeds (label routed, body
#                          validated against the per-label Pydantic model).
#   3. CRUD round-trip  — Belief + Convention POST, then GET /api/search
#                          returns the planted Decision.
#   4. Scope partition  — a scope=personal node is NOT returned by a
#                          project:demo search.
#   5. Feedback signal  — POST /api/sessions/{id}/score returns a
#                          ScoringReport (utility-score writer wired).
#   6. Clean exit       — script completes exit 0.
#
# Failure mode: the script exits non-zero on the FIRST gate that fails, so
# Task 20's iteration loop can focus on one gate without scrolling. Every
# step prints a `[demo] ===` banner so the failing gate is grep-able from
# stack-up logs.
#
# Dependencies: bash, curl, python3 (used for JSON assertions — keeps us
# off jq, which isn't on every macOS by default).

set -euo pipefail

SIDECAR=${GALVO_MEMORY_SIDECAR_URL:-http://localhost:7575}
# Unique-per-run id so re-runs against the same Neo4j volume don't collide
# on name-based search assertions. `date +%s` gives second precision which
# is enough — operators won't re-run within the same second.
DEMO_ID="demo_$(date +%s)_$$"

# Compose dir is relative to the repo root; this script lives at
# memory/examples/, so docker compose runs from memory/docker/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="${SCRIPT_DIR}/../docker"

# --- helpers ---------------------------------------------------------------

# step / ok / fail: status logging on stderr-adjacent stdout so a wrapping
# Task-20 runner can tee the output and still grep for the banner.
step() { printf '\n[demo] === %s ===\n' "$*"; }
ok()   { printf '[demo] ok: %s\n' "$*"; }
fail() { printf '[demo] FAIL: %s\n' "$*" >&2; exit 1; }

# curl_json METHOD URL [JSON_BODY] — fail-fast curl that emits the response
# body on stdout. `-f` would swallow the body on 4xx/5xx; we want the body
# for diagnostics, so we use `--fail-with-body` (curl 7.76+). macOS ships
# 8.x by default, Linux distros >= Ubuntu 22.04 do too.
curl_json() {
    local method=$1 url=$2 body=${3:-}
    if [ -n "$body" ]; then
        curl -sS --fail-with-body \
            -X "$method" \
            -H 'content-type: application/json' \
            "$url" -d "$body"
    else
        curl -sS --fail-with-body -X "$method" "$url"
    fi
}

# json_assert "<python expr>" "<input json>" "<fail msg>"
# Runs `python3 -c` with `import json,sys` + the user expr already in scope;
# stdin = input json. Used so assertion bodies stay short at call sites.
json_assert() {
    local expr=$1 input=$2 msg=$3
    if ! printf '%s' "$input" \
        | python3 -c "import json,sys
d=json.loads(sys.stdin.read())
assert $expr, f'json_assert: {d!r}'" 2>/dev/null
    then
        fail "$msg | body=$input"
    fi
}

# --- optional stack-up -----------------------------------------------------

if [ "${GALVO_MEMORY_DEMO_STARTUP:-0}" = "1" ]; then
    step "Starting stack via docker compose (GALVO_MEMORY_DEMO_STARTUP=1)"
    (cd "$COMPOSE_DIR" && docker compose up -d)
    printf '[demo] waiting for sidecar /health'
    # Up to ~90s — first cold start downloads the MiniLM checkpoint
    # (~90MB) and warms the embedder. Compose healthcheck has its own
    # 8-retry/15s policy; we poll independently so this script's wait
    # is visible to the operator.
    for _ in $(seq 1 90); do
        if curl -fsS "$SIDECAR/health" > /dev/null 2>&1; then
            printf ' up\n'
            break
        fi
        printf '.'
        sleep 1
    done
fi

# --- Gate 1: /health -------------------------------------------------------

step "Gate 1 — sidecar /health"
health=$(curl_json GET "$SIDECAR/health") \
    || fail "sidecar unreachable at $SIDECAR (is docker compose up -d done?)"

# Health shape (sidecar/app.py::health):
#   {"status": "ok", "neo4j": "ok", "embedder": "<model_name>",
#    "embedding_dimensions": 384, "stats": {...}}
# We assert the two strict-equal fields + that embedder is a non-empty
# string (the model name varies depending on env override; cycle-1
# default is all-MiniLM-L6-v2 but we don't pin it here).
json_assert \
    "d.get('status') == 'ok' and d.get('neo4j') == 'ok' and isinstance(d.get('embedder'), str) and d['embedder']" \
    "$health" \
    "/health body did not match {status=ok, neo4j=ok, embedder=<str>}"

ok "/health returned 200 with neo4j=ok"

# --- Gate 2: ontology in place (POST Decision) -----------------------------

step "Gate 2 — ontology in place (POST Decision)"
# Decision schema (sidecar/models.py::DecisionCreate): scope + title + rationale.
# We bake $DEMO_ID into the title so the Gate-3 search query can find it
# even when prior demo runs accumulated nodes in the same Neo4j volume.
dec_body=$(python3 -c "
import json
print(json.dumps({
    'scope': 'project:demo',
    'title': 'demo decision $DEMO_ID',
    'rationale': 'verify the 12-label ontology adopt() ran',
    'confidence': 0.8,
}))")
decision=$(curl_json POST "$SIDECAR/api/Decision" "$dec_body") \
    || fail "POST /api/Decision failed — is the custom ontology applied? \
(see memory/ontology/label_mapping.py)"
dec_id=$(printf '%s' "$decision" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["id"])')
ok "Decision created: id=$dec_id"

# --- Gate 3: Belief + Convention CRUD + search round-trip ------------------

step "Gate 3 — Belief + Convention CRUD"

# BeliefCreate: scope + claim (NOT title/name; design §4 belief immutability).
bel_body=$(python3 -c "
import json
print(json.dumps({
    'scope': 'project:demo',
    'claim': 'demo $DEMO_ID belief — sidecar honors immutability',
    'confidence': 0.9,
}))")
curl_json POST "$SIDECAR/api/Belief" "$bel_body" > /dev/null \
    || fail "POST /api/Belief failed"
ok "Belief created"

# ConventionCreate: scope + name + description (NOT rule_text — that was
# the strawman that the design walked back; verified in sidecar/models.py).
conv_body=$(python3 -c "
import json
print(json.dumps({
    'scope': 'project:demo',
    'name': 'demo $DEMO_ID convention',
    'description': 'always smoke-test before shipping',
    'source': 'explicit',
}))")
curl_json POST "$SIDECAR/api/Convention" "$conv_body" > /dev/null \
    || fail "POST /api/Convention failed"
ok "Convention created"

step "Gate 3 — search returns the planted Decision"
# /api/search/{label} returns a FLAT list (sidecar/routers/nodes.py — NOT
# a {results: [...]} envelope). The embedder is loaded by lifespan so the
# vector path is exercised; we still match on the unique DEMO_ID to dodge
# any "the embedder is sleepy on first call" noise.
search=$(curl_json GET \
    "$SIDECAR/api/search/Decision?q=demo+decision+$DEMO_ID&scope=project%3Ademo&limit=10")

# Decision's name property is `title` (per ontology.label_mapping::NAME_PROPERTY_PER_LABEL).
# We check both `title` and `name` because the row projection includes the
# library's `name` key too — either match is fine for the gate's intent.
json_assert \
    "isinstance(d, list) and any('$DEMO_ID' in ((h.get('title') or h.get('name') or '')) for h in d)" \
    "$search" \
    "search did not return the planted Decision"
ok "search returned planted Decision"

# --- Gate 4: scope partitioning --------------------------------------------

step "Gate 4 — scope partitioning (personal node hidden from project search)"
# Plant a personal-scope Decision with a distinguishing title fragment.
personal_marker="personal_only_${DEMO_ID}"
personal_body=$(python3 -c "
import json
print(json.dumps({
    'scope': 'personal',
    'title': '$personal_marker',
    'rationale': 'should not appear in project:demo searches',
}))")
curl_json POST "$SIDECAR/api/Decision" "$personal_body" > /dev/null \
    || fail "POST /api/Decision (personal scope) failed"

# Search the SAME marker text under project:demo scope. Per the scope rules
# (cypher_helpers.build_scoped_search_with_embedding): a project query sees
# its own scope + universal, but NEVER personal. So zero hits = correct.
proj_search=$(curl_json GET \
    "$SIDECAR/api/search/Decision?q=$personal_marker&scope=project%3Ademo&limit=10")

json_assert \
    "isinstance(d, list) and not any('$personal_marker' in ((h.get('title') or h.get('name') or '')) for h in d)" \
    "$proj_search" \
    "personal-scope node LEAKED into project:demo search"
ok "personal node correctly hidden from project search"

# --- Gate 5: feedback signal -----------------------------------------------

step "Gate 5 — feedback signal (session scoring endpoint)"
# Create the Session node first via the generic CRUD. SessionCreate fields
# (sidecar/models.py::SessionCreate): scope + title + task_description (+
# optional agent_tool, started_at). We pass `id` so the score endpoint can
# address it by the same key.
session_db_id="sess_$DEMO_ID"
sess_body=$(python3 -c "
import json
print(json.dumps({
    'id': '$session_db_id',
    'scope': 'project:demo',
    'title': 'demo session $DEMO_ID',
    'task_description': 'cycle-1 end-to-end smoke',
    'agent_tool': 'claude-code',
}))")
# Don't `fail` on this — if the Session writer ever regresses, the scorer
# below will still exercise the endpoint and return a zero-edge report,
# which is what Gate 5 actually verifies (the wire is connected).
curl_json POST "$SIDECAR/api/Session" "$sess_body" > /dev/null \
    || printf '[demo] note: POST /api/Session non-2xx; scorer endpoint still tested below\n'

# Scoring payload (sidecar/scoring.py::ScoringPayload): session_id,
# assistant_outputs (list[str]), task_outcome (str), requeries (list[str]).
# Cycle-1 default outcomes that aren't lowercase 'success' don't fire the
# +0.3 task-success signal — we send 'success' so the scorer at least
# exercises the positive branch even when there are zero RETRIEVED_IN
# edges for this session (a brand-new session has none until the
# UserPromptSubmit hook logs retrievals).
score_body=$(python3 -c "
import json
print(json.dumps({
    'session_id': '$session_db_id',
    'assistant_outputs': ['demo completed successfully'],
    'task_outcome': 'success',
    'requeries': [],
}))")
score=$(curl_json POST "$SIDECAR/api/sessions/$session_db_id/score" "$score_body") \
    || fail "POST /api/sessions/{id}/score failed — Task-10 scorer not wired?"

# ScoringReport shape (sidecar/scoring.py::ScoringReport): session_id +
# edges_scored + edges_skipped + scores[]. We don't require any edges in
# a fresh session — just confirm the endpoint returned the right shape.
json_assert \
    "d.get('session_id') == '$session_db_id' and 'edges_scored' in d and 'edges_skipped' in d and isinstance(d.get('scores'), list)" \
    "$score" \
    "score response did not match ScoringReport shape"
ok "session scored — feedback edge-writer reachable (edges_scored=$(printf '%s' "$score" | python3 -c 'import json,sys; print(json.load(sys.stdin)["edges_scored"])'))"

# --- Gate 6: clean exit ----------------------------------------------------

if [ "${GALVO_MEMORY_DEMO_STARTUP:-0}" = "1" ]; then
    step "Stopping stack (GALVO_MEMORY_DEMO_STARTUP=1)"
    (cd "$COMPOSE_DIR" && docker compose down)
fi

step "Gate 6 — clean exit"
ok "demo complete; all 6 cycle-1 acceptance gates passed for demo_id=$DEMO_ID"
