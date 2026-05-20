"""Task 8 — REST CRUD router for the 12-label memory ontology.

Two tiers, matching the rest of the suite:

* Unit tests patch :class:`MemoryClient` so the lifespan never touches
  Neo4j. They exercise route wiring + per-label Pydantic validation +
  the Cypher composition.
* ``@pytest.mark.integration`` tests boot the real lifespan against
  ``bolt://localhost:7687`` (matches ``memory/docker`` compose stack).
  Skipped by ``pytest -m 'not integration'``.

What we're protecting:

* Each of the 12 labels has a working ``POST /api/{label}`` validated
  against its create model.
* The label whitelist is enforced (``404`` for unknown labels).
* ``GET /api/{label}/{id}`` round-trips a created node.
* ``PATCH /api/{label}/{id}`` accepts the per-label update model and
  rejects extra keys.
* ``PATCH /api/Belief/{id}`` returns ``405 Method Not Allowed`` — Beliefs
  are immutable per design §4.
* ``GET /api/search/{label}`` calls the Task-5 vector-search helper
  with the scope filter and threshold passed through.
* Auto-generated ids follow the ``node_<hex>`` shape.
* The router's Cypher writes both labels (``:Entity:<Label>``) — without
  this, Task-5's helpers can never find the node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Mock helpers — the unit tests rely on a structured mock client.
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Build a :class:`MagicMock` shaped like a real ``MemoryClient``.

    Attaches:

    * ``connect`` / ``close`` / ``get_stats`` — lifespan hooks.
    * ``graph.execute_read`` / ``execute_write`` — the router's primary
      Cypher entry points.
    * ``long_term.embedder.embed`` — the embedding generator path
      :func:`sidecar.routers.nodes._embed_text` calls.

    The execute_* methods default to returning empty lists; tests
    override the return value per scenario via ``.side_effect`` or
    ``.return_value``.
    """
    instance = MagicMock()
    instance.connect = AsyncMock(return_value=None)
    instance.close = AsyncMock(return_value=None)
    instance.get_stats = AsyncMock(return_value={"entities": 0, "facts": 0})

    instance.graph = MagicMock()
    instance.graph.execute_write = AsyncMock(return_value=[])
    instance.graph.execute_read = AsyncMock(return_value=[])

    # The embedder is a real-looking object with an ``embed`` coroutine.
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 384)
    instance.long_term = MagicMock()
    instance.long_term.embedder = embedder

    return instance


@pytest.fixture
def mocked_client() -> "Iterator[MagicMock]":
    """Patch ``sidecar.app.MemoryClient`` so the lifespan returns our
    fixture mock. The patch swaps the class symbol — instantiations
    inside the lifespan yield the same :class:`MagicMock`.
    """
    with patch("sidecar.app.MemoryClient") as MockClient:
        instance = _make_mock_client()
        MockClient.return_value = instance
        yield instance


def _captured_cypher(mock_call: MagicMock) -> str:
    """Return the first positional arg from a ``call_args`` capture.

    ``execute_read`` / ``execute_write`` are called as
    ``(query, params)`` — we sometimes want to assert on the query text.
    """
    return mock_call.call_args.args[0]


def _captured_params(mock_call: MagicMock) -> dict:
    """Return the params dict from a ``call_args`` capture."""
    return mock_call.call_args.args[1]


# ---------------------------------------------------------------------------
# Helper: a CREATE-call mock that echoes the written props back as a row.
# ---------------------------------------------------------------------------


def _echo_props_on_create(client: MagicMock) -> None:
    """Wire ``graph.execute_write`` so it pretends Neo4j wrote the
    props dict and returns it as a row.

    The router's ``CREATE`` query passes ``{"props": props}``. We pull
    the props back out and wrap them in the row shape the router
    expects (``{"n": {...}}``).
    """

    async def _fake_write(query: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        if "CREATE" in query and "props" in params:
            row_props = dict(params["props"])
            # Server-stamped fields the router expects to find on the row.
            row_props.setdefault("created_at", "2026-05-19T12:00:00+00:00")
            return [{"n": row_props}]
        if "SET" in query and "vals" in params:
            # PATCH path: combine the existing node id with the SET diff
            # so the router's response model has the keys it needs.
            return [
                {
                    "n": {
                        "id": params.get("id"),
                        # Minimal stub — tests that need more should
                        # patch ``execute_read``/``execute_write`` themselves.
                        **params["vals"],
                    }
                }
            ]
        return []

    client.graph.execute_write.side_effect = _fake_write


# ===========================================================================
# Router wiring — does the FastAPI app actually mount these routes?
# ===========================================================================


def test_router_mounted_on_app(mocked_client: MagicMock) -> None:
    """The app exposes routes prefixed ``/api`` after the lifespan runs.

    A regression that forgot ``app.include_router(nodes.router)`` would
    leave the routes 404 even though the module imports cleanly.
    """
    from sidecar.app import app

    with TestClient(app) as _:
        paths = {
            getattr(r, "path", None)
            for r in app.routes
            if getattr(r, "path", None) is not None
        }

    # We don't pin every path; just confirm the prefix is present and
    # that at least one of the 12 labels has a POST/GET path mounted.
    assert any(p.startswith("/api/") for p in paths if p), (
        f"No /api/* routes mounted; routes={sorted(p for p in paths if p)}"
    )


# ===========================================================================
# Label validation — unknown labels 404.
# ===========================================================================


@pytest.mark.parametrize(
    "verb,url",
    [
        ("post", "/api/UnknownLabel"),
        ("get", "/api/UnknownLabel/some-id"),
        ("patch", "/api/UnknownLabel/some-id"),
        ("get", "/api/search/UnknownLabel?q=hello"),
    ],
)
def test_unknown_label_returns_404(
    mocked_client: MagicMock, verb: str, url: str
) -> None:
    """The router rejects labels not in :data:`LABEL_TO_TYPE` with 404.

    All four verbs share the same ``_ensure_label`` gate, so a
    parametrize gives us coverage without four near-identical tests.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        if verb == "post":
            response = client.post(url, json={"scope": "personal"})
        elif verb == "patch":
            response = client.patch(url, json={"description": "x"})
        else:
            response = client.get(url)

    assert response.status_code == 404, response.text
    body = response.json()
    assert "Unknown label" in body["detail"], body


# ===========================================================================
# POST /api/{label} — happy path for every label.
# ===========================================================================


# Minimum valid create body per label. Mirrors design §4 properties.
_VALID_PAYLOADS = {
    "Decision": {
        "scope": "project:galvo",
        "title": "Use Neo4j over KuzuDB",
        "rationale": "In-index filtering matters for cycle-1 scope partitioning",
    },
    "Pattern": {
        "scope": "project:galvo",
        "name": "TDD per feature",
        "description": "Write failing test, implement, verify green.",
    },
    "Convention": {
        "scope": "project:galvo",
        "name": "snake_case for Python",
        "description": "PEP-8 baseline for the Galvo codebase.",
    },
    "Constraint": {
        "scope": "project:galvo",
        "name": "All routes async",
        "description": "FastAPI in async mode — no sync endpoints.",
        "constraint_type": "performance",
    },
    "Task": {
        "scope": "project:galvo",
        "title": "Implement Task 8",
        "description": "REST CRUD for 12 node types",
    },
    "Session": {
        "scope": "project:galvo",
        "title": "Memory layer Phase 2 wave E",
        "task_description": "Build out the sidecar surface",
    },
    "Mistake": {
        "scope": "project:galvo",
        "summary": "Misread API signature",
        "description": "Assumed add_entity accepted extra_labels in v0.2.1.",
    },
    "Commit": {
        "scope": "project:galvo",
        "sha": "abc1234deadbeef",
        "message": "memory/sidecar: Task 8 REST CRUD",
    },
    "Failure": {
        "scope": "project:galvo",
        "error_signature": "ImportError: No module named 'sidecar.routers'",
        "failure_type": "build",
    },
    "Artifact": {
        "scope": "project:galvo",
        "path": "memory/sidecar/routers/nodes.py",
        "language": "python",
    },
    "Test": {
        "scope": "project:galvo",
        "identifier": "tests/test_nodes_router.py::test_create_decision_round_trip",
    },
    "Belief": {
        "scope": "project:galvo",
        "claim": "MemoryClient.long_term.add_entity does NOT accept extra_labels in v0.2.1",
        "confidence": 0.95,
    },
}


@pytest.mark.parametrize("label", sorted(_VALID_PAYLOADS))
def test_create_each_label_round_trips(
    mocked_client: MagicMock, label: str
) -> None:
    """Every label accepts its valid create body and returns a 201.

    Twelve assertions per parametrize — one per label. We test the
    happy-path round-trip: POST → 201 → body has the auto-minted id
    + the props we sent. The mock's ``execute_write`` echoes the
    written props back as a row.
    """
    from sidecar.app import app

    _echo_props_on_create(mocked_client)

    with TestClient(app) as client:
        response = client.post(f"/api/{label}", json=_VALID_PAYLOADS[label])

    assert response.status_code == 201, response.text
    body = response.json()
    # Auto-minted id, prefixed with ``node_``.
    assert body["id"].startswith("node_"), body
    assert body["scope"] == _VALID_PAYLOADS[label]["scope"]
    # The library-machine fields are stamped server-side.
    assert body["type"] in {"CONCEPT", "EVENT", "OBJECT", "FACT"}, body


def test_create_decision_validates_required_fields(
    mocked_client: MagicMock,
) -> None:
    """Missing required Decision fields → 422 with Pydantic detail.

    ``rationale`` is required by :class:`DecisionCreate`; a body that
    omits it must not write anything. We assert the mock was never
    called so the validation gate runs before the Cypher.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        response = client.post(
            "/api/Decision",
            json={"scope": "project:galvo", "title": "Use FastAPI"},
            # rationale missing
        )

    assert response.status_code == 422, response.text
    mocked_client.graph.execute_write.assert_not_awaited()


def test_create_rejects_extra_fields(mocked_client: MagicMock) -> None:
    """Bodies with unknown keys → 422 (Pydantic ``extra='forbid'``).

    The Task-8 plan declares strict per-label validation. Allowing
    unknown keys would silently drop bad client data and let typos
    persist (``alteranatives_considered`` → stored in metadata, never
    surfaced). 422 is the canonical FastAPI/Pydantic response.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        response = client.post(
            "/api/Decision",
            json={
                "scope": "project:galvo",
                "title": "X",
                "rationale": "Y",
                "nonexistent_field": "should reject",
            },
        )

    assert response.status_code == 422, response.text


def test_create_accepts_caller_supplied_id(mocked_client: MagicMock) -> None:
    """When the body includes ``id``, the router uses it verbatim.

    Hooks that want idempotent writes can hash a salt + payload and
    pass the result as the id. We confirm the router's CREATE Cypher
    bound that exact id in the params.
    """
    from sidecar.app import app

    _echo_props_on_create(mocked_client)

    with TestClient(app) as client:
        response = client.post(
            "/api/Decision",
            json={
                "id": "node_supplied123",
                "scope": "project:galvo",
                "title": "T",
                "rationale": "R",
            },
        )

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "node_supplied123"
    # The CREATE Cypher should have $props bound with that id.
    params = _captured_params(mocked_client.graph.execute_write)
    assert params["props"]["id"] == "node_supplied123"


def test_create_writes_multilabel_node(mocked_client: MagicMock) -> None:
    """The CREATE Cypher tags the node with BOTH ``:Entity`` and ``:<Label>``.

    This is the load-bearing invariant that lets Task-5's helpers find
    these nodes via ``MATCH (n:Decision)`` and via the
    ``CALL db.index.vector.queryNodes`` filter that demands
    ``'Decision' IN labels(n)``. A regression that dropped the second
    label would silently break every search endpoint.
    """
    from sidecar.app import app

    _echo_props_on_create(mocked_client)

    with TestClient(app) as client:
        client.post("/api/Decision", json=_VALID_PAYLOADS["Decision"])

    cypher = _captured_cypher(mocked_client.graph.execute_write)
    assert ":Entity:Decision" in cypher, cypher


def test_create_passes_embedding_when_embedder_available(
    mocked_client: MagicMock,
) -> None:
    """The router asks the library embedder for a vector + writes it.

    Without this the vector index stays empty for our custom nodes and
    Phase-2 acceptance gate §3 (semantic retrieval at UserPromptSubmit)
    breaks. We assert the embedder was awaited with the label's
    name-property value AND that the embedding made it into the props.
    """
    from sidecar.app import app

    _echo_props_on_create(mocked_client)

    with TestClient(app) as client:
        client.post("/api/Decision", json=_VALID_PAYLOADS["Decision"])

    # ``title`` is the name-property for Decision per NAME_PROPERTY_PER_LABEL.
    mocked_client.long_term.embedder.embed.assert_awaited_once_with(
        "Use Neo4j over KuzuDB"
    )
    params = _captured_params(mocked_client.graph.execute_write)
    assert "embedding" in params["props"]
    assert len(params["props"]["embedding"]) == 384, (
        "Embedding dimensions changed — Phase-2 D2 locks 384 for cycle 1."
    )


def test_create_propagates_409_on_constraint_violation(
    mocked_client: MagicMock,
) -> None:
    """If Neo4j raises a constraint error, surface 409 Conflict.

    The Task-2 schema sets ``commit_sha_unique`` + per-label ``id``
    uniqueness. A duplicate POST must NOT 500 — the standard HTTP
    semantic for "ID collision" is 409, and the hook layer treats 409
    as "already there, no-op."
    """
    from sidecar.app import app

    mocked_client.graph.execute_write.side_effect = RuntimeError(
        "Node already exists with property `sha` = 'abc1234'"
    )

    with TestClient(app) as client:
        response = client.post("/api/Commit", json=_VALID_PAYLOADS["Commit"])

    assert response.status_code == 409, response.text
    assert "collision" in response.json()["detail"].lower()


# ===========================================================================
# GET /api/{label}/{id} — read one.
# ===========================================================================


def test_get_node_returns_row(mocked_client: MagicMock) -> None:
    """A successful read returns the node's properties as JSON.

    The mocked driver returns one row; the router projects it through
    ``_row_to_dict`` and we assert the id round-trips. ``execute_read``
    receives the id binding.
    """
    from sidecar.app import app

    mocked_client.graph.execute_read.return_value = [
        {
            "n": {
                "id": "node_xyz",
                "scope": "project:galvo",
                "title": "Use Neo4j",
                "type": "CONCEPT",
                "name": "Use Neo4j",
                "rationale": "scope filtering",
                "confidence": 0.9,
                "alternatives_considered": [],
                "created_at": "2026-05-19T12:00:00+00:00",
            }
        }
    ]

    with TestClient(app) as client:
        response = client.get("/api/Decision/node_xyz")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "node_xyz"
    params = _captured_params(mocked_client.graph.execute_read)
    assert params["id"] == "node_xyz"


def test_get_node_404_when_missing(mocked_client: MagicMock) -> None:
    """An empty Cypher result → 404 with the id in the detail."""
    from sidecar.app import app

    mocked_client.graph.execute_read.return_value = []

    with TestClient(app) as client:
        response = client.get("/api/Decision/does_not_exist")

    assert response.status_code == 404, response.text
    assert "does_not_exist" in response.json()["detail"]


# ===========================================================================
# PATCH /api/{label}/{id} — update path.
# ===========================================================================


def test_patch_decision_writes_only_supplied_fields(
    mocked_client: MagicMock,
) -> None:
    """PATCH body with two fields → SET clause for exactly those two.

    A regression that switched to ``SET n += $props`` would clobber
    library-managed properties (``type``, ``name``, ``embedding``) when
    a buggy client included them. We verify the SET clause is precise.
    """
    from sidecar.app import app

    _echo_props_on_create(mocked_client)

    with TestClient(app) as client:
        response = client.patch(
            "/api/Decision/node_x",
            json={"rationale": "updated", "confidence": 0.85},
        )

    assert response.status_code == 200, response.text
    cypher = _captured_cypher(mocked_client.graph.execute_write)
    assert "SET n.rationale = $vals.rationale" in cypher, cypher
    assert "SET n.confidence" in cypher or "n.confidence = $vals.confidence" in cypher
    # No clobber of library fields.
    assert "n.type" not in cypher
    assert "n.embedding" not in cypher


def test_patch_rejects_extra_fields(mocked_client: MagicMock) -> None:
    """Update models forbid extras — 422 on unknown keys."""
    from sidecar.app import app

    with TestClient(app) as client:
        response = client.patch(
            "/api/Decision/node_x", json={"nonexistent": "x"}
        )

    assert response.status_code == 422, response.text
    mocked_client.graph.execute_write.assert_not_awaited()


def test_patch_404_when_missing(mocked_client: MagicMock) -> None:
    """When the write returns no rows, surface 404."""
    from sidecar.app import app

    mocked_client.graph.execute_write.return_value = []

    with TestClient(app) as client:
        response = client.patch(
            "/api/Decision/missing_id", json={"rationale": "x"}
        )

    assert response.status_code == 404, response.text


def test_patch_belief_returns_405_method_not_allowed(
    mocked_client: MagicMock,
) -> None:
    """Beliefs are immutable per design §4.

    The router must reject PATCH explicitly with 405 + an Allow header
    pointing at the supported verbs. Mutating a Belief would corrupt
    the cycle-2 consolidation logic that walks SUPERSEDES edges.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        response = client.patch(
            "/api/Belief/node_belief123",
            json={"confidence": 0.5},
        )

    assert response.status_code == 405, response.text
    # Allow header is part of the HTTP spec for 405 responses.
    assert "Allow" in response.headers
    assert "GET" in response.headers["Allow"]
    # The error message mentions the design rationale so the developer
    # who hits this has a path to the right fix (create new + SUPERSEDES).
    detail = response.json()["detail"].lower()
    assert "immutable" in detail
    assert "supersedes" in detail


# ===========================================================================
# GET /api/search/{label} — semantic search.
# ===========================================================================


def test_search_calls_embedder_with_query(
    mocked_client: MagicMock,
) -> None:
    """The query string ``q`` is what we embed — not a default.

    A regression that hardcoded the embedded text would return the same
    hits for every query.
    """
    from sidecar.app import app

    mocked_client.graph.execute_read.return_value = []

    with TestClient(app) as client:
        client.get("/api/search/Decision?q=use+neo4j&scope=project:galvo")

    mocked_client.long_term.embedder.embed.assert_awaited_once_with("use neo4j")


def test_search_passes_scope_to_cypher_helpers(
    mocked_client: MagicMock,
) -> None:
    """The scope query param flows into the Cypher params dict.

    Task-5's :func:`build_scoped_search_with_embedding` binds ``$scope``
    when the scope is project/personal. We confirm the router populated
    that key with the requested value.
    """
    from sidecar.app import app

    mocked_client.graph.execute_read.return_value = []

    with TestClient(app) as client:
        client.get("/api/search/Decision?q=test&scope=project:galvo")

    params = _captured_params(mocked_client.graph.execute_read)
    assert params["scope"] == "project:galvo"
    # The embedding made it through too.
    assert "embedding" in params
    assert len(params["embedding"]) == 384


def test_search_universal_scope_omits_scope_param(
    mocked_client: MagicMock,
) -> None:
    """``scope=universal`` does NOT bind ``$scope`` — the helper inlines
    the literal predicate for the universal-only case (see Task-5).
    """
    from sidecar.app import app

    mocked_client.graph.execute_read.return_value = []

    with TestClient(app) as client:
        client.get("/api/search/Decision?q=test&scope=universal")

    params = _captured_params(mocked_client.graph.execute_read)
    assert "scope" not in params


def test_search_returns_score_alongside_node(
    mocked_client: MagicMock,
) -> None:
    """Vector hits surface their similarity score as ``_score`` on each item.

    The SessionEnd scorer (Task 10) reads this when computing utility
    weights. Without surfacing it the per-hit score is lost.
    """
    from sidecar.app import app

    mocked_client.graph.execute_read.return_value = [
        {
            "n": {
                "id": "node_a",
                "scope": "project:galvo",
                "title": "Decision A",
                "rationale": "R",
                "confidence": 0.7,
                "alternatives_considered": [],
                "type": "CONCEPT",
                "name": "Decision A",
            },
            "score": 0.91,
        }
    ]

    with TestClient(app) as client:
        response = client.get("/api/search/Decision?q=hello&scope=project:galvo")

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["_score"] == pytest.approx(0.91)


def test_search_falls_back_to_listing_without_embedder(
    mocked_client: MagicMock,
) -> None:
    """If the embedder isn't attached, search degrades to a recent-by-scope listing.

    Heavily-mocked tests / future configurations may not load the
    embedder. The endpoint should still return SOMETHING actionable,
    not an empty list and not a 500.
    """
    from sidecar.app import app

    mocked_client.long_term.embedder = None
    mocked_client.graph.execute_read.return_value = [
        {
            "n": {
                "id": "node_recent",
                "scope": "project:galvo",
                "title": "T",
                "rationale": "R",
                "confidence": 0.7,
                "alternatives_considered": [],
                "type": "CONCEPT",
                "name": "T",
            }
        }
    ]

    with TestClient(app) as client:
        response = client.get(
            "/api/search/Decision?q=anything&scope=project:galvo"
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "node_recent"


# ===========================================================================
# Integration — boots the real lifespan against live Neo4j (skip by default).
# ===========================================================================


@pytest.mark.integration
def test_decision_round_trip_against_live_neo4j() -> None:
    """End-to-end: create + read against the live ``memory/docker`` stack.

    Requires the Compose stack running with default password. First
    boot downloads the MiniLM checkpoint (~90MB) so the embedder is
    available for the search test.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        # CREATE
        create_resp = client.post(
            "/api/Decision",
            json={
                "scope": "project:integration-test",
                "title": f"Integration test {pytest.__version__}",
                "rationale": "test_nodes_router live-Neo4j integration",
                "confidence": 0.9,
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        node_id = create_resp.json()["id"]

        # READ
        get_resp = client.get(f"/api/Decision/{node_id}")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["id"] == node_id

        # SEARCH — over-tolerant because integration runs may have data
        # from other test modules; we only assert the call works.
        search_resp = client.get(
            "/api/search/Decision?q=integration%20test&scope=project:integration-test"
        )
        assert search_resp.status_code == 200, search_resp.text
        # Cleanup: cycle-1 has no DELETE endpoint by design; the
        # integration test deliberately accumulates nodes and relies
        # on ``docker compose down -v`` between integration runs.
