"""Task 10 — SessionEnd utility scorer tests.

Two tiers, matching the rest of the suite:

* Unit tests build a ``MagicMock`` :class:`MemoryClient` whose
  ``graph.execute_read`` / ``graph.execute_write`` are :class:`AsyncMock`s
  returning canned edge rows. The scorer is pure logic plus two Neo4j
  calls, so mocking those is enough to cover every signal in isolation
  and in combination.
* The endpoint tests reuse :class:`fastapi.testclient.TestClient` against
  the real ``sidecar.app:app``, overriding the :func:`get_memory_client`
  dependency to inject the same mocked client. This exercises the
  router → scoring.py wiring, including the 400 mismatch guard.

What we're protecting:

* Each of the four signals from design §5 fires when its precondition
  holds and *only* then. A regression that flipped a weight's sign or
  inverted a boolean would visibly change the per-edge breakdown.
* The clamp keeps ``utility_score`` in ``[-1.0, +1.0]`` even if a future
  fifth signal is added without re-checking the bounds.
* Per-edge write failures are absorbed into ``edges_skipped`` — one
  transient Neo4j blip MUST NOT abort the rest of the session's signal.
* Idempotency: re-running the scorer on the same session overwrites
  ``utility_score`` in place (no duplicate edges, no stale state).
* The router's path/body session-id mismatch returns 400 — a typo guard
  the hook layer relies on for cycle-1 debugging.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from sidecar.scoring import (
    ScoringPayload,
    ScoringReport,
    _score_edge,
    _semantic_overlap,
    score_session,
)


# ---------------------------------------------------------------------------
# Test helpers — shared edge-row + client builders.
# ---------------------------------------------------------------------------


def _make_edge(
    *,
    edge_id: str = "e1",
    node_id: str = "n1",
    rank: int = 0,
    context: str = "what is the snake_case rule",
    content: str = "Python uses snake_case",
) -> dict[str, object]:
    """Build a single edge-row dict matching the shape ``_fetch_edges``
    returns from Neo4j.

    Keeping the shape in one place means a refactor of the Cypher's
    ``RETURN`` aliases only has to update one fixture builder, not every
    test that constructs one.
    """
    return {
        "edge_id": edge_id,
        "node_id": node_id,
        "rank": rank,
        "context": context,
        "content": content,
    }


def _make_memory(edges: list[dict[str, object]]) -> MagicMock:
    """Build a :class:`MagicMock` :class:`MemoryClient` whose
    ``graph.execute_read`` returns ``edges`` and ``graph.execute_write``
    silently succeeds.

    Tests that want to assert per-call arguments grab the ``write_mock``
    off the returned object after the scorer has run.
    """
    memory = MagicMock()
    memory.graph = MagicMock()
    memory.graph.execute_read = AsyncMock(return_value=edges)
    memory.graph.execute_write = AsyncMock(return_value=None)
    return memory


# ---------------------------------------------------------------------------
# _semantic_overlap helper — the cycle-1 Jaccard.
# ---------------------------------------------------------------------------


def test_semantic_overlap_helper() -> None:
    """Word-set Jaccard returns the expected ratio.

    Inputs share ``"the"`` and ``"fox"`` of four total unique tokens (the,
    quick, fox, lazy) — 2/4 = 0.5. We use ``pytest.approx`` because float
    division can introduce tiny noise on some platforms.
    """
    assert _semantic_overlap("the quick fox", "the lazy fox") == pytest.approx(0.5)


def test_semantic_overlap_empty_returns_zero() -> None:
    """Either-side empty returns 0.0, not NaN or a div-by-zero.

    Sessions with no recorded requeries are normal — we don't want the
    scorer to crash on those.
    """
    assert _semantic_overlap("", "anything") == 0.0
    assert _semantic_overlap("anything", "") == 0.0
    assert _semantic_overlap("", "") == 0.0


def test_semantic_overlap_identical_returns_one() -> None:
    """Identical inputs Jaccard to 1.0 — sanity check the upper bound.

    A regression that double-counted intersection or single-counted union
    would push this above 1.0; a regression that swapped the operands
    would push it below.
    """
    assert _semantic_overlap("matched query", "matched query") == 1.0


# ---------------------------------------------------------------------------
# _score_edge — pure signal logic, no I/O.
# ---------------------------------------------------------------------------


def test_referenced_only() -> None:
    """Content appears in an assistant output → +0.5, nothing else.

    Single positive signal, no negatives. Rank > top-3 so the unreferenced
    negative can't fire. Task outcome neither success nor a known failure
    state — neutral.
    """
    edge = _make_edge(rank=5, content="snake_case", context="")
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["Python uses snake_case for variable names"],
        task_outcome="partial",
    )
    score = _score_edge(edge, payload)
    assert score.referenced is True
    assert score.task_success is False
    assert score.requeried_after is False
    assert score.was_top3_unreferenced is False
    assert score.utility_score == pytest.approx(0.5)


def test_task_success_only() -> None:
    """Outcome == "success" → +0.3, no other signal fires.

    Content not mentioned in any output (no textual reference), no
    requery, and rank > top-3 so the unreferenced negative can't fire.
    """
    edge = _make_edge(rank=5, content="something obscure")
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["nothing relevant here"],
        task_outcome="success",
    )
    score = _score_edge(edge, payload)
    assert score.referenced is False
    assert score.task_success is True
    assert score.requeried_after is False
    assert score.was_top3_unreferenced is False
    assert score.utility_score == pytest.approx(0.3)


def test_task_outcome_case_insensitive() -> None:
    """``"SUCCESS"``, ``"Success"``, ``"success"`` all fire the signal.

    Hook implementations vary in casing convention (Python lowercase vs
    HTTP-style uppercase); cycle-1 design says the scorer is permissive.
    """
    edge = _make_edge(rank=5, content="x")
    for outcome in ("SUCCESS", "Success", "success", "sUcCeSs"):
        payload = ScoringPayload(
            session_id="s1", assistant_outputs=[], task_outcome=outcome
        )
        score = _score_edge(edge, payload)
        assert score.task_success is True, f"outcome={outcome!r} failed"


def test_requeried_after() -> None:
    """Context overlaps with a requery above the threshold → -0.4.

    Original context "snake_case naming rule" and requery
    "snake_case naming convention" share {snake_case, naming} of 4
    unique tokens — 2/4 = 0.5 Jaccard, hits the threshold exactly. The
    threshold is ``>=`` so equality counts.
    """
    edge = _make_edge(
        rank=5,  # >top-3 so unreferenced-negative can't fire
        content="something not mentioned",
        context="snake_case naming rule",
    )
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["other text"],
        task_outcome="partial",
        requeries=["snake_case naming convention"],
    )
    score = _score_edge(edge, payload)
    assert score.requeried_after is True
    assert score.utility_score == pytest.approx(-0.4)


def test_top3_unreferenced() -> None:
    """Rank ≤ 2 and content not in any output → -0.2.

    Pure top-3-unreferenced signal: rank=0 (top hit), task_outcome
    neutral, no requeries. Score is exactly -0.2.
    """
    edge = _make_edge(rank=0, content="unreferenced fact", context="")
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["something completely different"],
        task_outcome="partial",
    )
    score = _score_edge(edge, payload)
    assert score.was_top3_unreferenced is True
    assert score.referenced is False
    assert score.utility_score == pytest.approx(-0.2)


def test_top3_unreferenced_combined_with_success() -> None:
    """Top-3 unreferenced (-0.2) + task success (+0.3) → +0.1 net.

    Documents the partial-credit case: a memory ranked top but not
    referenced still earns net positive when the session succeeds —
    arguably because "it was relevant context even if not quoted".
    """
    edge = _make_edge(rank=1, content="unreferenced", context="")
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["unrelated"],
        task_outcome="success",
    )
    score = _score_edge(edge, payload)
    assert score.was_top3_unreferenced is True
    assert score.task_success is True
    assert score.utility_score == pytest.approx(0.1)


def test_all_positive_signals_no_clamp_needed() -> None:
    """referenced (+0.5) + task_success (+0.3) → 0.8 (within bounds).

    Cycle-1 max positive is 0.8 — under the clamp ceiling. We verify the
    sum is the correct unclamped value so the clamp logic doesn't
    accidentally cap below 1.0.
    """
    edge = _make_edge(rank=5, content="referenced item", context="")
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["The referenced item is useful here"],
        task_outcome="success",
    )
    score = _score_edge(edge, payload)
    assert score.referenced is True
    assert score.task_success is True
    assert score.utility_score == pytest.approx(0.8)


def test_clamp_above_one() -> None:
    """Synthetic over-budget case: directly call ``_score_edge`` on an
    edge whose four signals all fire positive AND construct via
    monkey-patched weights to exceed 1.0.

    We exercise the clamp by bumping the constants temporarily via
    ``monkeypatch``-style attribute override. This is the only test that
    verifies the upper clamp branch; cycle-1 signals can't naturally
    exceed +0.8.
    """
    import sidecar.scoring as scoring_mod

    # Save originals so we can restore them.
    orig_ref = scoring_mod._WEIGHT_REFERENCED
    orig_succ = scoring_mod._WEIGHT_TASK_SUCCESS
    try:
        scoring_mod._WEIGHT_REFERENCED = 0.8  # type: ignore[misc]
        scoring_mod._WEIGHT_TASK_SUCCESS = 0.7  # type: ignore[misc]
        # 0.8 + 0.7 = 1.5 — clamp should cap at 1.0.
        edge = _make_edge(rank=5, content="item", context="")
        payload = ScoringPayload(
            session_id="s1",
            assistant_outputs=["item shows up here"],
            task_outcome="success",
        )
        score = _score_edge(edge, payload)
        assert score.utility_score == pytest.approx(1.0)
    finally:
        scoring_mod._WEIGHT_REFERENCED = orig_ref  # type: ignore[misc]
        scoring_mod._WEIGHT_TASK_SUCCESS = orig_succ  # type: ignore[misc]


def test_all_negative_signals_min_clamp() -> None:
    """requeried (-0.4) + top3-unreferenced (-0.2) → -0.6 (within bounds).

    Cycle-1 minimum is -0.6. We verify both negative signals stack and
    that the result is under the clamp floor. (The clamp's lower branch
    is exercised by ``test_clamp_below_neg_one``.)
    """
    edge = _make_edge(
        rank=0,
        content="not mentioned",
        context="how to use the api",
    )
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["completely different topic"],
        task_outcome="failure",
        requeries=["how do I use the api correctly"],
    )
    score = _score_edge(edge, payload)
    assert score.requeried_after is True
    assert score.was_top3_unreferenced is True
    assert score.utility_score == pytest.approx(-0.6)


def test_clamp_below_neg_one() -> None:
    """Force the floor branch by bumping negative weights past -1.0.

    Mirrors ``test_clamp_above_one`` for the lower bound. Without the
    clamp the score would be -1.4 — verifies the ``max(-1.0, ...)``
    behavior.
    """
    import sidecar.scoring as scoring_mod

    orig_re = scoring_mod._WEIGHT_REQUERIED
    orig_top = scoring_mod._WEIGHT_TOP3_UNREFERENCED
    try:
        scoring_mod._WEIGHT_REQUERIED = -0.8  # type: ignore[misc]
        scoring_mod._WEIGHT_TOP3_UNREFERENCED = -0.6  # type: ignore[misc]
        edge = _make_edge(rank=0, content="missing", context="needle in haystack")
        payload = ScoringPayload(
            session_id="s1",
            assistant_outputs=["nothing"],
            task_outcome="failure",
            requeries=["another needle in haystack"],
        )
        score = _score_edge(edge, payload)
        assert score.utility_score == pytest.approx(-1.0)
    finally:
        scoring_mod._WEIGHT_REQUERIED = orig_re  # type: ignore[misc]
        scoring_mod._WEIGHT_TOP3_UNREFERENCED = orig_top  # type: ignore[misc]


def test_score_edge_handles_missing_content() -> None:
    """When ``content`` is None (node had no title/name/etc.), the
    ``referenced`` signal cannot fire — but other signals still can.

    Documents the graceful-degradation guard: an edge pointing at a node
    with no humane label still gets scored, just without textual evidence.
    """
    edge = _make_edge(rank=5, content="", context="")  # type: ignore[arg-type]
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["some text"],
        task_outcome="success",
    )
    score = _score_edge(edge, payload)
    assert score.referenced is False
    assert score.task_success is True
    # Only the success signal — +0.3.
    assert score.utility_score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# score_session — Neo4j I/O integration via mocks.
# ---------------------------------------------------------------------------


async def test_zero_edges_returns_empty_report() -> None:
    """Session with no RETRIEVED_IN edges → ScoringReport with
    edges_scored=0 and an empty ``scores`` array.

    A session that the agent skipped retrieval on (or that the
    SessionStart hook fired for but the agent never queried) MUST NOT
    raise. The hook needs a 200 to know the scorer ran.
    """
    memory = _make_memory(edges=[])
    payload = ScoringPayload(session_id="s1")
    report = await score_session(memory, payload=payload)
    assert isinstance(report, ScoringReport)
    assert report.session_id == "s1"
    assert report.edges_scored == 0
    assert report.edges_skipped == 0
    assert report.scores == []
    # Write was never called because there were no edges.
    memory.graph.execute_write.assert_not_awaited()


async def test_score_session_writes_back_utility_score() -> None:
    """One edge, all positive signals → score gets written with the
    correct utility_score, and the write call carries the right edge_id.

    Asserts both the count (edges_scored == 1) and that the
    ``execute_write`` was invoked with the matching ``edge_id`` + a
    score in the expected ballpark.
    """
    edge = _make_edge(
        edge_id="el-99", rank=5, content="vector index", context=""
    )
    memory = _make_memory(edges=[edge])
    payload = ScoringPayload(
        session_id="s1",
        assistant_outputs=["The vector index makes lookups fast."],
        task_outcome="success",
    )
    report = await score_session(memory, payload=payload)
    assert report.edges_scored == 1
    assert report.edges_skipped == 0
    # Write was called exactly once with our edge.
    memory.graph.execute_write.assert_awaited_once()
    call_kwargs = memory.graph.execute_write.call_args.kwargs
    assert call_kwargs["edge_id"] == "el-99"
    # All positive signals: 0.5 + 0.3 = 0.8.
    assert call_kwargs["score"] == pytest.approx(0.8)


async def test_write_failure_continues_other_edges() -> None:
    """One write raises mid-loop → other edges still get scored.

    The first edge's write raises a transient Neo4j error. The scorer
    MUST log + skip + carry on, scoring + writing the remaining edge.
    Returned report shows edges_scored=1, edges_skipped=1, but the
    ``scores`` array has both per-edge breakdowns regardless of write
    outcome.
    """
    edges = [
        _make_edge(edge_id="e1", rank=5, content="x", context=""),
        _make_edge(edge_id="e2", rank=5, content="y", context=""),
    ]
    memory = _make_memory(edges=edges)
    # First write raises, second succeeds.
    memory.graph.execute_write.side_effect = [
        ConnectionError("neo4j timeout"),
        None,
    ]
    payload = ScoringPayload(
        session_id="s1", assistant_outputs=[], task_outcome="success"
    )
    report = await score_session(memory, payload=payload)
    assert report.edges_scored == 1
    assert report.edges_skipped == 1
    assert len(report.scores) == 2  # both still in the breakdown
    # Both writes were attempted.
    assert memory.graph.execute_write.await_count == 2


async def test_score_session_idempotent_rescore() -> None:
    """Running score_session twice on the same session overwrites the
    utility_score — second run scores the same number of edges, both
    runs return matching reports.

    Critical for the operator "re-score from dashboard" flow (cycle 2)
    and for hook flapping resilience (a SessionEnd hook that fires twice
    must not corrupt the data).
    """
    edges = [_make_edge(rank=5, content="alpha", context="")]
    memory = _make_memory(edges=edges)
    payload = ScoringPayload(
        session_id="s1", assistant_outputs=[], task_outcome="success"
    )

    report1 = await score_session(memory, payload=payload)
    report2 = await score_session(memory, payload=payload)

    assert report1.edges_scored == report2.edges_scored == 1
    assert report1.scores[0].utility_score == report2.scores[0].utility_score
    # Two read calls + two write calls — second run didn't short-circuit.
    assert memory.graph.execute_read.await_count == 2
    assert memory.graph.execute_write.await_count == 2


async def test_score_session_fetch_includes_session_id_param() -> None:
    """The fetch query binds ``$session_id`` so different sessions don't
    bleed into each other.

    A regression that inlined a string-formatted session_id would let
    Cypher injection happen and would not parameterize properly; assert
    the kwarg is what got passed.
    """
    memory = _make_memory(edges=[])
    payload = ScoringPayload(session_id="abc-123")
    await score_session(memory, payload=payload)

    memory.graph.execute_read.assert_awaited_once()
    call_kwargs = memory.graph.execute_read.call_args.kwargs
    assert call_kwargs == {"session_id": "abc-123"}


# ---------------------------------------------------------------------------
# Router-level — POST /api/sessions/{id}/score.
# ---------------------------------------------------------------------------


def test_payload_mismatch_400() -> None:
    """``POST /api/sessions/s1/score`` with body.session_id="s2" → 400.

    The mismatch check is a typo guard — without it a client that builds
    the path one way and the payload another could silently score the
    wrong session. The detail string names both ids so client-side
    debugging is trivial.
    """
    from sidecar.app import app
    from sidecar.deps import get_memory_client

    # Override the memory dep so the lifespan doesn't have to open Neo4j.
    fake_memory = _make_memory(edges=[])
    app.dependency_overrides[get_memory_client] = lambda: fake_memory

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sessions/s1/score",
                json={
                    "session_id": "s2",  # mismatch
                    "assistant_outputs": [],
                    "task_outcome": "unknown",
                    "requeries": [],
                },
            )
        assert response.status_code == 400, response.text
        body = response.json()
        # Both ids in the detail for debuggability.
        assert "s1" in body["detail"]
        assert "s2" in body["detail"]
        # Importantly the scorer was never called — the mismatch short-circuits.
        fake_memory.graph.execute_read.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(get_memory_client, None)


def test_endpoint_happy_path_returns_report() -> None:
    """Round-trip: POST a valid payload → 200 → ScoringReport JSON.

    Exercises the full router → scoring.py wiring including the
    pydantic serialization of the response model. We use the same
    override pattern as :func:`test_payload_mismatch_400` to skip the
    lifespan's Neo4j connect.
    """
    from sidecar.app import app
    from sidecar.deps import get_memory_client

    edge = _make_edge(edge_id="ex1", rank=5, content="A", context="")
    fake_memory = _make_memory(edges=[edge])
    app.dependency_overrides[get_memory_client] = lambda: fake_memory

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sessions/s1/score",
                json={
                    "session_id": "s1",
                    "assistant_outputs": ["A is needed here"],
                    "task_outcome": "success",
                    "requeries": [],
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == "s1"
        assert body["edges_scored"] == 1
        assert body["edges_skipped"] == 0
        assert len(body["scores"]) == 1
        # Both signals fired (referenced + task_success) → 0.8.
        assert body["scores"][0]["utility_score"] == pytest.approx(0.8)
    finally:
        app.dependency_overrides.pop(get_memory_client, None)
