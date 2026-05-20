"""Task 9 — RETRIEVED_IN edge writer unit tests.

The :func:`sidecar.feedback.log_retrieval` function executes the Cypher
:func:`sidecar.cypher_helpers.build_retrieved_in_writer` builds against
a live :class:`MemoryClient`. The Cypher itself is covered by
``test_cypher_helpers.py``; this module exercises the *executor*: hit
iteration, rank assignment, parameter binding, context truncation,
graceful degradation on partial failure, and the hit-shape duck-typing
adapters (:func:`_hit_node_id`, :func:`_hit_score`).

What we're protecting:

* **Cardinality:** N hits → N execute_write calls, one per hit, in
  rank-ascending order. A regression that swapped to a batch UNWIND
  would break the per-call assertion semantics in Task 10 (scorer
  expects edges to be addressable by rank).
* **Rank assignment:** rank starts at 0 (top hit) and increments
  monotonically. The scorer treats ``rank <= 2`` ("top-3") as a
  special bucket — a regression that started rank at 1 would shift
  the bucket boundaries.
* **Truncation contract:** ``retrieval_context`` is hard-capped at
  :data:`MAX_CONTEXT_CHARS`. A 2-line prompt is fine; a 50-line
  pasted error trace would otherwise bloat edge storage when a
  long session writes thousands of edges.
* **Graceful degradation:** an individual write failure is logged
  but doesn't abort the batch or raise. Retrieval logging is
  observability, never the critical path.
* **Hit-shape duck typing:** vector hits (``.score`` attribute) and
  structural hits (no score) both work. Dict-shaped hits work too.
  A hit missing ``.id`` is skipped + logged, not raised.

No Neo4j needed for any of these — the :class:`MemoryClient` is fully
mocked via :class:`MagicMock` with an :class:`AsyncMock` on
``memory.graph.execute_write``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sidecar.cypher_helpers import build_retrieved_in_writer
from sidecar.feedback import MAX_CONTEXT_CHARS, log_retrieval


# ---------------------------------------------------------------------------
# Shared fixtures + fakes.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory() -> MagicMock:
    """A :class:`MemoryClient` stand-in with an awaitable ``execute_write``.

    The real client's :attr:`MemoryClient.graph` is a property returning a
    ``Neo4jClient`` whose :meth:`execute_write` is an async function. Our
    mock mirrors that exact attribute path so the production call site
    (``memory.graph.execute_write(...)``) lights up the assertions here
    without any indirection. Tests inspect
    ``mock_memory.graph.execute_write.call_args_list`` to verify the
    parameters bound per hit.
    """
    mem = MagicMock()
    mem.graph = MagicMock()
    mem.graph.execute_write = AsyncMock(return_value=None)
    return mem


class _FakeHit:
    """Minimal object-with-attributes hit shape.

    Mirrors the library's ``Entity`` / ``Fact`` surface (an ``id`` plus
    a ``score`` when produced by vector search). Defined at module
    scope rather than per-test so tests can construct lists of hits
    without redefining the class each time.
    """

    def __init__(self, id_: str, score: float) -> None:
        self.id = id_
        self.score = score


# ---------------------------------------------------------------------------
# Happy path — N hits produces N writes with monotonic rank.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_retrieval_writes_one_edge_per_hit(mock_memory: MagicMock) -> None:
    """N hits → N ``execute_write`` calls + return count equals N.

    This is the cardinality guarantee: a regression that batched the
    writes (single UNWIND-based query) would still produce the right
    edges in Neo4j but break Task 10's scorer, which expects each edge
    to be reachable per-call for assertion + reconciliation.
    """
    hits = [_FakeHit("n1", 0.9), _FakeHit("n2", 0.8), _FakeHit("n3", 0.7)]
    n = await log_retrieval(mock_memory, session_id="s1", query="q", hits=hits)
    assert n == 3
    assert mock_memory.graph.execute_write.call_count == 3


@pytest.mark.asyncio
async def test_rank_increments_from_zero(mock_memory: MagicMock) -> None:
    """``rank`` starts at 0 (top hit) and increments per hit.

    Design §D5 puts ``rank <= 2`` in the "top-3 bucket" the scorer
    treats specially. A regression that started rank at 1 (or skipped
    entries) would silently shift bucket boundaries — the kind of bug
    that produces plausible-but-wrong utility scores.
    """
    hits = [_FakeHit("a", 0.5), _FakeHit("b", 0.4), _FakeHit("c", 0.3)]
    await log_retrieval(mock_memory, session_id="s1", query="q", hits=hits)
    calls = mock_memory.graph.execute_write.call_args_list
    assert [c.args[1]["rank"] for c in calls] == [0, 1, 2]


@pytest.mark.asyncio
async def test_all_call_parameters_bound(mock_memory: MagicMock) -> None:
    """Every required Cypher parameter is bound on every call.

    The build_retrieved_in_writer Cypher references ``$node_id``,
    ``$session_id``, ``$rank``, ``$score``, ``$context``. A missing
    binding causes a runtime Neo4j error that's easy to miss in
    development (it only fires when the edge would actually be
    written). Belt-and-braces here protects against a refactor that
    drops one keyword.
    """
    hit = _FakeHit("node-xyz", 0.42)
    await log_retrieval(
        mock_memory, session_id="sess-abc", query="user query", hits=[hit]
    )
    kwargs = mock_memory.graph.execute_write.call_args.args[1]
    assert kwargs["node_id"] == "node-xyz"
    assert kwargs["session_id"] == "sess-abc"
    assert kwargs["rank"] == 0
    assert kwargs["score"] == pytest.approx(0.42)
    assert kwargs["context"] == "user query"


@pytest.mark.asyncio
async def test_query_passed_to_writer_as_cypher(mock_memory: MagicMock) -> None:
    """The Cypher executed matches :func:`build_retrieved_in_writer` exactly.

    A refactor that changed the Cypher (e.g. swapped MERGE for CREATE)
    would break the cardinality contract documented in
    :mod:`sidecar.cypher_helpers`. The two modules are decoupled, so we
    explicitly assert the writer Cypher is what feedback.py runs.
    """
    expected = build_retrieved_in_writer()
    await log_retrieval(
        mock_memory, session_id="s1", query="q", hits=[_FakeHit("n", 0.5)]
    )
    # The Cypher is positional arg 0.
    actual_cypher = mock_memory.graph.execute_write.call_args.args[0]
    assert actual_cypher == expected


# ---------------------------------------------------------------------------
# Truncation — retrieval_context capped at MAX_CONTEXT_CHARS.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_truncated_to_max_chars(mock_memory: MagicMock) -> None:
    """Queries longer than :data:`MAX_CONTEXT_CHARS` are truncated.

    The cap exists because edge property storage adds up — a long
    session can write thousands of edges, each carrying the user's
    prompt. A 50-line pasted error trace would otherwise multiply the
    on-disk footprint by 100x.
    """
    long_query = "x" * 2000
    await log_retrieval(
        mock_memory, session_id="s1", query=long_query, hits=[_FakeHit("n", 0.5)]
    )
    ctx = mock_memory.graph.execute_write.call_args.args[1]["context"]
    assert len(ctx) == MAX_CONTEXT_CHARS
    # Truncation is leading-N, not random-N — confirm we kept the start.
    assert ctx == "x" * MAX_CONTEXT_CHARS


@pytest.mark.asyncio
async def test_short_context_not_padded(mock_memory: MagicMock) -> None:
    """Queries shorter than the cap pass through unchanged.

    A regression that left-padded short contexts to the cap would
    waste storage and produce misleading edge inspection in Neo4j
    Browser. Truncation is one-sided: only trim, never extend.
    """
    short = "hi"
    await log_retrieval(mock_memory, session_id="s1", query=short, hits=[_FakeHit("n", 0.5)])
    ctx = mock_memory.graph.execute_write.call_args.args[1]["context"]
    assert ctx == "hi"
    assert len(ctx) == 2


@pytest.mark.asyncio
async def test_empty_query_becomes_empty_string(mock_memory: MagicMock) -> None:
    """An empty (or ``None``) query writes ``""`` rather than failing.

    Callers passing ``None`` for query (a structural list-by-label
    query that has no semantic text) shouldn't trigger a TypeError on
    the slice. Coercing to ``""`` makes the edge property
    well-defined for the scorer.
    """
    await log_retrieval(
        mock_memory, session_id="s1", query="", hits=[_FakeHit("n", 0.5)]
    )
    ctx = mock_memory.graph.execute_write.call_args.args[1]["context"]
    assert ctx == ""

    mock_memory.graph.execute_write.reset_mock()

    # ``None`` is coerced the same way — type checker may flag but
    # the runtime contract is explicit in the docstring.
    await log_retrieval(
        mock_memory,
        session_id="s1",
        query=None,  # type: ignore[arg-type]
        hits=[_FakeHit("n", 0.5)],
    )
    assert mock_memory.graph.execute_write.call_args.args[1]["context"] == ""


# ---------------------------------------------------------------------------
# Edge cases — empty hits, missing fields, dict shapes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_hits_writes_zero_edges(mock_memory: MagicMock) -> None:
    """Empty ``hits`` returns ``0`` and skips Neo4j entirely.

    Search endpoints will often return zero hits for an unfamiliar
    query, and we shouldn't burn a round-trip writing nothing. Also,
    a zero-call no-op is the cheapest possible signal to monitoring:
    a non-zero edge count for zero hits would mean a bug.
    """
    n = await log_retrieval(mock_memory, session_id="s1", query="q", hits=[])
    assert n == 0
    mock_memory.graph.execute_write.assert_not_called()


@pytest.mark.asyncio
async def test_hit_without_id_skipped(mock_memory: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
    """A hit lacking ``.id`` is skipped with a WARNING, not raised.

    The retrieval may have produced a hit shape we don't recognize
    (library refactor, custom Cypher returning a row dict without
    ``id``). Skipping the bad hit + logging is preferable to crashing
    the whole feedback write — the good hits should still land.
    """
    bad_hit = MagicMock(spec=[])  # spec=[] strips ALL attributes
    good_hit = _FakeHit("n", 0.9)
    with caplog.at_level(logging.WARNING, logger="sidecar.feedback"):
        n = await log_retrieval(
            mock_memory, session_id="s1", query="q", hits=[bad_hit, good_hit]
        )
    assert n == 1
    assert mock_memory.graph.execute_write.call_count == 1
    # Good hit is written with rank=1 (NOT re-ranked to 0 — the bad hit
    # still consumed rank 0). This preserves alignment with whatever
    # rank-meaning the caller had upstream.
    assert mock_memory.graph.execute_write.call_args.args[1]["rank"] == 1
    assert mock_memory.graph.execute_write.call_args.args[1]["node_id"] == "n"
    # The warning is informational, not actionable per-event.
    assert any("no .id attribute" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_hit_without_score_defaults_to_zero(mock_memory: MagicMock) -> None:
    """A hit without ``.score`` writes ``score=0.0``.

    Structural lookups (e.g. "list all Decisions in scope") don't have
    a similarity concept; their hits arrive without a score attribute.
    Task 10's scorer interprets ``score == 0.0`` as "non-semantic hit"
    (no boost or penalty from the score-based heuristics).
    """
    hit = MagicMock(spec=["id"])
    hit.id = "n"
    await log_retrieval(mock_memory, session_id="s1", query="q", hits=[hit])
    score = mock_memory.graph.execute_write.call_args.args[1]["score"]
    assert score == 0.0


@pytest.mark.asyncio
async def test_dict_hit_works(mock_memory: MagicMock) -> None:
    """A dict-shaped hit ``{"id": ..., "score": ...}`` is accepted.

    Raw Cypher returns rows as dicts; supporting this shape avoids
    forcing every Task 8 search endpoint to wrap rows in an object
    before calling :func:`log_retrieval`. A regression that required
    attribute-access only would break dict-passing callers silently.
    """
    n = await log_retrieval(
        mock_memory,
        session_id="s1",
        query="q",
        hits=[{"id": "n", "score": 0.7}],
    )
    assert n == 1
    kwargs = mock_memory.graph.execute_write.call_args.args[1]
    assert kwargs["node_id"] == "n"
    assert kwargs["score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_dict_hit_without_score_defaults_to_zero(mock_memory: MagicMock) -> None:
    """Dict hit lacking ``"score"`` key falls back to ``0.0``.

    Mirror of :func:`test_hit_without_score_defaults_to_zero` for the
    dict shape — same scorer semantics apply to both shapes.
    """
    await log_retrieval(
        mock_memory, session_id="s1", query="q", hits=[{"id": "n"}]
    )
    score = mock_memory.graph.execute_write.call_args.args[1]["score"]
    assert score == 0.0


@pytest.mark.asyncio
async def test_id_coerced_to_string(mock_memory: MagicMock) -> None:
    """Integer ``id`` (legacy/system nodes) is coerced to ``str``.

    The Cypher ``MATCH (n) WHERE n.id = $node_id`` matches whatever
    type Neo4j has on the property; the library writes string ids
    everywhere, so we normalize to ``str`` defensively to match.
    """
    hit = MagicMock(spec=["id", "score"])
    hit.id = 42
    hit.score = 0.5
    await log_retrieval(mock_memory, session_id="s1", query="q", hits=[hit])
    assert mock_memory.graph.execute_write.call_args.args[1]["node_id"] == "42"


# ---------------------------------------------------------------------------
# Graceful degradation — partial failure must not raise.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_failure_does_not_raise(mock_memory: MagicMock) -> None:
    """An exception from ``execute_write`` is logged + counted as 0.

    The whole point of D5 feedback is observability — making the
    critical path depend on the observability layer would be a tail
    wagging the dog. A retrieval that succeeded but couldn't write
    a feedback edge is *still* a successful retrieval.
    """
    mock_memory.graph.execute_write.side_effect = ConnectionError("db down")
    n = await log_retrieval(
        mock_memory, session_id="s1", query="q", hits=[_FakeHit("n", 0.5)]
    )
    assert n == 0  # nothing written; nothing raised


@pytest.mark.asyncio
async def test_partial_failure_continues_batch(mock_memory: MagicMock) -> None:
    """One failing write doesn't abort the rest of the batch.

    Pathological pattern: hit #2 has a malformed id that triggers a
    Neo4j parser error, but hits #1 and #3 are fine. We should write
    #1 + #3 and report ``n == 2``. A regression that broke out of the
    loop on first failure would drop most of the feedback.
    """
    call_count = 0

    async def flaky_write(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated mid-batch failure")
        return None

    mock_memory.graph.execute_write = AsyncMock(side_effect=flaky_write)
    hits = [_FakeHit("a", 0.9), _FakeHit("b", 0.8), _FakeHit("c", 0.7)]
    n = await log_retrieval(mock_memory, session_id="s1", query="q", hits=hits)
    assert n == 2  # a + c succeeded; b raised
    assert mock_memory.graph.execute_write.call_count == 3  # all attempted


@pytest.mark.asyncio
async def test_write_failure_logs_diagnostic(
    mock_memory: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Failure path emits a WARNING with node + session ids.

    Operators triage feedback drops by rate, but the per-event log
    needs enough context (node id + session id + exception repr) to
    investigate when the rate spikes. Pin the load-bearing fields
    rather than the exact message string.
    """
    mock_memory.graph.execute_write.side_effect = ConnectionError("db down")
    with caplog.at_level(logging.WARNING, logger="sidecar.feedback"):
        await log_retrieval(
            mock_memory,
            session_id="session-xyz",
            query="q",
            hits=[_FakeHit("node-abc", 0.5)],
        )
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("node-abc" in m and "session-xyz" in m for m in msgs), (
        f"missing diagnostic in warning logs: {msgs!r}"
    )


# ---------------------------------------------------------------------------
# Cross-call invariants.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_call_uses_same_cypher_text(mock_memory: MagicMock) -> None:
    """The Cypher is identical across every call in a batch.

    A regression that re-built the Cypher per-hit (or worse,
    constructed slightly different Cypher per-hit) would defeat
    Neo4j's query-plan cache + slow down every retrieval. Build once
    at the top of :func:`log_retrieval`, reuse for every hit.
    """
    hits = [_FakeHit(f"n{i}", 0.5) for i in range(4)]
    await log_retrieval(mock_memory, session_id="s1", query="q", hits=hits)
    cyphers = {c.args[0] for c in mock_memory.graph.execute_write.call_args_list}
    assert len(cyphers) == 1, f"Cypher varied across calls: {cyphers}"
