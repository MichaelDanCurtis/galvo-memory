"""SessionEnd utility scorer for the D5 cycle-1 SAGE-lite loop.

Walks every ``RETRIEVED_IN`` edge a session accumulated, applies the four
design-§5 signals to each, and writes the resulting ``utility_score`` (a
clamped ``[-1.0, +1.0]`` float) back to the edge so cycle-2 consolidation
queries (deferred to D5 cycle 2) can ``WHERE r.utility_score >= 0.X`` and
rank memories by their proven usefulness.

Signal table (locked by design §5):

    + 0.5  if memory content appears verbatim in any assistant turn output
           ("textual evidence" — the strongest positive signal we have in
           cycle 1; cycle 2 swaps for embedding similarity)
    + 0.3  if the task completed successfully ("no revert, no terminal
           error" — collected by the SessionEnd hook in Task 15)
    - 0.4  if the agent re-queried for semantically-overlapping info after
           this retrieval ("insufficient retrieval" — what the scorer
           penalizes most heavily because it's the strongest "this memory
           did not help" signal)
    - 0.2  if the memory was ranked top-3 but never referenced ("we put it
           in front of you and you ignored it" — softer negative because
           agent may have used it as context-only)

Sum, then ``max(-1.0, min(1.0, sum))`` — clamping only matters once cycle
2 adds a fifth signal; cycle-1 signals max at +0.8 / min at -0.6, but we
clamp anyway so the column type stays a strict [-1, +1] float.

Idempotency: re-running ``score_session`` on the same session_id is safe;
the ``SET r.utility_score = $score`` overwrites in place rather than
appending. The SessionEnd hook fires the endpoint exactly once per
session boundary, but a flapping hook (or an operator triggering a
re-score from the dashboard) won't corrupt the data.

Error policy: per-edge write failures are logged WARNING and counted in
``ScoringReport.edges_skipped``; we don't raise mid-loop, because a
single failed edge shouldn't poison the rest of the session's signal.
The endpoint returns 200 even when ``edges_skipped > 0``; the hook
layer (Task 15) eyeballs the report to decide whether to alert.

References:

* Design §5 "Per-retrieval logging + per-session scoring"
* Phase 2 plan §Task 10
* :func:`sidecar.cypher_helpers.build_retrieved_in_writer` — the writer
  that populates the edges this module scores

Cycle-2 swap points (called out so the cycle-2 PR knows where to look):

* :func:`_semantic_overlap` — replace word-overlap with embedding cosine
  similarity using the same MiniLM/Qwen3 embedder the search pipeline uses
* Add a fifth signal once consolidation lands (e.g. +0.2 if the memory
  was merged into a higher-scope :Belief during consolidation)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]


__all__ = [
    "EdgeScore",
    "ScoringPayload",
    "ScoringReport",
    "score_session",
]


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal weights — kept module-level constants so a future "tune the
# weights" PR can lift these into config without touching the loop body.
# ---------------------------------------------------------------------------


_WEIGHT_REFERENCED: float = 0.5
"""Positive weight when a memory's content appears in an assistant turn."""

_WEIGHT_TASK_SUCCESS: float = 0.3
"""Positive weight when the session reported a successful task outcome."""

_WEIGHT_REQUERIED: float = -0.4
"""Negative weight when the agent re-queried for semantically-similar info."""

_WEIGHT_TOP3_UNREFERENCED: float = -0.2
"""Negative weight when a top-3 memory was never referenced in output."""

_TOP3_RANK_THRESHOLD: int = 2
"""Top-3 cutoff — ranks are 0-indexed, so ``rank <= 2`` means positions 1-3."""

_SEMANTIC_OVERLAP_REQUERY_THRESHOLD: float = 0.5
"""Jaccard overlap above which a follow-up query counts as a re-query.

Cycle 1 uses word-set Jaccard (cheap, no embedder dependency); cycle 2
swaps for embedding cosine similarity with a similar 0.5-ish threshold.
The threshold is intentionally permissive because false negatives
(missing a re-query) silently zero out a strong negative signal, while
false positives (over-counting re-queries) just nudge the score
moderately downward.
"""


# ---------------------------------------------------------------------------
# Payload + report models.
# ---------------------------------------------------------------------------


class ScoringPayload(BaseModel):
    """SessionEnd input — collected by the Task-15 SessionEnd hook.

    The hook gathers the session transcript (assistant outputs + tool
    invocations + retry queries) and posts the trimmed version here.
    Field shapes are deliberately permissive so the hook can grow extra
    signal sources in cycle 2 without breaking the cycle-1 wire format.

    Fields:
        session_id: The id of the :Session node in Neo4j. Must match the
            session-id the SessionStart hook (Task 12) wrote — otherwise
            the scorer finds zero edges and returns an empty report.
        assistant_outputs: One entry per assistant turn — the full text
            output the agent produced. Used for textual-reference matching
            (signal +0.5). Empty list → signal can never fire.
        task_outcome: One of ``"success" | "failure" | "partial" |
            "unknown"``. Case-insensitive; anything that doesn't lowercase
            to ``"success"`` means the task-success signal does not fire.
        requeries: Queries the agent issued *after* the retrieval being
            scored, used to detect the "agent re-queried for similar info"
            signal (-0.4). The hook is responsible for filtering to
            queries that happened temporally after a given retrieval —
            cycle-1 we treat the list as global and let the semantic
            overlap check decide; cycle-2 should add timestamps.
    """

    session_id: str
    assistant_outputs: list[str] = Field(
        default_factory=list,
        description="Each assistant turn's textual output. Used for textual-reference matching.",
    )
    task_outcome: str = Field(
        default="unknown",
        description="One of: success | failure | partial | unknown",
    )
    requeries: list[str] = Field(
        default_factory=list,
        description=(
            "Queries the agent issued AFTER an earlier retrieval; used to detect "
            "insufficient-retrieval signal."
        ),
    )


class EdgeScore(BaseModel):
    """Per-edge scoring breakdown — useful for the SessionEnd dashboard
    drill-down and for spotting "this signal is firing in surprising
    ways" patterns during cycle-1 evaluation.

    The booleans mirror the four signals exactly; ``utility_score`` is
    the clamped sum the writer commits to Neo4j.
    """

    node_id: str
    """The retrieved node's ``id`` — points back at the memory that earned
    the score. Returned to the caller so a UI can deep-link."""

    retrieval_rank: int
    """The rank the node had in the retrieval (0 = top hit). Returned to
    the caller for sorting / display; the scorer itself uses ``rank``
    only to test the top-3 condition."""

    referenced: bool
    """Did the memory's content appear in any assistant output? (+0.5)"""

    task_success: bool
    """Did the session's task complete successfully? (+0.3)"""

    requeried_after: bool
    """Did the agent re-query for semantically-overlapping info? (-0.4)"""

    was_top3_unreferenced: bool
    """Was the memory ranked top-3 but never referenced? (-0.2)"""

    utility_score: float
    """The clamped sum of all four signals. Always in ``[-1.0, +1.0]``."""


class ScoringReport(BaseModel):
    """Top-level response from ``POST /api/sessions/{id}/score``.

    The hook layer reads ``edges_scored`` + ``edges_skipped`` to decide
    whether to alert the operator (e.g. "this session generated 30 edges
    but only 12 got scored — investigate"). The ``scores`` array is
    primarily for the dashboard's session-detail view.
    """

    session_id: str
    edges_scored: int
    """Number of edges where the ``utility_score`` write succeeded."""

    edges_skipped: int
    """Number of edges where the write raised — usually transient Neo4j
    hiccups. Each skip is logged at WARNING level with the edge_id."""

    scores: list[EdgeScore]
    """Per-edge breakdown, in fetch order (no guaranteed sort)."""


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------


async def score_session(
    memory: "MemoryClient",
    *,
    payload: ScoringPayload,
) -> ScoringReport:
    """Walk every ``RETRIEVED_IN`` edge for a session, score it, write back.

    Three-step flow:

    1. Read all edges (and the joined node's content + the edge's rank +
       context properties) via :func:`_fetch_edges`.
    2. Per edge, apply the four signals to derive an :class:`EdgeScore`
       via :func:`_score_edge`. Pure function — no I/O.
    3. Write ``r.utility_score`` (and a ``r.scored_at`` timestamp for
       observability) back via :func:`_write_score`. Per-edge writes
       failure-isolated so one transient Neo4j blip doesn't lose every
       score for the session.

    Idempotent: re-running on the same session overwrites the prior
    ``utility_score``. The Task-15 hook should call this exactly once
    per session boundary; the dashboard re-score button (cycle 2) relies
    on the overwrite semantics.

    Args:
        memory: Live :class:`MemoryClient`. In production from
            :data:`sidecar.deps.MemoryDep`; in tests a :class:`MagicMock`
            with :class:`~unittest.mock.AsyncMock` stubs on
            ``memory.graph.execute_read`` and
            ``memory.graph.execute_write``.
        payload: The SessionEnd signal bundle. See :class:`ScoringPayload`.

    Returns:
        :class:`ScoringReport` summarizing the writes and the per-edge
        breakdown. Always returns; never raises (per-edge failures are
        absorbed into ``edges_skipped``).
    """
    # Step 1 — pull edges + joined node content from Neo4j.
    edges = await _fetch_edges(memory, payload.session_id)

    scores: list[EdgeScore] = []
    written = 0
    skipped = 0

    # Step 2 + 3 — score + write per edge.
    for edge in edges:
        breakdown = _score_edge(edge, payload)
        try:
            await _write_score(memory, edge["edge_id"], breakdown.utility_score)
            written += 1
        except Exception as exc:  # noqa: BLE001 — logged, never raised
            # Treat write failures the same way log_retrieval treats
            # edge-write failures: log + count + carry on. The score is
            # still useful in the report (caller may retry on a per-edge
            # basis later), and one transient Neo4j hiccup shouldn't
            # blackhole an entire session's signal.
            _log.warning(
                "scoring write failed for edge %s: %r", edge.get("edge_id"), exc
            )
            skipped += 1
        scores.append(breakdown)

    return ScoringReport(
        session_id=payload.session_id,
        edges_scored=written,
        edges_skipped=skipped,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Neo4j I/O — the only impure parts of the module.
# ---------------------------------------------------------------------------


async def _fetch_edges(
    memory: "MemoryClient", session_id: str
) -> list[dict[str, Any]]:
    """Return every ``RETRIEVED_IN`` edge for the session, joined with the
    retrieved node's "content" property.

    "Content" is hand-picked across the 12 node labels — different labels
    use different identifying properties (a :Decision has ``title``, a
    :File has ``path``, etc.). The ``coalesce(...)`` falls through the
    candidates in cycle-1 priority order:

    * ``title`` (Decision, Belief)
    * ``name`` (Service, Pattern, Person)
    * ``claim`` (Belief alternate)
    * ``path`` (File, Module)
    * ``sha`` (Commit)
    * ``identifier`` (catch-all fallback the library writes when nothing
      else fits)

    Result rows are dicts with keys: ``edge_id``, ``rank``, ``context``,
    ``content``, ``node_id``. The ``edge_id`` is Neo4j's ``elementId(r)``
    — opaque to the caller, used only to MATCH the edge for the write.

    Why opaque elementId rather than a (node_id, session_id) pair?
    Because ``MATCH (n)-[r:RETRIEVED_IN]->(s)`` would have to filter on
    both endpoints, and that's two more parameter bindings per write.
    ``elementId(r)`` is a stable handle the driver returns for free.
    """
    cypher = (
        "MATCH (n)-[r:RETRIEVED_IN]->(s:Session {id: $session_id}) "
        "RETURN elementId(r) AS edge_id, "
        "r.retrieval_rank AS rank, "
        "r.retrieval_context AS context, "
        "coalesce(n.title, n.name, n.claim, n.path, n.sha, n.identifier) AS content, "
        "n.id AS node_id"
    )
    result = await memory.graph.execute_read(cypher, {"session_id": session_id})
    return list(result or [])


async def _write_score(
    memory: "MemoryClient", edge_id: str, score: float
) -> None:
    """Write ``utility_score`` + ``scored_at`` back to a specific edge.

    Uses ``elementId(r) = $edge_id`` rather than ``MATCH (n)-[r]->(s)``
    so the write touches a single edge by handle — no full-relationship
    scan, no risk of mis-matching on a duplicate session row. The
    ``scored_at`` timestamp is for observability only; the scorer itself
    doesn't read it.
    """
    cypher = (
        "MATCH ()-[r:RETRIEVED_IN]-() WHERE elementId(r) = $edge_id "
        "SET r.utility_score = $score, r.scored_at = datetime() "
        "RETURN r"
    )
    await memory.graph.execute_write(cypher, {"edge_id": edge_id, "score": score})


# ---------------------------------------------------------------------------
# Pure scoring helpers — no I/O, fully unit-testable.
# ---------------------------------------------------------------------------


def _score_edge(edge: dict[str, Any], payload: ScoringPayload) -> EdgeScore:
    """Apply the four signals to one edge; return the breakdown.

    Algorithm matches design §5 exactly:

    1. Lowercase the joined ``content`` once for substring matching —
       case-insensitive textual-reference is the cheapest cycle-1
       implementation; cycle-2 swaps for embedding similarity.
    2. Resolve each signal as a bool.
    3. Sum the signed weights for the True bools.
    4. Clamp to ``[-1.0, +1.0]`` (cycle-1 signals can't exceed that range
       anyway — but the clamp is cheap and future-proofs the property
       type contract).

    Edge cases:

    * Missing content (``coalesce`` returned ``None``): the
       ``referenced`` signal can never fire; ``was_top3_unreferenced``
       still may. This is correct — a memory with no matchable content
       can't be "referenced", but it can still occupy a top-3 slot.
    * Empty ``assistant_outputs``: ``referenced`` is False for every
       edge — no positive textual evidence available.
    * Missing edge ``context`` (older retrievals predating Task 9): the
       ``requeried_after`` signal is False — we can't detect requery
       without the original query text.
    """
    content_raw = edge.get("content") or ""
    content_lower = content_raw.lower()
    rank = edge.get("rank") or 0

    # Signal 1: textual reference (+0.5). Substring match in either
    # direction would let "Python" match an assistant turn that just says
    # "Python is great", which is what we want. We use `in` (substring)
    # not equality because content is usually a label/title fragment that
    # shows up inside larger sentences.
    referenced = bool(content_lower) and any(
        content_lower in (out or "").lower() for out in payload.assistant_outputs
    )

    # Signal 2: task success (+0.3). Case-insensitive equality on the
    # canonical "success" outcome string. Other outcomes ("partial",
    # "failure", "unknown") do not trigger the positive signal — there's
    # no partial-credit; cycle-2 may revisit.
    task_success = payload.task_outcome.lower() == "success"

    # Signal 3: re-query (-0.4). Compares the retrieval's stored context
    # (the original query text saved by Task 9's RETRIEVED_IN writer)
    # against every requery the payload supplied. Any single overlap
    # above the threshold fires the signal — we don't multiply for many.
    context_raw = edge.get("context") or ""
    context_lower = context_raw.lower()
    requeried_after = bool(context_lower) and any(
        _semantic_overlap(context_lower, q) >= _SEMANTIC_OVERLAP_REQUERY_THRESHOLD
        for q in payload.requeries
    )

    # Signal 4: top-3 unreferenced (-0.2). Inverse correlated with
    # ``referenced`` — a memory can't be both "in top 3 and unreferenced"
    # AND "referenced" at once.
    was_top3_unreferenced = rank <= _TOP3_RANK_THRESHOLD and not referenced

    # Apply weights.
    score = 0.0
    if referenced:
        score += _WEIGHT_REFERENCED
    if task_success:
        score += _WEIGHT_TASK_SUCCESS
    if requeried_after:
        score += _WEIGHT_REQUERIED
    if was_top3_unreferenced:
        score += _WEIGHT_TOP3_UNREFERENCED

    # Clamp — cheap belt-and-braces for the property-type contract.
    score = max(-1.0, min(1.0, score))

    return EdgeScore(
        node_id=str(edge.get("node_id") or "?"),
        retrieval_rank=rank,
        referenced=referenced,
        task_success=task_success,
        requeried_after=requeried_after,
        was_top3_unreferenced=was_top3_unreferenced,
        utility_score=score,
    )


def _semantic_overlap(a: str, b: str) -> float:
    """Quick-and-dirty word-set Jaccard similarity for cycle 1.

    Returns ``|A ∩ B| / |A ∪ B|`` where A and B are the lowercased word
    sets of the two inputs. Always in ``[0.0, 1.0]``; ``0.0`` when
    either input is empty or there's no overlap.

    Why Jaccard and not embeddings?

    * No additional model dependency in the scorer — keeps the SessionEnd
      endpoint snappy (no embedder warm-up cost for the score request).
    * Catches the most common "agent re-queried for similar info" case
      where the requery shares concrete tokens with the original (variable
      names, library names, etc.).
    * Cycle 2 swaps in embedding cosine similarity via the same
      MiniLM/Qwen3 embedder the search pipeline uses — at that point we
      can also bump the threshold higher (~0.7) because embeddings give
      better signal-to-noise.

    The function lowercases internally so callers don't have to.

    Args:
        a: First string. Typically the edge's stored ``retrieval_context``.
        b: Second string. Typically one of the SessionEnd payload's
            ``requeries`` entries.

    Returns:
        Jaccard similarity in ``[0.0, 1.0]``. Returns ``0.0`` when either
        input is empty rather than raising — sessions with no recorded
        requeries are normal, not exceptional.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union
