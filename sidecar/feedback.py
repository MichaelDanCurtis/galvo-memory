"""Feedback logging for the D5 cycle-1 SAGE-lite loop.

Every retrieval gets a ``RETRIEVED_IN`` edge written from the returned
node to the requesting :Session. Task 10's ``SessionEnd`` scorer
populates the ``utility_score`` property later by walking these edges
and computing the signed [-1, +1] score per design §5.

Design §11 step-5 (locked decision D5): logging + per-session scoring
ship in cycle 1; consolidation ("dream-state") is explicitly deferred
to cycle 2.

Why a separate module from :mod:`sidecar.cypher_helpers`? The helpers
are pure Cypher composition — no I/O, no library imports. This module
*executes* the writer against a live :class:`MemoryClient`. Keeping the
two split lets ``test_cypher_helpers`` stay Neo4j-free, while
``test_feedback`` mocks the client cleanly.

Why doesn't the writer also MERGE the :Session node? Because the
SessionStart hook (Task 12) is responsible for creating the :Session
the moment a session begins — it has the cwd, environment variables,
and Claude Code session id available, none of which are accessible
here. If we MERGEd a stub :Session, we'd need a follow-up backfill
pass at SessionEnd (Task 15) to populate ``started_at``, ``scope``,
``task_description``, etc. Cleaner to make the caller's responsibility
explicit: "ensure the :Session node exists before calling this." See
:func:`log_retrieval` for the contract.

Error policy: writes that fail (transient Neo4j hiccup, mis-bound
parameter, deadlock) are logged via :mod:`logging` and silently
counted as 0. Retrieval logging is *observability* — never the
critical path. A retrieval that succeeded but couldn't be logged is
still a successful retrieval; raising here would force every search
handler to wrap the call in try/except, defeating the purpose of the
helper.

References:

* Design §D5 "Feedback loop (logging + scoring)"
* Phase 2 plan §Task 9
* :func:`sidecar.cypher_helpers.build_retrieved_in_writer` — the Cypher
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sidecar.cypher_helpers import build_retrieved_in_writer

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]


__all__ = [
    "MAX_CONTEXT_CHARS",
    "log_retrieval",
]


_log = logging.getLogger(__name__)


# Hard cap on ``retrieval_context`` length. The property is informational
# (used by the SessionEnd scorer to spot "agent re-queried for similar
# info" patterns), not load-bearing for ranking, so truncation is safe.
# 500 chars accommodates a typical user prompt without bloating edge
# storage when retrieval fires once per user turn over a long session.
MAX_CONTEXT_CHARS: int = 500


async def log_retrieval(
    memory: "MemoryClient",
    *,
    session_id: str,
    query: str,
    hits: list[Any],
) -> int:
    """Write one ``RETRIEVED_IN`` edge per hit, all pointing at one :Session.

    Per design §D5 each retrieval the sidecar performs writes a feedback
    edge from the returned node to the active :Session. Cycle-1 the
    ``utility_score`` property is left null; Task 10's ``SessionEnd``
    scorer populates it later by walking these same edges.

    The :Session node MUST already exist — this function does *not*
    MERGE one. Callers (the search endpoint in Task 8, or the
    SessionStart hook in Task 12) are responsible for creating the
    :Session at session boot. If the :Session is missing, the underlying
    Cypher's ``MATCH (s:Session ...)`` fails silently (no rows, no
    edge written) and this function logs a warning per-hit and returns
    the count of writes that *did* land. See module docstring for
    rationale.

    Args:
        memory: Live :class:`MemoryClient`. In production this comes
            from :data:`sidecar.deps.MemoryDep`; in tests it's a
            :class:`unittest.mock.MagicMock` with an
            :class:`~unittest.mock.AsyncMock` on ``memory.graph.execute_write``.
        session_id: The id of the active :Session node. Must match the
            ``id`` property the SessionStart hook used.
        query: The retrieval's query string. Stored verbatim on each
            edge as ``retrieval_context`` after truncation to
            :data:`MAX_CONTEXT_CHARS`. ``None`` or empty strings are
            treated as ``""``.
        hits: Ordered list of retrieval hits (index 0 = top hit = rank 0).
            Each entry must carry ``.id`` (or be a dict with an ``"id"``
            key); ``.score`` is optional and defaults to ``0.0`` when
            missing. See :func:`_hit_node_id` and :func:`_hit_score` for
            the exact shape contract.

    Returns:
        Count of edges successfully written. Equals ``len(hits)`` on the
        happy path; less when individual writes failed (transient DB
        error, hit missing ``id``, etc.) — each shortfall is logged
        with WARNING-level context so the operator can spot patterns.
    """
    context = (query or "")[:MAX_CONTEXT_CHARS]
    cypher = build_retrieved_in_writer()
    count = 0
    for rank, hit in enumerate(hits):
        node_id = _hit_node_id(hit)
        if node_id is None:
            _log.warning("hit at rank %d has no .id attribute; skipping", rank)
            continue
        score = _hit_score(hit)
        try:
            await memory.graph.execute_write(
                cypher,
                {
                    "node_id": node_id,
                    "session_id": session_id,
                    "rank": rank,
                    "score": score,
                    "context": context,
                },
            )
            count += 1
        except Exception as exc:  # noqa: BLE001 — logged, never raised
            # Feedback is observability; a failure here MUST NOT abort
            # the retrieval the caller is still inside the middle of.
            # We log with WARNING (not ERROR) because a single dropped
            # feedback row is not actionable — operators triage via the
            # rate of these in the sidecar log, not individual events.
            _log.warning(
                "RETRIEVED_IN edge write failed for node=%s session=%s: %r",
                node_id,
                session_id,
                exc,
            )
    return count


# ---------------------------------------------------------------------------
# Hit-shape adapters — duck-typed extraction with safe fallbacks.
# ---------------------------------------------------------------------------


def _hit_node_id(hit: Any) -> str | None:
    """Extract the node id from a hit.

    Accepts two shapes:

    * Object with an ``.id`` attribute (the common case — both the
      library's ``Entity``/``Fact`` and our Task-8 response models
      expose it).
    * Dict with an ``"id"`` key (raw Cypher results from the library
      sometimes arrive this way).

    Returns ``None`` when neither shape matches; the caller logs and
    skips that hit. We don't raise because a single ill-shaped hit
    shouldn't poison the whole batch — log + skip + carry on.
    """
    if hasattr(hit, "id"):
        return str(hit.id)
    if isinstance(hit, dict):
        node_id = hit.get("id")
        return str(node_id) if node_id is not None else None
    return None


def _hit_score(hit: Any) -> float:
    """Extract the similarity score from a hit, defaulting to ``0.0``.

    Vector-search hits expose ``.score`` (a float in ``[0, 1]``);
    structural lookups (e.g. "list all Decisions in scope") don't have a
    score concept. We default to ``0.0`` rather than ``None`` so the
    edge property is always a concrete float — Task 10's scorer
    short-circuits on ``score == 0.0`` to mean "non-semantic hit".
    """
    if hasattr(hit, "score"):
        return float(hit.score)
    if isinstance(hit, dict):
        return float(hit.get("score", 0.0))
    return 0.0
