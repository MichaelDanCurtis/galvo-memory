"""Compose scope-aware Cypher queries.

All Phase 2 retrieval goes through these helpers. Design §D4 scope tiers
(``project:<id>`` / ``personal`` / ``universal``) become ``WHERE``-clause
fragments inserted into the returned Cypher string. The helpers are pure:
no I/O, no Neo4j calls, no library imports — output is a
``(cypher_string, params_dict)`` tuple ready for
``client.graph.execute_read(query, **params)``.

Cycle-1 scope semantics (per design §D4):

* ``scope = "project:<id>"`` — return both project-scoped *and*
  universal-scoped nodes. Universal nodes are explicitly opted-in by an
  operator promoting a fact, so surfacing them inside a project query is
  intended behavior.
* ``scope = "personal"`` — same composition rule: own scope *plus*
  ``universal``.
* ``scope = "universal"`` — return ONLY universal nodes. No project bleed
  in either direction.
* ``scope = None`` — disable scope filtering entirely (cross-scope query).
  This branch exists for explicit-callers-only — Hub federation, ad-hoc
  ops queries, etc. Hooks must never pass ``None``.

Why the asymmetry? A project author who memorized "X is the convention"
inside their project shouldn't have it leak into another project's
context. Universal facts (e.g. "Python uses snake_case") are the
deliberate cross-project channel.

References:
    * Design §D4 "3-tier scope partitioning"
    * Phase 2 plan §Task 5
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "build_retrieved_in_writer",
    "build_scoped_match",
    "build_scoped_search_with_embedding",
    "scope_filter_clause",
]


# ---------------------------------------------------------------------------
# Scope filter primitive — every other helper composes through this.
# ---------------------------------------------------------------------------


def scope_filter_clause(scope: str | None, *, alias: str = "n") -> str:
    """Return the ``WHERE``-clause fragment for a scope filter.

    The result is a bare predicate fragment with no leading keyword: the
    caller wraps it in ``WHERE`` or ``AND`` depending on whether there is
    already a ``WHERE`` clause in the query. Use :func:`build_scoped_match`
    when you don't want to do that wrapping yourself.

    Args:
        scope: The scope to filter by. ``None`` disables filtering entirely
            (cross-scope queries — explicit-callers-only). ``"universal"``
            returns only universal nodes (no project bleed). Anything else
            (``"project:<id>"``, ``"personal"``) returns scope-OR-universal.
        alias: The Cypher node variable name (default ``"n"``).

    Returns:
        Empty string when ``scope`` is ``None``. Otherwise a parenthesized
        predicate fragment using a ``$scope`` parameter — for example
        ``"(n.scope = $scope OR n.scope = 'universal')"`` — or a bare
        ``"n.scope = 'universal'"`` predicate when ``scope == "universal"``.

    The composition rule (own scope OR universal) is asymmetric on purpose:
    a query scoped to ``"universal"`` does NOT surface project-scoped nodes,
    but a project query DOES surface universal-scoped nodes. See module
    docstring for the design rationale.
    """
    if scope is None:
        return ""
    if scope == "universal":
        # Universal queries get ONLY universal nodes. No project bleed.
        # Hardcoded literal (no $scope param) so callers don't have to
        # remember to omit the scope param for the universal case.
        return f"{alias}.scope = 'universal'"
    # project:<id> OR personal — own scope plus universal opt-ins.
    return f"({alias}.scope = $scope OR {alias}.scope = 'universal')"


# ---------------------------------------------------------------------------
# MATCH-by-label helper — the common case for listing / filtering.
# ---------------------------------------------------------------------------


def build_scoped_match(
    label: str,
    scope: str | None,
    *,
    extra_where: str | None = None,
    order_by: str | None = None,
    limit: int = 10,
    alias: str = "n",
) -> tuple[str, dict[str, Any]]:
    """Build a scoped ``MATCH ... RETURN`` query for a single label.

    Composition steps (in order):

    1. ``MATCH ({alias}:{label})``
    2. ``WHERE`` clause, if any — built from the scope filter
       (:func:`scope_filter_clause`) AND any caller-supplied ``extra_where``.
    3. ``RETURN {alias}``
    4. ``ORDER BY {order_by}`` if supplied
    5. ``LIMIT $limit``

    Args:
        label: Node label to match (e.g. ``"Decision"``, ``"Belief"``).
            Inlined into the query — callers are responsible for using a
            label from :data:`memory.ontology.label_mapping.LABEL_TO_TYPE`.
        scope: Scope filter (``None`` = no filter; ``"project:galvo"`` /
            ``"personal"`` = own-scope OR universal; ``"universal"`` =
            universal only). See module docstring.
        extra_where: Optional additional ``WHERE``-clause fragment to
            ``AND`` in. Caller is responsible for binding any parameters
            it references — they won't appear in the returned ``params``
            dict unless the caller adds them after the fact.
        order_by: Optional ``ORDER BY`` clause body
            (e.g. ``"n.created_at DESC"``). The keyword is added; pass only
            the expression.
        limit: ``LIMIT`` value. Bound as ``$limit``.
        alias: Cypher node variable name. Must match what ``extra_where``
            and ``order_by`` reference.

    Returns:
        ``(query_string, params_dict)``. Caller does::

            q, p = build_scoped_match("Decision", "project:galvo")
            results = await client.graph.execute_read(q, **p)

    The ``params`` dict always contains ``limit``. It contains ``scope``
    only when ``scope`` is a project/personal value — ``None`` and
    ``"universal"`` scopes do not need the parameter (they're handled by
    the absence of filtering and a literal predicate, respectively).
    """
    parts: list[str] = [f"MATCH ({alias}:{label})"]

    where_clauses: list[str] = []
    scope_clause = scope_filter_clause(scope, alias=alias)
    if scope_clause:
        where_clauses.append(scope_clause)
    if extra_where:
        where_clauses.append(extra_where)
    if where_clauses:
        parts.append("WHERE " + " AND ".join(where_clauses))

    parts.append(f"RETURN {alias}")
    if order_by:
        parts.append(f"ORDER BY {order_by}")
    parts.append("LIMIT $limit")

    params: dict[str, Any] = {"limit": limit}
    if scope is not None and scope != "universal":
        params["scope"] = scope

    return " ".join(parts), params


# ---------------------------------------------------------------------------
# Vector-search helper — the semantic-retrieval common case.
# ---------------------------------------------------------------------------


def build_scoped_search_with_embedding(
    label: str,
    scope: str | None,
    *,
    embedding_index: str = "entity_embedding_idx",
    limit: int = 10,
    threshold: float = 0.7,
    alias: str = "n",
) -> tuple[str, dict[str, Any]]:
    """Build a vector-search query with scope filter.

    Uses Neo4j's ``CALL db.index.vector.queryNodes`` procedure for cycle 1.
    We over-fetch by ``3x`` and then filter post-hoc by label + scope, then
    truncate to ``limit``. This is the pattern recommended by Neo4j for
    "vector + structural filter" until the GA ``SEARCH`` clause is wired
    through ``neo4j-agent-memory`` (deferred to cycle 2).

    Args:
        label: Node label the hit must carry. The query filters with
            ``'{label}' IN labels({alias})`` because the procedure returns
            ``:Entity`` super-label nodes; we then verify the multi-label
            tag that ``adopt_existing_graph`` writes.
        scope: Scope filter (semantics as in :func:`scope_filter_clause`).
        embedding_index: Name of the vector index to query. Default
            ``"entity_embedding_idx"`` matches
            :data:`memory.ontology.label_mapping.VECTOR_INDEX_NAME` —
            the index created by ``adopt_existing_graph`` at 384 dims.
        limit: Maximum hits returned to the caller. The procedure is
            asked for ``3x`` candidates to leave headroom for the label +
            scope filter without empty pages.
        threshold: Minimum vector-similarity score. Bound as ``$threshold``.
        alias: Cypher variable name for the node.

    Returns:
        ``(query_string, params_dict)``. Caller must add the embedding
        vector to ``params`` (key ``"embedding"``)::

            q, p = build_scoped_search_with_embedding("Decision", scope)
            p["embedding"] = embedder.encode(query_text).tolist()
            hits = await client.graph.execute_read(q, **p)

    The returned ``params`` always contains ``vector_limit``, ``threshold``,
    and ``limit``. It contains ``scope`` only when ``scope`` is a
    project/personal value.

    The embedding-index name is parameterized so cycle 2's Qwen3 swap
    (D2 deferral, 4096-dim) only needs to pass a new index name — no
    helper changes.
    """
    scope_clause = scope_filter_clause(scope, alias=alias)
    where_extra = f" AND {scope_clause}" if scope_clause else ""

    cypher = (
        f"CALL db.index.vector.queryNodes('{embedding_index}', $vector_limit, $embedding) "
        f"YIELD node AS {alias}, score "
        f"WHERE score >= $threshold "
        f"AND '{label}' IN labels({alias})"
        f"{where_extra} "
        f"RETURN {alias}, score "
        f"ORDER BY score DESC "
        f"LIMIT $limit"
    )
    params: dict[str, Any] = {
        "vector_limit": limit * 3,  # over-fetch then filter
        "threshold": threshold,
        "limit": limit,
    }
    if scope is not None and scope != "universal":
        params["scope"] = scope
    return cypher, params


# ---------------------------------------------------------------------------
# RETRIEVED_IN edge writer — D5 cycle-1 feedback logging Cypher.
# ---------------------------------------------------------------------------


def build_retrieved_in_writer(
    *,
    session_id_param: str = "session_id",
    node_id_param: str = "node_id",
) -> str:
    """Build the Cypher that upserts a ``RETRIEVED_IN`` edge.

    Per design §D5 (cycle 1 logging surface), every retrieval writes one
    ``RETRIEVED_IN`` edge per hit, pointing the retrieved node at the
    current ``Session``. The SessionEnd scorer (Task 10) later populates
    ``utility_score``; this helper leaves that property null.

    The query uses ``MERGE`` so re-retrieval of the same node within a
    session updates the rank/score/context in place instead of creating
    a duplicate edge. ``created_at`` is reset on every merge — it reflects
    the most-recent retrieval, not the first. (If we ever want the first-
    retrieval timestamp, switch to ``ON CREATE SET created_at = datetime()``
    and stop setting it in the trailing ``SET``.)

    Args:
        session_id_param: Parameter name to bind the session id under.
            Default ``"session_id"``. Rarely changed; exposed for callers
            that have other parameters of the same name to avoid.
        node_id_param: Parameter name to bind the retrieved node id under.
            Default ``"node_id"``.

    Returns:
        A Cypher query string. The caller binds the parameters at execute
        time::

            q = build_retrieved_in_writer()
            await client.graph.execute_write(
                q,
                node_id=hit.id,
                session_id=session_id,
                rank=rank,
                score=hit.score,
                context=query_text,
            )

        Required parameters: ``${session_id_param}``, ``${node_id_param}``,
        ``rank`` (int), ``score`` (float), ``context`` (str).

    See ``memory/ontology/edges.py`` for the edge-property indexes that
    make ``utility_score`` and ``retrieval_rank`` queryable for cycle-2
    consolidation.
    """
    return (
        f"MATCH (n) WHERE n.id = ${node_id_param} "
        f"MATCH (s:Session {{id: ${session_id_param}}}) "
        f"MERGE (n)-[r:RETRIEVED_IN]->(s) "
        f"SET r.retrieval_rank = $rank, "
        f"r.retrieval_score = $score, "
        f"r.retrieval_context = $context, "
        f"r.created_at = datetime() "
        f"RETURN r"
    )
