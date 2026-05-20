"""Task 5 — scope-aware Cypher composition helpers.

Pure-Python unit tests; no Neo4j substrate required for any of these. The
helpers in :mod:`sidecar.cypher_helpers` are intentionally pure — they
emit ``(cypher_string, params_dict)`` tuples and never touch a driver —
so assertions here run anywhere ``pytest`` does.

What we're protecting:

* The scope-OR-universal composition rule (design §D4) — a project query
  surfaces universal nodes; a universal query does *not* surface project
  nodes. A regression here would silently leak cross-project context or
  hide universal facts.
* The params-dict contract — callers pass ``**params`` straight to the
  Neo4j driver, so a missing ``scope`` key (or a spurious one) means
  every query fails on the driver side. We test both directions.
* The vector-search shape — the ``CALL db.index.vector.queryNodes``
  procedure has rigid parameter ordering; an over-eager refactor that
  swaps argument positions would compile-pass and runtime-fail.
* The ``RETRIEVED_IN`` writer MERGE shape — the difference between MERGE
  and CREATE here is "one edge per (node, session)" vs. "one edge per
  retrieval". Cycle-1 design says one edge per session per node.

An optional ``@pytest.mark.integration`` test at the bottom of the file
asserts that the composed ``build_scoped_match`` query actually runs
against a live Neo4j; it's gated behind the ``integration`` marker so
``pytest -m 'not integration'`` skips it.
"""

from __future__ import annotations

import re

import pytest

from sidecar.cypher_helpers import (
    build_retrieved_in_writer,
    build_scoped_match,
    build_scoped_search_with_embedding,
    scope_filter_clause,
)


# ---------------------------------------------------------------------------
# scope_filter_clause — the predicate-fragment primitive.
# ---------------------------------------------------------------------------


def test_scope_filter_clause_none() -> None:
    """``None`` scope disables filtering — return empty string so callers
    can detect "no clause" without parsing.

    This is the explicit-callers-only branch; hooks must never pass
    ``None``. An empty string lets :func:`build_scoped_match` skip the
    ``WHERE`` keyword entirely when there's also no ``extra_where``.
    """
    assert scope_filter_clause(None) == ""


def test_scope_filter_clause_universal() -> None:
    """``"universal"`` returns a single-scope predicate — no project bleed.

    A universal-only query MUST NOT surface project-scoped nodes; otherwise
    asking "what universal facts do we know" returns project memories the
    operator never opted in to. The fragment uses a literal so callers
    don't need to bind a ``$scope`` parameter.
    """
    assert scope_filter_clause("universal") == "n.scope = 'universal'"


def test_scope_filter_clause_project() -> None:
    """``"project:galvo"`` returns the OR-universal composition.

    Project queries surface universal-scoped nodes (operator-promoted
    facts) in addition to own-scope nodes. The fragment uses ``$scope``
    so the caller can bind ``"project:galvo"`` — the helper is
    label-agnostic so this works for any project id.
    """
    fragment = scope_filter_clause("project:galvo")
    assert fragment == "(n.scope = $scope OR n.scope = 'universal')"


def test_scope_filter_clause_personal() -> None:
    """``"personal"`` returns the same OR-universal composition.

    Personal scope is treated identically to project scope by the filter
    composition — own scope plus universal opt-ins. The difference is
    only in the parameter value, not the predicate shape.
    """
    fragment = scope_filter_clause("personal")
    assert fragment == "(n.scope = $scope OR n.scope = 'universal')"


def test_scope_filter_clause_custom_alias() -> None:
    """Non-default alias propagates into the predicate.

    The helper is used by queries that already have a node variable named
    something other than ``n`` (e.g. a ``MATCH (m:Mistake)`` query where
    ``n`` is the project's :Session). A regression that hardcodes ``n.``
    would silently filter the wrong variable.
    """
    project_fragment = scope_filter_clause("project:galvo", alias="m")
    assert project_fragment == "(m.scope = $scope OR m.scope = 'universal')"
    universal_fragment = scope_filter_clause("universal", alias="m")
    assert universal_fragment == "m.scope = 'universal'"
    none_fragment = scope_filter_clause(None, alias="m")
    assert none_fragment == ""


# ---------------------------------------------------------------------------
# build_scoped_match — full query composition.
# ---------------------------------------------------------------------------


def test_build_scoped_match_no_scope() -> None:
    """``None`` scope produces no WHERE clause at all.

    Cross-scope queries (federation, ops) shouldn't be filtered. The
    output should be a clean MATCH + RETURN + LIMIT with no ``WHERE``.
    Params still contain ``limit`` but never ``scope``.
    """
    query, params = build_scoped_match("Decision", None)
    assert "WHERE" not in query, f"None scope produced WHERE clause: {query!r}"
    assert "MATCH (n:Decision)" in query
    assert "RETURN n" in query
    assert "LIMIT $limit" in query
    assert params == {"limit": 10}
    assert "scope" not in params


def test_build_scoped_match_with_scope_project() -> None:
    """Project scope produces a parenthesized OR-universal WHERE and
    binds ``$scope`` in params.

    Inverse of the no-scope case: ``WHERE`` appears, the predicate is
    the OR-universal fragment, and ``params["scope"]`` is set to the
    project string.
    """
    query, params = build_scoped_match("Decision", "project:galvo")
    assert "WHERE (n.scope = $scope OR n.scope = 'universal')" in query
    assert params == {"limit": 10, "scope": "project:galvo"}


def test_build_scoped_match_with_extra_where() -> None:
    """Caller-supplied ``extra_where`` is AND'd in after the scope clause.

    The scope clause comes first (it's the structural filter); the caller's
    predicate is layered on top with ``AND``. We check both that the
    keyword is present and that the order is scope-first — a swap would
    let a poorly-bound ``extra_where`` short-circuit the scope filter.
    """
    query, params = build_scoped_match(
        "Decision",
        "project:galvo",
        extra_where="n.confidence > 0.5",
    )
    assert "WHERE (n.scope = $scope OR n.scope = 'universal') AND n.confidence > 0.5" in query
    assert params == {"limit": 10, "scope": "project:galvo"}


def test_build_scoped_match_extra_where_without_scope() -> None:
    """``extra_where`` with ``None`` scope produces a clean ``WHERE extra_where``.

    The composer must not emit a leading ``AND`` when the scope clause is
    empty — ``WHERE AND foo = bar`` is a syntax error. This guards against
    a regression where the AND-joiner doesn't notice the empty fragment.
    """
    query, params = build_scoped_match(
        "Decision",
        None,
        extra_where="n.confidence > 0.5",
    )
    assert "WHERE n.confidence > 0.5" in query
    assert "WHERE AND" not in query
    assert "AND n.confidence" not in query  # no spurious leading AND
    assert params == {"limit": 10}


def test_build_scoped_match_order_limit() -> None:
    """ORDER BY + LIMIT both present + in the right order.

    The Neo4j parser requires ``ORDER BY`` before ``LIMIT``; a swap
    compiles to a syntax error. The helper's contract is that callers
    pass only the ORDER BY *expression* (we add the keyword), so the
    test asserts on the full expanded form.
    """
    query, params = build_scoped_match(
        "Decision",
        "project:galvo",
        order_by="n.created_at DESC",
        limit=25,
    )
    assert "ORDER BY n.created_at DESC" in query
    assert "LIMIT $limit" in query
    # ORDER BY must come before LIMIT in Cypher.
    assert query.index("ORDER BY") < query.index("LIMIT")
    assert params["limit"] == 25


def test_build_scoped_match_params_excludes_universal() -> None:
    """``"universal"`` scope must NOT add ``scope`` to params.

    The clause is a literal (``n.scope = 'universal'``) — no bound
    parameter — so a spurious ``scope`` key would either be unused
    (best case) or override a caller's later mutation (worst case).
    """
    query, params = build_scoped_match("Decision", "universal")
    assert "n.scope = 'universal'" in query
    assert "scope" not in params
    assert params == {"limit": 10}


def test_build_scoped_match_custom_alias() -> None:
    """Non-default alias propagates through every clause.

    MATCH, WHERE, RETURN, and ORDER BY must all use the same alias.
    A regression where one clause hardcodes ``n`` while others use the
    custom alias would compile to an "unknown variable" error at
    execution time.
    """
    query, params = build_scoped_match(
        "Mistake",
        "project:galvo",
        order_by="m.created_at DESC",
        alias="m",
    )
    assert "MATCH (m:Mistake)" in query
    assert "(m.scope = $scope OR m.scope = 'universal')" in query
    assert "RETURN m" in query
    assert "ORDER BY m.created_at DESC" in query


def test_build_scoped_match_label_inlined() -> None:
    """The label string is inlined (not bound as a parameter).

    Neo4j can't parameterize node labels — they must be literals in the
    Cypher source. The helper's contract puts the responsibility on the
    caller to pass a known-safe label from
    :data:`memory.ontology.label_mapping.LABEL_TO_TYPE`.
    """
    query, _ = build_scoped_match("Belief", "project:galvo")
    assert "MATCH (n:Belief)" in query
    # Verify NO label parameter is referenced.
    assert "$label" not in query


# ---------------------------------------------------------------------------
# build_scoped_search_with_embedding — vector search composition.
# ---------------------------------------------------------------------------


def test_build_scoped_search_with_embedding_shape() -> None:
    """The procedure call shape matches Neo4j's
    ``db.index.vector.queryNodes`` signature exactly.

    The procedure takes positional args ``(index_name, k, vector)``;
    swapping their order is a runtime error with a confusing message,
    so we test the literal call site here.
    """
    query, params = build_scoped_search_with_embedding("Decision", "project:galvo")
    assert (
        "CALL db.index.vector.queryNodes('entity_embedding_idx', $vector_limit, $embedding)"
        in query
    )
    assert "YIELD node AS n, score" in query
    assert "WHERE score >= $threshold" in query
    assert "AND 'Decision' IN labels(n)" in query
    assert "AND (n.scope = $scope OR n.scope = 'universal')" in query
    assert "RETURN n, score" in query
    assert "ORDER BY score DESC" in query
    assert "LIMIT $limit" in query


def test_build_scoped_search_with_embedding_params() -> None:
    """The params dict has ``vector_limit`` over-fetched 3x, plus
    ``threshold`` and ``limit`` for the caller's tuning.

    Over-fetching by 3x is the cycle-1 compromise: the vector index
    can't filter by label or scope on its own, so we ask for more
    candidates than we need and trim post-hoc. A regression that
    drops the multiplier (or makes it 1x) would silently return
    empty pages whenever the top-k all fall outside the scope.
    """
    query, params = build_scoped_search_with_embedding(
        "Decision",
        "project:galvo",
        limit=10,
        threshold=0.8,
    )
    assert params["vector_limit"] == 30  # 10 * 3
    assert params["threshold"] == 0.8
    assert params["limit"] == 10
    assert params["scope"] == "project:galvo"
    # Caller is responsible for binding `embedding`; helper must not.
    assert "embedding" not in params
    assert query  # non-empty


def test_build_scoped_search_with_embedding_universal_omits_scope_param() -> None:
    """``"universal"`` search produces a literal predicate and no
    ``scope`` param — mirrors :func:`build_scoped_match`.

    Same rationale: the universal case uses a Cypher literal so there's
    no parameter to bind. A regression that adds ``scope`` to params here
    would mean every universal search has an unused parameter (harmless
    but signals confusion).
    """
    query, params = build_scoped_search_with_embedding("Belief", "universal")
    assert "n.scope = 'universal'" in query
    assert "scope" not in params
    assert "$scope" not in query  # no bound-param reference either


def test_build_scoped_search_with_embedding_no_scope() -> None:
    """``None`` scope produces no scope filter at all in the query."""
    query, params = build_scoped_search_with_embedding("Decision", None)
    assert "n.scope" not in query
    assert "scope" not in params
    # Still has label filter, threshold, etc.
    assert "AND 'Decision' IN labels(n)" in query


def test_build_scoped_search_with_embedding_custom_index() -> None:
    """The ``embedding_index`` kwarg is the cycle-2 Qwen3 swap surface.

    Passing a different index name (matching the 4096-dim Qwen3 index a
    future operator will provision) must produce a query that targets
    that index, not the default. No helper changes — just a kwarg.
    """
    query, _ = build_scoped_search_with_embedding(
        "Decision",
        "project:galvo",
        embedding_index="qwen3_entity_embedding_idx",
    )
    assert "'qwen3_entity_embedding_idx'" in query
    assert "'entity_embedding_idx'" not in query  # no stale default


# ---------------------------------------------------------------------------
# build_retrieved_in_writer — feedback-edge MERGE Cypher.
# ---------------------------------------------------------------------------


def test_build_retrieved_in_writer_merge_shape() -> None:
    """The writer uses MERGE (not CREATE) so re-retrieval in the same
    session updates instead of duplicating.

    Design §D5 says one ``RETRIEVED_IN`` edge per (node, session); the
    properties on the edge reflect the most recent retrieval. A
    regression from MERGE to CREATE would produce N edges for N retrievals
    of the same node, breaking the session-end utility scorer's
    cardinality assumptions.
    """
    cypher = build_retrieved_in_writer()
    assert "MERGE (n)-[r:RETRIEVED_IN]->(s)" in cypher
    # Belt-and-braces: explicit absence of CREATE for the edge.
    assert "CREATE (n)-[r:RETRIEVED_IN]->(s)" not in cypher


def test_build_retrieved_in_writer_required_params() -> None:
    """All five expected parameters are referenced in the query body.

    The writer expects: ``$node_id``, ``$session_id``, ``$rank``, ``$score``,
    ``$context``. ``utility_score`` is intentionally NOT set here (Task
    10's SessionEnd scorer writes it later). A regression dropping any
    of the SET-properties means the feedback row is missing data.
    """
    cypher = build_retrieved_in_writer()
    assert "$node_id" in cypher
    assert "$session_id" in cypher
    assert "$rank" in cypher
    assert "$score" in cypher
    assert "$context" in cypher
    # utility_score is the Task 10 surface — not set here.
    assert "utility_score" not in cypher


def test_build_retrieved_in_writer_property_assignments() -> None:
    """All four edge properties are SET in a single SET clause.

    Multiple SET statements would still work but make the writer harder
    to grep when debugging. Test the literal text so a future "split
    SETs" refactor doesn't slip through unreviewed.
    """
    cypher = build_retrieved_in_writer()
    assert "SET r.retrieval_rank = $rank," in cypher
    assert "r.retrieval_score = $score," in cypher
    assert "r.retrieval_context = $context," in cypher
    assert "r.created_at = datetime()" in cypher


def test_build_retrieved_in_writer_custom_param_names() -> None:
    """The session_id_param + node_id_param kwargs let callers avoid
    name collisions when this Cypher is composed into a larger
    multi-statement transaction.

    Most callers won't change them; the kwargs exist for the hooks-layer
    composer (Task 11) which might be merging several queries.
    """
    cypher = build_retrieved_in_writer(
        session_id_param="sid",
        node_id_param="nid",
    )
    assert "$sid" in cypher
    assert "$nid" in cypher
    # Default param names must NOT appear when overridden.
    assert "$session_id" not in cypher
    assert "$node_id" not in cypher


def test_build_retrieved_in_writer_returns_edge() -> None:
    """The query returns the upserted edge so the caller can verify the
    write landed. A regression dropping the RETURN clause would mean
    callers couldn't distinguish "edge created" from "MATCH failed
    silently because the node id was wrong".
    """
    cypher = build_retrieved_in_writer()
    assert cypher.rstrip().endswith("RETURN r")


# ---------------------------------------------------------------------------
# Cross-helper invariants.
# ---------------------------------------------------------------------------


def test_helpers_are_pure_no_side_effects() -> None:
    """Repeated calls with the same args return identical output.

    The helpers' purity contract is what lets us avoid Neo4j in unit
    tests. A regression introducing module-level mutable state (e.g.
    a cache) would silently invalidate this contract.
    """
    q1, p1 = build_scoped_match("Decision", "project:galvo")
    q2, p2 = build_scoped_match("Decision", "project:galvo")
    assert q1 == q2
    assert p1 == p2
    assert p1 is not p2, "params dict must be a fresh object per call (no shared state)"


def test_no_sql_keyword_injection_via_label() -> None:
    """Sanity check: the label is inlined verbatim.

    Callers are responsible for using labels from
    :data:`memory.ontology.label_mapping.LABEL_TO_TYPE` — but if they
    don't, the helper does not sanitize, on the principle that the
    caller is in-process (not user-facing) so injection is a code
    bug not an attack surface. We test the inlining to make the
    behavior explicit, not to claim safety.
    """
    weird = "WeirdLabel"
    query, _ = build_scoped_match(weird, "project:galvo")
    assert f"MATCH (n:{weird})" in query


def test_scope_param_value_round_trip() -> None:
    """For each scope-value family, the ``scope`` param round-trips
    (or is correctly omitted)."""
    cases = [
        ("project:galvo", "project:galvo"),
        ("project:a-different-id", "project:a-different-id"),
        ("personal", "personal"),
    ]
    for scope_in, expected_param in cases:
        _, params = build_scoped_match("Decision", scope_in)
        assert params.get("scope") == expected_param, (
            f"scope_in={scope_in!r} produced params={params!r}"
        )

    # Universal + None both omit `scope`.
    for omit_case in ("universal", None):
        _, params = build_scoped_match("Decision", omit_case)
        assert "scope" not in params, (
            f"scope_in={omit_case!r} unexpectedly produced scope param: {params!r}"
        )


def test_whitespace_well_formed() -> None:
    """Composed Cypher has single spaces, no double spaces / leading
    keyword junk. A double-space won't fail the parser but indicates a
    composer regression that may produce wrong syntax for some inputs.
    """
    queries = [
        build_scoped_match("Decision", None)[0],
        build_scoped_match("Decision", "project:galvo")[0],
        build_scoped_match("Decision", "universal", order_by="n.created_at DESC")[0],
        build_scoped_match(
            "Decision",
            "project:galvo",
            extra_where="n.confidence > 0.5",
            order_by="n.created_at DESC",
            limit=20,
        )[0],
        build_scoped_search_with_embedding("Decision", "project:galvo")[0],
        build_scoped_search_with_embedding("Decision", None)[0],
        build_retrieved_in_writer(),
    ]
    for q in queries:
        assert "  " not in q, f"Double space in composed query: {q!r}"
        assert not q.startswith(" "), f"Leading space in composed query: {q!r}"
        assert not re.search(r"\s+,", q), f"Whitespace before comma in: {q!r}"


# ---------------------------------------------------------------------------
# Live-Neo4j integration — bonus test per Task 5 §"Tests".
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scoped_match_actually_runs_against_neo4j() -> None:
    """End-to-end: write two Decisions with different scopes, query with
    ``build_scoped_match`` for one project's scope, assert only that
    project's Decision comes back.

    This is the load-bearing assertion for the whole helper module: the
    composed Cypher actually executes, the parameters bind correctly, and
    the scope-OR-universal filter does what the unit tests claim.

    The test creates fresh ``Decision`` nodes with synthetic ids to avoid
    colliding with anything the rest of the suite has written. It does
    NOT use the library's ``MemoryClient`` — only the raw Cypher path —
    to keep the test independent of Task 8's REST-CRUD machinery.
    """
    from neo4j import AsyncGraphDatabase  # type: ignore[import-untyped]

    from ontology.label_mapping import (
        NEO4J_DATABASE,
        NEO4J_PASSWORD,
        NEO4J_URI,
        NEO4J_USERNAME,
    )

    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Cleanup any leftover synthetic data from prior runs.
            await session.run(
                "MATCH (d:Decision) WHERE d.id STARTS WITH 'test-cypher-helpers-' DETACH DELETE d"
            )
            # Two project-scope Decisions + one universal.
            await session.run(
                "CREATE (:Decision {id: 'test-cypher-helpers-1', title: 'galvo decision',"
                " scope: 'project:galvo-test', created_at: datetime()})"
            )
            await session.run(
                "CREATE (:Decision {id: 'test-cypher-helpers-2', title: 'other decision',"
                " scope: 'project:other-test', created_at: datetime()})"
            )
            await session.run(
                "CREATE (:Decision {id: 'test-cypher-helpers-3', title: 'universal decision',"
                " scope: 'universal', created_at: datetime()})"
            )

            # Query for galvo-test scope: expect 2 (project:galvo-test + universal).
            query, params = build_scoped_match(
                "Decision",
                "project:galvo-test",
                extra_where="n.id STARTS WITH 'test-cypher-helpers-'",
                order_by="n.id ASC",
                limit=10,
            )
            result = await session.run(query, **params)  # type: ignore[arg-type]
            rows = [record async for record in result]
            ids = [row["n"]["id"] for row in rows]
            assert ids == ["test-cypher-helpers-1", "test-cypher-helpers-3"], (
                f"Expected galvo + universal, got {ids}"
            )

            # Universal-only query: expect 1 (no project leak).
            query, params = build_scoped_match(
                "Decision",
                "universal",
                extra_where="n.id STARTS WITH 'test-cypher-helpers-'",
                order_by="n.id ASC",
                limit=10,
            )
            result = await session.run(query, **params)  # type: ignore[arg-type]
            rows = [record async for record in result]
            ids = [row["n"]["id"] for row in rows]
            assert ids == ["test-cypher-helpers-3"], (
                f"Universal-only query leaked project rows: {ids}"
            )

            # Cleanup.
            await session.run(
                "MATCH (d:Decision) WHERE d.id STARTS WITH 'test-cypher-helpers-' DETACH DELETE d"
            )
    finally:
        await driver.close()
