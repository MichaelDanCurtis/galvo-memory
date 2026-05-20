"""Generic REST CRUD router for the 12-label memory ontology (Task 8).

One router file with dynamic per-label dispatch — the alternative was
12 near-identical handlers per HTTP verb. The label is a path parameter
validated against :data:`ontology.label_mapping.LABEL_TO_TYPE`; the
body is validated against the matching Pydantic model in
:mod:`sidecar.models`.

**Implementation choice (B) — raw Cypher, not library** ``add_entity`` **.**

The library's ``client.long_term.add_entity()`` in v0.2.1 does not accept
an ``extra_labels`` or ``properties`` kwarg (probed Task-8 first action;
signature is::

    add_entity(name, entity_type, *, subtype, description, aliases,
               attributes, resolve, generate_embedding, deduplicate, ...)

and the internal Cypher only writes the library's pascal-cased label
plus ``:Entity``). Our 12 custom labels (``:Decision``, ``:Pattern``, …)
must be on the node for Task 5's :func:`build_scoped_match` /
:func:`build_scoped_search_with_embedding` to find them via
``MATCH (n:Decision)`` / ``'Decision' IN labels(n)``.

So we drop to ``client.graph.execute_write`` with hand-built Cypher
that creates ``(:Entity:<Label> {…})`` directly. This bypasses the
library's dedup machinery — for cycle 1 that's a feature, not a bug:
our nodes are user-authored memory, not auto-extracted entities, and
deduplication semantics are different (we want CREATE-conflict on
ids, not merge-on-name-similarity).

We DO still go through the library's embedder
(:attr:`client.long_term.embedder`) so the vector index stays populated.
The embedder is what makes Phase-2 acceptance gate §3 work (semantic
retrieval at UserPromptSubmit).

**Scope routing.**

Every create body carries an explicit ``scope`` per design §D4. The
router stores it verbatim and the Task-5 cypher helpers filter on it.
Search endpoints take ``scope`` as a query param; the helper's
own-OR-universal rule applies.

**Endpoint surface (per the 12 labels):**

* ``POST /api/{label}`` — create. 201 on success, 422 on validation
  failure, 404 if ``label`` is not registered.
* ``GET /api/{label}/{node_id}`` — fetch by id. 404 if missing.
* ``PATCH /api/{label}/{node_id}`` — partial update. 405 for
  ``Belief`` (design §4 immutability). 404 if missing.
* ``GET /api/search/{label}?q=…&scope=…&limit=…`` — semantic search.
  Uses Task-5 :func:`build_scoped_search_with_embedding`.

DELETE is intentionally omitted in cycle 1 — design §4 says corrections
create new nodes, never destroy. Cycle 2 consolidation may add a
soft-delete path; not in scope here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Path, Query, status
from pydantic import ValidationError

from ontology.label_mapping import LABEL_TO_TYPE, NAME_PROPERTY_PER_LABEL
from sidecar.cypher_helpers import (
    build_scoped_match,
    build_scoped_search_with_embedding,
)
from sidecar.deps import MemoryDep
from sidecar.models import (
    IMMUTABLE_LABELS,
    LABEL_CREATE_MODELS,
    LABEL_UPDATE_MODELS,
    generate_node_id,
    serialize_for_create,
)


__all__ = ["router"]


router = APIRouter(prefix="/api", tags=["nodes"])


# ---------------------------------------------------------------------------
# Internal helpers — kept private to this module.
# ---------------------------------------------------------------------------


def _ensure_label(label: str) -> None:
    """Validate ``label`` against :data:`LABEL_TO_TYPE` or raise 404.

    Centralized so every endpoint surfaces the same error shape. We
    return 404 (not 422) because the path looks like a resource: an
    unknown label is more "no such endpoint" than "bad payload".
    """
    if label not in LABEL_TO_TYPE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown label: {label!r}. Valid: {sorted(LABEL_TO_TYPE)}",
        )


def _validate_create_body(label: str, payload: dict[str, Any]) -> Any:
    """Dispatch payload validation through the per-label create model.

    Returns the validated model instance.

    We don't use FastAPI's native body-typed parameter because the body
    schema varies per label — passing a generic ``dict`` and dispatching
    manually is cleaner than 12 endpoint functions or a Union type
    (Union would expand the OpenAPI surface needlessly). That trades
    one thing away: Pydantic ``ValidationError`` raised inside the
    handler bypasses FastAPI's automatic 422 conversion, so we catch it
    here and re-raise as :class:`HTTPException` with the canonical 422
    status + a stable error envelope that mirrors FastAPI's native
    shape (``{"detail": [...]}``).
    """
    model_cls = LABEL_CREATE_MODELS[label]
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc


def _validate_update_body(label: str, payload: dict[str, Any]) -> Any:
    """Dispatch PATCH body through the per-label update model.

    Returns the model instance. Caller has already verified the label
    is updateable via :data:`IMMUTABLE_LABELS`. Same 422-translation
    pattern as :func:`_validate_create_body`.
    """
    model_cls = LABEL_UPDATE_MODELS[label]
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc


def _row_to_dict(row: dict[str, Any], *, node_key: str = "n") -> dict[str, Any]:
    """Project a raw ``execute_read`` result row into a flat dict.

    Neo4j's Python driver wraps each row's node value in a ``Node``
    object (or dict-of-properties depending on driver version) under
    the projected key. We accept both shapes:

    * ``{"n": {"id": ..., ...}}`` — neo4j-agent-memory's wrapper.
    * ``{"n": <Node with .items()>}`` — raw driver shape.

    The returned dict has the node's properties at the top level so the
    response model can validate against it. ``created_at`` arrives as
    :class:`neo4j.time.DateTime` and is coerced to :class:`datetime` so
    Pydantic v2's strict datetime accepts it.
    """
    node = row.get(node_key, row)
    if hasattr(node, "items"):  # neo4j.Node, dict
        props = dict(node.items())
    else:
        props = dict(node)
    # Coerce neo4j.time.DateTime → stdlib datetime if present.
    if "created_at" in props and props["created_at"] is not None:
        props["created_at"] = _coerce_datetime(props["created_at"])
    if "started_at" in props and props["started_at"] is not None:
        props["started_at"] = _coerce_datetime(props["started_at"])
    if "ended_at" in props and props["ended_at"] is not None:
        props["ended_at"] = _coerce_datetime(props["ended_at"])
    if "last_touched" in props and props["last_touched"] is not None:
        props["last_touched"] = _coerce_datetime(props["last_touched"])
    if "last_run_at" in props and props["last_run_at"] is not None:
        props["last_run_at"] = _coerce_datetime(props["last_run_at"])
    if "valid_from" in props and props["valid_from"] is not None:
        props["valid_from"] = _coerce_datetime(props["valid_from"])
    if "valid_to" in props and props["valid_to"] is not None:
        props["valid_to"] = _coerce_datetime(props["valid_to"])
    return props


def _coerce_datetime(value: Any) -> datetime:
    """Coerce assorted datetime-like objects to :class:`datetime`.

    The Neo4j Python driver returns its own ``neo4j.time.DateTime`` type
    which has a ``to_native()`` method. Stdlib datetimes and ISO strings
    both round-trip through here too. Pydantic v2 strict mode wants the
    real type — coercing here keeps the route handlers simple.
    """
    if isinstance(value, datetime):
        return value
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        return to_native()
    if isinstance(value, str):
        # Tolerate trailing ``Z`` (UTC marker) that fromisoformat doesn't
        # parse on Python < 3.11. (We require 3.12 but defensive doesn't hurt.)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    msg = f"Cannot coerce {type(value).__name__} to datetime"
    raise TypeError(msg)


async def _embed_text(memory: Any, text: str) -> list[float] | None:
    """Generate an embedding via the library's embedder, returning
    ``None`` if no embedder is attached (test fixtures may patch it out).

    Cycle-1 design D2 hardcodes ``MemorySettings.embedding =
    MiniLM-L6-v2 / 384-dim``. We call :meth:`Embedder.embed` directly
    rather than going through ``client.long_term.add_entity`` because
    we're writing raw Cypher (see module docstring (B) decision).
    """
    embedder = getattr(memory.long_term, "embedder", None)
    if embedder is None:
        return None
    embedding = await embedder.embed(text)
    # Embedder returns ``list[float]`` per its protocol.
    return list(embedding) if embedding is not None else None


# ---------------------------------------------------------------------------
# POST /api/{label} — create.
# ---------------------------------------------------------------------------


@router.post(
    "/{label}",
    status_code=status.HTTP_201_CREATED,
    summary="Create a node of the given label",
    description=(
        "Body is validated against the per-label Pydantic create model "
        "in sidecar.models. Server mints an opaque id when the body "
        "omits one. Returns the persisted node properties."
    ),
)
async def create_node(
    memory: MemoryDep,
    label: str = Path(..., description="One of the 12 ontology labels"),
    payload: dict[str, Any] = Body(..., description="Per-label create body"),
) -> dict[str, Any]:
    """Create a new node tagged with both ``:Entity`` and ``:<Label>``.

    The Cypher uses ``CREATE`` (not ``MERGE``) because the design treats
    every node as a new fact — corrections write a NEW node and add a
    ``SUPERSEDES`` edge to the old one. If a caller passes a duplicate
    id, Neo4j's id uniqueness constraint (Task 2) rejects it with a
    ``ConstraintError``; we surface that as 409 Conflict.

    The ``type`` property is set from :data:`LABEL_TO_TYPE` so the
    library's vector search (which filters by ``type``) keeps working
    alongside our label-based filters. ``name`` is set to the value of
    the label's name-property (per :data:`NAME_PROPERTY_PER_LABEL`) so
    the library's machinery has its expected key.
    """
    _ensure_label(label)
    create = _validate_create_body(label, payload)

    # Mint id if absent. The Pydantic model allows None on input.
    node_id = create.id or generate_node_id()
    entity_type = LABEL_TO_TYPE[label]
    name_prop = NAME_PROPERTY_PER_LABEL[label]
    name_value = getattr(create, name_prop)

    props = serialize_for_create(create)
    props["id"] = node_id
    # Library's ``name`` + ``type`` keys live on the same node so its
    # vector index keys + EntityType filter find us.
    props["name"] = name_value
    props["type"] = entity_type

    embedding = await _embed_text(memory, name_value)
    if embedding is not None:
        props["embedding"] = embedding

    # Build CREATE with both labels. Label is path-validated against
    # LABEL_TO_TYPE so it's not user-controlled here.
    cypher = (
        f"CREATE (n:Entity:{label} $props) "
        "SET n.created_at = datetime() "
        "RETURN n"
    )
    try:
        rows = await memory.graph.execute_write(cypher, {"props": props})
    except Exception as exc:  # noqa: BLE001 — surface DB errors as 409/500
        # The library's Neo4jClient surfaces constraint violations as a
        # subclass of Neo4jError. Without importing the library type
        # here (keeps test mocks simple), we detect by string match —
        # the message reliably contains "already exists" / "ConstraintError".
        msg = str(exc).lower()
        if "constraint" in msg or "already exists" in msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Node id collision: {node_id}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"create failed: {exc!r}",
        ) from exc

    if not rows:
        # Driver returned no rows — defensive, shouldn't happen with CREATE.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="create returned no rows",
        )
    return _row_to_dict(rows[0])


# ---------------------------------------------------------------------------
# GET /api/search/{label} — semantic search (registered BEFORE the
# `/{label}/{node_id}` route so FastAPI's path-matcher finds the literal
# ``search`` prefix first; otherwise a request like
# ``/api/search/Decision`` matches ``/{label}/{node_id}`` with
# ``label='search'`` and 404s out at _ensure_label.
# ---------------------------------------------------------------------------


@router.get(
    "/search/{label}",
    summary="Semantic search over a label, scope-filtered",
    description=(
        "Uses the library's embedder + the entity_embedding_idx vector "
        "index. Filters by scope per design §D4 (project queries surface "
        "universal nodes; universal queries do not surface project nodes). "
        "Returns the top ``limit`` hits ordered by similarity score."
    ),
)
async def search_nodes(
    memory: MemoryDep,
    label: str = Path(...),
    q: str = Query(..., description="Free-text query; embedded server-side"),
    scope: str | None = Query(
        None,
        description=(
            "Scope filter. ``project:<id>`` / ``personal`` returns own scope + "
            "universal; ``universal`` returns only universal; omit (None) for "
            "cross-scope queries (admin only)."
        ),
    ),
    limit: int = Query(10, ge=1, le=50),
    threshold: float = Query(0.7, ge=0.0, le=1.0),
) -> list[dict[str, Any]]:
    """Run the Task-5 vector-search helper with the scope filter applied.

    The threshold default 0.7 is the Task-5 default — kept identical so
    operators have one number to remember. Cycle-2 may tune it per label
    based on retrieval-utility scores.
    """
    _ensure_label(label)

    embedding = await _embed_text(memory, q)
    if embedding is None:
        # Without an embedder we cannot do semantic search. Fall back to
        # listing-by-scope so the endpoint at least returns SOMETHING; the
        # operator's intent is "give me top hits for this query" and an
        # empty list is misleading. The fallback is rare in production —
        # the embedder is loaded at lifespan startup and only None in
        # heavily-mocked tests.
        cypher, params = build_scoped_match(
            label=label,
            scope=scope,
            order_by="n.created_at DESC",
            limit=limit,
        )
        rows = await memory.graph.execute_read(cypher, params)
        return [_row_to_dict(r) for r in rows]

    cypher, params = build_scoped_search_with_embedding(
        label=label,
        scope=scope,
        limit=limit,
        threshold=threshold,
    )
    params["embedding"] = embedding
    rows = await memory.graph.execute_read(cypher, params)

    # The vector-search query returns ``n, score`` per row. We surface
    # the score in the response so clients can de-rank in their own UI
    # (the SessionEnd scorer, Task 10, may also use it).
    out: list[dict[str, Any]] = []
    for r in rows:
        node = _row_to_dict(r, node_key="n")
        score = r.get("score")
        if score is not None:
            node["_score"] = float(score)
        out.append(node)
    return out


# ---------------------------------------------------------------------------
# GET /api/{label}/{node_id} — read one.
# ---------------------------------------------------------------------------


@router.get(
    "/{label}/{node_id}",
    summary="Fetch a node by id",
    description="404 when the id exists but not under this label, or doesn't exist at all.",
)
async def get_node(
    memory: MemoryDep,
    label: str = Path(...),
    node_id: str = Path(...),
) -> dict[str, Any]:
    """Look up a node by id with a label filter.

    The label filter is important: ids should be unique globally, but
    asking for ``GET /api/Decision/<artifact_id>`` should 404 rather than
    return the wrong-shape node. Cypher ``MATCH (n:<Label> {id: ...})``
    delivers that filter.
    """
    _ensure_label(label)
    cypher = f"MATCH (n:{label}) WHERE n.id = $id RETURN n LIMIT 1"
    rows = await memory.graph.execute_read(cypher, {"id": node_id})
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found: id={node_id}",
        )
    return _row_to_dict(rows[0])


# ---------------------------------------------------------------------------
# PATCH /api/{label}/{node_id} — update mutable properties.
# ---------------------------------------------------------------------------


@router.patch(
    "/{label}/{node_id}",
    summary="Partial update of a node's mutable properties",
    description=(
        "Returns 405 for immutable labels (currently only Belief — design §4). "
        "Returns 404 when the node doesn't exist. Body is validated against "
        "the per-label update model; only provided keys are written."
    ),
)
async def update_node(
    memory: MemoryDep,
    label: str = Path(...),
    node_id: str = Path(...),
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    """Apply a partial update via per-property SET.

    We use a manually-built ``SET n.k = $params.k`` clause per supplied
    key rather than ``SET n += $props`` because ``+=`` would clobber the
    library's ``type`` / ``name`` / ``embedding`` keys if a buggy update
    model included them. Belt-and-braces: the update models forbid extra
    keys (``ConfigDict(extra='forbid')``) AND list only mutable fields.
    """
    _ensure_label(label)
    if label in IMMUTABLE_LABELS:
        # 405 = Method Not Allowed — the standard HTTP code for "this
        # verb isn't supported on this resource." Tells curl / clients
        # the URL is reachable but the verb is denied.
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=(
                f"{label} is immutable — design §4 mandates new node + "
                f"SUPERSEDES edge for corrections."
            ),
            headers={"Allow": "GET, POST"},
        )

    update = _validate_update_body(label, payload)
    # Only write the fields the caller actually supplied. ``model_dump
    # (exclude_none=True)`` omits ``None`` — but a caller who wants to
    # null a field explicitly cannot do so through this path. Acceptable
    # cycle 1; cycle 2 may add an explicit-null encoding if needed.
    diff = update.model_dump(exclude_none=True, mode="json")
    if not diff:
        # No-op: return the current node as the response so the client
        # gets a consistent shape.
        return await get_node(memory=memory, label=label, node_id=node_id)

    set_clauses = ", ".join(f"n.{key} = $vals.{key}" for key in diff)
    cypher = (
        f"MATCH (n:{label}) WHERE n.id = $id "
        f"SET {set_clauses}, n.updated_at = datetime() "
        f"RETURN n"
    )
    rows = await memory.graph.execute_write(
        cypher,
        {"id": node_id, "vals": diff},
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found: id={node_id}",
        )
    return _row_to_dict(rows[0])
