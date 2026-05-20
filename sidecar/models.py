"""Per-label Pydantic models for the 12-node REST CRUD surface (Task 8).

Each of the 12 labels from :data:`ontology.label_mapping.LABEL_TO_TYPE`
gets three models:

* ``<Label>Create`` — POST body. Carries all required design-§4 properties
  for that label plus ``scope`` and (optionally) a caller-supplied ``id``.
* ``<Label>Update`` — PATCH body. Every property is :class:`Optional` so
  the route handler can apply only the supplied keys via ``COALESCE``.
* ``<Label>Response`` — what the router returns on GET / POST / PATCH.
  Adds ``id`` + ``created_at`` to the create-shape.

``Belief`` has NO update model — design §4 mandates immutability for
beliefs (corrections create a new Belief node and a ``SUPERSEDES`` edge).
The router enforces this by returning ``405 Method Not Allowed`` on
``PATCH /api/Belief/{id}``.

Property names mirror design §4 verbatim. We don't fabricate fields —
every property here corresponds to a bullet in the design doc. If a label
gains a new property in cycle 2, this is the file that changes.

The three label-keyed dicts at the bottom (:data:`LABEL_CREATE_MODELS`,
:data:`LABEL_UPDATE_MODELS`, :data:`LABEL_RESPONSE_MODELS`) are the
generic router's entry point: it looks up the right Pydantic class by
``label`` string and dispatches validation through it. Adding a new label
means adding the three model classes AND registering them in these dicts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


__all__ = [
    "ArtifactCreate",
    "ArtifactResponse",
    "ArtifactUpdate",
    "BeliefCreate",
    "BeliefResponse",
    "CommitCreate",
    "CommitResponse",
    "CommitUpdate",
    "ConstraintCreate",
    "ConstraintResponse",
    "ConstraintUpdate",
    "ConventionCreate",
    "ConventionResponse",
    "ConventionUpdate",
    "DecisionCreate",
    "DecisionResponse",
    "DecisionUpdate",
    "FailureCreate",
    "FailureResponse",
    "FailureUpdate",
    "IMMUTABLE_LABELS",
    "LABEL_CREATE_MODELS",
    "LABEL_RESPONSE_MODELS",
    "LABEL_UPDATE_MODELS",
    "MistakeCreate",
    "MistakeResponse",
    "MistakeUpdate",
    "PatternCreate",
    "PatternResponse",
    "PatternUpdate",
    "SessionCreate",
    "SessionResponse",
    "SessionUpdate",
    "TaskCreate",
    "TaskResponse",
    "TaskUpdate",
    "TestCreate",
    "TestResponse",
    "TestUpdate",
    "generate_node_id",
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def generate_node_id() -> str:
    """Mint a stable opaque id for a freshly-created node.

    Format: ``node_<12-hex>``. 12 hex chars from a UUID4 give ~48 bits of
    entropy — collisions are negligible at the per-developer-process scale
    we ship at in cycle 1. The ``node_`` prefix makes IDs grep-able in
    logs and Cypher dumps without confusing them with the library's
    ``entity_<uuid>`` shape (which we don't use because we own the multi-
    label tagging via raw Cypher per the Task-8 (B) decision).
    """
    return f"node_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Base classes — every label inherits from these.
# ---------------------------------------------------------------------------


class _BaseCreate(BaseModel):
    """Shared by all 12 ``<Label>Create`` models.

    ``id`` is optional on the wire — when missing, the router calls
    :func:`generate_node_id`. Allowing the caller to set it makes hook
    code reproducible (a hook can hash the input + check exists-by-id
    before deciding to write).

    ``scope`` is required because the design §D4 partitioning rule has
    no defensible default at the API boundary — the hook layer always
    knows the scope by the time it talks to the sidecar, so requiring it
    here surfaces hook bugs early.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    """Optional opaque id. Auto-generated when omitted."""

    scope: str = Field(min_length=1)
    """One of ``project:<id>`` / ``personal`` / ``universal`` per design §D4.
    Format validation is delegated to :mod:`scope.detector`; this field only
    asserts non-empty."""


class _BaseResponse(BaseModel):
    """Shared by all 12 ``<Label>Response`` models.

    Adds the two server-assigned fields. ``created_at`` is a UTC
    datetime stamped by Neo4j's ``datetime()`` Cypher function at write
    time — using server time keeps timestamps consistent across clients
    in different timezones.
    """

    model_config = ConfigDict(from_attributes=True, extra="allow")

    id: str
    scope: str
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Decision — design §4: rationale, alternatives_considered, confidence, scope.
# ---------------------------------------------------------------------------


class DecisionCreate(_BaseCreate):
    """A non-trivial choice made during a session.

    ``title`` is the name property (per :data:`NAME_PROPERTY_PER_LABEL`)
    — short imperative summary that doubles as the search-key text the
    embedder hashes.
    """

    title: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class DecisionUpdate(BaseModel):
    """PATCH body — every property is nullable for partial updates.

    The ``title`` (name property) is intentionally NOT updateable here
    because the library's vector index is keyed off ``name`` + ``type``;
    changing it post-create would require re-embedding and re-indexing.
    Rename via SUPERSEDES semantics in cycle 2 instead.
    """

    model_config = ConfigDict(extra="forbid")

    rationale: str | None = None
    alternatives_considered: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DecisionResponse(_BaseResponse):
    title: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)
    confidence: float


# ---------------------------------------------------------------------------
# Pattern — design §4: description, evidence_count, success_rate, codebase_scope.
# ---------------------------------------------------------------------------


class PatternCreate(_BaseCreate):
    """A recurring approach observed across sessions."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    evidence_count: int = Field(default=1, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    codebase_scope: str | None = None


class PatternUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    evidence_count: int | None = Field(default=None, ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    codebase_scope: str | None = None


class PatternResponse(_BaseResponse):
    name: str
    description: str
    evidence_count: int
    success_rate: float
    codebase_scope: str | None = None


# ---------------------------------------------------------------------------
# Convention — design §4: description, source, strength.
# ---------------------------------------------------------------------------


class ConventionCreate(_BaseCreate):
    """An established way of doing things in a specific codebase."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    source: str = Field(default="inferred")
    """One of ``inferred`` / ``explicit`` / ``from_AGENTS.md`` (per design)."""
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class ConventionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    source: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)


class ConventionResponse(_BaseResponse):
    name: str
    description: str
    source: str
    strength: float


# ---------------------------------------------------------------------------
# Constraint — design §4: description, type, source.
# ---------------------------------------------------------------------------


class ConstraintCreate(_BaseCreate):
    """A hard requirement."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    constraint_type: str = Field(default="performance")
    """One of ``performance`` / ``security`` / ``compatibility`` / ``regulatory``.

    Named ``constraint_type`` instead of ``type`` to avoid colliding with the
    library's ``type`` property on the ``:Entity`` super-label.
    """
    source: str | None = None


class ConstraintUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    constraint_type: str | None = None
    source: str | None = None


class ConstraintResponse(_BaseResponse):
    name: str
    description: str
    constraint_type: str
    source: str | None = None


# ---------------------------------------------------------------------------
# Task — design §4: description, status, priority.
# ---------------------------------------------------------------------------


class TaskCreate(_BaseCreate):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="")
    status: str = Field(default="open")
    """``open`` / ``in_progress`` / ``blocked`` / ``done`` / ``abandoned``."""
    priority: str = Field(default="medium")
    """``low`` / ``medium`` / ``high`` / ``urgent``."""


class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    status: str | None = None
    priority: str | None = None


class TaskResponse(_BaseResponse):
    title: str
    description: str
    status: str
    priority: str


# ---------------------------------------------------------------------------
# Session — design §4: started_at, ended_at, agent_tool, task_description, outcome.
# ---------------------------------------------------------------------------


class SessionCreate(_BaseCreate):
    """A unit of work.

    ``title`` is the search-key text; ``task_description`` is the longer
    free-form summary. Both are present because the design lists
    ``task_description`` explicitly and Phase 2 hooks (Task 11+) want a
    short label to surface in top-of-mind injections.
    """

    title: str = Field(min_length=1, max_length=200)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    agent_tool: str = Field(default="claude-code")
    task_description: str = Field(default="")
    outcome: str | None = None
    """``success`` / ``partial`` / ``failed`` / ``abandoned`` / ``null``
    (still in progress)."""


class SessionUpdate(BaseModel):
    """Sessions ARE updateable — the SessionEnd hook needs to set
    ``ended_at`` + ``outcome`` after the session opens. This is the only
    real-world mutation path that's safe by design (the create writer is
    the SessionStart hook; the update writer is the SessionEnd hook;
    nothing else touches the node)."""

    model_config = ConfigDict(extra="forbid")

    ended_at: datetime | None = None
    task_description: str | None = None
    outcome: str | None = None


class SessionResponse(_BaseResponse):
    title: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    agent_tool: str
    task_description: str
    outcome: str | None = None


# ---------------------------------------------------------------------------
# Mistake — design §4: description, root_cause, fix_applied, time_to_discover.
# ---------------------------------------------------------------------------


class MistakeCreate(_BaseCreate):
    """Something that went wrong.

    ``summary`` is the name property — short headline; ``description``
    is the longer narrative.
    """

    summary: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    root_cause: str | None = None
    fix_applied: str | None = None
    time_to_discover_seconds: int | None = Field(default=None, ge=0)
    """Wall-clock seconds from mistake occurring to being noticed.
    ``None`` when not measured. Suffix to disambiguate units."""


class MistakeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    root_cause: str | None = None
    fix_applied: str | None = None
    time_to_discover_seconds: int | None = Field(default=None, ge=0)


class MistakeResponse(_BaseResponse):
    summary: str
    description: str
    root_cause: str | None = None
    fix_applied: str | None = None
    time_to_discover_seconds: int | None = None


# ---------------------------------------------------------------------------
# Commit — design §4: sha, message, intent, reverted_by.
# ---------------------------------------------------------------------------


class CommitCreate(_BaseCreate):
    """A code change.

    The ``sha`` is the name property (per :data:`NAME_PROPERTY_PER_LABEL`)
    AND the uniqueness key per the Task-2 constraint
    ``commit_sha_unique``. The router rejects duplicate SHAs at the DB
    layer (Neo4j raises a constraint-violation error which we surface as
    409 Conflict).
    """

    sha: str = Field(min_length=4, max_length=64)
    """Git SHA. Full 40-char or short 7+ are both accepted; longer is
    just stricter uniqueness."""
    message: str = Field(min_length=1)
    intent: str | None = None
    """Free-form: what the author was trying to achieve. Distinct from
    ``message`` which can be a one-line commit subject."""
    reverted_by: str | None = None
    """SHA of the commit that reverted this one, if any. Cycle-2
    consolidation may add an edge instead."""


class CommitUpdate(BaseModel):
    """SHA + message are immutable post-create (commits are facts).

    Only ``intent`` and ``reverted_by`` make sense to update later —
    intent because we may learn the real reason after the fact;
    reverted_by because it's set when the revert lands.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    reverted_by: str | None = None


class CommitResponse(_BaseResponse):
    sha: str
    message: str
    intent: str | None = None
    reverted_by: str | None = None


# ---------------------------------------------------------------------------
# Failure — design §4: type, error_signature, resolved (bool).
# ---------------------------------------------------------------------------


class FailureCreate(_BaseCreate):
    """A specific run failure."""

    error_signature: str = Field(min_length=1, max_length=500)
    """Short canonical fingerprint of the failure — e.g. exception class
    + first traceback frame. Used as the name property + dedup key."""
    failure_type: str = Field(default="runtime")
    """``test`` / ``build`` / ``lint`` / ``runtime``. Named
    ``failure_type`` to avoid colliding with the library's ``type``
    property on ``:Entity``."""
    resolved: bool = False
    full_message: str | None = None
    """Full error message + traceback for forensics. Optional because
    some failure signatures are self-describing."""


class FailureUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved: bool | None = None
    full_message: str | None = None
    failure_type: str | None = None


class FailureResponse(_BaseResponse):
    error_signature: str
    failure_type: str
    resolved: bool
    full_message: str | None = None


# ---------------------------------------------------------------------------
# Artifact — design §4: path, language, role, last_touched.
# ---------------------------------------------------------------------------


class ArtifactCreate(_BaseCreate):
    """A file, module, component, or function of significance."""

    path: str = Field(min_length=1, max_length=1024)
    """Absolute or repo-relative path. Repo-relative recommended for
    portability across worktrees. Used as the name property + dedup key."""
    language: str | None = None
    """Programming language identifier (``python``, ``rust``, etc.).
    ``None`` for non-code files like ``README.md``."""
    role: str | None = None
    """Free-form role tag — ``entrypoint`` / ``test`` / ``config`` / etc.
    Used by retrieval to prioritize high-role files."""
    last_touched: datetime | None = None
    """When a session most recently touched this artifact. Updated by
    PostToolUse hook (Task 14)."""


class ArtifactUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    role: str | None = None
    last_touched: datetime | None = None


class ArtifactResponse(_BaseResponse):
    path: str
    language: str | None = None
    role: str | None = None
    last_touched: datetime | None = None


# ---------------------------------------------------------------------------
# Test — design §4: identifier, last_run_status, last_run_at.
# ---------------------------------------------------------------------------


class TestCreate(_BaseCreate):
    """A test case or run result."""

    identifier: str = Field(min_length=1, max_length=500)
    """Test path — ``tests/test_foo.py::TestBar::test_baz`` or its
    equivalent in non-pytest harnesses. Used as the name property +
    dedup key."""
    last_run_status: str = Field(default="unknown")
    """``passed`` / ``failed`` / ``skipped`` / ``error`` / ``unknown``."""
    last_run_at: datetime | None = None
    runner: str | None = None
    """Framework — ``pytest`` / ``vitest`` / ``cargo test`` / etc."""


class TestUpdate(BaseModel):
    """PostToolUse hook updates ``last_run_status`` + ``last_run_at`` after
    every test invocation."""

    model_config = ConfigDict(extra="forbid")

    last_run_status: str | None = None
    last_run_at: datetime | None = None
    runner: str | None = None


class TestResponse(_BaseResponse):
    identifier: str
    last_run_status: str
    last_run_at: datetime | None = None
    runner: str | None = None


# ---------------------------------------------------------------------------
# Belief — design §4: claim, confidence, valid_from, valid_to, source_session_id.
# ---------------------------------------------------------------------------
# IMMUTABLE — corrections create new Belief + SUPERSEDES edge. No update model.


class BeliefCreate(_BaseCreate):
    """An inferred fact about the codebase or environment.

    Beliefs are the only nodes with explicit temporal validity. Per
    design §4, the writer never mutates a Belief — corrections create a
    new Belief node and a ``SUPERSEDES`` edge from new to old. The old
    belief's ``valid_to`` is set in the SAME transaction that creates
    the new one (Task 8 doesn't implement that orchestration — it's
    cycle-2 consolidation; cycle 1 just enforces no mutation).
    """

    claim: str = Field(min_length=1, max_length=500)
    """The asserted fact. Used as the name property + embedding key."""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    """Set when this belief is superseded. NULL while the belief is current."""
    source_session_id: str | None = None
    """Which session minted this belief. Lets us audit which sessions
    were producing high-confidence vs low-confidence beliefs."""


class BeliefResponse(_BaseResponse):
    claim: str
    confidence: float
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_session_id: str | None = None


# ---------------------------------------------------------------------------
# Label → model dispatch tables. The generic router uses these.
# ---------------------------------------------------------------------------

# Note: BaseModel covers both _BaseCreate and the (Belief-only) sentinel-less
# update map. Keep this typed as BaseModel for the broadest compatibility.

LABEL_CREATE_MODELS: dict[str, type[BaseModel]] = {
    "Decision": DecisionCreate,
    "Pattern": PatternCreate,
    "Convention": ConventionCreate,
    "Constraint": ConstraintCreate,
    "Task": TaskCreate,
    "Session": SessionCreate,
    "Mistake": MistakeCreate,
    "Commit": CommitCreate,
    "Failure": FailureCreate,
    "Artifact": ArtifactCreate,
    "Test": TestCreate,
    "Belief": BeliefCreate,
}
"""Maps a label string to its Pydantic create model.

The router validates incoming POST bodies through this dict — a label
missing from here means the endpoint won't accept POST. Adding a 13th
label means adding a row here AND in :data:`LABEL_RESPONSE_MODELS`
(and ``LABEL_UPDATE_MODELS`` unless the new label is immutable).
"""

LABEL_UPDATE_MODELS: dict[str, type[BaseModel]] = {
    "Decision": DecisionUpdate,
    "Pattern": PatternUpdate,
    "Convention": ConventionUpdate,
    "Constraint": ConstraintUpdate,
    "Task": TaskUpdate,
    "Session": SessionUpdate,
    "Mistake": MistakeUpdate,
    "Commit": CommitUpdate,
    "Failure": FailureUpdate,
    "Artifact": ArtifactUpdate,
    "Test": TestUpdate,
    # Belief is intentionally omitted — design §4 immutability rule.
}
"""Maps a label string to its Pydantic update (PATCH) model.

A label NOT in this dict is immutable: the router returns 405 on
``PATCH /api/<Label>/{id}``. Currently only ``Belief`` is in this state.
"""

LABEL_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "Decision": DecisionResponse,
    "Pattern": PatternResponse,
    "Convention": ConventionResponse,
    "Constraint": ConstraintResponse,
    "Task": TaskResponse,
    "Session": SessionResponse,
    "Mistake": MistakeResponse,
    "Commit": CommitResponse,
    "Failure": FailureResponse,
    "Artifact": ArtifactResponse,
    "Test": TestResponse,
    "Belief": BeliefResponse,
}
"""Maps a label string to its Pydantic response model. Used by the router
to serialize Cypher rows back to clients with the right field shape."""

IMMUTABLE_LABELS: frozenset[str] = frozenset(
    set(LABEL_CREATE_MODELS) - set(LABEL_UPDATE_MODELS)
)
"""Labels whose nodes cannot be PATCHed. The router checks this set
explicitly so the 405 path is grep-able and the failure mode is
documentable (not "we forgot to register an Update model")."""


def get_create_model(label: str) -> type[BaseModel]:
    """Look up the create model for a label, raising :class:`KeyError`
    on unknown labels.

    Routers prefer this helper over a raw ``LABEL_CREATE_MODELS[label]``
    so the failure mode is consistent across endpoints.
    """
    return LABEL_CREATE_MODELS[label]


def get_update_model(label: str) -> type[BaseModel] | None:
    """Look up the update model for a label.

    Returns ``None`` for labels in :data:`IMMUTABLE_LABELS` (currently
    just ``Belief``). The router maps a ``None`` return to a 405 response.
    """
    return LABEL_UPDATE_MODELS.get(label)


def get_response_model(label: str) -> type[BaseModel]:
    """Look up the response model for a label.

    Falls back to a permissive ``_BaseResponse`` if the label isn't
    registered, so newly-added labels don't 500 on read paths before
    the rest of the registration lands.
    """
    return LABEL_RESPONSE_MODELS.get(label, _BaseResponse)


def serialize_for_create(create_model: BaseModel) -> dict[str, Any]:
    """Project a Pydantic create model into a property dict for Cypher.

    Strips ``id`` (handled separately by the router so it can mint one
    when absent) and uses ``mode='json'`` so :class:`datetime` values
    become ISO 8601 strings — Neo4j Bolt accepts those and stores them
    as ``DATETIME`` when the property index is typed. List fields stay
    as lists (Neo4j supports arrays natively).
    """
    return create_model.model_dump(exclude={"id"}, mode="json", exclude_none=True)
