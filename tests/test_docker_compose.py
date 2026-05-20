"""Task 18 — docker-compose shape tests.

These are pure YAML-parsing checks: no Docker daemon required, no network
required, no embedder warmup. The goal is to fail loudly when the compose
file drifts away from what the rest of Phase 2 assumes about service names,
ports, env vars, and dependency ordering.

Why a static shape test rather than `docker compose config`:

* `docker compose config` requires a working Docker install in CI, which
  the SDK / memory test matrix doesn't guarantee.
* The interesting bugs (renamed env var, missing `service_healthy`
  condition, build context off-by-one) are pure YAML mistakes that
  ``yaml.safe_load`` is already 100% capable of catching.

If PyYAML isn't installed — which shouldn't happen because it's a
transitive dep of the sidecar's pinned ``uvicorn[standard]`` (via
``websockets``) and ``neo4j-agent-memory`` — the test is skipped with a
clear message so a stripped-down env doesn't turn this into a hard
failure.

Cross-checks ``SidecarSettings`` field names against env-var declarations
in the compose file. If somebody renames a field on the Python side, this
test catches the drift before the sidecar boots inside Compose and
silently falls back to localhost defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sidecar.config import SidecarSettings

yaml = pytest.importorskip("yaml", reason="PyYAML not installed in this env")


COMPOSE_PATH = Path(__file__).parent.parent / "docker" / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    """Parse the compose file once and share across tests."""
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict), "compose root must be a mapping"
    assert "services" in data, "compose file missing top-level `services` key"
    return data


# ---------------------------------------------------------------------------
# Service presence + identity.
# ---------------------------------------------------------------------------


def test_neo4j_service_still_present(compose: dict[str, Any]) -> None:
    """The pre-existing Neo4j service must not have been disturbed.

    Task 18 only adds the sidecar; the neo4j service config is load-bearing
    for Phase 1 spike data + Tasks 1-5. A subagent that accidentally rewrote
    the file from scratch would lose the heap settings / plugin list.
    """
    services = compose["services"]
    assert "neo4j" in services, "neo4j service deleted from compose file"
    neo4j = services["neo4j"]
    assert neo4j["image"] == "neo4j:2026.04-community"
    assert neo4j["container_name"] == "galvo-memory-neo4j"
    # Bolt port mapping intact — agent-memory client points here.
    assert "7687:7687" in neo4j["ports"]


def test_sidecar_service_exists(compose: dict[str, Any]) -> None:
    """The sidecar service was added by Task 18."""
    services = compose["services"]
    assert "sidecar" in services, "sidecar service missing — Task 18 not applied"
    sidecar = services["sidecar"]
    assert sidecar["container_name"] == "galvo-memory-sidecar"
    assert sidecar["restart"] == "unless-stopped"


# ---------------------------------------------------------------------------
# Build context / ports / startup ordering.
# ---------------------------------------------------------------------------


def test_sidecar_build_context_one_level_up(compose: dict[str, Any]) -> None:
    """The Dockerfile expects to run with `memory/` as the build root.

    Its `COPY pyproject.toml` / `COPY sidecar/` / `COPY ontology/` /
    `COPY scope/` instructions all assume the context is the memory/
    directory — one level up from the compose file at memory/docker/.
    A context of `.` would fail at the first COPY.
    """
    build = compose["services"]["sidecar"]["build"]
    assert build["context"] == "..", (
        f"sidecar build context must be `..` (= memory/), got {build['context']!r}. "
        "The Dockerfile's COPY instructions break otherwise."
    )
    assert build["dockerfile"] == "sidecar/Dockerfile", (
        f"sidecar Dockerfile path must be `sidecar/Dockerfile` (relative to "
        f"the memory/ context), got {build['dockerfile']!r}"
    )


def test_sidecar_port_mapping(compose: dict[str, Any]) -> None:
    """:7575 is hardcoded in the hook layer; the host-side mapping must match."""
    ports = compose["services"]["sidecar"]["ports"]
    assert "7575:7575" in ports, (
        f"sidecar must expose 7575:7575 (hook layer hits localhost:7575/health "
        f"with no env discovery), got {ports!r}"
    )


def test_sidecar_depends_on_neo4j_healthy(compose: dict[str, Any]) -> None:
    """Sidecar's lifespan calls ``MemoryClient.connect()`` which opens a Bolt
    session immediately — startup must block on Neo4j's healthcheck, not
    just on the container being up.

    The plain ``depends_on: [neo4j]`` short form only waits for the
    container process to exist; the long-form ``condition:
    service_healthy`` waits for healthcheck success. Getting this wrong
    causes the sidecar to crash-loop on cold boot until Neo4j finishes
    JVM warmup (~30-60s).
    """
    deps = compose["services"]["sidecar"]["depends_on"]
    # Must be long-form mapping, not list.
    assert isinstance(deps, dict), (
        f"sidecar.depends_on must use long-form mapping for `condition:`, "
        f"got {type(deps).__name__}"
    )
    assert "neo4j" in deps, "sidecar must depend on neo4j"
    assert deps["neo4j"]["condition"] == "service_healthy", (
        f"sidecar must wait for Neo4j's healthcheck, got "
        f"{deps['neo4j'].get('condition')!r}"
    )


# ---------------------------------------------------------------------------
# Healthcheck contract.
# ---------------------------------------------------------------------------


def test_sidecar_healthcheck_hits_health_endpoint(compose: dict[str, Any]) -> None:
    """The sidecar's healthcheck calls /health on :7575.

    The endpoint comes from Task 6 (sidecar.app). If a refactor renamed
    /health to /healthz or changed the port, this test surfaces it before
    `docker compose up` quietly marks the service permanently unhealthy.
    """
    hc = compose["services"]["sidecar"]["healthcheck"]
    test_cmd = hc["test"]
    # Joined form makes substring matching robust to list-vs-string format.
    joined = " ".join(test_cmd) if isinstance(test_cmd, list) else str(test_cmd)
    assert "http://localhost:7575/health" in joined, (
        f"healthcheck must hit /health on :7575, got {test_cmd!r}"
    )
    # Cold-start grace — embedder model download is ~10s but can spike on
    # slow networks. < 15s start_period would flap.
    start_period = hc.get("start_period", "0s")
    assert start_period.endswith("s"), f"start_period must be a duration, got {start_period!r}"
    assert int(start_period.rstrip("s")) >= 15, (
        f"start_period must be >=15s to allow embedder download on cold start, "
        f"got {start_period!r}"
    )


# ---------------------------------------------------------------------------
# Env-var prefix + field-name cross-check against SidecarSettings.
# ---------------------------------------------------------------------------


# The set of SidecarSettings fields the compose file is supposed to set.
# We don't require coverage of `host` / `port` because those are baked into
# the container at the uvicorn CMD level (`--host 0.0.0.0 --port 7575`) —
# overriding them via env would skip the EXPOSE 7575 in the Dockerfile.
COMPOSE_DRIVEN_FIELDS = {
    "neo4j_uri",
    "neo4j_user",
    "neo4j_password",
    "neo4j_database",
    "embedding_model",
    "embedding_dimensions",
}


def test_env_var_prefix_matches_settings(compose: dict[str, Any]) -> None:
    """All env vars use the prefix declared by ``SidecarSettings.model_config``.

    pydantic-settings derives env names from `env_prefix + field_name.upper()`.
    If somebody changes the prefix on the Python side (or in the compose
    file), this test fails before the sidecar boots and silently uses
    SidecarSettings defaults (which point at bolt://localhost:7687 — wrong
    inside the compose network).
    """
    prefix = SidecarSettings.model_config.get("env_prefix")
    assert prefix == "GALVO_MEMORY_SIDECAR_", (
        f"SidecarSettings.env_prefix changed to {prefix!r} — update "
        f"memory/docker/docker-compose.yml env keys to match."
    )

    env = compose["services"]["sidecar"]["environment"]
    assert isinstance(env, dict), "expected env mapping form, not list"
    for key in env:
        assert key.startswith(prefix), (
            f"env var {key!r} does not match SidecarSettings prefix {prefix!r}"
        )


def test_env_var_names_match_settings_fields(compose: dict[str, Any]) -> None:
    """Every compose env var maps to a real ``SidecarSettings`` field.

    pydantic-settings is silent about unknown env vars (we explicitly set
    ``extra="ignore"``), so a typo like
    ``GALVO_MEMORY_SIDECAR_NEO4J_URL`` (URL instead of URI) would simply
    be discarded and the sidecar would fall back to the localhost default.
    Catch the typo at YAML-parse time instead.
    """
    prefix = SidecarSettings.model_config["env_prefix"]
    valid_fields = set(SidecarSettings.model_fields.keys())

    env = compose["services"]["sidecar"]["environment"]
    for key in env:
        assert key.startswith(prefix)
        field_name = key[len(prefix) :].lower()
        assert field_name in valid_fields, (
            f"env var {key!r} → field {field_name!r} is not a SidecarSettings "
            f"field. Known fields: {sorted(valid_fields)}"
        )


def test_compose_driven_fields_are_all_set(compose: dict[str, Any]) -> None:
    """The fields a container deploy MUST override are all present in env.

    `neo4j_uri` in particular is load-bearing: the default
    `bolt://localhost:7687` would target the sidecar container itself, not
    the neo4j service. Forgetting to override it inside the compose
    network would silently break MemoryClient.connect().
    """
    prefix = SidecarSettings.model_config["env_prefix"]
    env = compose["services"]["sidecar"]["environment"]
    present_fields = {key[len(prefix) :].lower() for key in env if key.startswith(prefix)}
    missing = COMPOSE_DRIVEN_FIELDS - present_fields
    assert not missing, (
        f"compose env block missing required SidecarSettings overrides: "
        f"{sorted(missing)}. The sidecar will fall back to localhost / "
        f"placeholder defaults inside the compose network."
    )


def test_neo4j_uri_uses_service_name_not_localhost(compose: dict[str, Any]) -> None:
    """Inside the compose network the Bolt host must be ``neo4j``.

    `localhost` resolves to the sidecar container itself, which has nothing
    listening on :7687. This is the single most common docker-compose
    networking footgun for newly-added sidecar services.
    """
    env = compose["services"]["sidecar"]["environment"]
    uri = env["GALVO_MEMORY_SIDECAR_NEO4J_URI"]
    assert uri == "bolt://neo4j:7687", (
        f"sidecar must dial Neo4j via the compose service name, got {uri!r}"
    )
    assert "localhost" not in uri, "do NOT use localhost — wrong container"


def test_neo4j_password_matches_neo4j_auth(compose: dict[str, Any]) -> None:
    """Sidecar password must match the password baked into NEO4J_AUTH.

    If they diverge the sidecar's first ``MemoryClient.connect()`` returns
    AuthenticationRateLimit after a few retries — confusing because Neo4j
    itself is healthy. Catch the mismatch in YAML parsing.
    """
    neo4j_auth = compose["services"]["neo4j"]["environment"]["NEO4J_AUTH"]
    expected_user, expected_pw = neo4j_auth.split("/", 1)

    sidecar_env = compose["services"]["sidecar"]["environment"]
    assert sidecar_env["GALVO_MEMORY_SIDECAR_NEO4J_USER"] == expected_user
    assert sidecar_env["GALVO_MEMORY_SIDECAR_NEO4J_PASSWORD"] == expected_pw


def test_embedding_dimensions_match_default(compose: dict[str, Any]) -> None:
    """Compose's embedding_dimensions must equal the SidecarSettings default.

    Cycle 1 locked all-MiniLM-L6-v2 / 384 dims; a mismatch between the
    compose value and the SidecarSettings default would let the sidecar
    boot with mismatched vector index dims and reject every write at
    SchemaManager.adopt_existing_graph().
    """
    env = compose["services"]["sidecar"]["environment"]
    # Compose YAML preserves strings; SidecarSettings coerces to int.
    assert env["GALVO_MEMORY_SIDECAR_EMBEDDING_DIMENSIONS"] == "384"
    assert env["GALVO_MEMORY_SIDECAR_EMBEDDING_MODEL"] == "all-MiniLM-L6-v2"
    # Confirm those match the Python defaults — a drift on either side fails the test.
    defaults = SidecarSettings()
    assert defaults.embedding_dimensions == 384
    assert defaults.embedding_model == "all-MiniLM-L6-v2"
