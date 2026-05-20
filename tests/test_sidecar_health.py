"""Task 6 — FastAPI sidecar ``/health`` endpoint tests.

Two tiers, matching the rest of the suite:

* Unit tests patch ``sidecar.app.MemoryClient`` so the lifespan never
  touches a real Neo4j. These run anywhere ``pytest`` does and form the
  primary regression net.
* ``@pytest.mark.integration`` boots the real lifespan against the
  ``bolt://localhost:7687`` substrate (matches the ``docker compose``
  stack in ``memory/docker/``). Skipped by ``pytest -m 'not integration'``.

What we're protecting:

* Lifespan attaches ``app.state.memory`` and ``app.state.config`` — Task 7
  depends on these names exactly.
* Healthy path returns 200 with the four required keys: ``status``,
  ``neo4j``, ``embedder``, ``embedding_dimensions``.
* When :meth:`MemoryClient.get_stats` raises, the endpoint returns 503
  with a ``detail`` field that contains the exception text — the hook
  layer logs that string when degrading to no-op mode.
* The lifespan teardown is exercised (``close()`` is called). A regression
  that swallowed the ``finally:`` would leak Neo4j sessions in production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Iterator
    from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Shared patching infrastructure — the MemoryClient stand-in for unit tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_client() -> "Iterator[MagicMock]":
    """Patch :class:`MemoryClient` inside :mod:`sidecar.app` so the
    lifespan never opens a real Neo4j connection.

    The patch swaps the class symbol, so ``MemoryClient(settings=...)``
    inside the lifespan returns the same :class:`MagicMock` instance
    every test gets a fresh copy of. Three async methods are pre-stubbed
    because the lifespan + /health together exercise all of them:
    ``connect`` (boot), ``close`` (teardown), ``get_stats`` (the probe).
    """
    with patch("sidecar.app.MemoryClient") as MockClient:
        instance = MockClient.return_value
        instance.connect = AsyncMock(return_value=None)
        instance.close = AsyncMock(return_value=None)
        instance.get_stats = AsyncMock(return_value={"entities": 0, "facts": 0})
        yield instance


# ---------------------------------------------------------------------------
# Healthy-path tests.
# ---------------------------------------------------------------------------


def test_health_returns_200_when_neo4j_ok(mocked_client: "MagicMock") -> None:
    """The happy path: Neo4j up, embedder loaded, ``get_stats`` answers.

    ``TestClient(app)``'s context manager runs lifespan startup +
    shutdown around the request — important because /health reads
    ``app.state.memory``, which the lifespan attaches. A regression that
    moved the attachment out of lifespan would 500 here with an
    ``AttributeError``.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["neo4j"] == "ok"
    # Embedder name comes from SidecarSettings default (D2: MiniLM).
    assert body["embedder"] == "all-MiniLM-L6-v2"
    assert body["embedding_dimensions"] == 384
    # The stats blob is opaque to the hook layer but should round-trip.
    assert body["stats"] == {"entities": 0, "facts": 0}


def test_health_includes_required_keys(mocked_client: "MagicMock") -> None:
    """All four cycle-1 acceptance-gate keys are present in the 200 body.

    Acceptance gate §1 (PHASE-2-PLAN.md) requires ``status`` + ``neo4j``;
    the embedder + dimensions fields are operational additions the hooks
    layer uses to verify the running config matches what they expect.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        body = client.get("/health").json()

    for required_key in ("status", "neo4j", "embedder", "embedding_dimensions"):
        assert required_key in body, f"missing {required_key!r} in {body!r}"


def test_lifespan_calls_connect_and_close(mocked_client: "MagicMock") -> None:
    """The lifespan boots :class:`MemoryClient.connect` on entry and
    :meth:`close` on exit.

    A regression that dropped the ``finally:`` in lifespan would leak
    Neo4j connections — particularly painful in tests where TestClient
    instances come and go. We assert both methods were awaited exactly
    once across the lifespan window.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        client.get("/health")  # trigger the request → ensures lifespan ran

    mocked_client.connect.assert_awaited_once()
    mocked_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Failure-path tests — 503 with diagnostic.
# ---------------------------------------------------------------------------


def test_health_returns_503_when_neo4j_down(mocked_client: "MagicMock") -> None:
    """When :meth:`get_stats` raises, the endpoint returns 503.

    The hook layer reads the response's ``detail`` field to decide
    whether to degrade to no-op or surface the error to the user. We
    don't lock in the exact message format — only that the lower-cased
    text contains the word "unhealthy" so log greps stay stable.
    """
    from sidecar.app import app

    # Swap in a raising stats call.
    mocked_client.get_stats.side_effect = ConnectionError("neo4j unreachable")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503, response.text
    assert "unhealthy" in response.json()["detail"].lower()
    # Belt-and-braces: the exception's repr should appear in detail so
    # the hook layer's log gives an operator a starting point.
    assert "neo4j unreachable" in response.json()["detail"]


def test_health_503_does_not_block_shutdown(mocked_client: "MagicMock") -> None:
    """503 from a failing /health must still let the lifespan tear down.

    A regression that re-raised inside the lifespan body would prevent
    the ``close()`` call from running. We trigger the failing endpoint
    inside the TestClient, exit the context, and assert ``close()``
    was still called.
    """
    from sidecar.app import app

    mocked_client.get_stats.side_effect = RuntimeError("ouch")
    with TestClient(app) as client:
        client.get("/health")  # 503

    mocked_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Lifespan-state contract — Task 7 (deps) depends on these names.
# ---------------------------------------------------------------------------


def test_lifespan_attaches_memory_and_config_to_app_state(
    mocked_client: "MagicMock",
) -> None:
    """``app.state.memory`` is the MemoryClient instance and
    ``app.state.config`` is a :class:`SidecarSettings`.

    Task 7's ``get_memory_client`` / ``get_settings`` dependencies pull
    these by name. A renaming refactor would break the deps without
    failing this file's other tests.
    """
    from sidecar.app import app
    from sidecar.config import SidecarSettings

    with TestClient(app) as client:
        # Inside the context the lifespan has run.
        client.get("/health")
        assert app.state.memory is mocked_client, (
            "lifespan didn't attach mocked client at app.state.memory"
        )
        assert isinstance(app.state.config, SidecarSettings), (
            "lifespan didn't attach SidecarSettings at app.state.config"
        )


# ---------------------------------------------------------------------------
# Integration — boots the real lifespan against live Neo4j.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_health_integration() -> None:
    """End-to-end: real Neo4j + real embedder.

    Requires the ``memory/docker`` Compose stack to be running with the
    default password. First boot in a clean env downloads the MiniLM
    checkpoint (~90MB) which takes 10-30s — pytest's default timeout
    should not need extending for cached runs.

    The assertion shape mirrors the unit tests so regressions show up
    in the same form regardless of which tier ran.
    """
    from sidecar.app import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["neo4j"] == "ok"
    assert body["embedder"] == "all-MiniLM-L6-v2"
    assert body["embedding_dimensions"] == 384
    # Real Neo4j produces a non-None stats dict; we don't pin the shape.
    assert body["stats"] is not None
