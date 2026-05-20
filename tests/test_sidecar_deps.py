"""Task 7 — MemoryClient + SidecarSettings dependency injection tests.

The deps in :mod:`sidecar.deps` pull from ``app.state``, so a test wires
up a tiny ``FastAPI()`` instance with the lifespan-equivalent state
attached (or deliberately missing) and exercises the provider functions
through FastAPI's ``Depends`` machinery.

What we're protecting:

* The provider functions return whatever ``app.state.memory`` /
  ``app.state.config`` point at — a regression that hard-coded a
  singleton would silently ignore lifespan-attached objects.
* When ``app.state`` is missing the attribute, the provider raises a
  :class:`RuntimeError` with a hint that mentions ``lifespan`` and
  ``dependency_overrides`` — so a stack trace points at the fix.
* :attr:`FastAPI.dependency_overrides` replaces the provider in
  exactly the way Tasks 8/9/10 will use to inject fakes — the standard
  FastAPI pattern, but worth a regression test because the dependency
  is wired through an :class:`Annotated` alias and a refactor could
  break the override path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sidecar.config import SidecarSettings
from sidecar.deps import (
    MemoryDep,
    SettingsDep,
    get_memory_client,
    get_settings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_client() -> "Iterator[MagicMock]":
    """Same patch as test_sidecar_health — replace the lifespan's
    :class:`MemoryClient` with an :class:`AsyncMock`-backed fake.

    This fixture is used by tests that want to assert the dep returns
    the very client the lifespan attached. Tests that just want a
    standalone instance to attach manually can build their own
    ``MagicMock`` inline.
    """
    with patch("sidecar.app.MemoryClient") as MockClient:
        instance = MockClient.return_value
        instance.connect = AsyncMock(return_value=None)
        instance.close = AsyncMock(return_value=None)
        instance.get_stats = AsyncMock(return_value={"entities": 0, "facts": 0})
        yield instance


# ---------------------------------------------------------------------------
# Provider — happy-path tests.
# ---------------------------------------------------------------------------


def test_get_memory_client_returns_client(mocked_client: MagicMock) -> None:
    """When ``app.state.memory`` is set, the provider returns it.

    We boot the real ``sidecar.app:app`` lifespan (mocked client) and
    call the provider via a route that uses :data:`MemoryDep`. The
    response body confirms the client identity by exercising one of the
    mock's stubbed methods through the dep.
    """
    from sidecar.app import app

    # Add a route that simply echoes whether the injected client is the
    # one the lifespan attached. We can't compare objects in JSON, so
    # use the mock's `get_stats` return value as a proxy.
    @app.get("/_test_dep_identity")
    async def _identity_probe(memory: MemoryDep) -> dict:  # type: ignore[no-untyped-def]
        stats = await memory.get_stats()
        return {"stats": stats}

    try:
        with TestClient(app) as client:
            response = client.get("/_test_dep_identity")
        assert response.status_code == 200, response.text
        assert response.json() == {"stats": {"entities": 0, "facts": 0}}
    finally:
        # Remove the probe route so it doesn't leak across tests.
        app.routes[:] = [r for r in app.routes if getattr(r, "path", "") != "/_test_dep_identity"]


def test_get_settings_returns_sidecar_settings(mocked_client: MagicMock) -> None:
    """``get_settings`` returns the lifespan-attached
    :class:`SidecarSettings`.

    We assert it's the exact class (not a duck-typed dict) so refactors
    that swap the storage type fail loudly.
    """
    from sidecar.app import app

    @app.get("/_test_settings_identity")
    async def _settings_probe(cfg: SettingsDep) -> dict:  # type: ignore[no-untyped-def]
        return {
            "type": type(cfg).__name__,
            "embedder": cfg.embedding_model,
            "port": cfg.port,
        }

    try:
        with TestClient(app) as client:
            response = client.get("/_test_settings_identity")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["type"] == "SidecarSettings"
        assert body["embedder"] == "all-MiniLM-L6-v2"
        assert body["port"] == 7575
    finally:
        app.routes[:] = [
            r for r in app.routes if getattr(r, "path", "") != "/_test_settings_identity"
        ]


# ---------------------------------------------------------------------------
# Provider — failure-path tests.
# ---------------------------------------------------------------------------


def test_get_memory_client_raises_when_lifespan_not_run() -> None:
    """A bare ``FastAPI()`` with no lifespan raises :class:`RuntimeError`.

    The error message contains ``lifespan`` and ``dependency_overrides``
    so a developer reading the trace immediately sees the fix. We don't
    pin the full message, only the load-bearing keywords.
    """
    bare_app = FastAPI()  # no lifespan attached → no app.state.memory

    @bare_app.get("/_test_missing_dep")
    async def _missing(memory: MemoryDep) -> dict:  # type: ignore[no-untyped-def]
        return {"ok": True}

    with TestClient(bare_app) as client:
        # FastAPI wraps the dep RuntimeError into a 500; we still want
        # the error visible. Either path is acceptable as long as we
        # can confirm the dep raised.
        with pytest.raises(RuntimeError) as excinfo:
            client.get("/_test_missing_dep")
        msg = str(excinfo.value)
        assert "lifespan" in msg
        assert "dependency_overrides" in msg


def test_get_settings_raises_when_lifespan_not_run() -> None:
    """Mirror of the memory test: missing settings raises with the same
    hint structure.

    Both providers follow the same idiom, so a regression that fixed one
    but not the other would slip through if we only tested one.
    """
    bare_app = FastAPI()

    @bare_app.get("/_test_missing_settings")
    async def _missing(cfg: SettingsDep) -> dict:  # type: ignore[no-untyped-def]
        return {"ok": True}

    with TestClient(bare_app) as client:
        with pytest.raises(RuntimeError) as excinfo:
            client.get("/_test_missing_settings")
        msg = str(excinfo.value)
        assert "lifespan" in msg
        assert "dependency_overrides" in msg


def test_get_memory_client_called_directly_with_unconfigured_app() -> None:
    """Calling the provider against a bare :class:`Request`-stand-in
    without ``app.state.memory`` raises :class:`RuntimeError`.

    Sanity check: the provider doesn't rely on FastAPI's middleware to
    surface the error — it raises in the function body. A regression
    that silently returned ``None`` would let downstream routes get an
    ``AttributeError`` on the first method call.
    """
    bare_app = FastAPI()
    fake_request = MagicMock()
    fake_request.app = bare_app

    with pytest.raises(RuntimeError):
        get_memory_client(fake_request)

    with pytest.raises(RuntimeError):
        get_settings(fake_request)


# ---------------------------------------------------------------------------
# dependency_overrides — the canonical test-substitution path.
# ---------------------------------------------------------------------------


def test_dependency_override_for_tests(mocked_client: MagicMock) -> None:
    """``app.dependency_overrides[get_memory_client] = ...`` swaps the
    provider for the whole app.

    This is the documented FastAPI test pattern; we verify it works
    through the :data:`MemoryDep` Annotated alias (which wraps the
    provider in :class:`Depends`). A bug in the alias wiring would let
    the override "succeed" silently while routes kept seeing the real
    provider.
    """
    from sidecar.app import app

    # A separate stand-in client distinct from the lifespan's mock so
    # we can confirm the override took effect.
    override_client = MagicMock()
    override_client.get_stats = AsyncMock(return_value={"sentinel": True})

    @app.get("/_test_dep_override")
    async def _probe(memory: MemoryDep) -> dict:  # type: ignore[no-untyped-def]
        return await memory.get_stats()

    app.dependency_overrides[get_memory_client] = lambda: override_client
    try:
        with TestClient(app) as client:
            response = client.get("/_test_dep_override")
        assert response.status_code == 200, response.text
        assert response.json() == {"sentinel": True}
        override_client.get_stats.assert_awaited_once()
        # The lifespan's "real" mock client should NOT have been called.
        mocked_client.get_stats.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(get_memory_client, None)
        app.routes[:] = [r for r in app.routes if getattr(r, "path", "") != "/_test_dep_override"]


def test_settings_dependency_override(mocked_client: MagicMock) -> None:
    """Same override mechanism works for :func:`get_settings`.

    Use case: a test wants to verify route behavior with a non-default
    port / embedder config without re-running the lifespan.
    """
    from sidecar.app import app

    override_settings = SidecarSettings(
        host="127.0.0.1",
        port=9999,
        embedding_model="custom-model",
        embedding_dimensions=128,
    )

    @app.get("/_test_settings_override")
    async def _probe(cfg: SettingsDep) -> dict:  # type: ignore[no-untyped-def]
        return {"port": cfg.port, "embedder": cfg.embedding_model}

    app.dependency_overrides[get_settings] = lambda: override_settings
    try:
        with TestClient(app) as client:
            response = client.get("/_test_settings_override")
        assert response.status_code == 200, response.text
        assert response.json() == {"port": 9999, "embedder": "custom-model"}
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.routes[:] = [
            r for r in app.routes if getattr(r, "path", "") != "/_test_settings_override"
        ]


# ---------------------------------------------------------------------------
# Annotated alias sanity — what the route author actually types.
# ---------------------------------------------------------------------------


def test_memory_dep_alias_is_annotated_depends() -> None:
    """:data:`MemoryDep` is ``Annotated[MemoryClient, Depends(get_memory_client)]``.

    Routes inherit the dep wiring purely through Python's typing module
    — a regression that switched to a plain :class:`Depends` (no
    Annotated) would break new code that uses ``: MemoryDep`` because
    FastAPI only honors Annotated form for default-less parameters.
    """
    import typing

    # __origin__ + __metadata__ are the public Annotated introspection API.
    assert typing.get_origin(MemoryDep) is typing.Annotated, (
        "MemoryDep must use typing.Annotated"
    )
    metadata = typing.get_args(MemoryDep)[1:]
    assert any(
        getattr(item, "dependency", None) is get_memory_client for item in metadata
    ), f"MemoryDep metadata missing Depends(get_memory_client): {metadata!r}"


def test_settings_dep_alias_is_annotated_depends() -> None:
    """Mirror of the MemoryDep alias test for :data:`SettingsDep`."""
    import typing

    assert typing.get_origin(SettingsDep) is typing.Annotated
    metadata = typing.get_args(SettingsDep)[1:]
    assert any(getattr(item, "dependency", None) is get_settings for item in metadata)
