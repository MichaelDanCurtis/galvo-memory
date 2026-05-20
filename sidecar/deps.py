"""FastAPI dependencies — surfaces the lifespan's MemoryClient + settings.

Every endpoint that touches Neo4j or needs the active embedder config
declares the dependency via the type alias at the bottom of this module::

    from sidecar.deps import MemoryDep, SettingsDep

    @router.get("/foo")
    async def foo(memory: MemoryDep, cfg: SettingsDep) -> ...:
        ...

The :class:`Annotated` aliases keep the Depends boilerplate out of every
route signature — fewer characters to typo, easier to grep for callers.

Why pull from ``request.app.state`` rather than a module-level singleton?
:mod:`sidecar.app` attaches the client + settings inside the FastAPI
lifespan context. The lifespan runs per-app-instance, so a test that
spins up a fresh :class:`fastapi.FastAPI` (or :class:`TestClient`) gets
a fresh client. A module-level singleton would leak state between tests
and make :attr:`fastapi.FastAPI.dependency_overrides` harder to use.

Test substitution:

* Replace the dependency for the whole app via
  :attr:`app.dependency_overrides`::

      app.dependency_overrides[get_memory_client] = lambda: fake_client

  This is the canonical FastAPI pattern; tests that need
  per-test-method overrides should reset ``dependency_overrides``
  in a fixture teardown.

* The :class:`RuntimeError` raised when ``app.state.memory`` is missing
  is intentional: it fires the moment a developer wires a router into
  an app that has no lifespan, which is a setup bug, not a runtime one.
  Letting it propagate keeps the failure loud + early.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from sidecar.config import SidecarSettings

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]


__all__ = [
    "MemoryDep",
    "SettingsDep",
    "get_memory_client",
    "get_settings",
]


# ---------------------------------------------------------------------------
# Provider functions — these are what FastAPI's Depends() resolves to.
# ---------------------------------------------------------------------------


def get_memory_client(request: Request) -> "MemoryClient":
    """Return the :class:`MemoryClient` attached by the app's lifespan.

    Args:
        request: Injected by FastAPI; we only use it to reach
            ``request.app.state.memory``.

    Returns:
        The active :class:`MemoryClient`. Same instance for every
        request in the app's lifetime — the client maintains its own
        Neo4j connection pool internally, so per-request creation
        would defeat the pool.

    Raises:
        RuntimeError: When ``app.state.memory`` is missing — typically
            because the FastAPI app was created without the lifespan
            (e.g. by manually instantiating ``FastAPI()`` in a test
            and wiring routers in without invoking
            :func:`sidecar.app.lifespan`). The fix is to use
            ``app.dependency_overrides[get_memory_client]`` or to
            instantiate the app from :mod:`sidecar.app` so its
            lifespan registers.
    """
    client = getattr(request.app.state, "memory", None)
    if client is None:
        raise RuntimeError(
            "MemoryClient not attached to app.state — did lifespan run? "
            "Either use the sidecar.app:app instance or override "
            "get_memory_client via app.dependency_overrides."
        )
    return client


def get_settings(request: Request) -> SidecarSettings:
    """Return the :class:`SidecarSettings` snapshot taken at lifespan boot.

    The settings are captured once at boot rather than re-read per request
    so reproducing a failed health probe against the same config is
    straightforward — env var churn during a long-running process does
    not silently change the runtime.

    Raises:
        RuntimeError: Same conditions as :func:`get_memory_client`.
    """
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        raise RuntimeError(
            "SidecarSettings not attached to app.state — did lifespan run? "
            "Either use the sidecar.app:app instance or override "
            "get_settings via app.dependency_overrides."
        )
    return cfg


# ---------------------------------------------------------------------------
# Annotated type aliases — what route signatures should use.
# ---------------------------------------------------------------------------

# Imported as type-only so callers don't need to depend on the runtime
# import path of MemoryClient just to type a route. Lazy-resolves at
# request time via FastAPI's Depends machinery.
MemoryDep = Annotated["MemoryClient", Depends(get_memory_client)]
"""Type alias for routes: ``memory: MemoryDep`` injects the client."""

SettingsDep = Annotated[SidecarSettings, Depends(get_settings)]
"""Type alias for routes: ``cfg: SettingsDep`` injects the settings."""
