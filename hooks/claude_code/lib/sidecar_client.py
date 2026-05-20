"""HTTP client for the memory sidecar on ``http://127.0.0.1:7575`` (Task 11).

Wraps the sidecar's REST surface (Task 8 :8: ``/api/{Label}``, Task 8
``/api/search/{Label}``, Task 10 ``/api/sessions/{id}/score``) with tight
timeouts plus graceful degradation. Every public method returns a sentinel
(``None`` / empty list) on failure instead of raising — the calling hook
no-ops cleanly and logs the failure to ``~/.galvo-memory/logs/`` via the
shared rotating logger.

Synchronous on purpose — Claude Code invokes lifecycle hooks synchronously
(shell-out + wait). Async would buy nothing and complicate the integration.

**Why stdlib ``urllib`` and not ``httpx``?** Hooks run in the user's shell
environment, which is not guaranteed to have the sidecar's ``[sidecar]``
extra (``httpx`` is pulled in by ``fastapi.testclient`` and ``uvicorn``,
not by the base ``galvo-memory`` install). ``urllib.request`` is stdlib
— zero deps. The ergonomics tradeoff (no Session, no JSON helper, manual
encoding) is acceptable for cycle 1; cycle 2 may swap to httpx once we've
decided whether to ship the hook scripts inside the sidecar wheel.

**Critical UX constraint:** hooks must not block the user's session.
The default 3.0s timeout caps every HTTP call. Combined with the
sidecar being on loopback (127.0.0.1:7575), a healthy hook adds <100ms;
a failed hook with the sidecar process gone adds at most ``timeout_s``.

All errors during the request lifecycle (connection refused, HTTP error
status, JSON decode error, socket timeout) are caught and converted to
the sentinel return value. Logging happens at WARNING via the module
logger — callers should attach a :func:`logging.handlers.RotatingFileHandler`
via :func:`hooks.claude_code.lib.logging.setup_hook_logger` before invoking
the client.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib import error as _urlerr
from urllib import request as _urlreq
from urllib.parse import urlencode

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "SidecarClient",
]

_log = logging.getLogger(__name__)

# --- Hard caps -------------------------------------------------------------
# Hooks run synchronously inside the user's Claude Code session. Every
# HTTP call must complete (or fail) in bounded wall-clock time, regardless
# of sidecar state. 3.0s is the cycle-1 target; tighter values produced
# false positives during local dev when the sidecar paged the embedder.

DEFAULT_TIMEOUT_S: float = 3.0
"""Maximum wall-clock seconds the client will wait for any single request."""

DEFAULT_BASE_URL: str = "http://127.0.0.1:7575"
"""Loopback by default — the sidecar binds locally for the cycle-1 deploy.

Operators running the sidecar in Docker on a non-loopback interface
override via ``SidecarClient(base_url=...)`` from the hook scripts.
"""


class SidecarClient:
    """Synchronous HTTP client for the memory sidecar.

    All public methods are non-raising: on failure they return ``None``
    (single-object endpoints) or ``[]`` (list endpoints), log a WARNING
    line, and let the calling hook decide whether to degrade or skip the
    operation entirely.

    Args:
        base_url: Root URL of the sidecar. Trailing slash is normalized
            away. Defaults to :data:`DEFAULT_BASE_URL`.
        timeout_s: Per-request wall-clock cap in seconds. Applies to
            both connection and read. Defaults to :data:`DEFAULT_TIMEOUT_S`.

    Example::

        from hooks.claude_code.lib.sidecar_client import SidecarClient
        client = SidecarClient()
        if client.health() is None:
            return  # sidecar down — silently no-op
        hits = client.search("Decision", "ruff config", scope="project:galvo")
        for hit in hits[:5]:
            print(f"- {hit['title']}")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        # Strip trailing slash so f-string composition with leading-slash
        # paths produces a clean URL without doubled separators.
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # --- High-level endpoint wrappers --------------------------------------

    def health(self) -> dict[str, Any] | None:
        """``GET /health`` — returns the response body or ``None`` on failure.

        The sidecar's healthy response is a dict with at minimum a
        ``status`` field; this method does NOT inspect the dict, it just
        returns whatever JSON came back. Callers who care about the
        actual health (vs the sidecar being reachable) should check the
        body's ``status == "ok"`` themselves.
        """
        result = self._get("/health")
        if not isinstance(result, dict):
            return None
        return result

    def search(
        self,
        label: str,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 5,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """``GET /api/search/{label}?q=...&scope=...&limit=N&threshold=T`` — returns hits or ``[]``.

        Args:
            label: One of the 12 ontology labels (Decision / Pattern /
                Belief / ...). The sidecar 404s on unknown labels; we
                return ``[]`` rather than surface the 404 because hooks
                shouldn't break the session over a typo.
            query: Free-text search string. The sidecar embeds it
                server-side and ranks against the vector index.
            scope: Optional scope filter passed through as the ``scope``
                query param. Per design §D4, the sidecar returns own-scope
                plus universal when this is set.
            limit: Max number of hits to return (sidecar caps at 50;
                we don't validate here, let the server reject if needed).
            threshold: Optional vector-similarity floor. Sidecar default
                is 0.7 (tight match). For top-of-mind use cases where the
                query is a generic descriptor like "recent decision
                rationale", pass a lower threshold (e.g. 0.3) so that
                weakly-matching but recent items still surface. Cycle 2
                will add a recency-first list endpoint that makes the
                threshold-tuning workaround unnecessary.

        Returns:
            A list of node dicts (shape per :class:`sidecar.models`
            response models) in similarity-rank order. Empty list on
            any failure — including ``404`` (unknown label),
            ``503`` (sidecar booting), connection refused (sidecar dead),
            or timeout.
        """
        params: dict[str, Any] = {"q": query, "limit": limit}
        if scope is not None:
            params["scope"] = scope
        if threshold is not None:
            params["threshold"] = threshold
        result = self._get(f"/api/search/{label}", params=params)
        if not isinstance(result, list):
            return []
        return result

    def create(
        self,
        label: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """``POST /api/{label}`` — create a node. Returns response dict or ``None``.

        Args:
            label: One of the 12 ontology labels. The sidecar validates
                and 404s on unknown labels.
            payload: JSON body — the per-label Create model shape from
                :mod:`sidecar.models`. Caller is responsible for the
                shape; we don't validate client-side because the model
                classes live on the sidecar side of the wire.

        Returns:
            The created node's response dict (shape per
            :class:`sidecar.models` response models), or ``None`` if the
            sidecar refused the request, was unreachable, or timed out.
        """
        return self._post(f"/api/{label}", json_body=payload)

    def score_session(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """``POST /api/sessions/{id}/score`` — Task-10 utility scorer.

        Args:
            session_id: The Session node id. Must equal ``payload["session_id"]``
                or the sidecar returns 400.
            payload: The :class:`sidecar.scoring.ScoringPayload` shape —
                session_id + outcome signal + retrieved-ids list + assistant
                text body.

        Returns:
            The :class:`sidecar.scoring.ScoringReport` dict, or ``None``
            on any failure. The hook (Task 15) treats ``None`` as
            "scoring skipped" and continues.
        """
        return self._post(
            f"/api/sessions/{session_id}/score", json_body=payload
        )

    # --- Low-level HTTP helpers --------------------------------------------

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Issue a GET. Returns parsed JSON, or ``None`` on any failure.

        Catches every realistic error mode:

        * :class:`urllib.error.URLError` — DNS failure, connection refused
          (sidecar not running), TLS error, etc.
        * :class:`urllib.error.HTTPError` (subclass of URLError) — 4xx/5xx
          status. We treat any non-2xx as failure because the sidecar's
          contract is "200 on success" for these endpoints; a 503 means
          "boot still in progress" and we want to no-op identically to
          a connection refused.
        * :class:`TimeoutError` — read or connect timeout.
        * :class:`json.JSONDecodeError` — non-JSON body (rare; would mean
          the sidecar served HTML, e.g. a reverse-proxy error page).
        * Generic :class:`OSError` — socket errors not wrapped by URLError.
        """
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        try:
            req = _urlreq.Request(url, method="GET")
            with _urlreq.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (
            _urlerr.URLError,
            TimeoutError,
            json.JSONDecodeError,
            OSError,
        ) as exc:
            # `urllib.error.HTTPError` is a subclass of URLError and is
            # caught above. We don't differentiate — hooks treat all
            # failures uniformly as "sidecar unavailable, no-op".
            _log.warning("sidecar GET %s failed: %r", path, exc)
            return None

    def _post(
        self,
        path: str,
        *,
        json_body: dict[str, Any],
    ) -> Any:
        """Issue a POST with a JSON body. Returns parsed JSON or ``None`` on failure.

        Mirror of :meth:`_get`. Same error envelope; same sentinel.
        Content-Type is set explicitly because urllib doesn't infer it
        from the data shape.
        """
        url = f"{self.base_url}{path}"
        try:
            data = json.dumps(json_body).encode("utf-8")
        except (TypeError, ValueError) as exc:
            # The caller passed something non-JSON-serializable. We log
            # but don't raise — the hook contract is "never raise".
            _log.warning("sidecar POST %s body not serializable: %r", path, exc)
            return None
        req = _urlreq.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with _urlreq.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (
            _urlerr.URLError,
            TimeoutError,
            json.JSONDecodeError,
            OSError,
        ) as exc:
            _log.warning("sidecar POST %s failed: %r", path, exc)
            return None
