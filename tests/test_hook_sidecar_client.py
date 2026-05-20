"""Task 11 — hook framework + sidecar HTTP client unit tests.

What we're protecting:

* **Non-raising contract.** Every public :class:`SidecarClient` method
  returns a sentinel (``None`` / ``[]``) on failure instead of raising.
  This is the load-bearing UX guarantee for hooks running in the user's
  Claude Code session — a raised exception bubbles up through the
  lifecycle dispatcher and surfaces a traceback to the user.
* **HTTP semantics.** GETs hit ``/health`` and ``/api/search/{label}``
  with the expected query-param encoding; POSTs hit ``/api/{label}``
  and ``/api/sessions/{id}/score`` with JSON body + content-type
  header. Test via an in-process HTTP server that records what the
  client sent.
* **Timeout enforcement.** A slow server response past ``timeout_s``
  produces the sentinel, not a hang. This is the constraint that
  makes the hooks safe to wire into a real session.
* **Stdin parsing.** :class:`HookInputBase.from_stdin` is total —
  EOF, malformed JSON, wrong-type root, schema-mismatch all produce
  the ``"unknown"`` sentinel without raising.
* **Logger plumbing.** :func:`setup_hook_logger` writes to the right
  path under ``~/.galvo-memory/logs/``, doesn't pollute the root
  logger, and is idempotent across repeated calls.
* **Scope wrapper.** :func:`detect_scope_for_hook` honors an explicit
  cwd and falls back to :func:`Path.cwd` when given ``None``.

We use an in-process HTTP server (no fastapi, no httpx, no
TestClient) so the tests run anywhere and don't depend on the sidecar
extra being installed — matches the "hooks have zero deps" rationale
for picking stdlib :mod:`urllib` in :mod:`sidecar_client`.
"""

from __future__ import annotations

import http.server
import io
import json
import logging
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from hooks.claude_code.lib.logging import setup_hook_logger
from hooks.claude_code.lib.scope import detect_scope_for_hook
from hooks.claude_code.lib.sidecar_client import SidecarClient
from hooks.claude_code.lib.types import UNKNOWN_EVENT, HookInputBase


# ---------------------------------------------------------------------------
# In-process HTTP-server fixture.
# ---------------------------------------------------------------------------


@contextmanager
def mock_sidecar(
    handler_map: dict[str, tuple[int, Any]],
    *,
    delay_s: float = 0.0,
    record_requests: list[dict[str, Any]] | None = None,
):
    """Spin up a tiny HTTP server that returns canned responses.

    Args:
        handler_map: ``{"GET /health": (200, {...}), "POST /api/Decision": (201, {...})}``
            Maps ``"{METHOD} {PATH_WITHOUT_QUERY}"`` to the status code +
            JSON body to return. A missing key produces a 404 with no body.
        delay_s: Sleep this long before responding (used for the
            timeout test). Default 0 — no delay.
        record_requests: If provided, every served request gets a dict
            appended with the method, path, query, headers, and body.

    Yields:
        The base URL (``http://127.0.0.1:<port>``) the test should hand
        to :class:`SidecarClient`.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        # Silence default access-log spew which would clutter pytest output.
        def log_message(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def _record(self, body: bytes) -> None:
            if record_requests is None:
                return
            record_requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "query": dict(parse_qs(urlparse(self.path).query)),
                    "headers": dict(self.headers),
                    "body": body.decode("utf-8") if body else "",
                }
            )

        def _respond(self, body_bytes: bytes = b"") -> None:
            if delay_s:
                # Sleep BEFORE writing the response so the urllib timeout
                # path fires. Sleeping after would race the client's
                # read-side cancellation.
                time.sleep(delay_s)
            # Strip query string before lookup — handler_map keys are
            # method + path only.
            key = f"{self.command} {urlparse(self.path).path}"
            entry = handler_map.get(key)
            self._record(body_bytes)
            if entry is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status, body = entry
            raw = json.dumps(body).encode("utf-8") if body is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 — http.server convention
            self._respond()

        def do_POST(self) -> None:  # noqa: N802 — http.server convention
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            self._respond(body)

    # Find a free port (close immediately, race-free for localhost loopback).
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def closed_port_url():
    """Yield a URL to a port that nothing is listening on.

    The kernel will refuse the connection immediately — used to test
    the "sidecar dead" code path without waiting for a timeout.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# SidecarClient.health — happy + degradation
# ---------------------------------------------------------------------------


def test_health_returns_dict() -> None:
    """Mock returns 200 + dict → client returns the dict verbatim."""
    body = {"status": "ok", "neo4j": "ok", "embedder": "all-MiniLM-L6-v2"}
    with mock_sidecar({"GET /health": (200, body)}) as base_url:
        client = SidecarClient(base_url=base_url)
        assert client.health() == body


def test_health_returns_none_on_503() -> None:
    """5xx response → ``None`` (sidecar reports an internal issue → degrade)."""
    with mock_sidecar({"GET /health": (503, {"detail": "unhealthy"})}) as base_url:
        client = SidecarClient(base_url=base_url)
        assert client.health() is None


def test_health_returns_none_on_connection_refused() -> None:
    """Nothing listening → kernel refuses TCP → client returns ``None``.

    This is the most common failure mode in cycle 1: operator forgot
    to ``docker compose up`` the sidecar. The hook must silently no-op.
    """
    with closed_port_url() as base_url:
        client = SidecarClient(base_url=base_url, timeout_s=2.0)
        assert client.health() is None


# ---------------------------------------------------------------------------
# SidecarClient.search — list-shape contract + scope filter wiring
# ---------------------------------------------------------------------------


def test_search_returns_list() -> None:
    """Mock returns a JSON array → client returns the list."""
    hits = [{"id": "dec_001", "title": "Use ruff"}, {"id": "dec_002", "title": "MiniLM"}]
    with mock_sidecar({"GET /api/search/Decision": (200, hits)}) as base_url:
        client = SidecarClient(base_url=base_url)
        result = client.search("Decision", "ruff config")
        assert result == hits


def test_search_returns_empty_list_on_failure() -> None:
    """500 response → empty list (the search-endpoint contract).

    Empty list is semantically equivalent to "no hits" — the hook
    layer treats it identically to a successful empty result, which
    is correct: a failed search should not surface as different from
    "no relevant memory".
    """
    with mock_sidecar({"GET /api/search/Decision": (500, None)}) as base_url:
        client = SidecarClient(base_url=base_url)
        assert client.search("Decision", "q") == []


def test_search_passes_scope_query_param() -> None:
    """The ``scope=`` kwarg ends up in the URL query string.

    Crucial for design §D4: without this, the sidecar would default to
    cross-scope and the hook would leak personal memories into a
    project session (or vice versa).
    """
    recorded: list[dict[str, Any]] = []
    with mock_sidecar(
        {"GET /api/search/Decision": (200, [])},
        record_requests=recorded,
    ) as base_url:
        client = SidecarClient(base_url=base_url)
        client.search("Decision", "ruff", scope="project:galvo", limit=3)
    assert len(recorded) == 1
    qs = recorded[0]["query"]
    assert qs["q"] == ["ruff"]
    assert qs["scope"] == ["project:galvo"]
    assert qs["limit"] == ["3"]


def test_search_omits_scope_when_none() -> None:
    """``scope=None`` (default) → no ``scope=`` in the query string.

    Sidecar treats absent scope as "cross-scope query" — this exists
    for an admin path but should be rare.
    """
    recorded: list[dict[str, Any]] = []
    with mock_sidecar(
        {"GET /api/search/Decision": (200, [])},
        record_requests=recorded,
    ) as base_url:
        client = SidecarClient(base_url=base_url)
        client.search("Decision", "ruff")
    qs = recorded[0]["query"]
    assert "scope" not in qs


# ---------------------------------------------------------------------------
# SidecarClient.create — POST shape + body
# ---------------------------------------------------------------------------


def test_create_returns_dict_on_201() -> None:
    """A successful POST returns the response body dict verbatim."""
    response_body = {"id": "dec_001", "title": "Use ruff", "scope": "project:galvo"}
    with mock_sidecar({"POST /api/Decision": (201, response_body)}) as base_url:
        client = SidecarClient(base_url=base_url)
        result = client.create("Decision", {"title": "Use ruff"})
        assert result == response_body


def test_create_returns_none_on_failure() -> None:
    """500 → None. The hook treats None as "create failed, move on"."""
    with mock_sidecar({"POST /api/Decision": (500, {"detail": "db error"})}) as base_url:
        client = SidecarClient(base_url=base_url)
        result = client.create("Decision", {"title": "x"})
        assert result is None


def test_create_sets_json_content_type_and_body() -> None:
    """The POST body is JSON-encoded and the content-type header is set.

    Protects the contract with FastAPI: missing Content-Type would
    surface as a 422 because the body parser falls back to form data.
    """
    recorded: list[dict[str, Any]] = []
    with mock_sidecar(
        {"POST /api/Decision": (201, {"id": "dec_001"})},
        record_requests=recorded,
    ) as base_url:
        client = SidecarClient(base_url=base_url)
        client.create("Decision", {"title": "x", "rationale": "y"})
    assert len(recorded) == 1
    assert recorded[0]["headers"].get("Content-Type") == "application/json"
    parsed = json.loads(recorded[0]["body"])
    assert parsed == {"title": "x", "rationale": "y"}


def test_create_returns_none_on_unserializable_body() -> None:
    """A non-JSON-serializable body returns None instead of raising.

    Defensive: the hook layer shouldn't crash because someone passed a
    ``set()`` or a callable.
    """
    # No server needed — failure happens before any network call.
    client = SidecarClient(base_url="http://127.0.0.1:1")
    result = client.create("Decision", {"thing": {1, 2, 3}})  # set isn't JSON
    assert result is None


# ---------------------------------------------------------------------------
# SidecarClient.score_session
# ---------------------------------------------------------------------------


def test_score_session_returns_dict() -> None:
    """Happy path: 200 + ScoringReport-shaped dict → returned verbatim."""
    report = {"session_id": "sess_abc", "scored_edges": 5, "errors": []}
    with mock_sidecar({"POST /api/sessions/sess_abc/score": (200, report)}) as base_url:
        client = SidecarClient(base_url=base_url)
        result = client.score_session("sess_abc", {"session_id": "sess_abc"})
        assert result == report


def test_score_session_returns_none_on_400() -> None:
    """400 (path/body session_id mismatch) → ``None``.

    The Task-15 hook builds its body from the same session id it puts
    in the path, so this should never fire in practice — but we test
    it so a regression at the call site degrades safely.
    """
    with mock_sidecar(
        {"POST /api/sessions/sess_abc/score": (400, {"detail": "mismatch"})}
    ) as base_url:
        client = SidecarClient(base_url=base_url)
        result = client.score_session("sess_abc", {"session_id": "other"})
        assert result is None


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


def test_timeout_returns_sentinel() -> None:
    """Server delays response past ``timeout_s`` → client returns ``None``.

    The whole point of the hook timeout is the user's session can't
    block on a wedged sidecar. We use a 0.5s timeout and a 2.0s delay
    so the test runs quickly but the timeout path is unambiguously
    exercised.
    """
    with mock_sidecar(
        {"GET /health": (200, {"status": "ok"})},
        delay_s=2.0,
    ) as base_url:
        client = SidecarClient(base_url=base_url, timeout_s=0.5)
        t0 = time.monotonic()
        result = client.health()
        elapsed = time.monotonic() - t0
    assert result is None
    # Sanity check: we returned because of the timeout, not because the
    # response actually came back. 0.5s + small overhead, definitely
    # less than the 2.0s server-side delay.
    assert elapsed < 1.5, f"timeout took too long: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Base-URL normalization
# ---------------------------------------------------------------------------


def test_base_url_trailing_slash_stripped() -> None:
    """``SidecarClient(base_url="http://x/")`` strips the trailing slash.

    Without this, ``f"{base_url}{path}"`` would yield ``http://x//health``
    and some HTTP servers reject the double slash.
    """
    client = SidecarClient(base_url="http://127.0.0.1:7575/")
    assert client.base_url == "http://127.0.0.1:7575"


# ---------------------------------------------------------------------------
# HookInputBase.from_stdin
# ---------------------------------------------------------------------------


def test_hook_input_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid JSON on stdin parses into a typed model with all fields."""
    raw = json.dumps(
        {
            "hook_event_name": "SessionStart",
            "session_id": "sess_abc",
            "cwd": "/Volumes/Main External/Development/lucidity/Galvo",
            "transcript_path": "/tmp/transcript.json",
            "extra": {"foo": "bar"},
        }
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    parsed = HookInputBase.from_stdin()
    assert parsed.hook_event_name == "SessionStart"
    assert parsed.session_id == "sess_abc"
    assert parsed.cwd == "/Volumes/Main External/Development/lucidity/Galvo"
    assert parsed.transcript_path == "/tmp/transcript.json"
    assert parsed.extra == {"foo": "bar"}


def test_hook_input_malformed_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed JSON on stdin → ``hook_event_name == 'unknown'``, no exception."""
    monkeypatch.setattr("sys.stdin", io.StringIO("{this is not json}"))
    parsed = HookInputBase.from_stdin()
    assert parsed.hook_event_name == UNKNOWN_EVENT


def test_hook_input_empty_stdin_returns_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty stdin → ``"unknown"`` sentinel.

    Claude Code can emit empty stdin for events the hook doesn't
    receive payload for (rare; defensive coverage).
    """
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    parsed = HookInputBase.from_stdin()
    assert parsed.hook_event_name == UNKNOWN_EVENT


def test_hook_input_non_dict_root_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON that's not an object (list, string, number, null) → ``"unknown"``."""
    monkeypatch.setattr("sys.stdin", io.StringIO("[1, 2, 3]"))
    parsed = HookInputBase.from_stdin()
    assert parsed.hook_event_name == UNKNOWN_EVENT


def test_hook_input_extra_fields_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown top-level fields don't fail validation — Pydantic's ``extra='allow'``.

    Claude Code may add fields between versions; the hook shouldn't
    break the moment they do.
    """
    raw = json.dumps(
        {
            "hook_event_name": "SessionStart",
            "unexpected_new_field": "future-proof",
        }
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    parsed = HookInputBase.from_stdin()
    assert parsed.hook_event_name == "SessionStart"


def test_hook_input_missing_required_field_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON without ``hook_event_name`` fails validation → ``"unknown"``."""
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id": "x"}'))
    parsed = HookInputBase.from_stdin()
    assert parsed.hook_event_name == UNKNOWN_EVENT


# ---------------------------------------------------------------------------
# Logger plumbing
# ---------------------------------------------------------------------------


def test_logger_writes_to_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``setup_hook_logger("foo")`` writes to ``~/.galvo-memory/logs/foo.log``.

    We monkeypatch the module-level ``LOG_DIR`` to ``tmp_path`` and
    also point ``$HOME`` at ``tmp_path`` so the function's
    ``Path.home() / ".galvo-memory" / "logs"`` resolution lands inside
    the test sandbox. ``LOG_DIR`` was bound at module-import time,
    so we patch both the bound constant and the env to be safe.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".galvo-memory" / "logs"
    monkeypatch.setattr("hooks.claude_code.lib.logging.LOG_DIR", log_dir)

    logger = setup_hook_logger("test_logger_unit")
    logger.info("hello world")

    # Flush all handlers so the file content is visible to the assertion.
    for handler in logger.handlers:
        handler.flush()

    log_file = log_dir / "test_logger_unit.log"
    assert log_file.exists(), f"expected {log_file} to exist"
    content = log_file.read_text(encoding="utf-8")
    assert "hello world" in content
    assert "INFO" in content


def test_logger_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling :func:`setup_hook_logger` twice with the same name does NOT
    double-attach handlers — important for hook scripts imported during
    test collection (pytest may import the module multiple times)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".galvo-memory" / "logs"
    monkeypatch.setattr("hooks.claude_code.lib.logging.LOG_DIR", log_dir)

    logger_a = setup_hook_logger("test_idempotent")
    handler_count_a = len(logger_a.handlers)
    logger_b = setup_hook_logger("test_idempotent")
    handler_count_b = len(logger_b.handlers)

    assert logger_a is logger_b, "setup should return the same logger instance"
    assert handler_count_a == handler_count_b == 1, (
        f"expected exactly 1 handler, got {handler_count_a} -> {handler_count_b}"
    )


def test_logger_does_not_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook logger's propagation is disabled — log records don't
    bubble to the root logger (where they could land on stderr and
    pollute the Claude Code session transcript)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".galvo-memory" / "logs"
    monkeypatch.setattr("hooks.claude_code.lib.logging.LOG_DIR", log_dir)

    logger = setup_hook_logger("test_propagate")
    assert logger.propagate is False


# ---------------------------------------------------------------------------
# Scope wrapper
# ---------------------------------------------------------------------------


def test_scope_detection_from_hook_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """:func:`detect_scope_for_hook` with an explicit cwd walks up to find
    the marker.

    Mirrors the canonical ``scope/detector.py`` behavior, but via the
    hook-specific entry point that the four lifecycle hooks call.
    """
    from scope import write_marker

    write_marker(tmp_path, project_id="galvo", name="Galvo FACT")
    nested = tmp_path / "subdir" / "deep"
    nested.mkdir(parents=True)

    scope_string = detect_scope_for_hook(str(nested))
    assert scope_string == "project:galvo"


def test_scope_detection_with_none_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cwd=None`` falls back to :func:`Path.cwd` — defensive path.

    Realistic only if Claude Code's hook input omits cwd (rare). We
    test with the process cwd set to a directory under a fake ``$HOME``
    where no marker exists → result should be ``"personal"``.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    sub = tmp_path / "no_marker_here"
    sub.mkdir()
    monkeypatch.chdir(sub)

    scope_string = detect_scope_for_hook(None)
    assert scope_string == "personal"


# ---------------------------------------------------------------------------
# Sentinel / type-shape edge cases
# ---------------------------------------------------------------------------


def test_search_returns_empty_list_on_non_list_body() -> None:
    """If the sidecar returns a 200 + dict (wrong shape), we treat it
    as a failure and return ``[]`` rather than propagating the wrong
    type to the hook script.

    Belt-and-braces — protects against a server-side schema change
    that would otherwise cause a TypeError in the hook.
    """
    with mock_sidecar(
        {"GET /api/search/Decision": (200, {"unexpected": "shape"})}
    ) as base_url:
        client = SidecarClient(base_url=base_url)
        assert client.search("Decision", "q") == []


def test_health_returns_none_on_non_dict_body() -> None:
    """Health returning a JSON array (impossible per contract, but…) → ``None``.

    Same belt-and-braces argument as the search version above.
    """
    with mock_sidecar({"GET /health": (200, ["wrong", "shape"])}) as base_url:
        client = SidecarClient(base_url=base_url)
        assert client.health() is None


def test_warning_logged_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed request emits a WARNING line via the module logger.

    The operator's primary debugging surface is
    ``~/.galvo-memory/logs/*.log``; the WARNING level is what bridges
    the module logger to the rotating file handler the hook scripts
    set up via :func:`setup_hook_logger`.
    """
    with closed_port_url() as base_url:
        client = SidecarClient(base_url=base_url, timeout_s=1.0)
        with caplog.at_level(
            logging.WARNING,
            logger="hooks.claude_code.lib.sidecar_client",
        ):
            result = client.health()
    assert result is None
    # The exact message format is intentionally not pinned (operators
    # grep by substring); we just check the URL fragment is mentioned.
    assert any("/health" in rec.message for rec in caplog.records)
