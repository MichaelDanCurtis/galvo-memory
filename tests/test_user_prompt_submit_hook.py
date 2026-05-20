"""Task 13 — UserPromptSubmit hook unit tests.

What we're protecting:

* **Stdin contract.** The hook reads JSON from stdin via
  :meth:`HookInputBase.from_stdin`. Empty / malformed / unknown-event
  stdin all produce a clean ``exit 0`` with no output and no sidecar
  call. This is the "never break the session" guarantee — a raised
  exception or non-zero exit would surface a banner to the user.
* **Sidecar query shape.** The hook calls ``GET /api/search/Entity``
  with ``q={prompt[:500]}``, ``scope={detected}``, ``limit=5``. The
  ``Entity`` super-label is the load-bearing choice — it surfaces hits
  across all 12 custom labels in one round-trip, avoiding 12-call
  fan-out.
* **Context-block format.** When the sidecar returns N hits the hook
  writes a markdown block (header + N bullets) to stdout. Empty hits
  produce no output. The label prefix on each bullet uses the most
  specific known custom label, falling back to ``type``, and finally
  to ``"Entity"``.
* **Truncation.** Prompts longer than :data:`MAX_QUERY_CHARS` are
  truncated before being sent to the sidecar — protects the embedder's
  context window.

We use the same in-process HTTP-server pattern as
``test_hook_sidecar_client.py`` so the tests run anywhere without the
sidecar's ``[sidecar]`` extra. The hook script is exercised by
monkeypatching ``SidecarClient.__init__`` to point at the mock server's
URL (the hook constructs its own client; we can't inject one). This is
the minimum-surface-area mock that still goes through real urllib.
"""

from __future__ import annotations

import http.server
import io
import json
import socket
import threading
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from hooks.claude_code import user_prompt_submit
from hooks.claude_code.lib.sidecar_client import SidecarClient


# ---------------------------------------------------------------------------
# Mock-server fixture (copied verbatim from test_hook_sidecar_client.py — kept
# local rather than shared so test files remain independent; if a third hook
# test wants the same fixture we'll move it to conftest.py at that point).
# ---------------------------------------------------------------------------


@contextmanager
def mock_sidecar(
    handler_map: dict[str, tuple[int, Any]],
    *,
    record_requests: list[dict[str, Any]] | None = None,
):
    """Spin up an in-process HTTP server that returns canned JSON responses."""

    class Handler(http.server.BaseHTTPRequestHandler):
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


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _set_stdin(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | str) -> None:
    """Replace ``sys.stdin`` with an in-memory StringIO carrying ``payload``."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))


@contextmanager
def _redirect_client_base_url(base_url: str):
    """Force the hook's :class:`SidecarClient` instances to use ``base_url``.

    The hook constructs its own ``SidecarClient()`` (no DI), and the
    ``DEFAULT_BASE_URL`` constant is bound as the default arg on
    :class:`SidecarClient.__init__` at class-definition time — so
    overriding the module-level constant after import has no effect.

    Instead we replace the ``SidecarClient`` symbol the hook module
    captured at its own import time with a subclass whose default
    ``base_url`` points at ``base_url``. This is functionally identical
    to monkeypatching ``__init__`` but expressed via the subclass
    pattern, which is easier to reason about.
    """

    class _RedirectedClient(SidecarClient):
        def __init__(
            self,
            base_url_: str = base_url,
            timeout_s: float = 3.0,
        ) -> None:
            super().__init__(base_url=base_url_, timeout_s=timeout_s)

    original = user_prompt_submit.SidecarClient
    user_prompt_submit.SidecarClient = _RedirectedClient  # type: ignore[misc]
    try:
        yield
    finally:
        user_prompt_submit.SidecarClient = original  # type: ignore[misc]


def _silence_log(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the hook's log file at ``tmp_path`` instead of ``$HOME``.

    Without this, every test would write to the real
    ``~/.galvo-memory/logs/user_prompt_submit.log`` — fine but messy.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".galvo-memory" / "logs"
    monkeypatch.setattr("hooks.claude_code.lib.logging.LOG_DIR", log_dir)


# ---------------------------------------------------------------------------
# Stdin / no-op paths.
# ---------------------------------------------------------------------------


def test_empty_stdin_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Empty stdin → ``HookInputBase.from_stdin`` returns the ``unknown``
    sentinel → hook exits 0 with no stdout and no sidecar call."""
    _silence_log(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    rc = user_prompt_submit.main()

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_no_prompt_no_search(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """``extra`` lacks a ``prompt`` key → hook exits 0 without calling the
    sidecar. We assert this by pointing the client at a server that would
    404 any request and verifying no exception bubbles up + no stdout."""
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {},  # no prompt key
        },
    )

    with mock_sidecar({}) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_empty_prompt_string_no_search(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Whitespace-only prompt → hook treats it as "no prompt"; no output.

    Important for the edge case where Claude Code sends an empty event
    (e.g. user hit return on an empty input).
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "   \n\t  "},
        },
    )

    with mock_sidecar({}) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_empty_hits_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Sidecar returns ``[]`` → hook writes nothing to stdout.

    Empty hits means the retrieval found no relevant memory; injecting
    an empty header would pollute the model's context. Best to no-op.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "what is ruff?"},
        },
    )

    with mock_sidecar(
        {"GET /api/search/Entity": (200, [])}
    ) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_sidecar_down_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Sidecar unreachable → ``SidecarClient.search`` returns ``[]`` → no
    output. This is the most-common production failure mode (operator
    forgot ``docker compose up``), and it must silently no-op."""
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "what is ruff?"},
        },
    )

    # Bind a port + close it so connection-refused fires immediately.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    with _redirect_client_base_url(f"http://127.0.0.1:{port}"):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Happy path: hits → context block.
# ---------------------------------------------------------------------------


def test_happy_path_formats_hits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """3 hits → 4 stdout lines (header + 3 bullets), label-prefixed bullets.

    Spot-checks the markdown shape the model will see. The header echoes
    the (truncated) query and the count; each bullet is
    ``- [Label] Title``.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "ruff config"},
        },
    )

    hits = [
        {"id": "dec_001", "title": "Use ruff", "type": "CONCEPT", "labels": ["Entity", "Decision"]},
        {"id": "pat_002", "name": "Wave-converge merge", "type": "CONCEPT", "labels": ["Entity", "Pattern"]},
        {"id": "bel_003", "claim": "ScoreRow is nested", "type": "FACT", "labels": ["Entity", "Belief"]},
    ]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    out = capsys.readouterr().out
    lines = out.strip().split("\n")
    assert len(lines) == 4, f"expected header + 3 bullets, got: {out!r}"
    assert lines[0].startswith("# Memory: 3 relevant hits for")
    assert "ruff config" in lines[0]
    assert lines[1] == "- [Decision] Use ruff"
    assert lines[2] == "- [Pattern] Wave-converge merge"
    assert lines[3] == "- [Belief] ScoreRow is nested"


def test_prompt_truncated_to_500_chars(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """A 1000-char prompt → the sidecar's ``q`` query param is exactly 500.

    Protects the embedder's context window: MiniLM L6 v2 caps at 512
    tokens (~3000 chars) but we cap at 500 to leave slack. The test
    records the exact ``q`` value the sidecar received.
    """
    _silence_log(monkeypatch, tmp_path)
    long_prompt = "abcdefghij" * 100  # 1000 chars
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": long_prompt},
        },
    )

    recorded: list[dict[str, Any]] = []
    with mock_sidecar(
        {"GET /api/search/Entity": (200, [])},
        record_requests=recorded,
    ) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert len(recorded) == 1
    q_values = recorded[0]["query"]["q"]
    assert len(q_values) == 1
    assert len(q_values[0]) == user_prompt_submit.MAX_QUERY_CHARS == 500
    assert q_values[0] == long_prompt[: user_prompt_submit.MAX_QUERY_CHARS]


def test_scope_from_cwd_project(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """A cwd inside a marker-tagged project → sidecar receives
    ``scope=project:<id>``. Verifies the scope-detection wiring.
    """
    _silence_log(monkeypatch, tmp_path)
    # Drop a scope marker so detect_scope_for_hook resolves to project:demo.
    from scope import write_marker

    write_marker(tmp_path, project_id="demo", name="Demo Project")
    nested = tmp_path / "subdir"
    nested.mkdir()

    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(nested),
            "extra": {"prompt": "anything"},
        },
    )

    recorded: list[dict[str, Any]] = []
    with mock_sidecar(
        {"GET /api/search/Entity": (200, [])},
        record_requests=recorded,
    ) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert len(recorded) == 1
    assert recorded[0]["query"]["scope"] == ["project:demo"]


def test_search_request_uses_limit_5(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """The sidecar is asked for at most :data:`MAX_HITS` (5) hits.

    Cycle-1 chose 5 — anything more pollutes the context window.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )

    recorded: list[dict[str, Any]] = []
    with mock_sidecar(
        {"GET /api/search/Entity": (200, [])},
        record_requests=recorded,
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    assert recorded[0]["query"]["limit"] == [str(user_prompt_submit.MAX_HITS)]


# ---------------------------------------------------------------------------
# Label extraction — labels list / type fallback / generic fallback.
# ---------------------------------------------------------------------------


def test_label_extraction_from_labels_list(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """``labels=["Entity", "Decision"]`` → bullet shows ``[Decision]``.

    The most-specific known custom label wins over ``Entity``.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [
        {"id": "x", "title": "T", "type": "CONCEPT", "labels": ["Entity", "Decision"]},
    ]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    assert "- [Decision] T" in out


def test_label_extraction_extra_labels_alias(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Alternate key ``extra_labels`` (used by some Task-8 responses) → works.

    Defensive against the sidecar shape evolving — we accept either
    ``labels`` or ``extra_labels`` to surface the custom tag.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [
        {"id": "x", "name": "Wave merge", "type": "CONCEPT", "extra_labels": ["Pattern"]},
    ]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    assert "- [Pattern] Wave merge" in out


def test_label_fallback_to_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """No ``labels`` in hit → fallback uses the ``type`` discriminator.

    Task 8's current response shape (per ``sidecar.routers.nodes.
    _row_to_dict``) does NOT include the multi-label tag — only the
    node's properties. The library-side EntityType lives in ``type``
    (CONCEPT/EVENT/OBJECT/FACT). Falling back to that gives the model
    SOMETHING actionable.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [{"id": "x", "title": "Some node", "type": "CONCEPT"}]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    assert "- [CONCEPT] Some node" in out


def test_label_fallback_to_entity_when_no_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Hit with neither labels nor type → bullet shows ``[Entity]``.

    Last-resort fallback so the bullet never crashes the formatter.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [{"id": "node_xyz", "title": "Bare node"}]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    assert "- [Entity] Bare node" in out


# ---------------------------------------------------------------------------
# Title extraction — covers the per-label name-property variation.
# ---------------------------------------------------------------------------


def test_title_priority_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """When a hit has multiple name candidates, ``title`` wins.

    Per :data:`memory.ontology.label_mapping.NAME_PROPERTY_PER_LABEL`,
    the canonical name property varies by label. The hook probes in
    priority order; ``title`` (used by Decision/Task/Session) is first
    so most hits get the most-natural label.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [
        {
            "id": "x",
            "title": "Title wins",
            "name": "Name loses",
            "claim": "Claim loses",
            "type": "CONCEPT",
        }
    ]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    assert "- [CONCEPT] Title wins" in capsys.readouterr().out


def test_title_falls_back_to_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Hit with no recognized name field → bullet uses the node id.

    The bullet must never be blank — better to show an opaque id than
    a content-less line.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [{"id": "node_xyz", "type": "OBJECT"}]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    assert "- [OBJECT] node_xyz" in capsys.readouterr().out


def test_title_strips_to_first_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Multi-line titles → only the first line lands in the bullet.

    Prevents a single hit from spilling across multiple bullet lines —
    that would corrupt the markdown block's structure.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [{"id": "x", "title": "First line\nSecond line\nThird", "type": "CONCEPT"}]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    assert "- [CONCEPT] First line" in out
    assert "Second line" not in out


# ---------------------------------------------------------------------------
# MAX_HITS cap.
# ---------------------------------------------------------------------------


def test_caps_at_max_hits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """If the sidecar returns more than ``MAX_HITS``, only the first 5 surface.

    Defense-in-depth: the client passes ``limit=5`` but the sidecar's
    behavior on a buggy ``limit`` deserves an extra guardrail.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    hits = [
        {"id": f"n_{i}", "title": f"Title {i}", "type": "CONCEPT"}
        for i in range(10)
    ]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    bullet_lines = [line for line in out.splitlines() if line.startswith("- ")]
    assert len(bullet_lines) == user_prompt_submit.MAX_HITS == 5
    # Header reports the capped count, not the raw incoming count.
    assert out.splitlines()[0].startswith("# Memory: 5 relevant hits")


# ---------------------------------------------------------------------------
# Defensive: non-string prompt payload.
# ---------------------------------------------------------------------------


def test_non_string_prompt_payload_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """If Claude Code's ``extra.prompt`` is a non-string (defensive — schema
    drift), the hook silently no-ops. We don't want a TypeError to bubble
    when the model would happily accept a missing memory block."""
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": ["not", "a", "string"]},
        },
    )
    with mock_sidecar({}) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Header excerpt — quoting + newline handling.
# ---------------------------------------------------------------------------


def test_header_excerpt_truncates_long_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Header excerpt caps at 80 chars + ellipsis when the prompt is longer.

    Keeps the header on one line; the model can self-correct if the
    excerpt looks wrong.
    """
    _silence_log(monkeypatch, tmp_path)
    long_prompt = "x" * 200
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": long_prompt},
        },
    )
    hits = [{"id": "n", "title": "T", "type": "CONCEPT"}]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    out = capsys.readouterr().out
    header = out.splitlines()[0]
    # The excerpt has 80 'x' + the ellipsis; the rest of the header carries
    # the count and surrounding quotes.
    assert "x" * 80 + "…" in header


def test_header_excerpt_replaces_newlines(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Newlines in the prompt → spaces in the header excerpt.

    Without this, a multi-line prompt would break the single-line
    header convention and the model would see a malformed block.
    """
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "first line\nsecond line"},
        },
    )
    hits = [{"id": "n", "title": "T", "type": "CONCEPT"}]
    with mock_sidecar(
        {"GET /api/search/Entity": (200, hits)}
    ) as base_url, _redirect_client_base_url(base_url):
        user_prompt_submit.main()

    header = capsys.readouterr().out.splitlines()[0]
    # The whole header is a single line — no embedded newline.
    assert "\n" not in header
    # The prompt content (with the newline replaced) is in the header.
    assert "first line second line" in header


# ---------------------------------------------------------------------------
# Smoke: confirm the script entry point function returns 0 under happy path.
# Separate from "formats hits" because that fixture covers stdout shape;
# this one re-affirms the rc contract under realistic-ish stdin shape.
# ---------------------------------------------------------------------------


def test_returns_zero_on_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """End-to-end happy path: stdin payload → sidecar 200 → exit 0."""
    _silence_log(monkeypatch, tmp_path)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "extra": {"prompt": "anything"},
        },
    )
    with mock_sidecar(
        {
            "GET /api/search/Entity": (
                200,
                [{"id": "n", "title": "T", "type": "CONCEPT"}],
            )
        }
    ) as base_url, _redirect_client_base_url(base_url):
        rc = user_prompt_submit.main()

    assert rc == 0
    # And confirm stdout is non-empty (negative control vs the no-output
    # tests above).
    assert capsys.readouterr().out.strip() != ""
