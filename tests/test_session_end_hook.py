"""Task 15 — SessionEnd hook tests.

The hook script's contract is:

* Read stdin JSON via :class:`HookInputBase.from_stdin`. On malformed
  input or a missing ``session_id``: exit 0, no sidecar calls.
* If a transcript path is present and readable: parse it for
  ``assistant_outputs`` + ``requeries`` + first-user-prompt task description
  + best-guess outcome from the last assistant turn.
* :http:post:`/api/Session` to MERGE the final :Session node (title,
  started_at, ended_at, scope, outcome, task_description).
* :http:post:`/api/sessions/{id}/score` with the
  :class:`ScoringPayload` shape, applying the bounding limits the hook
  enforces locally (``_MAX_ASSISTANT_OUTPUTS=50``, ``_MAX_REQUERIES=20``).
* Return 0 from :func:`main` even when both HTTP calls fail — hooks must
  never fail the user's session shutdown.

We test these behaviors using the same in-process HTTP server pattern
as ``test_hook_sidecar_client.py`` (Task 11). That keeps the tests
dependency-light (no httpx, no fastapi.testclient) and matches the
hook's own choice of stdlib :mod:`urllib`.

The transcript-format assumptions are documented in
:mod:`hooks.claude_code.session_end._parse_transcript`. The tests
exercise the three concrete shapes the parser accepts:

* Top-level ``{"type": "user" | "assistant", "content": "..."}``
* Anthropic-envelope ``{"message": {"role": "...", "content": "..."}}``
* Anthropic-envelope with structured content blocks
  ``{"message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]}}``

A drift in Claude Code's actual transcript shape would surface here as
an under-counted signal in the integration tests at Task 20 acceptance
sweep, NOT as a test failure here — that's intentional. These tests
protect our parser's invariants; the integration tests protect the
parser-vs-real-transcript contract.
"""

from __future__ import annotations

import http.server
import io
import json
import socket
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import pytest

from hooks.claude_code import session_end


# ---------------------------------------------------------------------------
# In-process HTTP-server fixture (mirrors Task 11's helper).
# ---------------------------------------------------------------------------


@contextmanager
def mock_sidecar(
    handler_map: dict[str, tuple[int, Any]],
    *,
    record_requests: list[dict[str, Any]] | None = None,
):
    """Tiny canned-response HTTP server, on a random free port.

    Args:
        handler_map: ``{"POST /api/Session": (201, {...}), ...}``.
            Keys are ``"{METHOD} {PATH_NO_QUERY}"``.
        record_requests: Optional sink for inbound request metadata;
            tests use this to assert on the body the hook posted.

    Yields:
        The base URL the test should pass to ``SidecarClient(base_url=...)``
        via the hook script's :data:`session_end._BASE_URL` env or by
        monkeypatching :class:`SidecarClient`'s default URL.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_a: Any, **_kw: Any) -> None:
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

        def do_GET(self) -> None:  # noqa: N802
            self._respond()

        def do_POST(self) -> None:  # noqa: N802
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
# Helpers
# ---------------------------------------------------------------------------


def _patch_sidecar_base_url(base_url: str) -> Any:
    """Patch the sidecar client constructor to use the mock server's URL.

    The hook script constructs :class:`SidecarClient` with no args. Python
    binds default-argument values at function-definition time, so a naive
    ``patch("hooks.claude_code.lib.sidecar_client.DEFAULT_BASE_URL", ...)``
    wouldn't affect the already-bound default. We instead patch the
    *class* in the ``session_end`` module's namespace to a thin factory
    that forwards to the real class with ``base_url`` injected. This
    matches how production code reaches :class:`SidecarClient` (via the
    import in :mod:`session_end`) and doesn't depend on the constructor's
    parameter resolution order.
    """
    from hooks.claude_code.lib.sidecar_client import SidecarClient

    def factory(*_args: Any, **_kwargs: Any) -> SidecarClient:
        # Tight timeout — mock server replies in <100ms locally, so 2.0s
        # is plenty and the bounded value keeps a failing test from
        # hanging at the maximum of the production default (3 s).
        return SidecarClient(base_url=base_url, timeout_s=2.0)

    return patch(
        "hooks.claude_code.session_end.SidecarClient",
        factory,
    )


def _set_stdin(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    """Replace ``sys.stdin`` with a buffer containing ``payload`` as JSON.

    Mirrors how Claude Code dispatches hooks — the runtime spawns the
    script with stdin = the hook event JSON. The hook script reads it
    via :class:`HookInputBase.from_stdin` which calls ``sys.stdin.read()``.
    """
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def _write_transcript(path: Path, events: list[dict[str, Any]]) -> Path:
    """Materialize a JSONL transcript at ``path`` containing ``events``.

    Returns ``path`` to make chaining ergonomic in tests.
    """
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )
    return path


def _user_event(text: str) -> dict[str, Any]:
    """Build a top-level-discriminator user event matching the parser's
    first-shape branch."""
    return {"type": "user", "content": text}


def _assistant_event(text: str) -> dict[str, Any]:
    """Build a top-level-discriminator assistant event."""
    return {"type": "assistant", "content": text}


# ---------------------------------------------------------------------------
# Test 1 — empty/malformed stdin / missing session_id ⇒ no-op
# ---------------------------------------------------------------------------


def test_no_session_id_no_op(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stdin lacks ``session_id`` → hook exits 0, no sidecar calls.

    This is the defensive case for "Claude Code sent us a SessionEnd
    event without the session id" (shouldn't happen in practice, but the
    hook must degrade safely).
    """
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate the log file
    _set_stdin(monkeypatch, {"hook_event_name": "SessionEnd"})

    recorded: list[dict[str, Any]] = []
    with mock_sidecar({}, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            exit_code = session_end.main()

    assert exit_code == 0
    assert recorded == [], "no HTTP calls expected when session_id is absent"


# ---------------------------------------------------------------------------
# Test 2 — missing transcript ⇒ Session created with empty desc + "unknown"
# ---------------------------------------------------------------------------


def test_missing_transcript_creates_minimal_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transcript path doesn't exist → still write a :Session with
    ``task_description=""`` + ``outcome="unknown"``, then send a
    minimal scoring payload.

    Critical because Claude Code may invoke SessionEnd before the
    transcript file is fully synced to disk on the host filesystem.
    The hook still has work to do (close the :Session, fire the score
    request) — it should not give up just because the transcript
    isn't readable.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_abc",
            "cwd": str(tmp_path),
            "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_abc"}),
        "POST /api/sessions/sess_abc/score": (
            200,
            {"session_id": "sess_abc", "edges_scored": 0, "edges_skipped": 0, "scores": []},
        ),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            exit_code = session_end.main()

    assert exit_code == 0
    paths = [r["path"] for r in recorded]
    assert "/api/Session" in paths
    assert "/api/sessions/sess_abc/score" in paths

    session_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/Session")
    )
    assert session_body["task_description"] == ""
    assert session_body["outcome"] == "unknown"
    assert session_body["id"] == "sess_abc"


# ---------------------------------------------------------------------------
# Test 3 — happy path: transcript with user+assistant turns
# ---------------------------------------------------------------------------


def test_happy_path_writes_session_and_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transcript with a user task + assistant reply produces both an
    :Session POST and a /score POST with non-empty bodies.

    This is the canonical successful run — the cycle-1 "memory works"
    smoke test in a unit. We assert the bodies' key fields rather than
    pinning the entire JSON because field order / extra fields aren't
    semantically meaningful.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = _write_transcript(
        tmp_path / "transcript.jsonl",
        [
            _user_event("Implement task 15 SessionEnd hook"),
            _assistant_event("Here is the plan: ... PR merged successfully."),
        ],
    )
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_xyz",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_xyz"}),
        "POST /api/sessions/sess_xyz/score": (
            200,
            {"session_id": "sess_xyz", "edges_scored": 2, "edges_skipped": 0, "scores": []},
        ),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            exit_code = session_end.main()

    assert exit_code == 0

    by_path = {r["path"]: json.loads(r["body"]) for r in recorded}
    assert set(by_path) == {"/api/Session", "/api/sessions/sess_xyz/score"}

    session_body = by_path["/api/Session"]
    assert session_body["id"] == "sess_xyz"
    assert "Implement task 15" in session_body["task_description"]
    assert session_body["title"]  # non-empty
    assert session_body["agent_tool"] == "claude-code"

    score_body = by_path["/api/sessions/sess_xyz/score"]
    assert score_body["session_id"] == "sess_xyz"
    assert isinstance(score_body["assistant_outputs"], list)
    assert len(score_body["assistant_outputs"]) >= 1
    assert "PR merged successfully" in score_body["assistant_outputs"][-1]


# ---------------------------------------------------------------------------
# Test 4 — outcome inferred from last assistant turn keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "last_turn,expected_outcome",
    [
        ("Shipped to main; PR merged.", "success"),
        ("Everything passed — green CI.", "success"),
        ("Build failed: missing import.", "failure"),
        ("Aborted; user cancelled.", "failure"),
        ("I made some progress but need more info.", "partial"),
    ],
)
def test_outcome_inferred_from_last_assistant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    last_turn: str,
    expected_outcome: str,
) -> None:
    """The last assistant turn's wording maps to the outcome enum.

    Parametrized across the three outcome buckets the cycle-1 heuristic
    distinguishes. A regression in the keyword tables would flip one of
    these cases — easy to spot at a glance.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = _write_transcript(
        tmp_path / "transcript.jsonl",
        [
            _user_event("Do the thing."),
            _assistant_event(last_turn),
        ],
    )
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_p",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_p"}),
        "POST /api/sessions/sess_p/score": (200, {"session_id": "sess_p", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            session_end.main()

    session_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/Session")
    )
    assert session_body["outcome"] == expected_outcome


# ---------------------------------------------------------------------------
# Test 5 — requeries extracted from subsequent user messages
# ---------------------------------------------------------------------------


def test_requeries_extracted_from_subsequent_user_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """First user turn is the task; later user turns become ``requeries``.

    The scorer (-0.4 signal) penalizes memories that didn't satisfy the
    agent's first query — the requery field is how it knows the agent
    asked again. The first prompt is excluded because by definition it's
    not a re-query of an earlier retrieval.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = _write_transcript(
        tmp_path / "transcript.jsonl",
        [
            _user_event("Initial task"),
            _assistant_event("Working on it..."),
            _user_event("That didn't work — try a different angle?"),
            _assistant_event("Trying again..."),
            _user_event("Show me the test output."),
        ],
    )
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_q",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_q"}),
        "POST /api/sessions/sess_q/score": (200, {"session_id": "sess_q", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            session_end.main()

    score_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/sessions/sess_q/score")
    )
    assert len(score_body["requeries"]) == 2
    assert score_body["requeries"][0] == "That didn't work — try a different angle?"
    assert score_body["requeries"][1] == "Show me the test output."

    # And the first user prompt is the task, not a requery
    session_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/Session")
    )
    assert session_body["task_description"] == "Initial task"


# ---------------------------------------------------------------------------
# Test 6 — Session create failure still scores
# ---------------------------------------------------------------------------


def test_session_create_failure_still_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If :Session POST returns 500 we still POST /score.

    The earlier hooks (Task 12 SessionStart, Task 14 PostToolUse) may
    have already created a :Session node; even if our finalization
    write fails, the scoring endpoint should still find edges via the
    earlier-created node. Skipping /score because /Session 500'd would
    silently zero the cycle-1 feedback signal.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    transcript = _write_transcript(
        tmp_path / "transcript.jsonl",
        [
            _user_event("Task"),
            _assistant_event("Done."),
        ],
    )
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_f",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        # Session create returns 500 — the sidecar client returns None
        "POST /api/Session": (500, {"detail": "db hiccup"}),
        # /score should still be called
        "POST /api/sessions/sess_f/score": (
            200,
            {"session_id": "sess_f", "edges_scored": 1, "edges_skipped": 0, "scores": []},
        ),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            exit_code = session_end.main()

    assert exit_code == 0
    paths = [r["path"] for r in recorded]
    assert "/api/Session" in paths
    assert "/api/sessions/sess_f/score" in paths


# ---------------------------------------------------------------------------
# Test 7 — scope is detected and passed through to the Session payload
# ---------------------------------------------------------------------------


def test_scope_passed_to_session_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cwd inside a project (with a ``.galvo-mem`` marker) writes
    ``scope=project:<id>`` to the :Session payload.

    Validates the design-§D4 partitioning glue. Without this, all
    SessionEnd writes would land in the same bucket and the scope filter
    on retrieval would never find them.
    """
    from scope import write_marker

    monkeypatch.setenv("HOME", str(tmp_path))
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    write_marker(project_dir, project_id="myproj-id", name="My Project")
    transcript = _write_transcript(
        project_dir / "transcript.jsonl",
        [_user_event("hi"), _assistant_event("hello")],
    )
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_s",
            "cwd": str(project_dir),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_s"}),
        "POST /api/sessions/sess_s/score": (200, {"session_id": "sess_s", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            session_end.main()

    session_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/Session")
    )
    assert session_body["scope"] == "project:myproj-id"


# ---------------------------------------------------------------------------
# Test 8 — payload is bounded (≤50 assistant_outputs in /score body)
# ---------------------------------------------------------------------------


def test_payload_bounded_to_max_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transcript with >50 assistant turns trims to the LAST 50 in the
    /score payload.

    Verifies the hard cap (:data:`session_end._MAX_ASSISTANT_OUTPUTS`)
    that bounds the JSON body sent to the sidecar. Without this trim, a
    long-running session could push hundreds of KB through the loopback
    HTTP call — fine in isolation, but creates a slow / unbounded
    failure mode under load.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    events: list[dict[str, Any]] = [_user_event("Task")]
    for i in range(200):
        events.append(_assistant_event(f"turn-{i}"))
    transcript = _write_transcript(tmp_path / "transcript.jsonl", events)

    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_b",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_b"}),
        "POST /api/sessions/sess_b/score": (200, {"session_id": "sess_b", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            session_end.main()

    score_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/sessions/sess_b/score")
    )
    assert len(score_body["assistant_outputs"]) == session_end._MAX_ASSISTANT_OUTPUTS
    # We took the LAST 50, so the final entry should be turn-199
    assert "turn-199" in score_body["assistant_outputs"][-1]
    # And the first entry should be turn-150 (200 minus 50)
    assert "turn-150" in score_body["assistant_outputs"][0]


# ---------------------------------------------------------------------------
# Test 9 — Anthropic-envelope transcript shape is also parsed
# ---------------------------------------------------------------------------


def test_anthropic_envelope_shape_parsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transcript using the ``{"message": {"role": ..., "content": [...]}}``
    shape is parsed correctly.

    Claude Code's real transcript on macOS as of 2026-05 uses this
    envelope shape (verified empirically). The parser supports it as
    one of three accepted shapes; this test pins the support so a
    regression breaks the test, not the integration.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    events = [
        {
            "message": {
                "role": "user",
                "content": "First task with envelope shape",
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Working on it..."},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "x"}},
                    {"type": "text", "text": "Done — shipped."},
                ],
            }
        },
    ]
    transcript = _write_transcript(tmp_path / "transcript.jsonl", events)
    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_e",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_e"}),
        "POST /api/sessions/sess_e/score": (200, {"session_id": "sess_e", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            session_end.main()

    session_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/Session")
    )
    assert "First task" in session_body["task_description"]
    # The assistant content blocks are joined; "shipped" → success
    assert session_body["outcome"] == "success"

    score_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/sessions/sess_e/score")
    )
    assert len(score_body["assistant_outputs"]) >= 1
    last = score_body["assistant_outputs"][-1]
    assert "Working on it" in last
    assert "shipped" in last
    # tool_use blocks must NOT appear in extracted text
    assert "tool_use" not in last


# ---------------------------------------------------------------------------
# Test 10 — main() never raises on unparseable transcript lines
# ---------------------------------------------------------------------------


def test_malformed_transcript_lines_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSONL file with mixed valid + corrupt lines: hook still
    succeeds, using only the valid lines.

    Mid-file corruption (e.g. a partially-flushed final line) is the
    common failure mode for transcripts on a crash. The parser must
    treat each line independently — a bad line must not poison the
    surrounding good lines.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    bad_text = "\n".join(
        [
            json.dumps(_user_event("Real task")),
            "{this is not json",  # malformed
            "[1, 2, 3]",  # JSON but not a dict
            json.dumps(_assistant_event("All shipped.")),
            "",  # blank
        ]
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(bad_text, encoding="utf-8")

    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_m",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_m"}),
        "POST /api/sessions/sess_m/score": (200, {"session_id": "sess_m", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            exit_code = session_end.main()

    assert exit_code == 0
    session_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/Session")
    )
    # Despite the corruption, we picked up the good lines
    assert session_body["task_description"] == "Real task"
    assert session_body["outcome"] == "success"


# ---------------------------------------------------------------------------
# Test 11 — empty stdin still exits 0
# ---------------------------------------------------------------------------


def test_empty_stdin_no_op(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty stdin → :class:`HookInputBase.from_stdin` returns the
    "unknown" sentinel → :func:`main` exits 0 without any HTTP calls.

    This is the upper-bound defensive case: even with no stdin at all,
    the script must not raise.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    recorded: list[dict[str, Any]] = []
    with mock_sidecar({}, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            exit_code = session_end.main()

    assert exit_code == 0
    assert recorded == []


# ---------------------------------------------------------------------------
# Test 12 — requeries are also bounded
# ---------------------------------------------------------------------------


def test_requeries_bounded_to_max(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transcript with >20 user follow-ups trims requeries to the last 20."""
    monkeypatch.setenv("HOME", str(tmp_path))
    events: list[dict[str, Any]] = [_user_event("Task")]
    for i in range(100):
        events.append(_assistant_event(f"reply-{i}"))
        events.append(_user_event(f"follow-up-{i}"))
    transcript = _write_transcript(tmp_path / "transcript.jsonl", events)

    _set_stdin(
        monkeypatch,
        {
            "hook_event_name": "SessionEnd",
            "session_id": "sess_r",
            "cwd": str(tmp_path),
            "transcript_path": str(transcript),
        },
    )

    recorded: list[dict[str, Any]] = []
    handlers = {
        "POST /api/Session": (201, {"id": "sess_r"}),
        "POST /api/sessions/sess_r/score": (200, {"session_id": "sess_r", "edges_scored": 0, "edges_skipped": 0, "scores": []}),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        with _patch_sidecar_base_url(base_url):
            session_end.main()

    score_body = json.loads(
        next(r["body"] for r in recorded if r["path"] == "/api/sessions/sess_r/score")
    )
    assert len(score_body["requeries"]) == session_end._MAX_REQUERIES
    # The last requery in the trimmed list is the last one we wrote
    assert "follow-up-99" in score_body["requeries"][-1]
