"""Task 14 — PostToolUse hook unit tests.

What we're protecting
=====================

The :mod:`hooks.claude_code.post_tool_use` hook is the most-frequently-
fired hook in the lifecycle (every tool call). Its load-bearing
invariants are:

1. **Dispatch correctness** — the right Claude Code tool name routes to
   the right sidecar endpoint. Read / Edit / Write / NotebookEdit →
   ``POST /api/Artifact``; ``Bash`` matching git commit → ``POST
   /api/Commit``; ``Bash`` matching pytest etc. → ``POST /api/Test`` or
   ``POST /api/Failure`` based on exit code; everything else → no HTTP
   call.
2. **Conservative defaults** — when a tool call doesn't match any
   tracked pattern, we MUST NOT call the sidecar. Over-firing would
   swamp the graph with noise.
3. **Never-raise contract** — every realistic failure mode (sidecar
   down, malformed payload, unparseable output) is swallowed; the hook
   returns 0 even when nothing was logged.
4. **Field-naming gotchas** — ``Failure.failure_type`` (not ``type``),
   ``Test.last_run_status`` uses ``passed``/``failed`` (not
   ``pass``/``fail``). These are sidecar-contract issues that would
   surface as 422 if the hook used the wrong key.
5. **Scope wiring** — the resolved scope string is on every payload so
   the design §D4 partitioning works.

Test approach
=============

We reuse the in-process HTTP server pattern from
``test_hook_sidecar_client.py``: a small ``http.server.BaseHTTPRequestHandler``
records every request, lets the test assert on path + method + body.
The hook is invoked by calling :func:`main` after monkeypatching
:func:`SidecarClient.__init__` to point at the test server.

We deliberately avoid mocking with :mod:`unittest.mock` — going through
real HTTP exercises the urllib code path the hook uses in production,
and the in-process server is fast enough that it doesn't slow the test
run meaningfully.
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

import pytest

from hooks.claude_code import post_tool_use


# ---------------------------------------------------------------------------
# In-process HTTP-server fixture — mirrored from test_hook_sidecar_client.
# ---------------------------------------------------------------------------


@contextmanager
def mock_sidecar(
    handler_map: dict[str, tuple[int, Any]] | None = None,
    *,
    record_requests: list[dict[str, Any]] | None = None,
):
    """Spin up an HTTP server that records every request.

    Args:
        handler_map: ``{"POST /api/Artifact": (201, {...})}`` mapping.
            Defaults to a permissive 201-with-empty-body for any path —
            most tests in this file only care about WHAT was POSTed,
            not what came back.
        record_requests: List the server appends a request-dict to on
            every served request. Tests read from this to assert on
            method + path + body.

    Yields:
        ``http://127.0.0.1:<port>`` base URL the hook should use.
    """
    handler_map = handler_map or {}

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
                    "body_json": _safe_json(body),
                }
            )

        def _respond(self, body_bytes: bytes = b"") -> None:
            key = f"{self.command} {urlparse(self.path).path}"
            entry = handler_map.get(key)
            self._record(body_bytes)
            # Default: 201 with empty body — most tests don't care about
            # the response shape, only that the request landed.
            if entry is None:
                if self.command == "POST":
                    self.send_response(201)
                else:
                    self.send_response(200)
                body_raw = b"{}"
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_raw)))
                self.end_headers()
                self.wfile.write(body_raw)
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


@contextmanager
def closed_port_url():
    """Yield a URL to a port nothing is listening on — sidecar-dead path."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield f"http://127.0.0.1:{port}"


def _safe_json(body: bytes) -> dict[str, Any] | None:
    """Decode a JSON body to dict, returning ``None`` on failure.

    Lets the request log show structured data when available without
    forcing tests to manually json.loads the raw body field.
    """
    if not body:
        return None
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


# ---------------------------------------------------------------------------
# Helpers — drive the hook with synthetic stdin + a configurable sidecar URL.
# ---------------------------------------------------------------------------


def _build_stdin_json(
    *,
    tool_name: str,
    tool_input: dict[str, Any] | None = None,
    tool_response: dict[str, Any] | None = None,
    session_id: str = "sess_test",
    cwd: str | None = None,
) -> str:
    """Compose the stdin JSON Claude Code emits for a PostToolUse event.

    The schema mirrors what :class:`HookInputBase` parses. Per-event
    fields go into ``extra``.
    """
    payload: dict[str, Any] = {
        "hook_event_name": "PostToolUse",
        "session_id": session_id,
        "extra": {
            "tool_name": tool_name,
            "tool_input": tool_input or {},
            "tool_response": tool_response or {},
        },
    }
    if cwd is not None:
        payload["cwd"] = cwd
    return json.dumps(payload)


def _run_hook_with(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdin_json: str,
    base_url: str,
    tmp_home: Path,
) -> int:
    """Drive :func:`post_tool_use.main` with synthetic stdin + a redirected sidecar URL.

    Steers ``$HOME`` at ``tmp_home`` so the hook's log file writer
    doesn't touch the operator's real ``~/.galvo-memory/``. Patches
    :class:`SidecarClient`'s default ``base_url`` so the hook's
    HTTP traffic goes to the test server instead of the (presumed
    nonexistent) :7575.
    """
    monkeypatch.setenv("HOME", str(tmp_home))
    # Redirect the logger's LOG_DIR via $HOME and force the module
    # reload by patching the cached LOG_DIR — the module reads
    # ``Path.home()`` once at import and caches the result.
    monkeypatch.setattr(
        "hooks.claude_code.lib.logging.LOG_DIR",
        tmp_home / ".galvo-memory" / "logs",
    )
    # Stub out SidecarClient's __init__ to point at the test server.
    monkeypatch.setattr(
        "hooks.claude_code.lib.sidecar_client.DEFAULT_BASE_URL",
        base_url,
    )
    monkeypatch.setattr(
        "hooks.claude_code.post_tool_use.SidecarClient.__init__",
        lambda self, base_url=base_url, timeout_s=2.0: _init_client(
            self, base_url=base_url, timeout_s=timeout_s
        ),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_json))
    return post_tool_use.main()


def _init_client(self, *, base_url: str, timeout_s: float) -> None:
    """Replacement for :class:`SidecarClient.__init__` — just sets fields."""
    self.base_url = base_url.rstrip("/")
    self.timeout_s = timeout_s


def _posts_to(recorded: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    """Filter recorded requests to POSTs hitting the given path prefix.

    Helper to keep test assertions readable. The hook may make multiple
    HTTP calls per event (rare in cycle 1; defensive against future
    additions).
    """
    return [r for r in recorded if r["method"] == "POST" and r["path"].startswith(path)]


# ---------------------------------------------------------------------------
# 1. Read tool → Artifact node
# ---------------------------------------------------------------------------


def test_read_tool_creates_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read tool fires ``POST /api/Artifact`` with the file path payload."""
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Read",
        tool_input={"file_path": "/repo/src/foo.py"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    artifact_calls = _posts_to(recorded, "/api/Artifact")
    assert len(artifact_calls) == 1, recorded
    body = artifact_calls[0]["body_json"]
    assert body is not None
    assert body["path"] == "/repo/src/foo.py"
    assert body["scope"]  # non-empty


# ---------------------------------------------------------------------------
# 2. Edit tool with .py extension → language=python
# ---------------------------------------------------------------------------


def test_edit_tool_creates_artifact_with_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Edit on a ``.py`` file → Artifact body contains ``language=python``."""
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Edit",
        tool_input={"file_path": "/repo/src/bar.py"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    artifact_calls = _posts_to(recorded, "/api/Artifact")
    assert len(artifact_calls) == 1
    body = artifact_calls[0]["body_json"]
    assert body is not None
    assert body["language"] == "python"


# ---------------------------------------------------------------------------
# 3. Unknown tool → no sidecar call
# ---------------------------------------------------------------------------


def test_unknown_tool_no_sidecar_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``WebSearch`` (not tracked) → zero POSTs hit the sidecar.

    Conservative-by-default: untracked tools log locally only.
    """
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="WebSearch",
        tool_input={"query": "anything"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    posts = [r for r in recorded if r["method"] == "POST"]
    assert posts == []


# ---------------------------------------------------------------------------
# 4. Bash git commit → Commit node with parsed SHA
# ---------------------------------------------------------------------------


def test_bash_git_commit_creates_commit_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``git commit`` output ``[main abc1234] message`` → POST /api/Commit
    with sha=abc1234."""
    recorded: list[dict[str, Any]] = []
    git_output = (
        "[main abc1234] add the thing\n"
        " 1 file changed, 5 insertions(+), 0 deletions(-)\n"
    )
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": 'git commit -m "add the thing"'},
        tool_response={"exit_code": 0, "output": git_output},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    commit_calls = _posts_to(recorded, "/api/Commit")
    assert len(commit_calls) == 1, recorded
    body = commit_calls[0]["body_json"]
    assert body is not None
    assert body["sha"] == "abc1234"
    assert body["message"] == "add the thing"
    # The shell command is preserved as ``intent`` for forensics.
    assert "git commit" in body["intent"]


# ---------------------------------------------------------------------------
# 5. Bash pytest exit=0 → Test node
# ---------------------------------------------------------------------------


def test_bash_pytest_pass_creates_test_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``pytest`` exit 0 → POST /api/Test with ``last_run_status='passed'``.

    Note the status string is ``passed`` (Test model default), NOT
    ``pass``. The hook brief said ``pass``; we use the sidecar's actual
    enum value so the create body doesn't 422.
    """
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": "pytest tests/ -v"},
        tool_response={"exit_code": 0, "output": "12 passed in 0.5s\n"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    test_calls = _posts_to(recorded, "/api/Test")
    assert len(test_calls) == 1
    body = test_calls[0]["body_json"]
    assert body is not None
    assert body["last_run_status"] == "passed"
    assert body["runner"] == "pytest"


# ---------------------------------------------------------------------------
# 6. Bash pytest exit=1 → Failure node
# ---------------------------------------------------------------------------


def test_bash_pytest_fail_creates_failure_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``pytest`` exit 1 → POST /api/Failure with ``failure_type='test'``.

    The error_signature is mined from the tail of stdout — protects
    against the renamed-field-from-type gotcha.
    """
    recorded: list[dict[str, Any]] = []
    pytest_output = (
        "tests/test_foo.py::test_bar FAILED\n"
        "AssertionError: expected 42 got 7\n"
        "1 failed in 0.3s\n"
    )
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": "pytest tests/test_foo.py"},
        tool_response={"exit_code": 1, "output": pytest_output},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    failure_calls = _posts_to(recorded, "/api/Failure")
    assert len(failure_calls) == 1
    body = failure_calls[0]["body_json"]
    assert body is not None
    # IMPORTANT: ``failure_type`` (not ``type``). Sidecar model renames
    # to avoid library collision.
    assert body["failure_type"] == "test"
    assert body["resolved"] is False
    assert "FAILED" in body["error_signature"] or "Assert" in body["error_signature"]


# ---------------------------------------------------------------------------
# 7. Bash build fail → Failure node failure_type='build'
# ---------------------------------------------------------------------------


def test_bash_build_fail_creates_failure_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``make`` exit 2 → POST /api/Failure with ``failure_type='build'``.

    Successful builds do NOT emit Failure nodes — too frequent.
    """
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": "make build"},
        tool_response={"exit_code": 2, "output": "Error: missing target\n"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    failure_calls = _posts_to(recorded, "/api/Failure")
    assert len(failure_calls) == 1
    body = failure_calls[0]["body_json"]
    assert body is not None
    assert body["failure_type"] == "build"


# ---------------------------------------------------------------------------
# 8. Bash other command → no sidecar call
# ---------------------------------------------------------------------------


def test_bash_other_command_no_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``ls -la`` (no pattern match) → zero POSTs hit the sidecar."""
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": "ls -la"},
        tool_response={"exit_code": 0, "output": "file1\nfile2\n"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    posts = [r for r in recorded if r["method"] == "POST"]
    assert posts == []


# ---------------------------------------------------------------------------
# 9. Sidecar unreachable → hook still returns 0 (never-raise)
# ---------------------------------------------------------------------------


def test_artifact_create_failure_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the sidecar is dead, the hook still exits 0 — no exception."""
    stdin_json = _build_stdin_json(
        tool_name="Write",
        tool_input={"file_path": "/repo/x.py"},
        cwd=str(tmp_path),
    )
    with closed_port_url() as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0


# ---------------------------------------------------------------------------
# 10. Scope is on every payload
# ---------------------------------------------------------------------------


def test_scope_passed_to_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hook resolves scope from cwd and passes it through to the sidecar.

    Design §D4 partitioning rests on the ``scope`` property being on
    every node payload. We write a project marker so the resolved scope
    is ``project:galvo`` (distinguishable from the ``personal`` fallback).
    """
    from scope import write_marker  # local import — avoids touching top of file

    write_marker(tmp_path, project_id="galvo", name="Galvo Test")
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Read",
        tool_input={"file_path": str(tmp_path / "src" / "foo.py")},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    artifact_calls = _posts_to(recorded, "/api/Artifact")
    assert len(artifact_calls) == 1
    body = artifact_calls[0]["body_json"]
    assert body is not None
    assert body["scope"] == "project:galvo"


# ---------------------------------------------------------------------------
# 11. Empty stdin → exit 0, no HTTP calls
# ---------------------------------------------------------------------------


def test_empty_stdin_exits_zero_no_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty stdin → HookInputBase.from_stdin returns UNKNOWN_EVENT → no-op."""
    recorded: list[dict[str, Any]] = []
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json="", base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    assert recorded == []


# ---------------------------------------------------------------------------
# 12. NotebookEdit → Artifact via notebook_path
# ---------------------------------------------------------------------------


def test_notebook_edit_creates_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``NotebookEdit`` uses ``notebook_path`` (not ``file_path``)."""
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="NotebookEdit",
        tool_input={"notebook_path": "/repo/analysis.ipynb"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    artifact_calls = _posts_to(recorded, "/api/Artifact")
    assert len(artifact_calls) == 1
    body = artifact_calls[0]["body_json"]
    assert body is not None
    assert body["path"] == "/repo/analysis.ipynb"
    assert body["language"] == "jupyter"


# ---------------------------------------------------------------------------
# 13. Cargo test pass + fail
# ---------------------------------------------------------------------------


def test_bash_cargo_test_pass_creates_test_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``cargo test`` exit 0 → Test passed, runner=``cargo test``.

    Protects the test-runner inference path for non-pytest harnesses.
    """
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": "cargo test --workspace"},
        tool_response={"exit_code": 0, "output": "test result: ok. 12 passed"},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    test_calls = _posts_to(recorded, "/api/Test")
    assert len(test_calls) == 1
    assert test_calls[0]["body_json"]["runner"] == "cargo test"


# ---------------------------------------------------------------------------
# 14. Build success → no Failure node (conservative)
# ---------------------------------------------------------------------------


def test_bash_build_success_no_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``cargo build`` exit 0 → no node. Successful builds are not
    worth a graph write."""
    recorded: list[dict[str, Any]] = []
    stdin_json = _build_stdin_json(
        tool_name="Bash",
        tool_input={"command": "cargo build --release"},
        tool_response={"exit_code": 0, "output": "Compiling galvo ..."},
        cwd=str(tmp_path),
    )
    with mock_sidecar(record_requests=recorded) as base_url:
        rc = _run_hook_with(
            monkeypatch, stdin_json=stdin_json, base_url=base_url, tmp_home=tmp_path
        )
    assert rc == 0
    posts = [r for r in recorded if r["method"] == "POST"]
    assert posts == []


# ---------------------------------------------------------------------------
# 15. Helpers — pure-function coverage
# ---------------------------------------------------------------------------


def test_guess_language_known_extensions() -> None:
    """Sample of the language-map happy path + unknown fallback."""
    assert post_tool_use._guess_language("foo.py") == "python"
    assert post_tool_use._guess_language("foo.rs") == "rust"
    assert post_tool_use._guess_language("foo.tsx") == "typescript"
    assert post_tool_use._guess_language("README.md") == "markdown"
    # Unknown suffix falls back to "unknown" (not None — design intent).
    assert post_tool_use._guess_language("foo.xyz") == "unknown"


def test_extract_failure_signature_prefers_error_lines() -> None:
    """Mining function picks up Error/FAILED lines from the tail."""
    output = (
        "Running tests...\n"
        "test_a ... ok\n"
        "test_b ... FAILED\n"
        "AssertionError: expected 1 got 2\n"
        "1 failed, 1 passed\n"
    )
    sig = post_tool_use._extract_failure_signature(output)
    assert "FAILED" in sig or "AssertionError" in sig


def test_extract_failure_signature_empty_output() -> None:
    """Empty/whitespace-only output → sentinel string (sidecar requires
    error_signature min_length=1)."""
    assert post_tool_use._extract_failure_signature("") == "(empty output)"
    assert post_tool_use._extract_failure_signature("   \n  \n") == "(empty output)"


def test_coerce_exit_code_handles_various_shapes() -> None:
    """Exit code coercion tolerates int / str / None / bool."""
    assert post_tool_use._coerce_exit_code(0) == 0
    assert post_tool_use._coerce_exit_code(1) == 1
    assert post_tool_use._coerce_exit_code("2") == 2
    assert post_tool_use._coerce_exit_code(None) == 0
    assert post_tool_use._coerce_exit_code(True) == 0  # shell-style: True = success
    assert post_tool_use._coerce_exit_code(False) == 1
    assert post_tool_use._coerce_exit_code("notanumber") == 0


def test_coerce_output_supports_stdout_stderr_keys() -> None:
    """Older tool_response shapes (stdout/stderr) get joined into one string."""
    out = post_tool_use._coerce_output({"stdout": "hello", "stderr": "world"})
    assert "hello" in out
    assert "world" in out


def test_extract_commit_message_handles_branch_with_dashes() -> None:
    """Branch names with dashes / slashes don't confuse SHA parsing."""
    output = "[feature/foo-bar abc1234] my message\n"
    assert post_tool_use._extract_commit_message(output) == "my message"
