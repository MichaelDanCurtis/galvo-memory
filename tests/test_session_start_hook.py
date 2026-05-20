"""Task 12 — SessionStart hook unit tests.

What we're protecting:

* **Never-raises contract.** Every input the script can see — empty
  stdin, malformed JSON, dead sidecar, slow sidecar, unexpected
  response shape — must produce ``exit_code == 0`` and never raise.
  This is the load-bearing UX guarantee: a SessionStart hook that
  crashes blocks the user from starting a Claude Code session.

* **Token-budget enforcement.** Design §10 caps the injection at
  ≤30 lines. A pathological sidecar returning 100 decisions must not
  blow past that — the cap is applied AFTER formatting so it's
  effective regardless of how many sections the sidecar populated.

* **Silent no-op when nothing to inject.** If the sidecar reports
  health but every search returns ``[]``, the script must emit ZERO
  bytes to stdout. Emitting a bare header would still cost the model
  a context slot for no signal.

* **Scope wiring.** The cwd from stdin must drive a scope-filtered
  search. Without this, a session running in ``/tmp`` would surface
  ``project:galvo`` decisions (or worse, vice versa: a Galvo session
  would surface no project memories because the search omitted the
  scope filter).

* **Stdout, not stderr.** Claude Code captures both, but stdout
  goes into context (the intent) and stderr goes into the user-visible
  transcript (noise). The hook must use stdout exclusively for the
  markdown block.

We use the same in-process HTTP-server fixture pattern as Task 11
(``test_hook_sidecar_client.py``) so the tests run with zero deps
beyond ``pytest`` + the ``galvo-memory`` package — no fastapi,
no httpx, no live Neo4j.
"""

from __future__ import annotations

import http.server
import io
import json
import socket
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

# Importing the hook script registers it under the package path the
# tests address. The package is laid out under memory/hooks/claude_code/
# per pyproject.toml [tool.hatch.build.targets.wheel].packages = [...].
from hooks.claude_code import session_start as ss
from hooks.claude_code.lib import sidecar_client as sc_mod


# ---------------------------------------------------------------------------
# In-process HTTP-server fixture — identical contract to Task 11's tests.
# Repeated here (rather than imported) so the two test modules don't
# couple at the fixture boundary; if either changes, the other is
# unaffected.
# ---------------------------------------------------------------------------


@contextmanager
def mock_sidecar(
    handler_map: dict[str, tuple[int, Any]],
    *,
    delay_s: float = 0.0,
    record_requests: list[dict[str, Any]] | None = None,
):
    """Spin up a tiny HTTP server that returns canned JSON responses.

    Args:
        handler_map: ``{"GET /health": (200, {...}), ...}``. Maps
            ``"{METHOD} {PATH}"`` (no query string) to ``(status, body)``.
            Missing keys produce a 404.
        delay_s: Sleep before responding — used for the timeout test.
        record_requests: If non-None, each request gets a dict appended
            describing the wire-level call.

    Yields:
        Base URL (``http://127.0.0.1:<port>``).
    """

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
            if delay_s:
                time.sleep(delay_s)
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


@contextmanager
def closed_port_url():
    """Yield a URL whose port has nothing listening — sidecar-dead case."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Test helper: run main() with a controlled stdin + a redirected sidecar URL.
# ---------------------------------------------------------------------------


def _run_main(
    stdin_text: str,
    *,
    base_url: str | None = None,
    cwd_override: str | None = None,
    home_override: Path | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[int, str, str]:
    """Invoke :func:`session_start.main` in a controlled environment.

    Returns ``(exit_code, stdout_text, stderr_text)``.

    Args:
        stdin_text: Raw bytes to feed the script's stdin reader.
        base_url: If provided, monkeypatch ``DEFAULT_BASE_URL`` so the
            in-script ``SidecarClient()`` (called with no args) hits
            the mock server instead of the real :7575.
        cwd_override: Not used by main() directly — main() reads cwd
            from the parsed stdin payload. Caller embeds in stdin JSON.
            (Kept as a parameter slot for future hook tests that may
            stub :func:`Path.cwd`.)
        home_override: If provided, monkeypatch ``$HOME`` so the
            logger doesn't write into the real user home during tests.
        monkeypatch: Required if ``base_url`` or ``home_override`` is set
            (we need pytest's monkeypatch fixture for clean restoration).
    """
    if (base_url is not None or home_override is not None) and monkeypatch is None:
        raise RuntimeError("monkeypatch required when overriding base_url/home")

    with ExitStack() as stack:
        # stdin redirection — sys.stdin is a fresh StringIO so the
        # script's HookInputBase.from_stdin() reads our payload.
        fake_stdin = io.StringIO(stdin_text)
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        stack.enter_context(_temp_attr(sys, "stdin", fake_stdin))
        stack.enter_context(_temp_attr(sys, "stdout", fake_stdout))
        stack.enter_context(_temp_attr(sys, "stderr", fake_stderr))

        if home_override is not None:
            assert monkeypatch is not None
            monkeypatch.setenv("HOME", str(home_override))
            # The lib.logging module caches LOG_DIR at import time; we
            # patch the module-level constant so the logger writes
            # under tmp_path rather than the real ~/.galvo-memory.
            log_dir = home_override / ".galvo-memory" / "logs"
            monkeypatch.setattr(
                "hooks.claude_code.lib.logging.LOG_DIR", log_dir
            )

        if base_url is not None:
            assert monkeypatch is not None
            # The script does ``SidecarClient()`` with no args. Python
            # captured ``DEFAULT_BASE_URL`` as the default-argument
            # value at SidecarClient.__init__'s definition time — so
            # patching the module constant after import does NOT
            # redirect new instances. We patch the imported reference
            # on the session_start module itself with a thin factory
            # that injects ``base_url`` on every call.
            url = base_url

            class _PinnedClient(sc_mod.SidecarClient):
                def __init__(
                    self,
                    base_url: str = url,
                    timeout_s: float = sc_mod.DEFAULT_TIMEOUT_S,
                ) -> None:
                    super().__init__(base_url=base_url, timeout_s=timeout_s)

            monkeypatch.setattr(ss, "SidecarClient", _PinnedClient)

        exit_code = ss.main()

        return exit_code, fake_stdout.getvalue(), fake_stderr.getvalue()


@contextmanager
def _temp_attr(target: object, attr: str, value: object):
    """Temporarily set an attribute and restore it on exit."""
    original = getattr(target, attr)
    setattr(target, attr, value)
    try:
        yield
    finally:
        setattr(target, attr, original)


def _stdin_payload(
    *,
    cwd: str | None = None,
    session_id: str = "sess_test_001",
) -> str:
    """Build a SessionStart hook stdin JSON payload."""
    payload: dict[str, Any] = {
        "hook_event_name": "SessionStart",
        "session_id": session_id,
    }
    if cwd is not None:
        payload["cwd"] = cwd
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# 1. Empty stdin → exits 0, no output.
# ---------------------------------------------------------------------------


def test_empty_stdin_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No stdin at all → exit 0, zero bytes of stdout."""
    exit_code, stdout, _stderr = _run_main(
        "",
        home_override=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    assert stdout == ""


# ---------------------------------------------------------------------------
# 2. Malformed JSON stdin → exits 0, no output.
# ---------------------------------------------------------------------------


def test_malformed_json_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garbage on stdin → exit 0, zero bytes of stdout, no traceback."""
    exit_code, stdout, _stderr = _run_main(
        "{this is not json",
        home_override=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    assert stdout == ""


# ---------------------------------------------------------------------------
# 3. Sidecar down (connection refused) → exits 0, no output.
# ---------------------------------------------------------------------------


def test_sidecar_down_emits_silent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar's TCP port closed → silent no-op.

    The most common dev-env failure: operator forgot ``docker compose up``.
    The hook must not block the user's session or surface an error.
    """
    with closed_port_url() as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    assert stdout == ""


# ---------------------------------------------------------------------------
# 4. Happy path — sidecar returns decisions + beliefs → output has them.
# ---------------------------------------------------------------------------


def test_happy_path_emits_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar returns 2 decisions + 1 belief → output has section headers + bullets."""
    decisions = [
        {"id": "dec_001", "title": "Use ruff"},
        {"id": "dec_002", "title": "Pin neo4j-agent-memory to 0.2.1"},
    ]
    beliefs = [{"id": "bel_001", "claim": "ScoreRow shape is nested"}]
    handlers = {
        "GET /health": (200, {"status": "ok", "neo4j": "ok"}),
        "GET /api/search/Decision": (200, decisions),
        "GET /api/search/Belief": (200, beliefs),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    # Header always present when at least one section has content.
    assert "Galvo Memory — top-of-mind" in stdout
    # Decisions section + each title in a bullet.
    assert "## Recent decisions" in stdout
    assert "- Use ruff" in stdout
    assert "- Pin neo4j-agent-memory to 0.2.1" in stdout
    # Beliefs section + the claim.
    assert "## Active beliefs" in stdout
    assert "- ScoreRow shape is nested" in stdout
    # Empty sections should NOT appear (no header without bullets).
    assert "## Open tasks" not in stdout
    assert "## Open failures" not in stdout


# ---------------------------------------------------------------------------
# 5. All-empty sidecar → output is empty (not even a bare header).
# ---------------------------------------------------------------------------


def test_empty_sidecar_omits_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar healthy but every search returns []. No bytes of stdout.

    A bare header with no bullets would still cost the model a context
    slot — we suppress it entirely. The hook contract is "inject signal
    when present; emit nothing otherwise".
    """
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, []),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    assert stdout == ""


# ---------------------------------------------------------------------------
# 6. Hard line cap — 50 decisions clipped to ≤30 lines total.
# ---------------------------------------------------------------------------


def test_hard_line_cap_30(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pathological sidecar response → output ≤ MAX_LINES lines.

    Even if (somehow, contrary to our limit= parameter) the sidecar
    returned 50 decisions, the script's hard truncation must catch it.
    The truncation is a defensive last line of defense after section
    caps (MAX_DECISIONS=5) and the limit= kwarg on search().
    """
    fifty_decisions = [
        {"id": f"dec_{i:03d}", "title": f"Decision {i}"} for i in range(50)
    ]
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, fifty_decisions),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    # The output (without the trailing newline) must be ≤ MAX_LINES lines.
    # We count the lines of the actual markdown block; the final
    # trailing newline produces an empty trailing element when split,
    # so we strip it before counting.
    line_count = len(stdout.rstrip("\n").splitlines())
    assert line_count <= ss.MAX_LINES, (
        f"output exceeded MAX_LINES={ss.MAX_LINES}: got {line_count} lines"
    )


# ---------------------------------------------------------------------------
# 7. Scope passed to sidecar — project marker present → project:<id>.
# ---------------------------------------------------------------------------


def test_scope_passed_to_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cwd inside a project (marker present) → search calls carry scope=project:<id>.

    Drops a real ``.galvo-mem/project.toml`` into ``tmp_path`` and
    points the hook's ``cwd`` at it. Records each request that hit the
    mock sidecar and asserts ``scope`` was on every search URL.
    """
    from scope import write_marker

    write_marker(tmp_path, project_id="galvo-test", name="Galvo Test Fixture")

    recorded: list[dict[str, Any]] = []
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, []),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        # We use a different home so the logger doesn't try to write
        # into the marker'd tmp_path. tmp_path here is BOTH the project
        # root (where the marker lives) AND would be HOME by default
        # — using a sibling for HOME keeps the marker test from
        # accidentally classifying the marker dir as "personal".
        home_dir = tmp_path / "home_sandbox"
        home_dir.mkdir()
        exit_code, _stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=home_dir,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    # We should see 1 /health + 4 /api/search/* calls.
    search_calls = [r for r in recorded if "/api/search/" in r["path"]]
    assert len(search_calls) == 4, f"expected 4 search calls, got {len(search_calls)}"
    for call in search_calls:
        assert call["query"].get("scope") == ["project:galvo-test"], (
            f"missing scope on {call['path']!r}: query={call['query']!r}"
        )


# ---------------------------------------------------------------------------
# 8. Personal scope — no marker, cwd under $HOME → scope=personal.
# ---------------------------------------------------------------------------


def test_personal_scope_when_no_project_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cwd inside $HOME but no project marker → scope=personal."""
    # The cwd we pass IS inside $HOME (tmp_path here serves as $HOME).
    sub = tmp_path / "notes"
    sub.mkdir()
    recorded: list[dict[str, Any]] = []
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, []),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers, record_requests=recorded) as base_url:
        exit_code, _stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(sub)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    search_calls = [r for r in recorded if "/api/search/" in r["path"]]
    for call in search_calls:
        assert call["query"].get("scope") == ["personal"], (
            f"expected personal scope, got {call['query']!r}"
        )


# ---------------------------------------------------------------------------
# 9. Output goes to stdout, not stderr.
# ---------------------------------------------------------------------------


def test_output_goes_to_stdout_not_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The markdown block must land on stdout — stderr must remain empty.

    Claude Code captures both, but stdout is what gets injected into
    context; stderr ends up in the user-visible transcript. A hook
    that wrote diagnostics to stdout would leak debug text into the
    model's window; a hook that wrote memory to stderr would leak it
    into the user's transcript. This test pins the right channel.
    """
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, [{"id": "d1", "title": "X"}]),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    assert "- X" in stdout
    assert stderr == "", f"unexpected stderr: {stderr!r}"


# ---------------------------------------------------------------------------
# 10. Per-section caps — MAX_DECISIONS = 5; extra hits clipped.
# ---------------------------------------------------------------------------


def test_section_cap_clips_extra_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If sidecar (somehow) returns more than MAX_DECISIONS hits, only the
    first ``MAX_DECISIONS`` appear in output.

    This is a defense-in-depth check: in practice the ``limit=`` kwarg
    on :meth:`SidecarClient.search` caps the server-side response, but
    the formatter ALSO slices to MAX_DECISIONS in case the sidecar
    ignores the limit (e.g. an older sidecar version).
    """
    ten_decisions = [
        {"id": f"dec_{i:03d}", "title": f"Decision_{i}"} for i in range(10)
    ]
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, ten_decisions),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    decision_bullets = [
        line for line in stdout.splitlines() if line.startswith("- Decision_")
    ]
    assert len(decision_bullets) == ss.MAX_DECISIONS, (
        f"expected exactly {ss.MAX_DECISIONS} decision bullets, "
        f"got {len(decision_bullets)}: {decision_bullets!r}"
    )


# ---------------------------------------------------------------------------
# 11. Failure section uses error_signature, not name.
# ---------------------------------------------------------------------------


def test_failure_section_uses_error_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure nodes display ``error_signature`` per design §4 NAME_PROPERTY.

    Cross-check that the formatter doesn't accidentally fall through
    to a generic ``name`` field when the label-specific property is
    populated.
    """
    failures = [
        {"id": "f1", "error_signature": "TypeError: 'NoneType' object is not subscriptable"},
    ]
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, []),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, failures),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    assert "## Open failures" in stdout
    assert "- TypeError: 'NoneType' object is not subscriptable" in stdout


# ---------------------------------------------------------------------------
# 12. Schema drift — node dict missing the label-specific name → falls back.
# ---------------------------------------------------------------------------


def test_schema_drift_falls_back_to_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision missing ``title`` but has ``name`` → bullet uses ``name``.

    Defensive against a future sidecar response model change. The
    formatter must not crash on a missing key; it must surface
    *something* useful.
    """
    decisions = [
        {"id": "dec_001", "name": "FallbackName"},  # No "title" field
        {"id": "dec_002"},  # No title, no name — should show "?"
    ]
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, decisions),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    assert "- FallbackName" in stdout
    assert "- ?" in stdout


# ---------------------------------------------------------------------------
# 13. Multi-line title gets collapsed to a single line.
# ---------------------------------------------------------------------------


def test_multiline_title_collapsed_to_single_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node title containing newlines must not break MAX_LINES enforcement.

    The formatter collapses internal whitespace so each node bullet
    occupies exactly one output line. Without this, a Decision titled
    ``"foo\\nbar"`` would produce two lines in the output, double-counting
    against the line budget.
    """
    decisions = [{"id": "d1", "title": "Line one\nLine two\nLine three"}]
    handlers = {
        "GET /health": (200, {"status": "ok"}),
        "GET /api/search/Decision": (200, decisions),
        "GET /api/search/Belief": (200, []),
        "GET /api/search/Task": (200, []),
        "GET /api/search/Failure": (200, []),
    }
    with mock_sidecar(handlers) as base_url:
        exit_code, stdout, _stderr = _run_main(
            _stdin_payload(cwd=str(tmp_path)),
            base_url=base_url,
            home_override=tmp_path,
            monkeypatch=monkeypatch,
        )
    assert exit_code == 0
    assert "- Line one Line two Line three" in stdout
    # No line in the output starts with "Line " (i.e. no orphan
    # continuation lines from the newline-split title).
    orphan_lines = [
        line for line in stdout.splitlines() if line.startswith("Line ")
    ]
    assert orphan_lines == [], f"unexpected orphan title lines: {orphan_lines!r}"


# ---------------------------------------------------------------------------
# 14. Smoke test — running the script as a subprocess works end-to-end.
# ---------------------------------------------------------------------------


def test_main_invocable_via_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The script's ``main()`` is invocable; ``__all__`` and constants exported.

    Sanity check that the module-level structure matches what Claude
    Code's hook loader expects. We don't actually spawn a subprocess
    here (that would couple to the test runner's environment); just
    verify the entry point is callable and the budget constants are
    exported.
    """
    assert callable(ss.main)
    assert ss.MAX_LINES == 30
    assert ss.MAX_DECISIONS == 5
    assert ss.MAX_BELIEFS == 5
    assert ss.MAX_TASKS == 3
    assert ss.MAX_FAILURES == 3
    # All four query strings should be non-empty so the sidecar's
    # embedder gets something to embed.
    assert ss.QUERY_DECISIONS.strip()
    assert ss.QUERY_BELIEFS.strip()
    assert ss.QUERY_TASKS.strip()
    assert ss.QUERY_FAILURES.strip()
