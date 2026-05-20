"""Task 17 — ``memory.cli promote`` unit tests.

Surface protected:

* :func:`fetch_node` — label probing logic. Mocks the sidecar with an
  in-process HTTP server so we can drive the 200/404 cascade and assert
  the right URL was hit. Same pattern as :mod:`test_hook_sidecar_client`.
* :func:`format_for_markdown` — per-label markdown projection. Exercised
  for the five hand-crafted formatters + the generic fallback.
* :func:`insert_into_file` — append / section-insert / section-create
  branches. Uses :func:`tmp_path` fixtures for the file I/O.
* :func:`render_diff` — unified-diff output format.
* :func:`main` — end-to-end argv parsing + dry-run vs write semantics.
* ``--help`` printing without a live sidecar — the CLI must not require
  network reachability to surface usage info.

No live Neo4j / no live sidecar — all HTTP traffic goes through an
in-process loopback server. Tests run anywhere :mod:`memory` is importable.
"""

from __future__ import annotations

import http.server
import json
import pathlib
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import pytest

from cli import promote as promote_mod
from cli.promote import (
    CLIError,
    build_parser,
    fetch_node,
    format_for_markdown,
    insert_into_file,
    main,
    render_diff,
)


# ---------------------------------------------------------------------------
# In-process HTTP server fixture — sidecar stand-in.
# ---------------------------------------------------------------------------


@contextmanager
def mock_sidecar(handler_map: dict[str, tuple[int, Any]]):
    """Spin up an HTTP server that returns canned responses keyed by path.

    Args:
        handler_map: ``{"/api/Convention/conv_abc": (200, {...})}`` — maps
            request path (no method since the CLI only does GET) to a
            ``(status, body)`` tuple. A missing key produces 404 with no body.

    Yields:
        The base URL the CLI should hand to :func:`fetch_node`.

    Modeled after :func:`tests.test_hook_sidecar_client.mock_sidecar` but
    simpler — the promote CLI only does GETs.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            key = urlparse(self.path).path
            entry = handler_map.get(key)
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
    """Yield a URL to a port nothing is listening on (instant connection refused).

    Used to verify :class:`CLIError` is raised when the sidecar is down.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# fetch_node — happy path + 404 cascade + unreachable
# ---------------------------------------------------------------------------


def test_fetch_node_returns_label_and_dict() -> None:
    """Happy path: ``GET /api/Convention/<id>`` returns 200 → label='Convention'.

    Convention is the first label in :data:`PROBE_ORDER`, so this also
    asserts the probe loop short-circuits on first hit (the other 11
    endpoints don't even need to be registered in the mock).
    """
    body = {
        "id": "conv_abc",
        "name": "Use ruff",
        "description": "Run ruff before every commit",
        "source": "explicit",
        "strength": 0.9,
        "scope": "project:galvo",
    }
    with mock_sidecar({"/api/Convention/conv_abc": (200, body)}) as base_url:
        label, node = fetch_node("conv_abc", sidecar_url=base_url)
    assert label == "Convention"
    assert node == body


def test_fetch_node_404_raises_cli_error() -> None:
    """Every label returns 404 → :class:`CLIError` with the node id in the message."""
    # Empty handler_map → server returns 404 to every request.
    with mock_sidecar({}) as base_url:
        with pytest.raises(CLIError) as excinfo:
            fetch_node("node_missing", sidecar_url=base_url)
    assert "node_missing" in str(excinfo.value)
    assert "not found" in str(excinfo.value).lower()


def test_fetch_node_probes_in_priority_order() -> None:
    """The probe loop walks PROBE_ORDER and returns the FIRST match.

    We register only Decision (second in the order) and assert
    ``label == "Decision"``. The first probe (Convention) returns 404,
    second (Decision) returns 200 — the test passes only if the loop
    actually iterates past Convention.
    """
    body = {"id": "dec_001", "title": "Use ruff", "rationale": "consistency"}
    with mock_sidecar({"/api/Decision/dec_001": (200, body)}) as base_url:
        label, _ = fetch_node("dec_001", sidecar_url=base_url)
    assert label == "Decision"


def test_fetch_node_with_label_hint_skips_probe() -> None:
    """When ``label_hint="Belief"`` is passed, the CLI hits ``/api/Belief/<id>``
    directly and does NOT iterate the probe order.

    We register ONLY the Belief endpoint; if the probe loop were to run
    first it would 404 against Convention (the first probe-order entry)
    and miss the hint. The test passes only when the hint short-circuits
    the loop.
    """
    body = {"id": "bel_xyz", "claim": "ruff catches most lint", "confidence": 0.85}
    with mock_sidecar({"/api/Belief/bel_xyz": (200, body)}) as base_url:
        label, node = fetch_node("bel_xyz", sidecar_url=base_url, label_hint="Belief")
    assert label == "Belief"
    assert node["claim"] == "ruff catches most lint"


def test_fetch_node_with_wrong_label_hint_raises() -> None:
    """``label_hint="Decision"`` on an id that's actually a Convention → CLIError.

    Hint mode does NOT fall back to probing — the operator asked for a
    specific label so we honor that and surface the 404 explicitly.
    """
    with mock_sidecar({}) as base_url:
        with pytest.raises(CLIError) as excinfo:
            fetch_node("conv_x", sidecar_url=base_url, label_hint="Decision")
    err = str(excinfo.value)
    assert "Decision" in err
    assert "conv_x" in err


def test_fetch_node_unreachable_sidecar_raises() -> None:
    """Closed port → :class:`CLIError` with hint about ``docker compose up``.

    Operator's most likely failure mode — friendly error message > raw
    :class:`ConnectionRefusedError`.
    """
    with closed_port_url() as base_url:
        with pytest.raises(CLIError) as excinfo:
            fetch_node("conv_x", sidecar_url=base_url, timeout_s=1.0)
    err = str(excinfo.value)
    assert "sidecar unreachable" in err or "unreachable" in err
    assert "7575" in err or "docker compose" in err


def test_fetch_node_500_raises_cli_error() -> None:
    """A 5xx response (not 200 and not 404) → CLIError, not silent skip.

    A 5xx means the sidecar saw the request but failed; the operator
    deserves to know rather than have the CLI silently move on to the
    next label probe (which would also fail and produce a misleading
    "not found" error).
    """
    with mock_sidecar({"/api/Convention/conv_x": (500, None)}) as base_url:
        with pytest.raises(CLIError) as excinfo:
            fetch_node("conv_x", sidecar_url=base_url)
    assert "500" in str(excinfo.value)


# ---------------------------------------------------------------------------
# format_for_markdown — five labels + fallback
# ---------------------------------------------------------------------------


def test_format_for_convention() -> None:
    """Convention → ``## <name>`` + description + source/strength annotation."""
    node = {
        "name": "Use ruff",
        "description": "Run ruff before every commit",
        "source": "explicit",
        "strength": 0.9,
    }
    out = format_for_markdown("Convention", node)
    assert out.startswith("## Use ruff\n")
    assert "Run ruff before every commit" in out
    assert "Source: explicit" in out
    assert "0.90" in out  # strength rendered to 2 dp


def test_format_for_decision() -> None:
    """Decision → ``## Decision: <title>`` + rationale + alternatives block.

    Alternatives list with multiple entries renders as a bullet list.
    """
    node = {
        "title": "Use ruff",
        "rationale": "consistency across team",
        "alternatives_considered": ["black", "flake8", "no linter"],
    }
    out = format_for_markdown("Decision", node)
    assert out.startswith("## Decision: Use ruff\n")
    assert "Rationale: consistency across team" in out
    assert "Alternatives considered:\n" in out
    assert "- black" in out
    assert "- flake8" in out
    assert "- no linter" in out


def test_format_for_decision_single_alternative_inline() -> None:
    """A 0/1-item alternatives list renders inline, not as a bullet.

    A single-item bullet list looks like a typo in markdown — the inline
    form ``Alternatives considered: black`` reads better.
    """
    node = {
        "title": "x",
        "rationale": "y",
        "alternatives_considered": ["black"],
    }
    out = format_for_markdown("Decision", node)
    assert "Alternatives considered: black" in out
    assert "- black" not in out


def test_format_for_belief() -> None:
    """Belief → ``- **<claim>** (held since <valid_from>)`` bullet."""
    node = {
        "claim": "ruff catches most lint issues",
        "valid_from": "2026-05-01T00:00:00",
    }
    out = format_for_markdown("Belief", node)
    assert out.startswith("- **ruff catches most lint issues**")
    assert "held since 2026-05-01T00:00:00" in out
    assert out.endswith("\n")


def test_format_for_belief_falls_back_to_created_at() -> None:
    """When valid_from is missing, fall back to created_at."""
    node = {"claim": "x", "created_at": "2026-04-01"}
    out = format_for_markdown("Belief", node)
    assert "held since 2026-04-01" in out


def test_format_for_pattern() -> None:
    """Pattern → heading + description + evidence/success annotation."""
    node = {
        "name": "Strangler-fig refactor",
        "description": "Wrap legacy code behind a façade and replace it incrementally.",
        "evidence_count": 7,
        "success_rate": 0.86,
    }
    out = format_for_markdown("Pattern", node)
    assert out.startswith("## Pattern: Strangler-fig refactor\n")
    assert "evidence count: 7" in out
    assert "success rate: 86%" in out


def test_format_for_constraint() -> None:
    """Constraint → heading + description + constraint-type annotation."""
    node = {
        "name": "TLS 1.2+ only",
        "description": "Reject TLS 1.0 / 1.1 connections at the edge.",
        "constraint_type": "security",
    }
    out = format_for_markdown("Constraint", node)
    assert out.startswith("## TLS 1.2+ only\n")
    assert "Constraint type: security" in out


def test_format_for_unknown_label_uses_generic() -> None:
    """Labels without a hand-crafted formatter → key/value dump under a heading.

    We use ``Mistake`` (which has no formatter in cycle 1) and assert the
    fallback shape.
    """
    node = {
        "id": "mst_001",
        "summary": "forgot to run migrations",
        "description": "deployed without running alembic upgrade",
        "root_cause": "missed checklist item",
    }
    out = format_for_markdown("Mistake", node)
    assert out.startswith("## Mistake: forgot to run migrations\n")
    assert "- **description**: deployed without running alembic upgrade" in out
    assert "- **root_cause**: missed checklist item" in out
    # id is intentionally skipped from the dump (internal).
    assert "**id**:" not in out


# ---------------------------------------------------------------------------
# insert_into_file — append / section-insert / section-create
# ---------------------------------------------------------------------------


def test_insert_when_no_section_appends(tmp_path: pathlib.Path) -> None:
    """With ``section=None``, content is appended at the end of the file."""
    target = tmp_path / "AGENTS.md"
    target.write_text("# AGENTS\n\nExisting content.\n", encoding="utf-8")

    before, after = insert_into_file(target, "## New Item\n\nBody.\n", section=None)
    assert before == "# AGENTS\n\nExisting content.\n"
    assert after.endswith("## New Item\n\nBody.\n")
    # Original content preserved.
    assert "Existing content." in after


def test_insert_under_existing_section(tmp_path: pathlib.Path) -> None:
    """Section header exists in the file → splice content right after it."""
    target = tmp_path / "CLAUDE.md"
    target.write_text(
        "# Header\n\n## Conventions\n\nFirst convention.\n\n## Other\n\nThing.\n",
        encoding="utf-8",
    )

    block = "## Use ruff\n\nRun ruff.\n"
    before, after = insert_into_file(target, block, section="## Conventions")
    # The block is inserted between the "## Conventions" header and the
    # rest of the conventions section content.
    conv_idx = after.index("## Conventions")
    use_ruff_idx = after.index("## Use ruff")
    first_conv_idx = after.index("First convention")
    other_idx = after.index("## Other")
    assert conv_idx < use_ruff_idx < first_conv_idx < other_idx, (
        "block should land between the section header and existing content"
    )


def test_insert_creates_section_when_missing(tmp_path: pathlib.Path) -> None:
    """Section header NOT in the file → header + content appended at end."""
    target = tmp_path / "AGENTS.md"
    target.write_text("# AGENTS\n\nExisting.\n", encoding="utf-8")

    block = "Body line.\n"
    before, after = insert_into_file(target, block, section="## Conventions")
    assert "## Conventions" in after
    assert "Body line." in after
    # Header appears AFTER the existing content.
    assert after.index("Existing") < after.index("## Conventions")
    assert after.index("## Conventions") < after.index("Body line.")


def test_insert_creates_file_when_missing(tmp_path: pathlib.Path) -> None:
    """Missing target → treated as empty file; ``before`` is the empty string.

    The caller (``main``) is responsible for the actual write — we just
    return ``("", content)`` so the diff shows the file being created.
    """
    target = tmp_path / "does_not_exist.md"
    before, after = insert_into_file(target, "## New\n", section=None)
    assert before == ""
    assert "## New" in after
    # No write side effect from insert_into_file itself.
    assert not target.exists()


# ---------------------------------------------------------------------------
# render_diff
# ---------------------------------------------------------------------------


def test_render_diff_format(tmp_path: pathlib.Path) -> None:
    """The diff uses ``unified_diff`` and embeds the path in the headers."""
    path = tmp_path / "x.md"
    before = "Line A\n"
    after = "Line A\nLine B\n"
    diff = render_diff(before, after, path)
    assert "Line B" in diff
    assert str(path) in diff
    # Conventional unified-diff markers — these are what `git apply` parses.
    assert "---" in diff and "+++" in diff
    assert "+Line B" in diff


def test_render_diff_empty_when_identical(tmp_path: pathlib.Path) -> None:
    """Equal strings → empty diff string (not None, not a header-only block).

    ``main`` short-circuits on the empty case and prints a "no change"
    message instead of writing.
    """
    diff = render_diff("same\n", "same\n", tmp_path / "x.md")
    assert diff == ""


# ---------------------------------------------------------------------------
# main — end-to-end argv + dry-run vs write
# ---------------------------------------------------------------------------


def test_main_dry_run_does_not_write(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` prints the diff and leaves the file untouched.

    The original file content is preserved byte-for-byte.
    """
    target = tmp_path / "AGENTS.md"
    target.write_text("# AGENTS\n\nOriginal.\n", encoding="utf-8")
    original = target.read_text(encoding="utf-8")

    body = {
        "id": "conv_x",
        "name": "Use uv",
        "description": "Use uv for all Python tooling",
        "source": "explicit",
        "strength": 0.8,
        "scope": "project:galvo",
    }
    with mock_sidecar({"/api/Convention/conv_x": (200, body)}) as base_url:
        code = main([
            "conv_x",
            "--to", str(target),
            "--dry-run",
            "--sidecar-url", base_url,
        ])
    assert code == 0
    # File unchanged on disk.
    assert target.read_text(encoding="utf-8") == original
    # Diff content went to stdout.
    captured = capsys.readouterr()
    assert "Use uv" in captured.out


def test_main_writes_file_when_not_dry_run(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--dry-run``, the file content is updated and the diff printed."""
    target = tmp_path / "AGENTS.md"
    target.write_text("# AGENTS\n\nOriginal.\n", encoding="utf-8")

    body = {
        "id": "conv_y",
        "name": "Use ruff",
        "description": "ruff before commit",
        "source": "explicit",
        "strength": 0.7,
        "scope": "project:galvo",
    }
    with mock_sidecar({"/api/Convention/conv_y": (200, body)}) as base_url:
        code = main([
            "conv_y",
            "--to", str(target),
            "--sidecar-url", base_url,
        ])
    assert code == 0
    written = target.read_text(encoding="utf-8")
    assert "Original." in written  # original preserved
    assert "## Use ruff" in written
    assert "ruff before commit" in written


def test_main_creates_target_file_when_missing(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Target file (and parent dir) created on first promote.

    Operators often promote into a fresh memory/feedback_*.md file; the
    CLI must not require the file to pre-exist.
    """
    target = tmp_path / "memory" / "feedback_xyz.md"
    body = {
        "id": "conv_z",
        "name": "New convention",
        "description": "x",
        "source": "explicit",
        "strength": 0.5,
        "scope": "personal",
    }
    with mock_sidecar({"/api/Convention/conv_z": (200, body)}) as base_url:
        code = main([
            "conv_z",
            "--to", str(target),
            "--sidecar-url", base_url,
        ])
    assert code == 0
    assert target.exists()
    assert "New convention" in target.read_text(encoding="utf-8")


def test_main_returns_2_when_node_missing(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All-404 cascade → exit code 2 + stderr message.

    ``main`` translates :class:`CLIError` into a clean error path so the
    operator's shell scripts can branch on the exit code.
    """
    target = tmp_path / "AGENTS.md"
    with mock_sidecar({}) as base_url:
        code = main([
            "nope_id",
            "--to", str(target),
            "--sidecar-url", base_url,
        ])
    assert code == 2
    captured = capsys.readouterr()
    assert "nope_id" in captured.err


def test_main_with_label_hint_skips_probe(
    tmp_path: pathlib.Path,
) -> None:
    """``--label Belief`` hits ``/api/Belief/<id>`` directly."""
    target = tmp_path / "memory" / "feedback_x.md"
    body = {
        "id": "bel_1",
        "claim": "ruff is fast",
        "confidence": 0.9,
        "valid_from": "2026-04-01",
        "scope": "project:galvo",
    }
    with mock_sidecar({"/api/Belief/bel_1": (200, body)}) as base_url:
        code = main([
            "bel_1",
            "--to", str(target),
            "--label", "Belief",
            "--sidecar-url", base_url,
        ])
    assert code == 0
    assert "ruff is fast" in target.read_text(encoding="utf-8")


def test_main_with_section_inserts_under_header(
    tmp_path: pathlib.Path,
) -> None:
    """``--section`` splices under an existing header."""
    target = tmp_path / "CLAUDE.md"
    target.write_text(
        "# CLAUDE\n\n## Conventions\n\nExisting convention.\n",
        encoding="utf-8",
    )
    body = {
        "id": "conv_q",
        "name": "Always uv",
        "description": "use uv for python",
        "source": "explicit",
        "strength": 0.7,
        "scope": "project:galvo",
    }
    with mock_sidecar({"/api/Convention/conv_q": (200, body)}) as base_url:
        code = main([
            "conv_q",
            "--to", str(target),
            "--section", "## Conventions",
            "--sidecar-url", base_url,
        ])
    assert code == 0
    written = target.read_text(encoding="utf-8")
    # The new section's heading appears after "## Conventions" and
    # before "Existing convention.".
    assert written.index("## Conventions") < written.index("## Always uv")
    assert written.index("## Always uv") < written.index("Existing convention.")


# ---------------------------------------------------------------------------
# --help works without a sidecar
# ---------------------------------------------------------------------------


def test_help_runs_without_sidecar(capsys: pytest.CaptureFixture[str]) -> None:
    """``promote --help`` must NOT make any HTTP calls.

    Argparse prints help and raises SystemExit(0). We assert the help
    body mentions the key flags so a future refactor doesn't drop them.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--to" in out
    assert "--section" in out
    assert "--dry-run" in out
    assert "--label" in out
    assert "--sidecar-url" in out


def test_help_via_subprocess() -> None:
    """``python -m memory.cli promote --help`` runs cleanly as a subprocess.

    Belt-and-braces — :func:`main` works when called in-process, but the
    realistic invocation form is the module run line. We spawn it via
    subprocess to catch packaging mistakes (missing ``__main__.py``,
    typos in :data:`_SUBCOMMANDS`).
    """
    memory_root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "cli", "promote", "--help"],
        cwd=memory_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"--help should exit 0; got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "--to" in result.stdout


def test_dispatcher_no_args_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m memory.cli`` with no subcommand → exit 1 + usage on stderr."""
    from cli.__main__ import main as dispatcher_main

    code = dispatcher_main([])
    assert code == 1
    err = capsys.readouterr().err
    assert "usage" in err.lower()
    assert "promote" in err


def test_dispatcher_unknown_subcommand_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown subcommand → exit 1 + usage. Catches typos like ``promotes``."""
    from cli.__main__ import main as dispatcher_main

    code = dispatcher_main(["promotes"])
    assert code == 1
    err = capsys.readouterr().err
    assert "unknown" in err.lower()
    assert "promote" in err


def test_dispatcher_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m memory.cli --help`` → exit 0 + usage on stdout (not stderr)."""
    from cli.__main__ import main as dispatcher_main

    code = dispatcher_main(["--help"])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()
    assert "promote" in out


# ---------------------------------------------------------------------------
# Parser-level
# ---------------------------------------------------------------------------


def test_build_parser_requires_node_id_and_to() -> None:
    """Argparse rejects calls missing ``node_id`` or ``--to``.

    Belt-and-braces — protects against a refactor dropping the
    required-ness of either field.
    """
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing both
    with pytest.raises(SystemExit):
        parser.parse_args(["node_x"])  # missing --to


def test_build_parser_label_choices_restricted() -> None:
    """``--label`` only accepts one of the 12 ontology labels.

    A typo like ``--label decision`` (lowercase) is rejected by argparse,
    saving an embarrassing 404 from the sidecar.
    """
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["node_x", "--to", "x.md", "--label", "decision"])
    # Valid case parses cleanly.
    args = parser.parse_args(["node_x", "--to", "x.md", "--label", "Decision"])
    assert args.label == "Decision"


# ---------------------------------------------------------------------------
# Constants — guard against accidental changes
# ---------------------------------------------------------------------------


def test_probe_order_covers_all_12_labels() -> None:
    """:data:`PROBE_ORDER` covers all 12 ontology labels.

    Without this, adding a new label to the ontology would silently
    omit it from the probe loop and the CLI would 404 on the new label.
    """
    from ontology.label_mapping import LABEL_TO_TYPE

    assert set(promote_mod.PROBE_ORDER) == set(LABEL_TO_TYPE)
