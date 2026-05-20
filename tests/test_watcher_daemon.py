"""Task 16 — file watcher daemon dispatch + ingest tests.

What we're protecting
=====================

The :func:`memory.watcher.daemon.ingest_file` function is the
load-bearing piece of the watcher: it takes a path + scope + HTTP
callable and turns the file contents into ``POST /api/Convention``
calls. Watchdog itself is exercised indirectly (the
``--once`` mode + initial-pass scan use the same dispatch path) but
we don't spin up an Observer in these tests — the runtime is
single-threaded and that keeps the suite fast + deterministic.

We exercise:

* **POST shape.** ``ConventionCreate`` field mapping:
  ``title → name``, ``rule_text → description``,
  source-tag → ``"from_<tag>"``.
* **Scope tagging.** Every POSTed body carries the scope string the
  daemon resolved for the file.
* **Diff detection.** When a ``search`` callable returns a hit with
  the same normalized title + source, the result records a
  SUPERSEDES intent (cycle 1 still posts; cycle-2 will write the
  edge). When ``search`` is ``None``, every parse becomes a fresh
  POST.
* **Failure handling.** Sidecar down (POST → ``None``) → the
  convention lands in ``result.failed``; nothing raises.
* **ProjectRegistry scope routing.** Files under a registered root
  use that root's scope; files outside any registered root fall
  through to :func:`detect_scope` and ultimately ``"personal"``.
* **scan_paths_for_instruction_files.** Walks a tree and returns
  exactly the AGENTS.md / CLAUDE.md / .cursorrules / .codex/*.md
  files inside.
* **CLI parsing.** ``_parse_args`` honors positional paths,
  ``--sidecar-url``, ``--once``, ``--scope``.
* **Watchdog-missing path.** :func:`main` returns 1 with a helpful
  message when ``watchdog`` import fails (simulated via sys.modules).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from scope import write_marker
from watcher import daemon as watcher_daemon
from watcher.daemon import (
    ProjectRegistry,
    RetryEntry,
    RetryQueue,
    _convention_to_create_body,
    _normalize_title,
    _parse_args,
    ingest_file,
    main,
    run_once,
    scan_paths_for_instruction_files,
)
from watcher.parsers import ParsedConvention


# ---------------------------------------------------------------------------
# In-memory POST recorder
# ---------------------------------------------------------------------------


class _PostRecorder:
    """Configurable POST callable for tests.

    Mirrors the production :func:`default_post_callable` contract:
    every call records the (path, body) and returns either the
    configured response or ``None`` (failure).
    """

    def __init__(self, default_response: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.default_response = default_response
        self.fail_indices: set[int] = set()
        """Calls at these indices will return ``None`` regardless of the default."""

    def __call__(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        idx = len(self.calls)
        self.calls.append((path, body))
        if idx in self.fail_indices:
            return None
        return self.default_response


class _SearchRecorder:
    """Configurable search callable.

    ``hits_by_query`` is keyed by the literal ``query`` string;
    everything else returns ``[]``. The recorder also captures the
    actual call args so the test can assert on what the daemon
    searched for.
    """

    def __init__(self, hits_by_query: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.hits_by_query = hits_by_query or {}

    def __call__(self, label: str, query: str, scope: str | None) -> list[dict[str, Any]]:
        self.calls.append((label, query, scope))
        return list(self.hits_by_query.get(query, []))


# ---------------------------------------------------------------------------
# 1 — happy path: parse + POST with scope
# ---------------------------------------------------------------------------


def test_ingest_file_posts_convention_with_scope(tmp_path: Path) -> None:
    """A two-section AGENTS.md → two POSTs to /api/Convention with scope tag.

    Field shape: ``name`` (from title), ``description`` (from
    rule_text), ``source`` (``from_AGENTS.md`` per design §4),
    ``strength`` (0.8), ``scope`` (the passed-in value).
    """
    p = tmp_path / "AGENTS.md"
    p.write_text(
        "# Title\n"
        "\n"
        "## Use ruff\n"
        "Run ruff format before committing.\n"
        "\n"
        "## Pin deps\n"
        "Floor every dependency with `>=`.\n"
    )
    recorder = _PostRecorder(default_response={"id": "node_abc"})
    result = ingest_file(p, scope="project:galvo", post=recorder, search=None)

    assert len(result.parsed) == 2
    assert len(result.posted) == 2
    assert result.failed == []
    # Two POSTs to /api/Convention, both with the scope tag.
    paths = {call[0] for call in recorder.calls}
    assert paths == {"/api/Convention"}
    bodies = [call[1] for call in recorder.calls]
    titles = sorted(b["name"] for b in bodies)
    assert titles == ["Pin deps", "Use ruff"]
    for body in bodies:
        assert body["scope"] == "project:galvo"
        assert body["source"] == "from_AGENTS.md"
        assert body["strength"] == 0.8
        assert isinstance(body["description"], str) and body["description"]


# ---------------------------------------------------------------------------
# 2 — POST failure → result.failed records the convention
# ---------------------------------------------------------------------------


def test_ingest_file_records_post_failure(tmp_path: Path) -> None:
    """A sidecar-down POST (callable returns None) lands in ``result.failed``.

    Crucial because the daemon's main loop pushes failed records onto
    the :class:`RetryQueue`; if they don't show up in ``result.failed``
    the retry path is silently broken.
    """
    p = tmp_path / "CLAUDE.md"
    p.write_text("## Locked rule\nbody\n")
    recorder = _PostRecorder(default_response=None)  # every call fails
    result = ingest_file(p, scope="personal", post=recorder, search=None)
    assert len(result.parsed) == 1
    assert result.posted == []
    assert len(result.failed) == 1
    assert result.failed[0].title == "Locked rule"


# ---------------------------------------------------------------------------
# 3 — diff detection: search hit with same title → SUPERSEDES intent
# ---------------------------------------------------------------------------


def test_ingest_file_logs_supersedes_intent_when_search_hits(tmp_path: Path) -> None:
    """A search hit with matching normalized title + source records a
    ``SUPERSEDES`` intent (cycle 1 still POSTs).

    The graph still gets the new node — cycle 1 has no SUPERSEDES
    writer endpoint, so the intent is a log-only marker for cycle-2 to
    consume. The mere presence of the entry in
    ``result.supersedes_intents`` proves the diff detector ran.
    """
    p = tmp_path / "AGENTS.md"
    p.write_text("## Use ruff for formatting\nbody\n")
    post = _PostRecorder(default_response={"id": "node_new"})
    search = _SearchRecorder(
        hits_by_query={
            "Use ruff for formatting": [
                {
                    "id": "node_old",
                    "name": "Use ruff for formatting",
                    "source": "from_AGENTS.md",
                    "scope": "project:galvo",
                }
            ]
        }
    )
    result = ingest_file(
        p, scope="project:galvo", post=post, search=search
    )
    assert len(result.posted) == 1
    assert len(result.supersedes_intents) == 1
    intent_convention, intent_id = result.supersedes_intents[0]
    assert intent_convention.title == "Use ruff for formatting"
    assert intent_id == "node_old"


# ---------------------------------------------------------------------------
# 4 — diff detection: hit with different source NOT counted as supersede
# ---------------------------------------------------------------------------


def test_ingest_file_does_not_supersede_across_sources(tmp_path: Path) -> None:
    """A search hit with the same title but a different ``source`` tag
    is NOT a supersede candidate.

    A rule with the same name in AGENTS.md vs .cursorrules is two
    distinct conventions — the user authored them in different files
    on purpose.
    """
    p = tmp_path / "AGENTS.md"
    p.write_text("## Pin deps\nbody\n")
    post = _PostRecorder(default_response={"id": "node_new"})
    search = _SearchRecorder(
        hits_by_query={
            "Pin deps": [
                {
                    "id": "node_cursor",
                    "name": "Pin deps",
                    "source": "from_.cursorrules",  # different file format
                    "scope": "project:galvo",
                }
            ]
        }
    )
    result = ingest_file(
        p, scope="project:galvo", post=post, search=search
    )
    assert len(result.posted) == 1
    assert result.supersedes_intents == []


# ---------------------------------------------------------------------------
# 5 — search disabled (None) → no diff queries; every parse is fresh
# ---------------------------------------------------------------------------


def test_ingest_file_no_search_means_no_diff(tmp_path: Path) -> None:
    """With ``search=None`` the daemon doesn't try to detect duplicates.

    This is the cycle-1 daemon default at the main entry point —
    the diff is exercised in unit tests but the default is "ingest
    everything", because the SUPERSEDES writer doesn't exist yet so
    diff results aren't actionable.
    """
    p = tmp_path / "AGENTS.md"
    p.write_text("## A\nbody A\n\n## B\nbody B\n")
    post = _PostRecorder(default_response={"id": "node_new"})
    result = ingest_file(p, scope="personal", post=post, search=None)
    assert len(result.posted) == 2
    assert result.supersedes_intents == []


# ---------------------------------------------------------------------------
# 6 — empty / unparseable file → no POSTs
# ---------------------------------------------------------------------------


def test_ingest_file_empty_or_unparseable_no_posts(tmp_path: Path) -> None:
    """A file with no parsed sections → zero sidecar calls.

    Defensive: a parse failure (returns ``[]``) must not flood the
    sidecar with placeholder rows.
    """
    p = tmp_path / "AGENTS.md"
    p.write_text("Just prose, no headings.\n")
    post = _PostRecorder(default_response={"id": "node_new"})
    result = ingest_file(p, scope="personal", post=post, search=None)
    assert result.parsed == []
    assert result.posted == []
    assert post.calls == []


# ---------------------------------------------------------------------------
# 7 — search callable that raises is swallowed, treated as no hits
# ---------------------------------------------------------------------------


def test_ingest_file_swallows_search_exceptions(tmp_path: Path) -> None:
    """A buggy search callable must not poison the ingest path."""
    p = tmp_path / "AGENTS.md"
    p.write_text("## Rule\nbody\n")
    post = _PostRecorder(default_response={"id": "node_new"})

    def _broken_search(label: str, query: str, scope: str | None) -> list[dict[str, Any]]:
        raise RuntimeError("search blew up")

    result = ingest_file(p, scope="personal", post=post, search=_broken_search)
    assert len(result.posted) == 1
    assert result.supersedes_intents == []


# ---------------------------------------------------------------------------
# 8 — empty rule body → description gets a fallback
# ---------------------------------------------------------------------------


def test_ingest_file_handles_empty_body(tmp_path: Path) -> None:
    """A ``##`` heading with no body still POSTs (Pydantic
    ``min_length=1`` on description requires a non-empty placeholder).
    """
    p = tmp_path / "AGENTS.md"
    p.write_text("## Empty rule\n\n## Next rule\nbody\n")
    post = _PostRecorder(default_response={"id": "node_new"})
    result = ingest_file(p, scope="personal", post=post, search=None)
    assert len(result.posted) == 2
    bodies = {b["name"]: b for b in (call[1] for call in post.calls)}
    assert "Empty rule" in bodies
    # Synthesized fallback — non-empty, mentions the title.
    assert "Empty rule" in bodies["Empty rule"]["description"]


# ---------------------------------------------------------------------------
# 9 — ProjectRegistry.scope_for: marker resolution + nesting
# ---------------------------------------------------------------------------


def test_project_registry_scope_for_uses_registered_root(tmp_path: Path) -> None:
    """A file under a registered root inherits that root's scope.

    Doesn't fall through to detect_scope when an explicit entry
    covers the file — the registry is the authoritative source for
    daemon-level scope assignment.
    """
    root = tmp_path / "project_root"
    root.mkdir()
    (root / "sub" / "deeper").mkdir(parents=True)
    target = root / "sub" / "deeper" / "AGENTS.md"
    target.write_text("## R\nbody\n")
    registry = ProjectRegistry()
    registry.add(root, scope="project:galvo-test")
    assert registry.scope_for(target) == "project:galvo-test"


def test_project_registry_scope_for_picks_most_specific(tmp_path: Path) -> None:
    """When multiple registered roots cover a file, the longest root wins.

    Important for repos with sub-projects: registering ``/repo`` and
    ``/repo/sub`` separately should route ``/repo/sub/AGENTS.md`` to
    the sub scope.
    """
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    target = inner / "AGENTS.md"
    target.write_text("## R\nbody\n")
    registry = ProjectRegistry()
    registry.add(outer, scope="project:outer")
    registry.add(inner, scope="project:inner")
    assert registry.scope_for(target) == "project:inner"


def test_project_registry_add_resolves_scope_from_marker(tmp_path: Path) -> None:
    """``ProjectRegistry.add(root)`` without explicit scope calls
    :func:`detect_scope`, which uses the ``.galvo-mem/project.toml``
    marker if present.
    """
    root = tmp_path / "project"
    root.mkdir()
    write_marker(root, project_id="galvo-test", name="Galvo Test")
    registry = ProjectRegistry()
    entry = registry.add(root)  # scope=None — should auto-detect
    assert entry.scope == "project:galvo-test"


# ---------------------------------------------------------------------------
# 10 — scan_paths_for_instruction_files: walk a tree, find the right files
# ---------------------------------------------------------------------------


def test_scan_paths_finds_all_supported_files(tmp_path: Path) -> None:
    """The scanner picks up AGENTS.md / CLAUDE.md / .cursorrules /
    .codex/*.md anywhere in the tree, and nothing else."""
    (tmp_path / "AGENTS.md").write_text("# label\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "CLAUDE.md").write_text("# label\n")
    (tmp_path / "sub" / ".cursorrules").write_text("rule\n")
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "rules.md").write_text("# label\n")
    (tmp_path / "noise.md").write_text("# random\n")  # ignored
    (tmp_path / "other.txt").write_text("nope\n")  # ignored

    found = scan_paths_for_instruction_files([tmp_path])
    names = sorted(p.name for p in found)
    assert names == ["AGENTS.md", "CLAUDE.md", ".cursorrules", "rules.md"][::1] or names == [
        ".cursorrules",
        "AGENTS.md",
        "CLAUDE.md",
        "rules.md",
    ]


# ---------------------------------------------------------------------------
# 11 — run_once: scope routing + filter to instruction files
# ---------------------------------------------------------------------------


def test_run_once_filters_non_instruction_files(tmp_path: Path) -> None:
    """``run_once`` skips paths that aren't instruction files.

    The scanner's responsibility is to find them; ``run_once`` is the
    defense-in-depth filter so a hand-built path list doesn't smuggle
    a random file into the ingest pipeline.
    """
    agents = tmp_path / "AGENTS.md"
    agents.write_text("## R\nbody\n")
    junk = tmp_path / "junk.txt"
    junk.write_text("nope\n")
    registry = ProjectRegistry()
    registry.add(tmp_path, scope="personal")
    post = _PostRecorder(default_response={"id": "node_new"})
    results = run_once(
        [agents, junk], registry=registry, post=post, search=None
    )
    assert len(results) == 1
    assert results[0].path == agents
    assert len(post.calls) == 1


# ---------------------------------------------------------------------------
# 12 — convention → create-body projection
# ---------------------------------------------------------------------------


def test_convention_to_create_body_field_map() -> None:
    """The projection is the canonical contract between parser + sidecar.

    Lock it down at the unit level so a refactor doesn't silently
    rename a field on the wire.
    """
    convention = ParsedConvention(
        title="Use ruff",
        rule_text="Run ruff format before commit.",
        source="AGENTS.md",
        applies_to=["*.py"],
        examples=["ruff format ."],
        file_path="/repo/AGENTS.md",
    )
    body = _convention_to_create_body(convention, scope="project:galvo")
    assert body == {
        "name": "Use ruff",
        "description": "Run ruff format before commit.",
        "source": "from_AGENTS.md",
        "strength": 0.8,
        "scope": "project:galvo",
    }


# ---------------------------------------------------------------------------
# 13 — title normalization
# ---------------------------------------------------------------------------


def test_normalize_title_case_and_whitespace() -> None:
    """Diff key normalizer: case-fold + collapse whitespace.

    No stemming / punctuation stripping — over-normalizing leads to
    false merges that look like data loss to a user inspecting the
    graph.
    """
    assert _normalize_title("Use Ruff For Formatting") == "use ruff for formatting"
    assert _normalize_title("  use\truff  for\nformatting  ") == "use ruff for formatting"
    # Punctuation is preserved.
    assert _normalize_title("Use ruff!") == "use ruff!"


# ---------------------------------------------------------------------------
# 14 — RetryQueue: bounded FIFO behavior
# ---------------------------------------------------------------------------


def test_retry_queue_drops_oldest_when_full() -> None:
    """At capacity, the head is evicted on push (not the tail).

    Preserves the most-recent failures because those are most likely
    to still be relevant when the sidecar comes back up.
    """
    q = RetryQueue(capacity=2)
    conv = lambda title: ParsedConvention(  # noqa: E731 — concise factory
        title=title,
        rule_text="x",
        source="AGENTS.md",
        applies_to=[],
        examples=[],
        file_path="/x",
    )
    q.push(RetryEntry(convention=conv("a"), scope="s"))
    q.push(RetryEntry(convention=conv("b"), scope="s"))
    q.push(RetryEntry(convention=conv("c"), scope="s"))  # evicts "a"
    drained = q.drain()
    titles = [r.convention.title for r in drained]
    assert titles == ["b", "c"]
    assert len(q) == 0


# ---------------------------------------------------------------------------
# 15 — CLI arg parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_cwd_via_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """No positional args + no flags → the daemon would watch cwd.

    We test the parser directly so we don't have to spin up watchdog.
    """
    args = _parse_args([])
    assert args.paths == []
    assert args.once is False
    assert args.scope is None
    assert args.sidecar_url == "http://localhost:7575"


def test_parse_args_honors_flags(tmp_path: Path) -> None:
    """All four flags wire through to the Namespace."""
    args = _parse_args(
        [
            str(tmp_path),
            "--sidecar-url",
            "http://example:9999",
            "--once",
            "--scope",
            "personal",
        ]
    )
    assert args.paths == [tmp_path]
    assert args.sidecar_url == "http://example:9999"
    assert args.once is True
    assert args.scope == "personal"


def test_parse_args_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GALVO_MEMORY_SIDECAR_URL`` provides the default when
    ``--sidecar-url`` is absent.

    Re-import the module so the env var is re-read at argparse
    construction (the default is captured at parser-build time).
    """
    monkeypatch.setenv("GALVO_MEMORY_SIDECAR_URL", "http://env-set:1234")
    # _parse_args grabs the env at call time via os.environ.get default.
    import importlib

    importlib.reload(watcher_daemon)
    try:
        args = watcher_daemon._parse_args([])
        assert args.sidecar_url == "http://env-set:1234"
    finally:
        # Restore module-level state for the rest of the suite.
        monkeypatch.delenv("GALVO_MEMORY_SIDECAR_URL", raising=False)
        importlib.reload(watcher_daemon)


# ---------------------------------------------------------------------------
# 16 — main exits 1 with friendly message when watchdog missing
# ---------------------------------------------------------------------------


def test_main_exits_1_when_watchdog_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The watchdog import inside :func:`main` fails → exit 1 with hint.

    Simulated by inserting a hook into ``sys.meta_path`` that raises
    ``ImportError`` for any ``watchdog`` lookup. We arrange the test
    so the initial scan finds nothing (the daemon enters the
    watchdog branch immediately) and then the import fails.
    """
    # Make sure any cached watchdog (in case the optional extra ever
    # gets installed) gets re-imported through the hook.
    for name in list(sys.modules):
        if name == "watchdog" or name.startswith("watchdog."):
            del sys.modules[name]

    class _BlockWatchdog:
        def find_module(self, name: str, path: object = None) -> object | None:
            if name == "watchdog" or name.startswith("watchdog."):
                return self
            return None

        def find_spec(self, name: str, path: object = None, target: object = None) -> object | None:
            if name == "watchdog" or name.startswith("watchdog."):
                # Returning a spec-like object would let the import
                # proceed; we want it to fail. Raising here mimics the
                # "module not installed" state.
                raise ImportError(f"blocked: {name}")
            return None

        def load_module(self, name: str) -> object:
            raise ImportError(f"blocked: {name}")

    monkeypatch.setattr(sys, "meta_path", [_BlockWatchdog(), *sys.meta_path])
    # Redirect $HOME so the logger doesn't litter the real user dir.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        watcher_daemon, "LOG_DIR", tmp_path / ".galvo-memory" / "logs"
    )

    # Stub out the post / search callables so the initial scan
    # doesn't try to reach a real sidecar. We also point the watcher
    # at a tmp_path with no instruction files, so the initial scan
    # is trivially empty.
    monkeypatch.setattr(
        watcher_daemon,
        "default_post_callable",
        lambda *a, **kw: lambda path, body: None,
    )
    monkeypatch.setattr(
        watcher_daemon,
        "default_search_callable",
        lambda *a, **kw: lambda label, query, scope: [],
    )

    rc = main([str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "watchdog" in captured.err
    assert "pip install" in captured.err
