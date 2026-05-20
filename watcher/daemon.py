"""File watcher daemon — turns instruction-file changes into Convention upserts (Task 16).

The daemon process watches one or more project trees for changes to
known instruction files (``AGENTS.md`` / ``CLAUDE.md`` / ``.cursorrules``
/ ``.codex/*.md``). When a file changes:

1. Parse the file via :mod:`memory.watcher.parsers`.
2. For each parsed convention, search the sidecar for an existing
   Convention with the same title in the same scope (diff detection).
3. POST new conventions; log SUPERSEDES intent for ones that look like
   updates of existing nodes (cycle-2 will write the edge once the
   sidecar exposes an edge-writer endpoint — cycle 1 just stamps the
   intent in the description metadata + log).

Design constraints
==================

* **Non-canonical inputs.** Per design §10, files feed knowledge INTO
  the graph; the graph is the source of truth. We never delete graph
  nodes when a file is edited — old conventions are superseded, not
  removed. A user who wants to delete a rule does it through the graph
  CLI (Task 17 promote-action is the read side; cycle-2 will add a
  retire-action for the write side).
* **Best-effort.** Sidecar down → log + queue for retry. Bad parse →
  skip the file. Permission denied → drop the directory from the
  registry. Never blocks; never raises.
* **Opt-in.** This module imports ``watchdog`` lazily inside :func:`main`
  so the parsers + ``ingest_file`` can be tested without the watcher
  extra installed. Calling ``python -m memory.watcher`` without
  watchdog prints a friendly install hint and exits 1.
* **No daemonization.** We run as a foreground process the operator
  launches via tmux / systemd / launchd. The Unix-daemon double-fork
  dance is out of scope for cycle 1.

HTTP contract reminder
======================

The sidecar's ``ConventionCreate`` model uses ``name`` / ``description``
(not ``title`` / ``rule_text``). :func:`ingest_file` does the projection
at the POST boundary. The ``source`` field on the wire is the
design-§4 enum value (``"from_AGENTS.md"`` etc.); the parser produces
the short tag and :func:`memory.watcher.parsers.format_to_source_tag`
maps it to the enum.

The diff endpoint is ``GET /api/search/Convention?q=<title>&scope=<scope>&limit=10``.
We compare by title (case-folded, whitespace-normalized) because
descriptions tend to drift across edits while titles are stable for
the same rule. A hit with identical normalized title → existing
convention; we POST a new one anyway (since cycle 1 has no SUPERSEDES
writer) but log the intent so a human auditing the graph can manually
clean up.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error as _urlerr
from urllib import request as _urlreq
from urllib.parse import urlencode

from scope.detector import detect_scope

from .parsers import ParsedConvention, format_to_source_tag, parse_file

__all__ = [
    "DEBOUNCE_SECONDS",
    "DEFAULT_SIDECAR_URL",
    "LOG_DIR",
    "WATCH_FILENAMES",
    "PostResult",
    "ProjectRegistry",
    "RetryQueue",
    "default_post_callable",
    "default_search_callable",
    "ingest_file",
    "main",
    "run_once",
    "scan_paths_for_instruction_files",
    "setup_watcher_logger",
]


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------


DEFAULT_SIDECAR_URL: str = "http://localhost:7575"
"""Sidecar base URL. Matches the hook client's default — operators
running the sidecar in Docker on a non-loopback interface override via
:func:`main` ``--sidecar-url`` or the env var ``GALVO_MEMORY_SIDECAR_URL``."""

DEBOUNCE_SECONDS: float = 1.5
"""Wait this long after the last write event before parsing a file.

Tools like editor save-then-rename produce a flurry of fs events for a
single human-meaningful save; the debounce coalesces them so we parse
the file once."""

WATCH_FILENAMES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md", ".cursorrules")
"""Filenames matched at any depth in a watched tree. ``.codex/*.md``
files are matched separately by the ``.codex`` parent-dir rule inside
:func:`_is_instruction_file`."""

DEFAULT_TIMEOUT_S: float = 5.0
"""Per-request wall-clock cap for the watcher's sidecar calls.

Looser than the hook client's 3.0s because the watcher isn't blocking
a user session — it can afford to wait a bit longer for a busy
sidecar. Still bounded so the daemon never wedges."""

LOG_DIR: Path = Path.home() / ".galvo-memory" / "logs"
"""Where the watcher writes its diagnostic log (mirrors the hook log
location so an operator inspecting hook activity finds the watcher
events in the same directory)."""

RETRY_QUEUE_CAPACITY: int = 50
"""Maximum pending retries when the sidecar is unreachable. Once full,
the oldest entry is dropped — better to lose the stalest pending
ingest than to grow the queue without bound."""


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project registry — what we watch, in what scope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectEntry:
    """One project in the watcher's registry.

    Multiple entries with the same ``root`` are supported (e.g. a tree
    that has both a project marker AND a personal-scope sub-tree); the
    daemon handles each on its own scope.
    """

    root: Path
    """Absolute path to the directory to watch."""

    scope: str
    """The scope string to tag created Conventions with."""


class ProjectRegistry:
    """In-memory list of project roots + their scopes.

    The daemon resolves the scope per file when it processes a change,
    so the registry only needs to know the *roots*. We keep the
    explicit scope as a fallback for trees that have no
    ``.galvo-mem/project.toml`` (a personal-scope file outside any
    project tree, e.g. ``~/AGENTS.md``).
    """

    def __init__(self, entries: Iterable[ProjectEntry] | None = None) -> None:
        self._entries: list[ProjectEntry] = list(entries or [])

    def add(self, root: Path, scope: str | None = None) -> ProjectEntry:
        """Register ``root``, resolving scope on the fly if not given.

        Args:
            root: Directory to watch. Resolved on entry so symlinks
                don't shadow lookups later.
            scope: Optional explicit scope. When ``None`` (default), we
                run :func:`memory.scope.detector.detect_scope` against
                the root.

        Returns:
            The :class:`ProjectEntry` added. Duplicate adds are allowed
            — the daemon's dedup happens at the file-event layer, not
            the registry.
        """
        resolved_root = root.resolve()
        if scope is None:
            scope = detect_scope(resolved_root)
        entry = ProjectEntry(root=resolved_root, scope=scope)
        self._entries.append(entry)
        return entry

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[ProjectEntry]:
        """Read-only view of the registered entries (returns a copy)."""
        return list(self._entries)

    def scope_for(self, path: Path) -> str:
        """Find the scope that should tag a change at ``path``.

        Lookup order:

        1. Find the registered entry whose root is an ancestor of
           ``path``. If multiple entries match, pick the longest root
           (most-specific wins).
        2. Fall back to :func:`detect_scope` on ``path.parent``.
        3. Final fallback: ``"personal"``.
        """
        resolved = path.resolve()
        best: ProjectEntry | None = None
        best_len = -1
        for entry in self._entries:
            try:
                resolved.relative_to(entry.root)
            except ValueError:
                continue
            length = len(entry.root.parts)
            if length > best_len:
                best_len = length
                best = entry
        if best is not None:
            return best.scope
        # Fall through — try direct detection from the file's parent dir.
        try:
            return detect_scope(resolved.parent)
        except Exception:  # noqa: BLE001 — never raise from registry
            return "personal"


def _is_instruction_file(path: Path) -> bool:
    """``True`` iff ``path`` is one of the known instruction-file names.

    Used by both the recursive scanner and the runtime fs-event filter.
    Centralized so the two paths can never drift.
    """
    name = path.name
    if name in WATCH_FILENAMES:
        return True
    if path.parent.name == ".codex" and path.suffix == ".md":
        return True
    return False


def scan_paths_for_instruction_files(roots: Iterable[Path]) -> list[Path]:
    """Walk ``roots`` and return every existing instruction file inside.

    Used at daemon startup to do a first-pass ingest before the
    watchdog observer starts firing events. Without this initial pass,
    the daemon would miss the existing on-disk state.

    Symlinks are followed at the top level (so a registered root that
    IS a symlink works) but not traversed into (so we don't recurse
    into ``node_modules`` symlinks that some monorepo layouts use).
    """
    found: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if not resolved.is_dir():
            continue
        for path in resolved.rglob("*"):
            try:
                if path.is_file() and _is_instruction_file(path):
                    found.append(path)
            except (PermissionError, OSError) as exc:
                _log.warning(
                    "scan_paths: cannot stat %s: %r — skipping", path, exc
                )
                continue
    return found


# ---------------------------------------------------------------------------
# Retry queue — stash failures while the sidecar is unreachable
# ---------------------------------------------------------------------------


@dataclass
class RetryEntry:
    """One pending retry — a parsed convention + the scope it was tagged with."""

    convention: ParsedConvention
    scope: str
    attempts: int = 0
    """Number of POST attempts so far. The daemon does NOT cap retries
    in cycle 1 — a sidecar that's down for hours catches up when it
    comes back. Cycle-2 may add an exponential backoff."""


class RetryQueue:
    """Bounded FIFO of pending convention POSTs.

    ``RetryQueue`` is the recovery surface for sidecar-down events:
    every failed POST gets appended here, and :meth:`drain` retries
    them on the next file change (or when the operator explicitly
    triggers a re-scan).

    Capacity is :data:`RETRY_QUEUE_CAPACITY`; once full, the oldest
    entry is evicted. This bound matters because a long sidecar outage
    on a chatty repo could otherwise OOM the daemon — a finite drop
    rate at the head is preferable to unbounded growth.
    """

    def __init__(self, capacity: int = RETRY_QUEUE_CAPACITY) -> None:
        self._buf: deque[RetryEntry] = deque(maxlen=capacity)

    def push(self, entry: RetryEntry) -> None:
        """Append ``entry`` — evicts the head if at capacity."""
        self._buf.append(entry)

    def drain(self) -> list[RetryEntry]:
        """Return and clear the current queue contents.

        Caller is responsible for re-pushing failures encountered while
        retrying — the queue itself doesn't track which entries
        succeeded.
        """
        items = list(self._buf)
        self._buf.clear()
        return items

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# HTTP callable types
# ---------------------------------------------------------------------------


PostCallable = Callable[[str, dict[str, Any]], dict[str, Any] | None]
"""Function that POSTs JSON to ``<sidecar>/{path}`` and returns the
decoded response (or ``None`` on failure). Threaded through
:func:`ingest_file` so tests can supply an in-memory recorder."""

SearchCallable = Callable[[str, str, str | None], list[dict[str, Any]]]
"""Function ``(label, query, scope) -> list[dict]`` that hits the
sidecar's ``/api/search/{label}`` endpoint. Returns an empty list on
failure (matches the sidecar's contract — no hits ≡ "search failed")."""


def default_post_callable(
    base_url: str = DEFAULT_SIDECAR_URL,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> PostCallable:
    """Build the production POST callable bound to a sidecar base URL.

    Mirrors the hook client's error handling — all realistic failures
    (connection refused, HTTPError, timeout, decode error) are caught
    and converted to ``None``. We do NOT raise.
    """
    base = base_url.rstrip("/")

    def _post(path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{base}{path}"
        try:
            data = json.dumps(body).encode("utf-8")
        except (TypeError, ValueError) as exc:
            _log.warning("watcher POST %s body not serializable: %r", path, exc)
            return None
        req = _urlreq.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with _urlreq.urlopen(req, timeout=timeout_s) as resp:
                decoded = json.loads(resp.read().decode("utf-8"))
                return decoded if isinstance(decoded, dict) else None
        except (_urlerr.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            _log.warning("watcher POST %s failed: %r", path, exc)
            return None

    return _post


def default_search_callable(
    base_url: str = DEFAULT_SIDECAR_URL,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> SearchCallable:
    """Build the production search callable bound to a sidecar base URL.

    Returns a function with the :data:`SearchCallable` signature.
    Same error envelope as the POST counterpart — never raises.
    """
    base = base_url.rstrip("/")

    def _search(label: str, query: str, scope: str | None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"q": query, "limit": 10}
        if scope is not None:
            params["scope"] = scope
        url = f"{base}/api/search/{label}?{urlencode(params)}"
        try:
            with _urlreq.urlopen(_urlreq.Request(url, method="GET"), timeout=timeout_s) as resp:
                decoded = json.loads(resp.read().decode("utf-8"))
                return decoded if isinstance(decoded, list) else []
        except (_urlerr.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            _log.warning("watcher search %s failed: %r", label, exc)
            return []

    return _search


# ---------------------------------------------------------------------------
# Ingest — parse + diff + POST
# ---------------------------------------------------------------------------


@dataclass
class PostResult:
    """Outcome of :func:`ingest_file` — useful for tests + retry book-keeping."""

    path: Path
    """The file we processed."""

    scope: str
    """The scope string applied to every produced node."""

    parsed: list[ParsedConvention]
    """The full parse output. ``[]`` if the file couldn't be parsed."""

    posted: list[dict[str, Any]] = field(default_factory=list)
    """Bodies actually sent to ``POST /api/Convention`` (successful posts only)."""

    skipped: list[ParsedConvention] = field(default_factory=list)
    """Conventions we matched against existing nodes and chose to skip.

    Cycle 1 still POSTs everything because we have no SUPERSEDES
    writer — so this list is usually empty. Kept here so cycle-2 can
    flip the behavior to "skip identical, supersede different" without
    breaking the call site.
    """

    failed: list[ParsedConvention] = field(default_factory=list)
    """Conventions whose POST failed (sidecar down, 5xx, etc.).

    The daemon's main loop pushes these onto the :class:`RetryQueue`.
    """

    supersedes_intents: list[tuple[ParsedConvention, str]] = field(default_factory=list)
    """``(new_convention, existing_id_to_supersede)`` pairs.

    Cycle 1 just logs these for human cleanup. Cycle-2 will turn them
    into real ``SUPERSEDES`` edges via a new sidecar endpoint.
    """


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Case-fold + whitespace-collapse a title for diff comparison.

    Two titles that differ only in case or runs of whitespace are
    considered the same rule. We don't go further (no stemming /
    punctuation stripping) because over-normalizing leads to false
    merges that look like data loss to a user inspecting the graph.
    """
    return _WHITESPACE_RE.sub(" ", title.strip().casefold())


def ingest_file(
    path: Path,
    *,
    scope: str,
    post: PostCallable,
    search: SearchCallable | None = None,
) -> PostResult:
    """Parse ``path``, diff against the sidecar, POST new conventions.

    Args:
        path: File to ingest. Need not exist — :func:`parse_file`
            handles missing-file gracefully and we return an empty
            result.
        scope: Scope string to tag every produced node with.
        post: HTTP POST callable (production:
            :func:`default_post_callable`). Tests pass an in-memory
            recorder.
        search: HTTP search callable. ``None`` (the default) disables
            diff detection — every parsed convention is treated as
            new. This is the cycle-1 default at the daemon entry point
            because the SUPERSEDES writer doesn't exist yet; tests
            inject a real search to exercise the diff path.

    Returns:
        A :class:`PostResult` summarizing what happened. The function
        never raises — every realistic failure (parse error, sidecar
        down, malformed search response) is captured and reported via
        the result.

    Per the design-§10 "files are inputs only" rule, we never delete
    on the wire here. A file that lost a section produces no DELETE
    POST; the old section's Convention node lingers in the graph until
    a human retires it.
    """
    parsed = parse_file(path)
    result = PostResult(path=path, scope=scope, parsed=parsed)
    if not parsed:
        return result

    # Build a map of existing-title → existing-id for diff detection.
    # We bucket by source-tag-normalized to scope the diff: an
    # AGENTS.md rule and a .cursorrules rule with the same title are
    # treated as distinct (they live in different files; the user
    # probably DID intend two records).
    existing_by_title: dict[str, str] = {}
    if search is not None:
        for convention in parsed:
            hits = _safe_search(search, "Convention", convention.title, scope)
            for hit in hits:
                hit_name = hit.get("name")
                hit_source = hit.get("source")
                hit_id = hit.get("id")
                if not isinstance(hit_name, str) or not isinstance(hit_id, str):
                    continue
                # Only count it as a diff candidate when the source
                # tag matches — see comment above.
                expected_source = format_to_source_tag(convention.source)
                if hit_source != expected_source:
                    continue
                if _normalize_title(hit_name) == _normalize_title(convention.title):
                    existing_by_title[_normalize_title(convention.title)] = hit_id

    for convention in parsed:
        body = _convention_to_create_body(convention, scope=scope)
        existing_id = existing_by_title.get(_normalize_title(convention.title))
        response = post("/api/Convention", body)
        if response is None:
            result.failed.append(convention)
            continue
        result.posted.append(body)
        if existing_id is not None:
            result.supersedes_intents.append((convention, existing_id))
            _log.info(
                "watcher: SUPERSEDES intent — new convention %r supersedes %s "
                "(file=%s, scope=%s). Cycle-2 will write the edge; cycle 1 "
                "logs only.",
                convention.title,
                existing_id,
                path,
                scope,
            )

    return result


def _safe_search(
    search: SearchCallable, label: str, query: str, scope: str
) -> list[dict[str, Any]]:
    """Wrap the search callable so a misbehaving search never leaks."""
    try:
        result = search(label, query, scope)
    except Exception as exc:  # noqa: BLE001 — never-raise contract
        _log.warning("watcher: search callable raised: %r", exc)
        return []
    return result if isinstance(result, list) else []


def _convention_to_create_body(
    convention: ParsedConvention, *, scope: str
) -> dict[str, Any]:
    """Project a parser record into the sidecar's ``ConventionCreate`` shape.

    Field map (per :class:`sidecar.models.ConventionCreate`):

    * ``name`` ← ``title`` (clipped to 200 chars in the parser)
    * ``description`` ← ``rule_text`` (or a one-line fallback when
      the rule_text is empty, because Pydantic enforces min_length=1)
    * ``source`` ← :func:`format_to_source_tag` (turns ``"AGENTS.md"``
      into the design-§4 enum value ``"from_AGENTS.md"``)
    * ``scope`` ← caller's scope arg
    * ``strength`` ← 0.8 (high — these are explicit, file-authored
      rules. Conventions we *inferred* from session history will use
      lower strengths in cycle-2; here we know the user wrote it down.)

    The ``applies_to`` + ``examples`` + ``file_path`` parser fields are
    NOT on the wire — the sidecar's Convention model has no slot for
    them in cycle 1. They're still attached to the in-memory
    :class:`ParsedConvention` for the daemon's log output, and we may
    surface them in a future ``ConventionCreate.metadata`` field.
    """
    description = convention.rule_text.strip()
    if not description:
        # Pydantic min_length=1 — synthesize a one-liner from the title.
        # Choosing a stable fallback (vs raising) honors the never-raise
        # contract and gives the operator something readable in the graph.
        description = f"(rule body empty in source) — title: {convention.title}"
    return {
        "name": convention.title,
        "description": description,
        "source": format_to_source_tag(convention.source),
        "strength": 0.8,
        "scope": scope,
    }


# ---------------------------------------------------------------------------
# Driver — synchronous + watchdog-driven entry points
# ---------------------------------------------------------------------------


def run_once(
    paths: Iterable[Path],
    *,
    registry: ProjectRegistry,
    post: PostCallable,
    search: SearchCallable | None = None,
) -> list[PostResult]:
    """Synchronously ingest a fixed list of files (no watchdog).

    Useful for tests, CLI ``--once`` mode, and the initial-pass scan
    the daemon does at startup before the watchdog observer fires.
    """
    out: list[PostResult] = []
    for path in paths:
        if not _is_instruction_file(path):
            continue
        scope = registry.scope_for(path)
        out.append(
            ingest_file(path, scope=scope, post=post, search=search)
        )
    return out


def setup_watcher_logger() -> logging.Logger:
    """Mirror the hook logger but for the watcher daemon.

    Writes to ``~/.galvo-memory/logs/watcher.log`` (rotating, 1 MB ×
    3 backups — same caps as the hooks). Idempotent: re-attaching is a
    no-op so importing the module multiple times during tests doesn't
    duplicate log lines.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("galvo.memory.watcher")
    logger.setLevel(logging.INFO)

    expected_path = str(LOG_DIR / "watcher.log")
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and getattr(handler, "baseFilename", None) == expected_path
        ):
            return logger

    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "watcher.log",
        maxBytes=1_000_000,
        backupCount=3,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# CLI / module entry point
# ---------------------------------------------------------------------------


_WATCHDOG_MISSING_MSG = (
    "memory.watcher: the 'watchdog' package is not installed.\n"
    "Install the watcher extra to use the daemon:\n"
    "    pip install 'galvo-memory[watcher]'\n"
    "    # or: uv pip install -e '.[watcher]' from memory/\n"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``python -m memory.watcher``.

    Kept module-level + side-effect-free so tests can call the parser
    directly to verify the option surface.
    """
    parser = argparse.ArgumentParser(
        prog="memory.watcher",
        description="Watch instruction files (AGENTS.md / CLAUDE.md / .cursorrules / "
        ".codex/*.md) and ingest changes into the memory sidecar.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Project root directories to watch. Defaults to the current "
        "directory if not given.",
    )
    parser.add_argument(
        "--sidecar-url",
        default=os.environ.get("GALVO_MEMORY_SIDECAR_URL", DEFAULT_SIDECAR_URL),
        help=f"Sidecar base URL (default: {DEFAULT_SIDECAR_URL}; honors "
        "GALVO_MEMORY_SIDECAR_URL env).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan + ingest once, then exit (no watchdog).",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Force a specific scope string for ALL watched roots "
        "(default: auto-detect via .galvo-mem/project.toml markers).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Daemon entry point.

    Returns:
        Exit code. 0 on clean shutdown / successful ``--once`` pass;
        1 when the ``watchdog`` import fails (printing the install
        hint to stderr).

    Lifecycle
    ---------

    1. Parse args; build :class:`ProjectRegistry` from the positional
       ``paths`` (or the current dir).
    2. Set up the rotating logger.
    3. Do an initial-pass scan + ingest of every instruction file
       under every registered root.
    4. If ``--once`` was given, return.
    5. Otherwise lazy-import ``watchdog`` and start an
       :class:`watchdog.observers.Observer` for each root. Handler:
       :class:`_WatcherHandler` (defined inside :func:`main` so the
       module-level import doesn't depend on watchdog).
    6. Block until SIGINT — at which point we ``stop()`` + ``join()``
       every observer.
    """
    args = _parse_args(argv)
    log = setup_watcher_logger()

    paths = args.paths or [Path.cwd()]
    registry = ProjectRegistry()
    for root in paths:
        try:
            registry.add(root, scope=args.scope)
        except (PermissionError, OSError) as exc:
            log.warning("main: cannot register %s: %r", root, exc)

    post = default_post_callable(args.sidecar_url)
    search = default_search_callable(args.sidecar_url)

    # Initial pass — pick up everything currently on disk.
    initial_files = scan_paths_for_instruction_files(
        entry.root for entry in registry.entries
    )
    log.info("main: initial scan found %d instruction files", len(initial_files))
    results = run_once(
        initial_files, registry=registry, post=post, search=search
    )
    for result in results:
        log.info(
            "main: ingested %s — parsed=%d posted=%d failed=%d "
            "supersedes_intents=%d",
            result.path,
            len(result.parsed),
            len(result.posted),
            len(result.failed),
            len(result.supersedes_intents),
        )

    if args.once:
        return 0

    # Lazy import of watchdog — keeps the rest of the module usable
    # without the optional extra installed.
    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        sys.stderr.write(_WATCHDOG_MISSING_MSG)
        return 1

    retry_queue = RetryQueue()
    # debounce_state maps absolute file path → monotonic deadline.
    # When the timer fires we re-read the path and dispatch ingest_file.
    debounce_state: dict[str, float] = {}

    class _WatcherHandler(FileSystemEventHandler):
        """Watchdog event handler — debounces and dispatches."""

        def on_any_event(self, event: FileSystemEvent) -> None:
            try:
                if event.is_directory:
                    return
                src = Path(str(event.src_path))
                if not _is_instruction_file(src):
                    return
                # Schedule a deferred ingest after DEBOUNCE_SECONDS.
                debounce_state[str(src)] = time.monotonic() + DEBOUNCE_SECONDS
            except Exception as exc:  # noqa: BLE001 — never raise
                log.warning("WatcherHandler.on_any_event: %r", exc)

    handler = _WatcherHandler()
    observers: list[Any] = []
    for entry in registry.entries:
        observer = Observer()
        observer.schedule(handler, str(entry.root), recursive=True)
        observer.start()
        observers.append(observer)
        log.info("main: watching %s (scope=%s)", entry.root, entry.scope)

    try:
        while True:
            now = time.monotonic()
            due: list[str] = [
                p for p, deadline in debounce_state.items() if deadline <= now
            ]
            for path_str in due:
                debounce_state.pop(path_str, None)
                p = Path(path_str)
                scope = registry.scope_for(p)
                try:
                    result = ingest_file(
                        p, scope=scope, post=post, search=search
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("main: ingest_file raised on %s: %r", p, exc)
                    continue
                log.info(
                    "main: change %s — parsed=%d posted=%d failed=%d",
                    p,
                    len(result.parsed),
                    len(result.posted),
                    len(result.failed),
                )
                for convention in result.failed:
                    retry_queue.push(
                        RetryEntry(convention=convention, scope=scope)
                    )

            # Drain retries opportunistically — on every iteration so a
            # sidecar that flickers back to life catches up quickly.
            if retry_queue:
                pending = retry_queue.drain()
                log.info("main: retrying %d pending POSTs", len(pending))
                for retry in pending:
                    body = _convention_to_create_body(
                        retry.convention, scope=retry.scope
                    )
                    if post("/api/Convention", body) is None:
                        retry.attempts += 1
                        retry_queue.push(retry)

            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("main: SIGINT — shutting down")
    finally:
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join(timeout=2)

    return 0


if __name__ == "__main__":  # pragma: no cover — module entry
    sys.exit(main())
