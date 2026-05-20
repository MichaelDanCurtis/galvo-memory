"""Galvo memory layer — file watcher daemon (design §10, Phase 2 Task 16).

The watcher turns changes to instruction files (``AGENTS.md`` /
``CLAUDE.md`` / ``.cursorrules`` / ``.codex/*.md``) into ``Convention``
node upserts on the memory sidecar. Per design §10, these files are
non-canonical INPUTS to the graph — the graph is the source of truth,
the files are how humans express rules.

Public surface
==============

* :mod:`memory.watcher.parsers` — :func:`parse_file` and the underlying
  :class:`ParsedConvention` dataclass.
* :mod:`memory.watcher.daemon` — :func:`ingest_file` (the testable POST
  driver), :func:`run_once` (synchronous CLI mode), :class:`ProjectRegistry`,
  :class:`RetryQueue`, :func:`main` (the ``python -m memory.watcher``
  entry point).

The daemon's :func:`main` lazy-imports ``watchdog`` so callers who only
want the parsers + ingest helpers don't need the ``[watcher]`` extra
installed. Importing this package by itself never touches watchdog.
"""

from .daemon import (
    DEBOUNCE_SECONDS,
    DEFAULT_SIDECAR_URL,
    WATCH_FILENAMES,
    PostResult,
    ProjectEntry,
    ProjectRegistry,
    RetryEntry,
    RetryQueue,
    default_post_callable,
    default_search_callable,
    ingest_file,
    main,
    run_once,
    scan_paths_for_instruction_files,
)
from .parsers import ParsedConvention, format_to_source_tag, parse_file, parse_markdown, parse_plain

__all__ = [
    "DEBOUNCE_SECONDS",
    "DEFAULT_SIDECAR_URL",
    "WATCH_FILENAMES",
    "ParsedConvention",
    "PostResult",
    "ProjectEntry",
    "ProjectRegistry",
    "RetryEntry",
    "RetryQueue",
    "default_post_callable",
    "default_search_callable",
    "format_to_source_tag",
    "ingest_file",
    "main",
    "parse_file",
    "parse_markdown",
    "parse_plain",
    "run_once",
    "scan_paths_for_instruction_files",
]
