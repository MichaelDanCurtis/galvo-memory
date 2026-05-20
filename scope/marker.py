"""Read/write ``.galvo-mem/project.toml`` — the stable project identifier.

Per design §10 (`memory/docs/MEMORY-LAYER-DESIGN.md`) and Phase 2 plan §Task 4,
the marker file is created once at first use and survives directory renames,
moves, and worktrees. It anchors a stable ``project:<id>`` scope for everything
written from inside the project tree.

The file format is TOML (Python 3.11+ stdlib ``tomllib`` for reading; plain
string rendering for writing — we deliberately avoid pulling in a write-capable
TOML dep for cycle 1):

.. code-block:: toml

    # Created at first use. Stable repo identifier — survives renames, moves,
    # worktrees.

    [project]
    id = "galvo"                        # short stable id
    name = "Galvo FACT"                 # human-readable, can change
    created_at = "2026-05-17T00:00:00Z"

    [scope]
    default = "project:galvo"           # nodes written from this tree default here
    allowed_universal_topics = []       # opt-in writeable into universal scope
"""

from __future__ import annotations

import tomllib  # Python 3.11+ stdlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MARKER_REL_PATH = Path(".galvo-mem") / "project.toml"
"""Relative path to the marker file from the project root."""


@dataclass
class ScopeConfig:
    """The ``[scope]`` section of ``project.toml``."""

    default: str
    """Default scope string for writes originating from this tree, e.g.
    ``"project:galvo"`` or (rarely) ``"personal"``."""

    allowed_universal_topics: list[str] = field(default_factory=list)
    """Topics that may be written into the ``universal`` scope from this tree.

    Empty list means no universal writes are pre-authorized; explicit operator
    action is required for each universal write.
    """


@dataclass
class ProjectMarker:
    """The full ``project.toml`` payload plus the absolute path of its root."""

    id: str
    """Short stable repo id, e.g. ``"galvo"``. Used as the suffix of the
    ``project:<id>`` scope string."""

    name: str
    """Human-readable project name. May change without breaking ``id``."""

    created_at: datetime
    """When the marker was first written (UTC). Diagnostic only."""

    scope: ScopeConfig
    """The ``[scope]`` section, parsed."""

    root: Path
    """Absolute, resolved path to the directory containing ``.galvo-mem/``.

    NOT the path to ``project.toml`` itself — the directory above it. This is
    what callers join against to compute relative paths.
    """


def find_marker(start: Path) -> ProjectMarker | None:
    """Walk up from ``start`` until a ``.galvo-mem/project.toml`` is found.

    Returns the parsed ``ProjectMarker`` (whose ``root`` field is the directory
    containing ``.galvo-mem/``) or ``None`` if filesystem root is reached
    without finding one.

    Symlinks are resolved on entry so the walk traverses the canonical path.
    """
    current = start.resolve()
    while True:
        candidate = current / MARKER_REL_PATH
        if candidate.is_file():
            return _load_marker(candidate, root=current)
        if current.parent == current:
            # We've hit filesystem root.
            return None
        current = current.parent


def _load_marker(path: Path, *, root: Path) -> ProjectMarker:
    """Parse the TOML file at ``path``; ``root`` is the directory containing
    ``.galvo-mem/``."""
    data = tomllib.loads(path.read_text())
    project = data["project"]
    scope_data = data.get("scope", {})
    return ProjectMarker(
        id=project["id"],
        name=project["name"],
        created_at=_parse_dt(project["created_at"]),
        scope=ScopeConfig(
            default=scope_data.get("default", f"project:{project['id']}"),
            allowed_universal_topics=scope_data.get("allowed_universal_topics", []),
        ),
        root=root,
    )


def _parse_dt(value: str | datetime) -> datetime:
    """``tomllib`` may decode an ISO datetime literal into a ``datetime`` directly,
    but it also accepts ``"YYYY-MM-DDTHH:MM:SSZ"`` strings written by us — handle
    both paths.
    """
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_marker(
    root: Path,
    *,
    project_id: str,
    name: str,
    default_scope: str | None = None,
) -> Path:
    """Create ``.galvo-mem/project.toml`` at ``root``.

    Parameters
    ----------
    root:
        Directory that should become the project root. The function creates
        ``root/.galvo-mem/`` if it doesn't exist.
    project_id:
        Short stable identifier (e.g. ``"galvo"``). Used as the ``project:<id>``
        scope suffix.
    name:
        Human-readable project name (e.g. ``"Galvo FACT"``).
    default_scope:
        Optional override for the ``[scope].default`` value. Defaults to
        ``f"project:{project_id}"``.

    Returns the absolute path to the written ``project.toml``.
    """
    marker_dir = root / ".galvo-mem"
    marker_dir.mkdir(parents=True, exist_ok=True)
    path = marker_dir / "project.toml"
    scope = default_scope or f"project:{project_id}"
    path.write_text(
        f"""\
# Created at first use. Stable repo identifier — survives renames, moves, worktrees.

[project]
id = "{project_id}"
name = "{name}"
created_at = "{datetime.now(UTC).isoformat()}"

[scope]
default = "{scope}"
allowed_universal_topics = []
"""
    )
    return path
