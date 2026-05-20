"""Scope detection from current working directory.

Per design §D4 (3-tier scope partitioning), every write to the memory graph is
tagged with one of three scope strings:

* ``project:<repo-id>`` — found a ``.galvo-mem/project.toml`` walking up from
  cwd. The marker's ``[scope].default`` value is returned (almost always
  ``"project:<id>"`` matching the marker's ``project.id``).
* ``personal``           — no marker but cwd is somewhere inside ``$HOME``.
* ``universal``          — no marker and cwd is outside ``$HOME``. This is
  rare and ``universal`` is **never** auto-detected as a write target by hooks;
  explicit operator action is required to write into universal scope. The
  string is returned here purely so downstream code can distinguish the case.

The contract is intentionally narrow: ``detect_scope`` returns a string, and
that's it. Helpers ``is_project_scope`` and ``project_id_from_scope`` let
callers branch on the result without re-parsing.
"""

from __future__ import annotations

from pathlib import Path

from .marker import find_marker


def detect_scope(cwd: Path | None = None) -> str:
    """Return the scope string for ``cwd`` (defaulting to :func:`Path.cwd`).

    Lookup order:

    1. Walk up from ``cwd`` looking for ``.galvo-mem/project.toml`` — if found,
       return the marker's ``[scope].default``.
    2. If no marker, check whether ``cwd`` is inside ``$HOME``. If so, return
       ``"personal"``.
    3. Otherwise return ``"universal"`` — the explicit-writes-only scope.

    Symlinks in ``cwd`` and ``$HOME`` are resolved before comparison so the
    ``$HOME`` test does the right thing on macOS (where ``/tmp`` is a symlink
    to ``/private/tmp``, and ``$HOME`` itself can be a symlink under certain
    user configurations).
    """
    cwd = (cwd or Path.cwd()).resolve()
    marker = find_marker(cwd)
    if marker is not None:
        return marker.scope.default
    home = Path.home().resolve()
    try:
        cwd.relative_to(home)
    except ValueError:
        return "universal"
    return "personal"


def is_project_scope(scope: str) -> bool:
    """``True`` iff ``scope`` matches the ``"project:<id>"`` pattern."""
    return scope.startswith("project:")


def project_id_from_scope(scope: str) -> str | None:
    """Extract the ``<id>`` suffix from a ``"project:<id>"`` scope.

    Returns ``None`` for ``"personal"``, ``"universal"``, or any non-project
    scope string. Returns the empty string only if someone manages to pass
    ``"project:"`` (malformed) — we don't second-guess that here.
    """
    if not is_project_scope(scope):
        return None
    return scope.removeprefix("project:")
