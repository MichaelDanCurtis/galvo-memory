"""Galvo memory layer — scope partitioning (design §D4).

Public API for the 3-tier scope system:

* :func:`detect_scope` — walks up from a working directory and returns one of
  ``"project:<repo-id>"`` / ``"personal"`` / ``"universal"``.
* :func:`find_marker` / :func:`write_marker` — read/write the
  ``.galvo-mem/project.toml`` anchor file.
* :class:`ProjectMarker` / :class:`ScopeConfig` — parsed marker payload.
* :func:`is_project_scope` / :func:`project_id_from_scope` — string helpers
  for callers that need to branch on the scope value.
"""

from .detector import detect_scope, is_project_scope, project_id_from_scope
from .marker import MARKER_REL_PATH, ProjectMarker, ScopeConfig, find_marker, write_marker

__all__ = [
    "MARKER_REL_PATH",
    "ProjectMarker",
    "ScopeConfig",
    "detect_scope",
    "find_marker",
    "is_project_scope",
    "project_id_from_scope",
    "write_marker",
]
