"""Task 4 — scope detection + project marker round-trip.

Pure-Python tests; no Neo4j substrate required.

Each test uses ``tmp_path`` for filesystem isolation and ``monkeypatch.setenv``
to spoof ``$HOME`` so we can exercise the personal/universal branches without
touching the real ``$HOME``. ``Path.home()`` reads ``HOME`` on POSIX, so this
is sufficient for both macOS + Linux CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scope import (
    MARKER_REL_PATH,
    ProjectMarker,
    detect_scope,
    find_marker,
    is_project_scope,
    project_id_from_scope,
    write_marker,
)


# ---------------------------------------------------------------------------
# detect_scope — marker found
# ---------------------------------------------------------------------------


def test_detect_scope_returns_project_when_marker_exists(tmp_path: Path) -> None:
    """A directory containing ``.galvo-mem/project.toml`` resolves to the
    marker's configured default scope."""
    write_marker(tmp_path, project_id="galvo", name="Galvo FACT")
    assert detect_scope(tmp_path) == "project:galvo"


def test_detect_scope_walks_up_to_find_marker(tmp_path: Path) -> None:
    """``detect_scope`` walks up from a deeply nested subdir to find the
    marker at the project root."""
    write_marker(tmp_path, project_id="galvo", name="Galvo FACT")
    nested = tmp_path / "sdk" / "galvo" / "eval"
    nested.mkdir(parents=True)
    assert detect_scope(nested) == "project:galvo"


def test_detect_scope_uses_custom_default_scope(tmp_path: Path) -> None:
    """If the marker overrides ``[scope].default``, that value (not
    ``project:<id>``) is returned."""
    write_marker(
        tmp_path,
        project_id="galvo",
        name="Galvo FACT",
        default_scope="personal",  # someone explicitly chose personal-scope writes
    )
    assert detect_scope(tmp_path) == "personal"


# ---------------------------------------------------------------------------
# detect_scope — no marker
# ---------------------------------------------------------------------------


def test_detect_scope_returns_personal_when_no_marker_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$HOME`` set to ``tmp_path``; cwd is a subdir of ``$HOME`` with no
    marker → ``"personal"``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    sub = tmp_path / "Documents" / "notes"
    sub.mkdir(parents=True)
    assert detect_scope(sub) == "personal"


def test_detect_scope_returns_personal_when_cwd_is_home_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edge case: cwd == ``$HOME`` (no subdir). Still personal."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert detect_scope(tmp_path) == "personal"


def test_detect_scope_returns_universal_when_outside_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$HOME`` set to a *different* directory than ``tmp_path``; no marker.
    cwd is outside ``$HOME`` → ``"universal"``."""
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    assert detect_scope(outside) == "universal"


# ---------------------------------------------------------------------------
# Marker round-trip
# ---------------------------------------------------------------------------


def test_write_marker_round_trip(tmp_path: Path) -> None:
    """``write_marker`` then ``find_marker`` recovers identical id + scope."""
    written = write_marker(tmp_path, project_id="galvo", name="Galvo FACT")
    assert written.is_file()
    assert written == tmp_path / MARKER_REL_PATH

    marker = find_marker(tmp_path)
    assert marker is not None
    assert isinstance(marker, ProjectMarker)
    assert marker.id == "galvo"
    assert marker.name == "Galvo FACT"
    assert marker.scope.default == "project:galvo"
    assert marker.scope.allowed_universal_topics == []
    assert marker.root == tmp_path.resolve()


def test_find_marker_returns_none_when_no_marker(tmp_path: Path) -> None:
    """No marker anywhere → ``None``. We can't easily test "walks to root"
    without polluting the real filesystem, but verifying ``None`` for an
    isolated tree is enough — ``find_marker`` returns ``None`` only by
    reaching the filesystem root unsuccessfully.

    On macOS the resolve()'d ``tmp_path`` is under ``/private/var/...`` so
    the walk will hit ``/`` — a parent we don't control. To keep the test
    deterministic, we assert merely that no ``ProjectMarker`` for our
    ``project.id`` was found (i.e. the function returned either ``None`` or
    a *different* marker, never an erroneous one rooted under tmp_path).
    """
    marker = find_marker(tmp_path)
    if marker is not None:
        # The host happened to have a real marker above /tmp; that's fine
        # so long as it's NOT rooted at our tmp_path.
        assert marker.root != tmp_path.resolve()


def test_write_marker_creates_galvo_mem_dir(tmp_path: Path) -> None:
    """The ``.galvo-mem/`` directory is created if it didn't exist."""
    assert not (tmp_path / ".galvo-mem").exists()
    write_marker(tmp_path, project_id="galvo", name="Galvo FACT")
    assert (tmp_path / ".galvo-mem").is_dir()
    assert (tmp_path / ".galvo-mem" / "project.toml").is_file()


def test_write_marker_is_idempotent_on_existing_dir(tmp_path: Path) -> None:
    """Pre-existing ``.galvo-mem/`` directory doesn't break ``write_marker``."""
    (tmp_path / ".galvo-mem").mkdir()
    written = write_marker(tmp_path, project_id="galvo", name="Galvo FACT")
    assert written.is_file()


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def test_is_project_scope() -> None:
    assert is_project_scope("project:galvo") is True
    assert is_project_scope("project:any-id-here") is True
    assert is_project_scope("personal") is False
    assert is_project_scope("universal") is False
    assert is_project_scope("") is False


def test_project_id_from_scope() -> None:
    assert project_id_from_scope("project:galvo") == "galvo"
    assert project_id_from_scope("project:something-else") == "something-else"
    assert project_id_from_scope("personal") is None
    assert project_id_from_scope("universal") is None
    assert project_id_from_scope("") is None
