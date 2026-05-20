"""Task 16 — instruction-file parser unit tests.

What we're protecting
=====================

The :mod:`memory.watcher.parsers` module is the first hop in the
file → graph pipeline. If parsing is wrong, the daemon writes the
wrong Conventions; if it raises, the daemon crashes; if it misses
sections, rules silently disappear.

Test surface
============

* **Markdown carving** — ``##`` and ``###`` headings each start a
  section; level-1 headings are file-level labels and don't emit; the
  body of each section becomes ``rule_text``.
* **Plain-text paragraph split** — non-empty blank-line-separated
  paragraphs each become one record; title is the first line; full
  paragraph is the rule_text.
* **Source-tag dispatch** — :func:`parse_file` picks the right parser
  by filename + suffix + parent-dir-name (``.codex/*.md``).
* **Best-effort enrichment** — ``Applies to:`` lines feed
  ``applies_to``; fenced code blocks feed ``examples``.
* **Never-raise contract** — missing file, malformed content,
  permission denied, decode error all return ``[]``.
* **Source-tag → design-enum mapping** — ``"AGENTS.md"`` →
  ``"from_AGENTS.md"`` per design §4.
* **Title length cap** — long heading lines get clipped to 200 chars
  (the sidecar's Convention.name max_length).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from watcher.parsers import (
    ParsedConvention,
    format_to_source_tag,
    parse_file,
    parse_markdown,
    parse_plain,
)


# ---------------------------------------------------------------------------
# Source-tag → design-enum mapping
# ---------------------------------------------------------------------------


def test_format_to_source_tag_known_values() -> None:
    """All four canonical source tags map to ``from_<tag>``.

    The sidecar's ``ConventionCreate.source`` accepts an open string
    but the design §4 enum is ``inferred`` / ``explicit`` /
    ``from_AGENTS.md``. We extend with ``from_CLAUDE.md`` /
    ``from_.cursorrules`` / ``from_.codex`` to cover the other
    parser-supported formats; this is forward-compatible since the
    field is loose-typed on the wire.
    """
    assert format_to_source_tag("AGENTS.md") == "from_AGENTS.md"
    assert format_to_source_tag("CLAUDE.md") == "from_CLAUDE.md"
    assert format_to_source_tag(".cursorrules") == "from_.cursorrules"
    assert format_to_source_tag(".codex") == "from_.codex"


def test_format_to_source_tag_unknown_fallback() -> None:
    """Unknown tags get ``from_<tag>`` so projection is total.

    The mapping is open-ended on purpose — a parser extension that
    introduces a new source tag (e.g. ``"WARP.md"``) shouldn't have to
    update the mapping dict to be storable.
    """
    assert format_to_source_tag("WARP.md") == "from_WARP.md"


# ---------------------------------------------------------------------------
# parse_markdown — heading-based section carving
# ---------------------------------------------------------------------------


def test_parse_markdown_level_two_headings_become_sections() -> None:
    """``##`` headings each become one ParsedConvention.

    Body lines under a heading (up to the next ``##`` / ``###``) feed
    into ``rule_text``. Level-1 (``#``) is a file-level label and
    does NOT emit a record.
    """
    text = (
        "# Galvo — Project Instructions\n"
        "\n"
        "## Use ruff for formatting\n"
        "Always run `ruff format` before committing.\n"
        "\n"
        "## Pin dependencies\n"
        "Every dep gets a `>=` version floor.\n"
    )
    out = parse_markdown(text, source="AGENTS.md", file_path="/repo/AGENTS.md")
    assert len(out) == 2
    assert out[0].title == "Use ruff for formatting"
    assert "ruff format" in out[0].rule_text
    assert out[1].title == "Pin dependencies"
    assert ">=" in out[1].rule_text
    # Both records carry the source + file_path.
    for record in out:
        assert record.source == "AGENTS.md"
        assert record.file_path == "/repo/AGENTS.md"


def test_parse_markdown_level_three_headings_become_sections() -> None:
    """``###`` headings also emit — they're third-level rules under a
    common parent section, but the parser flattens that hierarchy."""
    text = (
        "## Top-level area\n"
        "Some narrative.\n"
        "\n"
        "### Sub-rule alpha\n"
        "Alpha body.\n"
        "\n"
        "### Sub-rule beta\n"
        "Beta body.\n"
    )
    out = parse_markdown(text, source="AGENTS.md", file_path="/r/A.md")
    titles = [s.title for s in out]
    assert titles == ["Top-level area", "Sub-rule alpha", "Sub-rule beta"]


def test_parse_markdown_extracts_applies_to() -> None:
    """A leading ``Applies to:`` line feeds ``applies_to``.

    The line is stripped from the resulting ``rule_text`` so the
    description doesn't redundantly carry the metadata header.
    """
    text = (
        "## No vendored binaries\n"
        "Applies to: *.so, *.dylib, target/**\n"
        "Build artifacts must never be committed.\n"
    )
    out = parse_markdown(text, source="AGENTS.md", file_path="/r/A.md")
    assert len(out) == 1
    rec = out[0]
    assert rec.applies_to == ["*.so", "*.dylib", "target/**"]
    # The Applies-to line shouldn't appear in rule_text.
    assert "Applies to" not in rec.rule_text
    assert "Build artifacts must never be committed" in rec.rule_text


def test_parse_markdown_extracts_code_examples() -> None:
    """Fenced code blocks feed ``examples`` and STAY in rule_text.

    Examples-as-structured-field is for retrieval ranking; keeping
    them in the body keeps the rule self-contained for a model
    reading it. The opening language tag on the fence (e.g.
    triple-backtick + ``python``) is stripped from the captured
    example body.
    """
    text = (
        "## Prefer Path over os.path\n"
        "Use pathlib for everything:\n"
        "\n"
        "```python\n"
        "from pathlib import Path\n"
        "config = Path.home() / '.config'\n"
        "```\n"
        "\n"
        "Avoid os.path.join.\n"
    )
    out = parse_markdown(text, source="AGENTS.md", file_path="/r/A.md")
    assert len(out) == 1
    rec = out[0]
    assert len(rec.examples) == 1
    assert "from pathlib import Path" in rec.examples[0]
    # The fenced block is still in rule_text — examples are duplicated.
    assert "from pathlib import Path" in rec.rule_text


def test_parse_markdown_headings_inside_code_blocks_ignored() -> None:
    """A ``##`` line INSIDE a fenced code block is not a section break.

    Without this the parser would split a code sample with a markdown
    comment in it into two phantom sections.
    """
    text = (
        "## Real section\n"
        "Some content.\n"
        "\n"
        "```python\n"
        "## this is a python comment, not a heading\n"
        "x = 1\n"
        "```\n"
        "\n"
        "More content under the real section.\n"
    )
    out = parse_markdown(text, source="AGENTS.md", file_path="/r/A.md")
    # Only ONE section — the in-code-block ## must not split.
    assert len(out) == 1
    assert out[0].title == "Real section"
    assert "More content" in out[0].rule_text


def test_parse_markdown_title_clipped_to_200_chars() -> None:
    """Very long heading lines are clipped to ConventionCreate.name's
    max_length=200 so the sidecar accepts the body without a 422."""
    long_title = "A" * 500
    text = f"## {long_title}\nbody\n"
    out = parse_markdown(text, source="AGENTS.md", file_path="/r/A.md")
    assert len(out) == 1
    assert len(out[0].title) == 200
    assert out[0].title == "A" * 200


def test_parse_markdown_empty_input_returns_empty_list() -> None:
    """An empty file produces no records (parser is total on edge cases)."""
    out = parse_markdown("", source="AGENTS.md", file_path="/r/A.md")
    assert out == []


def test_parse_markdown_no_headings_returns_empty_list() -> None:
    """A file with only prose (no headings) produces no records.

    We deliberately don't emit a "file-as-one-rule" fallback —
    operators with that intent should add a single ``##`` heading
    rather than relying on parser magic.
    """
    text = "Just some prose.\n\nMore prose.\n"
    out = parse_markdown(text, source="AGENTS.md", file_path="/r/A.md")
    assert out == []


# ---------------------------------------------------------------------------
# parse_plain — .cursorrules paragraph split
# ---------------------------------------------------------------------------


def test_parse_plain_paragraphs_become_records() -> None:
    """Each blank-line-separated paragraph emits one record.

    Title = first line; rule_text = full paragraph; applies_to /
    examples always empty (plain text has no structured fields).
    """
    text = (
        "Use ruff for formatting.\n"
        "Always run before committing.\n"
        "\n"
        "Pin dependencies with >= version floor.\n"
        "\n"
        "Never commit secrets.\n"
    )
    out = parse_plain(text, source=".cursorrules", file_path="/r/.cursorrules")
    assert len(out) == 3
    assert out[0].title == "Use ruff for formatting."
    assert "before committing" in out[0].rule_text
    assert out[1].title.startswith("Pin dependencies")
    assert out[2].title == "Never commit secrets."
    for record in out:
        assert record.source == ".cursorrules"
        assert record.applies_to == []
        assert record.examples == []


def test_parse_plain_collapses_repeated_blank_lines() -> None:
    """Multiple blank lines between paragraphs are treated as one separator."""
    text = "Rule one.\n\n\n\nRule two.\n"
    out = parse_plain(text, source=".cursorrules", file_path="/r/.cursorrules")
    assert len(out) == 2


def test_parse_plain_handles_trailing_paragraph_without_newline() -> None:
    """A final paragraph that doesn't end in a newline still emits.

    Editors sometimes save files without a trailing newline; we
    shouldn't lose the last rule.
    """
    text = "Rule one.\n\nRule two without trailing newline."
    out = parse_plain(text, source=".cursorrules", file_path="/r/.cursorrules")
    assert len(out) == 2
    assert out[1].title == "Rule two without trailing newline."


# ---------------------------------------------------------------------------
# parse_file — dispatch logic
# ---------------------------------------------------------------------------


def test_parse_file_dispatches_agents_md(tmp_path: Path) -> None:
    """``AGENTS.md`` → markdown parser, source tag ``"AGENTS.md"``."""
    p = tmp_path / "AGENTS.md"
    p.write_text("## Rule one\nBody.\n")
    out = parse_file(p)
    assert len(out) == 1
    assert out[0].source == "AGENTS.md"
    assert out[0].title == "Rule one"


def test_parse_file_dispatches_claude_md(tmp_path: Path) -> None:
    """``CLAUDE.md`` → markdown parser, source tag ``"CLAUDE.md"``."""
    p = tmp_path / "CLAUDE.md"
    p.write_text("## Pinned rule\nBody.\n")
    out = parse_file(p)
    assert len(out) == 1
    assert out[0].source == "CLAUDE.md"


def test_parse_file_dispatches_cursorrules(tmp_path: Path) -> None:
    """``.cursorrules`` → plain parser, source tag ``".cursorrules"``."""
    p = tmp_path / ".cursorrules"
    p.write_text("Rule one.\n\nRule two.\n")
    out = parse_file(p)
    assert len(out) == 2
    assert out[0].source == ".cursorrules"


def test_parse_file_dispatches_codex_subdir_md(tmp_path: Path) -> None:
    """``.codex/foo.md`` → markdown parser, source tag ``".codex"``.

    The source is the parent-dir name (``.codex``), not the filename,
    because a project can have multiple codex files (``rules.md`` /
    ``style.md`` / etc.) and the relevant grouping is by tool.
    """
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    p = codex_dir / "rules.md"
    p.write_text("## Codex rule\nBody.\n")
    out = parse_file(p)
    assert len(out) == 1
    assert out[0].source == ".codex"


def test_parse_file_missing_returns_empty(tmp_path: Path) -> None:
    """Missing file → empty list, no exception (never-raise contract)."""
    p = tmp_path / "doesnt-exist.md"
    out = parse_file(p)
    assert out == []


def test_parse_file_unsupported_extension_returns_empty(tmp_path: Path) -> None:
    """A ``.txt`` file (or any unknown name) returns ``[]``.

    Conservative dispatch: we don't speculate that a random file is
    an instruction file.
    """
    p = tmp_path / "rando.txt"
    p.write_text("nope\n")
    out = parse_file(p)
    assert out == []


def test_parse_file_handles_decode_error(tmp_path: Path) -> None:
    """A file containing invalid UTF-8 returns ``[]`` (never-raise).

    Real instruction files won't have this issue, but the daemon
    might encounter editor backup files / vendored binaries that
    pattern-match the watch list.
    """
    p = tmp_path / "AGENTS.md"
    # Bytes that don't decode as UTF-8. Choose a sequence that's
    # definitely invalid (a continuation byte without a leading byte).
    p.write_bytes(b"\xff\xfe## Not actually utf-8\n")
    out = parse_file(p)
    assert out == []


def test_parse_file_resolves_absolute_path(tmp_path: Path) -> None:
    """``file_path`` on the returned record is the resolved absolute path.

    Important for traceability — symlinks and relative paths should
    not leak through into the graph metadata.
    """
    p = tmp_path / "AGENTS.md"
    p.write_text("## R\nbody\n")
    out = parse_file(p)
    assert len(out) == 1
    assert out[0].file_path == str(p.resolve())


# ---------------------------------------------------------------------------
# ParsedConvention dataclass — defensive shape tests
# ---------------------------------------------------------------------------


def test_parsed_convention_is_frozen() -> None:
    """ParsedConvention is immutable — the daemon doesn't mutate parser output."""
    rec = ParsedConvention(
        title="t",
        rule_text="r",
        source="AGENTS.md",
        applies_to=[],
        examples=[],
        file_path="/x/y",
    )
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
        rec.title = "different"  # type: ignore[misc]
