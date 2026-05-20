"""Parse instruction files into Convention nodes (Task 16).

Per design §10, instruction files (``AGENTS.md`` / ``CLAUDE.md`` / ``.cursorrules``
/ ``.codex/*``) feed knowledge INTO the graph; the graph is canonical, files
are inputs only. This module is the read side: it takes raw file text and
produces a list of :class:`ParsedConvention` records that the daemon
turns into ``POST /api/Convention`` calls.

Supported formats
=================

We degrade gracefully across these layouts:

* ``AGENTS.md`` / ``CLAUDE.md`` / ``.codex/*.md`` — Markdown. Each ``##``
  (level-2) or ``###`` (level-3) heading starts a new ``ParsedConvention``.
  The heading text is the title; everything beneath until the next
  same-or-higher-level heading is ``rule_text``. The very top of the file
  (before any heading) is treated as an implicit "preamble" convention
  ONLY if it contains body text — a file that opens with a top-level
  ``# Title`` and goes straight into ``##`` sections has no preamble.
* ``.cursorrules`` — Plain text (no markdown structure expected). We
  split on blank lines; each non-empty paragraph becomes one
  ``ParsedConvention``. Title = first line, rule_text = full paragraph.

Sidecar contract carry
======================

The sidecar's :class:`ConventionCreate` model uses ``name`` /
``description`` (not ``title`` / ``rule_text``). The daemon does the
projection at the POST boundary — *this* module stays in
parser-natural language (``title`` / ``rule_text``) because that's how
operators reading the parser tests will think about the structure.

Per the sidecar contract, ``source`` is one of ``inferred`` /
``explicit`` / ``from_AGENTS.md`` (design §4). The parser sets a
canonical short-tag (``"AGENTS.md"`` / ``"CLAUDE.md"`` / etc.); the
daemon prefixes it with ``"from_"`` when projecting to the wire so the
stored value matches the design's enum.

Best-effort fields
==================

* ``applies_to`` — extracted from a leading ``Applies to:`` / ``Targets:``
  / ``Files:`` line at the top of a section body. Comma- or
  newline-separated globs / path fragments. Absent = empty list.
* ``examples`` — fenced code blocks (triple-backtick) inside the
  section body. We keep the raw code text (no language tag) so
  retrieval can surface "what does the example look like" without
  re-parsing the markdown.

Never raises
============

:func:`parse_file` catches every realistic failure (missing file,
permission denied, decode error, malformed markdown) and returns ``[]``.
The daemon treats the empty-list return identically to "no conventions
in this file" — the user's session never sees a watcher traceback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ParsedConvention",
    "format_to_source_tag",
    "parse_file",
    "parse_markdown",
    "parse_plain",
]


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedConvention:
    """A single rule extracted from an instruction file.

    The dataclass is intentionally narrow — only fields that the
    sidecar's ``ConventionCreate`` model can consume (after the
    daemon's title→name / rule_text→description projection) plus a few
    best-effort enrichment fields. Adding new fields here means adding
    matching support in the sidecar model OR documenting where they get
    dropped on the way to the wire.
    """

    title: str
    """Short imperative summary. Becomes ``ConventionCreate.name``.

    Bounded to <=200 chars to match the sidecar's ``name`` Field
    constraint — :func:`parse_markdown` and :func:`parse_plain` clip
    titles that exceed this length.
    """

    rule_text: str
    """The body of the rule. Becomes ``ConventionCreate.description``.

    Free-form markdown / plain text. The daemon does not transform it
    before POST; the sidecar stores it verbatim.
    """

    source: str
    """Short canonical tag for the origin file format.

    One of ``"AGENTS.md"`` / ``"CLAUDE.md"`` / ``".cursorrules"`` /
    ``".codex"``. The daemon maps this to the design-§4 enum value via
    :func:`format_to_source_tag` (``"AGENTS.md"`` →
    ``"from_AGENTS.md"`` etc.) before posting.
    """

    applies_to: list[str] = field(default_factory=list)
    """Best-effort target list parsed from a leading ``Applies to:`` /
    ``Targets:`` / ``Files:`` line. Empty when absent.

    Not currently consumed by the sidecar — kept here so cycle-2 can
    surface "this rule applies to these paths" in retrieval ranking
    without re-parsing the source file.
    """

    examples: list[str] = field(default_factory=list)
    """Fenced code-block bodies extracted from the section.

    The opening / closing fences and the language tag are stripped;
    the inner text is preserved with original indentation.
    """

    file_path: str = ""
    """Absolute path to the source file, for traceability.

    The daemon does NOT include this on the wire (the sidecar has no
    field for it), but it's load-bearing for the daemon's debug log
    output and tests.
    """


# ---------------------------------------------------------------------------
# Source-tag mapping
# ---------------------------------------------------------------------------


_SOURCE_TAG_TO_DESIGN_ENUM: dict[str, str] = {
    "AGENTS.md": "from_AGENTS.md",
    "CLAUDE.md": "from_CLAUDE.md",
    ".cursorrules": "from_.cursorrules",
    ".codex": "from_.codex",
}


def format_to_source_tag(source: str) -> str:
    """Map a parser-side source tag to the sidecar-side design enum value.

    Returns ``"from_<source>"`` for unknown tags so the projection is
    total (never raises, never silently drops the rule). The design's
    enum explicitly allows ``from_AGENTS.md``; per-other-format values
    are a forward-compatible extension that the sidecar's Convention
    model accepts because ``source`` is an open string field there.
    """
    return _SOURCE_TAG_TO_DESIGN_ENUM.get(source, f"from_{source}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_MARKDOWN_FILES: frozenset[str] = frozenset({"AGENTS.md", "CLAUDE.md"})
"""Filenames recognized as markdown rule files (case-sensitive — these
are the canonical filenames; tools using lowercase variants will simply
hit the suffix-based branch below)."""


def parse_file(path: Path) -> list[ParsedConvention]:
    """Dispatch parsing based on filename + suffix.

    Args:
        path: The file to parse. Need not exist — :func:`parse_file`
            returns ``[]`` on a missing file rather than raising.

    Returns:
        A list of :class:`ParsedConvention` records. Empty on any
        failure (missing file, permission denied, decode error, parser
        exception). The function never raises.

    Dispatch rules
    --------------

    1. Name == ``AGENTS.md`` / ``CLAUDE.md`` → :func:`parse_markdown`,
       source = the bare filename.
    2. Name == ``.cursorrules`` → :func:`parse_plain`, source =
       ``".cursorrules"``.
    3. Parent directory == ``.codex`` AND suffix == ``.md`` →
       :func:`parse_markdown`, source = ``".codex"``.
    4. Suffix == ``.md`` AND filename is otherwise unknown →
       :func:`parse_markdown`, source = the bare filename.
    5. Anything else → ``[]``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError) as exc:
        _log.warning("parse_file: cannot read %s: %r", path, exc)
        return []

    file_path = str(path.resolve()) if path.exists() else str(path)
    name = path.name

    try:
        if name in _MARKDOWN_FILES:
            return parse_markdown(text, source=name, file_path=file_path)
        if name == ".cursorrules":
            return parse_plain(text, source=".cursorrules", file_path=file_path)
        if path.parent.name == ".codex" and path.suffix == ".md":
            return parse_markdown(text, source=".codex", file_path=file_path)
        if path.suffix == ".md":
            return parse_markdown(text, source=name, file_path=file_path)
    except Exception as exc:  # noqa: BLE001 — never-raise contract
        _log.warning("parse_file: parser raised on %s: %r", path, exc)
        return []

    return []


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------


# A heading line. Up to 6 hashes, single space, then the title text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Fenced code block delimiters. Match an optional language tag on the
# opening fence (we strip it). We deliberately don't match ``~~~`` —
# it's vanishingly rare in instruction files and keeping the parser
# narrow makes the contract easier to reason about.
_FENCE_OPEN_RE = re.compile(r"^```([A-Za-z0-9_-]*)\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")

# A leading metadata line — ``Applies to:`` / ``Targets:`` / ``Files:``
# at the top of a section body. Case-insensitive match on the prefix
# only; the value is everything after the colon, trimmed.
_APPLIES_TO_RE = re.compile(
    r"^\s*(?:applies\s*to|targets|files?)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


# Max title length matches ConventionCreate.name's max_length=200.
_MAX_TITLE_LEN: int = 200


def parse_markdown(text: str, source: str, file_path: str) -> list[ParsedConvention]:
    """Carve a markdown document into one ``ParsedConvention`` per ``##`` / ``###`` section.

    Args:
        text: The full file body.
        source: Canonical source tag (e.g. ``"AGENTS.md"``). Passed
            through to every produced record.
        file_path: Absolute path string for traceability. Stored on
            every produced record.

    Returns:
        A list of records, one per second- or third-level heading.
        Level-1 (``#``) headings are treated as titles for the file
        as a whole and NOT individually emitted — most AGENTS.md files
        have one top-level heading that's purely a label. Levels 4-6
        also do not start new sections; their bodies are folded into
        the enclosing level-2/3 section.

    Empty sections (heading with no body) are still emitted with an
    empty ``rule_text`` — they're rare and harmless; dropping them
    would risk losing a section the operator added the heading for and
    will fill in later.
    """
    lines = text.splitlines()
    sections: list[ParsedConvention] = []

    # State machine. ``current_title`` is None when we haven't entered
    # a ## / ### section yet (or the file has no headings at all).
    current_title: str | None = None
    current_body: list[str] = []

    def _flush() -> None:
        """Materialize the in-flight section into a ParsedConvention."""
        if current_title is None:
            return
        body_text = "\n".join(current_body).strip("\n")
        applies_to, examples, rule_text = _extract_section_fields(body_text)
        sections.append(
            ParsedConvention(
                title=current_title[:_MAX_TITLE_LEN],
                rule_text=rule_text,
                source=source,
                applies_to=applies_to,
                examples=examples,
                file_path=file_path,
            )
        )

    in_fence = False
    for line in lines:
        # When we're inside a fenced code block, the heading regex must
        # NOT fire — a ``##`` line inside a code block is sample code,
        # not a section break. We still need to track the fence so we
        # don't accidentally treat the closing ``` as a new fence.
        if in_fence:
            current_body.append(line)
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
            continue

        if _FENCE_OPEN_RE.match(line):
            current_body.append(line)
            in_fence = True
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match is None:
            current_body.append(line)
            continue

        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()

        # Level 1 headings: file-level label. Don't emit as a section,
        # but DO flush any pending section so a pre-section preamble
        # gets captured if it was meaningful.
        if level == 1:
            _flush()
            current_title = None
            current_body = []
            continue

        # Levels 2 and 3 start new sections.
        if level in (2, 3):
            _flush()
            current_title = title
            current_body = []
            continue

        # Levels 4-6 fold into the current section's body verbatim
        # (preserving the heading text so the reader can still see the
        # sub-structure inside the rule).
        current_body.append(line)

    _flush()
    return sections


def _extract_section_fields(body: str) -> tuple[list[str], list[str], str]:
    """Pull ``applies_to`` + code-block examples out of a section body.

    Returns ``(applies_to, examples, rule_text)``:

    * ``applies_to`` — list of strings parsed from a leading
      ``Applies to:`` / ``Targets:`` / ``Files:`` line. The line itself
      is stripped from the returned ``rule_text``.
    * ``examples`` — list of raw code-block bodies (fence delimiters
      stripped, language tag dropped, inner text preserved). Code
      blocks STAY in ``rule_text`` too — they're the most useful part
      of the rule for a model reading it later, so we don't strip them
      from the body. The duplication is cheap and the extracted list
      makes them queryable as a structured field.
    * ``rule_text`` — the body with the ``applies_to`` line removed
      (if present) and trimmed of leading/trailing whitespace.
    """
    applies_to: list[str] = []
    examples: list[str] = []
    out_lines: list[str] = []

    in_fence = False
    fence_buf: list[str] = []

    lines = body.splitlines()
    leading_applies_to_consumed = False

    for line in lines:
        if in_fence:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
                examples.append("\n".join(fence_buf))
                fence_buf = []
                out_lines.append(line)
                continue
            fence_buf.append(line)
            out_lines.append(line)
            continue

        if _FENCE_OPEN_RE.match(line):
            in_fence = True
            out_lines.append(line)
            continue

        # Only consume an Applies-to line BEFORE any other content —
        # mid-body matches stay in the rule_text because they're
        # narrative references, not the structured metadata header.
        if not leading_applies_to_consumed and out_lines == [] or all(
            not s.strip() for s in out_lines
        ):
            match = _APPLIES_TO_RE.match(line)
            if match is not None:
                applies_to = _split_targets(match.group(1))
                leading_applies_to_consumed = True
                # NOTE: deliberately skip appending this line — drop it
                # from rule_text so the description doesn't redundantly
                # show the metadata.
                continue

        out_lines.append(line)

    # Defensive: if the file ended with an unterminated fence, flush
    # whatever we collected — losing the buffer would silently
    # discard sample code.
    if in_fence and fence_buf:
        examples.append("\n".join(fence_buf))

    rule_text = "\n".join(out_lines).strip("\n").strip()
    return applies_to, examples, rule_text


def _split_targets(raw: str) -> list[str]:
    """Split an ``Applies to:`` value into individual target strings.

    Commas are the canonical separator. We trim whitespace and drop
    empty entries so a trailing comma or double-comma doesn't produce
    a ghost target.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Plain-text parser
# ---------------------------------------------------------------------------


def parse_plain(text: str, source: str, file_path: str) -> list[ParsedConvention]:
    """Split a plain-text file into one convention per blank-line paragraph.

    Each non-empty paragraph (run of consecutive non-blank lines)
    becomes one :class:`ParsedConvention`:

    * ``title`` — the paragraph's first line, clipped to
      :data:`_MAX_TITLE_LEN`.
    * ``rule_text`` — the full paragraph, stripped of leading /
      trailing whitespace.
    * ``applies_to`` / ``examples`` — always empty. Plain-text files
      don't have a structured way to declare them; cycle-2 may add a
      ``# applies-to: ...`` comment convention if real ``.cursorrules``
      authors start using one.
    """
    paragraphs: list[list[str]] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        if raw_line.strip():
            current.append(raw_line)
            continue
        if current:
            paragraphs.append(current)
            current = []

    if current:
        paragraphs.append(current)

    out: list[ParsedConvention] = []
    for para in paragraphs:
        body = "\n".join(para).strip()
        if not body:
            continue
        title_line = para[0].strip()
        out.append(
            ParsedConvention(
                title=title_line[:_MAX_TITLE_LEN],
                rule_text=body,
                source=source,
                applies_to=[],
                examples=[],
                file_path=file_path,
            )
        )
    return out
