"""``promote`` — copy a graph node into a markdown instruction file.

Reads the node via the FastAPI sidecar (no direct Neo4j access) and
appends a markdown-formatted projection to the target file. Prints a
unified diff so the operator can see what was written; ``--dry-run``
skips the write and just prints the diff.

Per design §10 — "graph is canonical, files are inputs only, with explicit
promote action" — this is the only path that writes back into
``AGENTS.md`` / ``CLAUDE.md`` / ``memory/feedback_*.md`` after the
Phase-2 cutover. Hooks read those files at SessionStart; the graph is
written by the four lifecycle hooks. The promote action lets an operator
explicitly pin a learned Convention / Decision / Belief so a teammate
without graph access still sees it.

**Sidecar contract assumptions**

* ``GET /api/{label}/{node_id}`` returns 200 + the node's properties as a
  flat dict when the node exists under that label, else 404. The response
  does NOT include the label string (cycle-2 follow-up flagged in
  ``memory/sidecar/routers/nodes.py``); we record the label from whichever
  probe succeeded.
* No direct Neo4j access — the CLI runs on the operator's laptop and
  talks to the sidecar over loopback HTTP.
* No new sidecar endpoint added in cycle 1 — we get by with N round-trips
  via the existing per-label routes. N <= 12 (the ontology size) and each
  request is ~ms on loopback, so the probe latency is acceptable for an
  interactive command. Cycle-2 may add ``GET /api/node/{id}`` that returns
  ``(label, props)`` in one shot.

**Label-aware formatting**

Each label has a hand-crafted markdown projection in :func:`format_for_markdown`.
Five labels are covered explicitly (the most likely promote targets):
``Decision``, ``Belief``, ``Convention``, ``Pattern``, ``Constraint``.
Other labels fall through to a generic "key: value" dump — the operator
can always edit the file by hand after.

**Section insertion**

``--section "## Conventions"`` inserts the content immediately after that
header line. If the section doesn't exist, we append the section header
AND the content at the end of the file. ``--section`` omitted ⇒ plain
append.

**Output**

The unified diff between the file before and after the write goes to
stdout. ``--dry-run`` prints the same diff but skips :func:`pathlib.Path.write_text`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from typing import Any
from urllib import error as _urlerr
from urllib import request as _urlreq

__all__ = [
    "CLIError",
    "DEFAULT_SIDECAR_URL",
    "DEFAULT_TIMEOUT_S",
    "PROBE_ORDER",
    "fetch_node",
    "format_for_markdown",
    "insert_into_file",
    "main",
    "render_diff",
]


DEFAULT_SIDECAR_URL: str = "http://127.0.0.1:7575"
"""Loopback default — matches :data:`hooks.claude_code.lib.sidecar_client.DEFAULT_BASE_URL`.

Overridable via the ``--sidecar-url`` CLI flag and the
``GALVO_MEMORY_SIDECAR_URL`` env var (cycle-2 maybe). The CLI is
operator-facing so we don't need the auto-detect fallback the hooks have.
"""

DEFAULT_TIMEOUT_S: float = 3.0
"""Per-request wall-clock timeout. Matches
:data:`hooks.claude_code.lib.sidecar_client.DEFAULT_TIMEOUT_S` so the
operator's expectations are identical across hook + CLI invocations.
"""

PROBE_ORDER: tuple[str, ...] = (
    # Most likely promote targets first — operators promote conventions
    # and decisions far more often than session metadata.
    "Convention",
    "Decision",
    "Pattern",
    "Belief",
    "Constraint",
    "Mistake",
    "Failure",
    "Task",
    "Session",
    "Commit",
    "Artifact",
    "Test",
)
"""Order in which labels are probed when the user doesn't pass ``--label``.

Order chosen by promote-likelihood, not alphabetical, so the common case
short-circuits early. 12 round-trips worst case on a 404 cascade is
acceptable interactive latency on loopback (well under a second).
"""


class CLIError(RuntimeError):
    """Raised by helper functions when they can't proceed.

    The :func:`main` entry catches this and prints the message to stderr
    + returns a non-zero exit code. Helpers raise this instead of calling
    ``sys.exit`` directly so they remain unit-testable without subprocess
    isolation.
    """


# ---------------------------------------------------------------------------
# Sidecar probing.
# ---------------------------------------------------------------------------


def _http_get(url: str, *, timeout_s: float) -> tuple[int, dict[str, Any] | None]:
    """Issue a GET. Returns ``(status_code, parsed_json_or_none)``.

    Translates :class:`urllib.error.HTTPError` into a ``(status, None)``
    tuple so the caller can branch on 404 without a separate exception
    path. Connection-level failures (refused, timeout) raise
    :class:`CLIError` — the operator needs to know the sidecar's down.

    A non-JSON 200 body is treated as ``(200, None)`` — surprising but
    safe: the caller falls through to "label not found" which surfaces
    a clean error message instead of an uncaught :class:`json.JSONDecodeError`.
    """
    req = _urlreq.Request(url, method="GET")
    try:
        with _urlreq.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except _urlerr.HTTPError as exc:
        # HTTPError IS-A URLError but we want its status code.
        return exc.code, None
    except (_urlerr.URLError, TimeoutError, OSError) as exc:
        # Connection refused, DNS, timeout — sidecar unreachable.
        msg = (
            f"sidecar unreachable at {url}: {exc!r}. "
            f"Is the memory sidecar running on :7575? "
            f"(`docker compose -f memory/docker/docker-compose.yml up`)"
        )
        raise CLIError(msg) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return status, None
    if not isinstance(parsed, dict):
        return status, None
    return status, parsed


def fetch_node(
    node_id: str,
    *,
    sidecar_url: str = DEFAULT_SIDECAR_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    label_hint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Look up a node by id, returning ``(label, properties)``.

    Args:
        node_id: The opaque node id (e.g. ``node_abc123``).
        sidecar_url: Base URL of the sidecar.
        timeout_s: Per-probe HTTP timeout. With 12 probes worst-case the
            total wall-clock cap is ``12 * timeout_s``; in practice the
            short-circuit on the first 200 keeps it under one timeout.
        label_hint: When provided, skip the probe loop and go straight
            to ``GET /api/{label_hint}/{node_id}``. Useful when the
            operator already knows the label and wants to avoid the
            probe overhead.

    Returns:
        ``(label_string, node_props_dict)`` — label is one of the 12
        ontology labels; props is the raw dict the sidecar returned
        (which already includes ``id``, ``scope``, ``created_at`` etc.).

    Raises:
        CLIError: When the node isn't found under any label, or the
            sidecar is unreachable. The error message includes the node
            id so the operator can grep their shell history for context.
    """
    base = sidecar_url.rstrip("/")
    probes: tuple[str, ...] = (label_hint,) if label_hint else PROBE_ORDER

    for label in probes:
        url = f"{base}/api/{label}/{node_id}"
        status, body = _http_get(url, timeout_s=timeout_s)
        if status == 200 and body is not None:
            return label, body
        if status == 404:
            # Expected — this label doesn't match. Move on.
            continue
        # Anything else (5xx, 422, ...) — surface so the operator can
        # debug. We don't swallow it like a 404 because a 5xx means
        # the sidecar saw the request but failed to process it.
        msg = (
            f"sidecar returned status {status} for {url}; "
            f"expected 200 (found) or 404 (not under this label)"
        )
        raise CLIError(msg)

    # All probes 404'd OR the explicit label_hint 404'd.
    if label_hint:
        msg = (
            f"node {node_id!r} not found under label {label_hint!r}. "
            f"Try omitting --label to probe all 12 labels."
        )
    else:
        msg = (
            f"node {node_id!r} not found under any of the 12 ontology labels. "
            f"Check the id (graph node ids look like 'node_abc123')."
        )
    raise CLIError(msg)


# ---------------------------------------------------------------------------
# Markdown formatting.
# ---------------------------------------------------------------------------


def _coerce_str(value: Any) -> str:
    """Coerce a node-prop value into a readable string for markdown.

    Lists become comma-joined; dicts become ``json.dumps``; everything
    else gets ``str()``. ``None`` becomes the literal ``"(none)"`` so a
    missing optional field doesn't render as the four-letter Python
    repr "None" in a user-facing doc.
    """
    if value is None:
        return "(none)"
    if isinstance(value, list):
        if not value:
            return "(none)"
        return ", ".join(_coerce_str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _format_decision(node: dict[str, Any]) -> str:
    """``## Decision: <title>`` + rationale + alternatives.

    Three paragraphs. Alternatives renders as a bullet list when there
    are multiple, single-line "Alternatives: ..." when there's one or
    zero — the bullet form on a single item looks awkward.
    """
    title = _coerce_str(node.get("title", "(untitled decision)"))
    rationale = _coerce_str(node.get("rationale", ""))
    alts = node.get("alternatives_considered") or []
    if isinstance(alts, list) and len(alts) >= 2:
        alts_block = "Alternatives considered:\n" + "\n".join(
            f"- {_coerce_str(a)}" for a in alts
        )
    else:
        alts_block = f"Alternatives considered: {_coerce_str(alts)}"
    return f"## Decision: {title}\n\nRationale: {rationale}\n\n{alts_block}\n"


def _format_belief(node: dict[str, Any]) -> str:
    """``- **<claim>** (held since <valid_from>)`` — single bullet.

    Beliefs are immutable facts (design §4) and are short by nature.
    The valid_from timestamp is the most useful timestamp to a human
    reader; created_at is identical for current beliefs but valid_from
    is the design's first-class temporal property.
    """
    claim = _coerce_str(node.get("claim", "(no claim)"))
    valid_from = node.get("valid_from") or node.get("created_at")
    suffix = f" (held since {_coerce_str(valid_from)})" if valid_from else ""
    return f"- **{claim}**{suffix}\n"


def _format_convention(node: dict[str, Any]) -> str:
    """``## <name>`` + description + source/strength annotation.

    Conventions are what AGENTS.md / CLAUDE.md are mostly composed of, so
    this format must read cleanly as a top-level section.
    """
    name = _coerce_str(node.get("name", "(unnamed convention)"))
    description = _coerce_str(node.get("description", ""))
    source = _coerce_str(node.get("source", "inferred"))
    strength = node.get("strength")
    strength_note = f", strength {strength:.2f}" if isinstance(strength, (int, float)) else ""
    return f"## {name}\n\n{description}\n\n_Source: {source}{strength_note}._\n"


def _format_pattern(node: dict[str, Any]) -> str:
    """``## Pattern: <name>`` + description + evidence/success counts."""
    name = _coerce_str(node.get("name", "(unnamed pattern)"))
    description = _coerce_str(node.get("description", ""))
    evidence = node.get("evidence_count")
    success = node.get("success_rate")
    parts: list[str] = []
    if evidence is not None:
        parts.append(f"evidence count: {_coerce_str(evidence)}")
    if isinstance(success, (int, float)):
        parts.append(f"success rate: {success:.0%}")
    annotations = "; ".join(parts)
    annotation_block = f"\n_{annotations}._\n" if annotations else ""
    return f"## Pattern: {name}\n\n{description}\n{annotation_block}"


def _format_constraint(node: dict[str, Any]) -> str:
    """``## <name>`` + description + constraint type annotation.

    Constraints render close to conventions but call out the constraint
    type so the reader knows whether they're looking at a security rule
    vs a performance budget.
    """
    name = _coerce_str(node.get("name", "(unnamed constraint)"))
    description = _coerce_str(node.get("description", ""))
    ctype = _coerce_str(node.get("constraint_type", "performance"))
    return f"## {name}\n\n{description}\n\n_Constraint type: {ctype}._\n"


def _format_generic(label: str, node: dict[str, Any]) -> str:
    """Fallback dump for labels without a hand-crafted formatter.

    Renders as a definition list — ``- **key**: value`` — under a heading.
    The operator can hand-edit afterwards if they want a different shape;
    this just guarantees the promote operation always succeeds.
    """
    title = _coerce_str(
        node.get("name")
        or node.get("title")
        or node.get("sha")
        or node.get("identifier")
        or node.get("path")
        or node.get("summary")
        or node.get("claim")
        or node.get("error_signature")
        or node.get("id")
        or label
    )
    # Skip internal / generic-render-noisy keys when dumping.
    skip = {"id", "embedding", "name", "type"}
    body_lines: list[str] = []
    for key, value in sorted(node.items()):
        if key in skip:
            continue
        body_lines.append(f"- **{key}**: {_coerce_str(value)}")
    body = "\n".join(body_lines) if body_lines else "(no properties)"
    return f"## {label}: {title}\n\n{body}\n"


_FORMATTERS: dict[str, Any] = {
    "Decision": _format_decision,
    "Belief": _format_belief,
    "Convention": _format_convention,
    "Pattern": _format_pattern,
    "Constraint": _format_constraint,
}


def format_for_markdown(label: str, node: dict[str, Any]) -> str:
    """Render ``node`` as a markdown block for label ``label``.

    Dispatches to a label-specific formatter or falls back to
    :func:`_format_generic`. Output always ends with a single trailing
    newline so :func:`insert_into_file` can splice it without
    blank-line gymnastics.
    """
    formatter = _FORMATTERS.get(label)
    if formatter is None:
        return _format_generic(label, node)
    return formatter(node)


# ---------------------------------------------------------------------------
# File-insertion logic.
# ---------------------------------------------------------------------------


def insert_into_file(
    path: pathlib.Path,
    content: str,
    section: str | None,
) -> tuple[str, str]:
    """Append (or insert) ``content`` into ``path``.

    Args:
        path: Target markdown file. Missing-file is OK — we treat it as
            an empty existing file and create it on write.
        content: Markdown block to insert. Should end with a trailing newline.
        section: When provided, e.g. ``"## Conventions"``, the content is
            inserted immediately after that header line. When the header
            doesn't exist in the file, we append the header AND content
            at the end. When ``None``, content is plain-appended.

    Returns:
        ``(before, after)`` — the file content before and after the
        insertion. The caller diffs these and may decide not to write
        (``--dry-run``). We do NOT touch the filesystem here; the caller
        owns the write side-effect for testability.
    """
    before = path.read_text(encoding="utf-8") if path.exists() else ""

    if section is None:
        # Plain append with a blank-line separator if the file isn't empty.
        separator = "" if before == "" or before.endswith("\n\n") else (
            "\n" if before.endswith("\n") else "\n\n"
        )
        after = before + separator + content
        return before, after

    # Section mode: find the header line.
    target_header = section.strip()
    lines = before.splitlines(keepends=True) if before else []
    insert_index: int | None = None
    for i, line in enumerate(lines):
        if line.rstrip("\n").strip() == target_header:
            insert_index = i + 1
            break

    if insert_index is None:
        # Header missing → append header + content at the end.
        # Ensure there's a blank line before the new section if the file
        # has prior content.
        separator = "" if before == "" or before.endswith("\n\n") else (
            "\n" if before.endswith("\n") else "\n\n"
        )
        after = before + separator + target_header + "\n\n" + content
        return before, after

    # Header present → splice content immediately after.
    # Skip any blank lines directly under the header so consecutive
    # promotes don't pile blank lines between entries.
    splice_at = insert_index
    # Insert content (which already ends with \n). If the line at
    # splice_at isn't a blank line, prepend a blank line so the new
    # block is visually separated from whatever follows.
    prefix = "\n" if (
        splice_at < len(lines)
        and lines[splice_at].strip() != ""
    ) else ""
    suffix = "" if content.endswith("\n\n") else ("\n" if content.endswith("\n") else "\n\n")
    # The header line itself ends with \n already (kept via splitlines).
    # We insert: "\n" (blank line after header) + content + maybe-trailing-newline.
    new_block = "\n" + content + (suffix if prefix else "")
    after = "".join(lines[:splice_at]) + new_block + prefix + "".join(lines[splice_at:])
    return before, after


def render_diff(before: str, after: str, path: pathlib.Path) -> str:
    """Render a unified diff between ``before`` and ``after``.

    Returns the diff as a single string. Empty when the strings are
    identical (e.g. a no-op promote). Uses :func:`difflib.unified_diff`
    with the file path embedded in the ``a/`` and ``b/`` markers so the
    output looks like a normal ``git diff`` snippet.
    """
    if before == after:
        return ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


# ---------------------------------------------------------------------------
# CLI entry.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser. Factored out for ``--help`` testability."""
    parser = argparse.ArgumentParser(
        prog="memory.cli promote",
        description=(
            "Copy a graph node into a markdown instruction file. "
            "Reads from the FastAPI sidecar at :7575 (override via "
            "--sidecar-url), formats the node per its label, and "
            "appends/inserts it into the target markdown file."
        ),
    )
    parser.add_argument(
        "node_id",
        help="Node id from the graph (e.g. node_abc123).",
    )
    parser.add_argument(
        "--to",
        required=True,
        type=pathlib.Path,
        dest="target",
        metavar="PATH",
        help="Target markdown file (e.g. AGENTS.md, CLAUDE.md, memory/feedback_xyz.md).",
    )
    parser.add_argument(
        "--section",
        default=None,
        help=(
            "Section header to insert under, e.g. \"## Conventions\". When the "
            "header doesn't exist in the file, the header and content are both "
            "appended at the end. Omit for plain append."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        choices=list(PROBE_ORDER),
        help=(
            "Skip the label probe loop and go straight to "
            "GET /api/<LABEL>/<id>. Use when you already know the node's label."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the diff that WOULD be written, but don't touch the file.",
    )
    parser.add_argument(
        "--sidecar-url",
        default=DEFAULT_SIDECAR_URL,
        help=f"Sidecar base URL (default: {DEFAULT_SIDECAR_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        dest="timeout_s",
        metavar="SECONDS",
        help=f"Per-request HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m memory.cli promote``.

    Returns the exit code so the dispatcher in :mod:`memory.cli.__main__`
    can re-raise via :func:`sys.exit`.

    Flow:
    1. Parse args (argparse raises :class:`SystemExit` on bad input — fine).
    2. Fetch the node via the sidecar. Errors print to stderr + return 2.
    3. Format the node per its label.
    4. Compute the (before, after) diff against the target file.
    5. Print the diff to stdout.
    6. Unless ``--dry-run``, write ``after`` to the file.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        label, node = fetch_node(
            args.node_id,
            sidecar_url=args.sidecar_url,
            timeout_s=args.timeout_s,
            label_hint=args.label,
        )
    except CLIError as exc:
        print(f"promote: error: {exc}", file=sys.stderr)
        return 2

    block = format_for_markdown(label, node)
    before, after = insert_into_file(args.target, block, args.section)
    diff = render_diff(before, after, args.target)

    if not diff:
        # No-op — file already contains identical bytes after the
        # would-be insertion. Surface this rather than silently exit
        # so the operator can investigate.
        print(
            f"promote: no change — {args.target} already contains the block",
            file=sys.stderr,
        )
        return 0

    print(diff)

    if args.dry_run:
        print(
            f"\n--- dry-run: {args.target} NOT modified ---",
            file=sys.stderr,
        )
        return 0

    # Ensure the parent directory exists; promotes into a subfolder
    # (memory/feedback_*.md) are common.
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(after, encoding="utf-8")
    print(
        f"\n--- wrote {len(after) - len(before)} bytes to {args.target} ---",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via __main__.py
    sys.exit(main())
