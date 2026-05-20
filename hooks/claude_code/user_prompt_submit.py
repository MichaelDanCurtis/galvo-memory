#!/usr/bin/env python3
"""UserPromptSubmit hook — targeted semantic retrieval on the user's prompt (Task 13).

Invoked by Claude Code each time the user submits a prompt. The hook:

1. Reads the hook event JSON from stdin (``HookInputBase.from_stdin``).
2. Extracts ``extra.prompt`` (Claude Code's per-event payload key).
3. Resolves the scope filter from the hook's ``cwd``.
4. Queries the sidecar's ``GET /api/search/Entity`` endpoint with the prompt
   text (truncated to :data:`MAX_QUERY_CHARS` to fit the embedder's window).
5. Formats up to :data:`MAX_HITS` of the returned hits as a compact markdown
   context block on stdout — Claude Code injects this verbatim into the
   model's input context before the actual user prompt reaches the model.

We search ``Entity`` (the library's super-label that every 12-label custom
node carries; see :func:`memory.ontology.label_mapping.apply_ontology`) rather
than per-label so a single round-trip surfaces hits across all node types.
Per-label searches would have to fan out 12 calls and aggregate — that
violates the <500ms latency target for cycle 1.

**Latency target:** <500ms wall-clock. The sidecar's vector search runs on
loopback (127.0.0.1:7575) and typically responds in 100-300ms for a 384-dim
MiniLM index of <100k nodes. The 3.0s client default is the hard ceiling
(:data:`hooks.claude_code.lib.sidecar_client.DEFAULT_TIMEOUT_S`); a healthy
hook is well under that.

**Failure mode:** every error path (sidecar down, malformed input, empty
prompt, zero hits) silently exits 0 without writing to stdout. The user's
session is never blocked, never sees a stack trace, and never has the model
context polluted by an error message. Diagnostics go to
``~/.galvo-memory/logs/user_prompt_submit.log``.

**Stdout contract:** Claude Code captures whatever this script writes to
stdout and prepends it to the model's input context for the upcoming prompt.
We use a markdown header + bullet list so the model can either ignore it or
reference it naturally — same convention used by the existing
``# Memory`` system-reminder block.
"""
from __future__ import annotations

import sys
from typing import Any

from hooks.claude_code.lib.logging import setup_hook_logger
from hooks.claude_code.lib.scope import detect_scope_for_hook
from hooks.claude_code.lib.sidecar_client import SidecarClient
from hooks.claude_code.lib.types import UNKNOWN_EVENT, HookInputBase

__all__ = [
    "MAX_HITS",
    "MAX_QUERY_CHARS",
    "main",
]


MAX_HITS: int = 5
"""Maximum number of hits formatted into the context block.

Cycle 1 picks 5 as a compromise: enough to surface relevant context across
project, personal, and universal scopes without overwhelming the model's
input. Cycle 2 may tune this per-scope (e.g. 3 project + 2 personal) once
we have retrieval-utility scores from the SessionEnd scorer (Task 10).
"""

MAX_QUERY_CHARS: int = 500
"""Hard cap on the prompt text we hand to the sidecar's embedder.

MiniLM-L6-v2 (Phase-1 spike §1, design D2) tokenizes ~6 chars/token and
caps at 512 tokens, so 500 chars stays well inside the model's window
while preserving the user's intent. Truncation happens client-side so
the sidecar's failure mode (long inputs would degrade similarity scores)
never fires.
"""

# The 12 custom ontology labels — used to pick the most specific tag for the
# bullet prefix when the search response includes a multi-label hit.
_KNOWN_LABELS: frozenset[str] = frozenset(
    {
        "Decision",
        "Pattern",
        "Convention",
        "Constraint",
        "Task",
        "Session",
        "Mistake",
        "Commit",
        "Failure",
        "Artifact",
        "Test",
        "Belief",
    }
)


def main() -> int:
    """Hook entry point. NEVER raises; always returns 0.

    The contract with Claude Code is "exit cleanly, write retrieval context
    to stdout, write diagnostics to a private log". Any uncaught exception
    would bubble up as a traceback in the user's session — much worse than
    a missed retrieval injection.

    Returns:
        Always 0. The hook intentionally swallows every error path because
        Claude Code interprets non-zero exits as hook misconfiguration and
        surfaces a banner to the user. We'd rather silently no-op.
    """
    log = setup_hook_logger("user_prompt_submit")

    hook_input = HookInputBase.from_stdin()
    if hook_input.hook_event_name == UNKNOWN_EVENT:
        # Malformed stdin already logged at WARNING by from_stdin's caller
        # (none, in practice — but the sentinel is the standard signal).
        return 0

    # Claude Code puts the user-typed prompt under extra.prompt per its
    # hook protocol (reverse-engineered from the dispatcher; see
    # hooks.claude_code.lib.types module docstring for the source-of-truth
    # caveat).
    prompt_value = (hook_input.extra or {}).get("prompt", "")
    if not isinstance(prompt_value, str):
        log.info("non-string prompt payload (%r); skipping retrieval", type(prompt_value))
        return 0

    if not prompt_value.strip():
        log.info("empty prompt; skipping retrieval")
        return 0

    query = prompt_value[:MAX_QUERY_CHARS]
    scope = detect_scope_for_hook(hook_input.cwd)

    client = SidecarClient()
    # Search Entity (the super-label every custom-label node carries) so a
    # single round-trip surfaces hits across all 12 labels. Per-label
    # fan-out would be 12 calls and miss the latency target.
    hits = client.search("Entity", query, scope=scope, limit=MAX_HITS)

    if not hits:
        log.info("no retrieval hits for scope=%s query_len=%d", scope, len(query))
        return 0

    block = _format_block(query, hits)
    sys.stdout.write(block)
    sys.stdout.write("\n")
    log.info("injected %d hits for scope=%s", len(hits), scope)
    return 0


def _format_block(query: str, hits: list[dict[str, Any]]) -> str:
    """Format hits as a compact markdown context block.

    Format::

        # Memory: 3 relevant hits for "ruff config..."
        - [Decision] Use ruff for linting
        - [Belief] ScoreRow is nested, not flat
        - [Pattern] Wave-and-converge merge pattern

    The header includes the query excerpt so the model can self-correct if
    the retrieval pulled in unrelated hits (e.g. wrong scope). The bullets
    carry the label prefix because the model uses that as a strong signal
    for "this is a Decision I should respect" vs "this is just a pattern".

    Args:
        query: The (truncated) prompt text the sidecar was queried with.
            Echoed back in the header excerpt so the model can sanity-check.
        hits: The sidecar search response — a list of node-property dicts.
            Each dict has the node's properties at the top level (plus an
            optional ``_score`` from the vector index).

    Returns:
        A multi-line markdown string. NO trailing newline (the caller adds
        one before writing to stdout so the model's context boundary is
        clean).
    """
    # Excerpt the query for the header — newline-stripped + capped so the
    # block stays one line. The trailing ellipsis ("…", U+2026) signals
    # truncation without consuming three characters of width.
    excerpt = query[:80].replace("\n", " ").replace("\r", " ")
    if len(query) > 80:
        excerpt = excerpt + "…"

    capped = hits[:MAX_HITS]
    lines = [f'# Memory: {len(capped)} relevant hits for "{excerpt}"']
    for hit in capped:
        label = _hit_label(hit)
        title = _hit_title(hit)
        lines.append(f"- [{label}] {title}")
    return "\n".join(lines)


def _hit_label(hit: dict[str, Any]) -> str:
    """Extract the most specific custom label from a hit.

    The sidecar's ``GET /api/search/Entity`` response is a list of
    flat-property dicts (see :mod:`sidecar.routers.nodes` ``_row_to_dict``).
    Node labels are not currently surfaced in the response shape
    (Task 8's response models project node properties only) — so we
    inspect optional ``labels`` / ``extra_labels`` keys for forward
    compatibility, then fall back to the ``type`` property (the library's
    EntityType discriminator: CONCEPT/EVENT/OBJECT/FACT), and finally to
    a generic ``Entity`` sentinel.

    **Task 8 coordination note:** if Task 8's response shape is amended
    to include the multi-label tag (which would be the natural place
    to surface it), this function picks it up automatically — no
    coordination required, because we already check ``labels`` /
    ``extra_labels`` first.

    Args:
        hit: A single search-response dict.

    Returns:
        The label name (e.g. ``"Decision"``) — the most specific custom
        label that's known to the ontology, or the node's ``type`` if no
        custom label is present, or ``"Entity"`` as a last resort.
    """
    for key in ("labels", "extra_labels"):
        value = hit.get(key)
        if isinstance(value, list):
            for label in value:
                if isinstance(label, str) and label in _KNOWN_LABELS:
                    return label
    node_type = hit.get("type")
    if isinstance(node_type, str) and node_type:
        return node_type
    return "Entity"


def _hit_title(hit: dict[str, Any]) -> str:
    """Pick the best human-readable handle for a hit.

    Each ontology label has a different canonical name property (see
    :data:`memory.ontology.label_mapping.NAME_PROPERTY_PER_LABEL`):

    * Decision/Task/Session → ``title``
    * Pattern/Convention/Constraint → ``name``
    * Belief → ``claim``
    * Artifact → ``path``
    * Commit → ``sha``
    * Mistake → ``summary``
    * Failure → ``error_signature``
    * Test → ``identifier``

    Rather than dispatch on label (we don't always know it), we probe a
    priority list of keys. The first non-empty string wins. Final
    fallback is the node's ``id`` so the bullet always carries SOME
    identifier — better than a blank line.

    Args:
        hit: A single search-response dict.

    Returns:
        Trimmed, single-line title string. Never empty (falls back to id
        or the literal ``"?"`` if even id is missing).
    """
    for key in ("title", "name", "claim", "summary", "error_signature", "path", "identifier", "sha"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().splitlines()[0]
    node_id = hit.get("id")
    if isinstance(node_id, str) and node_id:
        return node_id
    return "?"


if __name__ == "__main__":
    sys.exit(main())
