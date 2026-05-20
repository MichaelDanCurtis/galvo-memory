"""Apply the edge-property indexes declared in ``edges.cypher``.

The 16 edge types themselves are created on first MERGE — Neo4j auto-creates
relationship types when they're first referenced. What this module does is
provision the supporting indexes (currently three; see ``ALL_EDGE_INDEX_NAMES``
in :mod:`ontology.edges`) so cycle-1 logging queries and cycle-2 consolidation
queries don't degrade into full scans.

Design references:
    * §4 "Edge types"
    * §5 "Per-retrieval logging"

The function is idempotent — every statement in ``edges.cypher`` uses
``IF NOT EXISTS``. It is safe to call on every sidecar boot.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]


_CYPHER_FILE: Path = Path(__file__).with_name("edges.cypher")


def _load_statements() -> list[str]:
    """Return Cypher statements from ``edges.cypher`` with comments stripped.

    Neo4j drivers reject multi-statement strings; we split on ``;`` and drop
    line-comments (``//``) so each ``execute_write`` call sees a single
    statement.
    """
    raw = _CYPHER_FILE.read_text(encoding="utf-8")
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        cleaned_lines.append(line)
    body = "\n".join(cleaned_lines)
    return [s.strip() for s in body.split(";") if s.strip()]


async def apply_edge_indexes() -> list[str]:
    """Apply every statement in ``edges.cypher`` against the live Neo4j.

    Connects via the helper in :mod:`ontology.label_mapping` so connection
    parameters stay in one place.

    Returns:
        The list of statements executed (useful for tests + logging).

    Raises:
        ClientError: If Neo4j rejects any statement — e.g. syntax error or
            permission denial.
    """
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    from .label_mapping import _build_memory_settings

    statements = _load_statements()
    async with MemoryClient(settings=_build_memory_settings()) as client:
        for statement in statements:
            await client.graph.execute_write(statement, {})
    return statements


async def list_edge_indexes(client: MemoryClient | None = None) -> list[dict]:
    """Return the index rows currently in Neo4j whose name starts with the
    edge-index prefix.

    Args:
        client: An open ``MemoryClient``. If ``None``, opens a new one using
            :func:`_build_memory_settings`.

    Returns:
        List of rows ``{name, type, entityType, ...}`` from ``SHOW INDEXES``,
        filtered to those we declared in :data:`ontology.edges.ALL_EDGE_INDEX_NAMES`.
    """
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    from .edges import ALL_EDGE_INDEX_NAMES
    from .label_mapping import _build_memory_settings

    async def _query(c: MemoryClient) -> list[dict]:
        return await c.graph.execute_read(
            "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties "
            "WHERE name IN $names "
            "RETURN name, type, entityType, labelsOrTypes, properties",
            {"names": ALL_EDGE_INDEX_NAMES},
        )

    if client is not None:
        return await _query(client)
    async with MemoryClient(settings=_build_memory_settings()) as opened:
        return await _query(opened)
