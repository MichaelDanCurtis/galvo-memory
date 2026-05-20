"""Apply per-label property constraints + indexes to the live Neo4j substrate.

Reads ``memory/ontology/properties.cypher`` and runs each statement via
:meth:`MemoryClient.graph.execute_write`. Every statement is ``IF NOT EXISTS``
so the operation is idempotent — calling :func:`apply_properties` twice is a
no-op the second time.

Per the Phase-2 plan §Task 2, this module is the runtime counterpart to
``properties.cypher``. Tests live in ``memory/tests/test_property_schema.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ontology.label_mapping import _build_memory_settings

if TYPE_CHECKING:  # pragma: no cover — typing-only import
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Resolve the DDL file relative to this module so callers can run from any cwd.
PROPERTIES_CYPHER_PATH: Final[Path] = Path(__file__).parent / "properties.cypher"


def _parse_statements(cypher_source: str) -> list[str]:
    """Split a Cypher script into individual statements.

    Strips ``//`` line comments + blank lines, then splits on ``;`` so each
    yielded statement is exactly one Neo4j command. Doing this client-side
    keeps :meth:`execute_write` calls one-statement-at-a-time, which is
    required because Neo4j refuses multi-statement transactional writes for
    ``CREATE CONSTRAINT`` / ``CREATE INDEX`` (schema commands run outside the
    usual transaction).

    The parser handles ``//`` comments only — we don't currently use ``/* */``
    in the DDL. If we ever need them, swap to a real Cypher tokenizer.
    """
    lines: list[str] = []
    for raw_line in cypher_source.splitlines():
        # Strip ``//`` comments — anything from the marker to end-of-line.
        comment_at = raw_line.find("//")
        if comment_at >= 0:
            raw_line = raw_line[:comment_at]
        stripped = raw_line.strip()
        if stripped:
            lines.append(stripped)

    joined = " ".join(lines)
    statements = [s.strip() for s in joined.split(";") if s.strip()]
    return statements


async def _apply_to_client(client: MemoryClient) -> int:
    """Apply ``properties.cypher`` against an open :class:`MemoryClient`.

    Returns the number of statements run. Each statement is logged at DEBUG
    level — fine to leave a CI run noisy because the count (28-ish) is small.

    Args:
        client: A connected :class:`MemoryClient` (caller owns lifecycle).

    Raises:
        FileNotFoundError: If ``properties.cypher`` is missing (broken build).
        neo4j.exceptions.CypherSyntaxError: If a DDL statement is malformed.
            Surface as-is so the test suite can identify the offending line.
    """
    cypher_source = PROPERTIES_CYPHER_PATH.read_text(encoding="utf-8")
    statements = _parse_statements(cypher_source)

    for stmt in statements:
        logger.debug("Applying DDL: %s", stmt)
        await client.graph.execute_write(stmt, {})

    return len(statements)


async def apply_properties() -> int:
    """Apply ``properties.cypher`` against the default Galvo Neo4j substrate.

    Opens a short-lived :class:`MemoryClient` using the same settings as
    Task 1's :func:`ontology.label_mapping.apply_ontology`. Returns the number
    of DDL statements applied (constraints + indexes).

    This is the canonical entry point for the sidecar bootstrap path (Task 6+)
    and for the test suite.
    """
    from neo4j_agent_memory import MemoryClient  # type: ignore[import-untyped]

    async with MemoryClient(settings=_build_memory_settings()) as client:
        return await _apply_to_client(client)
