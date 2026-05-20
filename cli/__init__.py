"""``memory.cli`` — operator-facing command line for the Galvo memory layer.

Cycle-1 surface is intentionally minimal:

* ``promote`` — copy a graph node into a markdown instruction file
  (per design §10: "graph is canonical, files are inputs only, with
  explicit promote action").

Subcommands are wired into :mod:`memory.cli.__main__` for the
``python -m memory.cli <subcommand>`` invocation. Each subcommand is a
self-contained module that exposes a ``main(argv) -> int`` callable so
the dispatcher can route to it without importing every submodule eagerly.

Why a package and not a single-file script: cycle-2 will add ``demote``,
``audit``, ``list-feedback``, and ``consolidate``. Splitting them into
sibling modules keeps each one independently testable and lets ``--help``
discovery scale without one giant argparse tree.
"""

from __future__ import annotations

__all__: list[str] = []
