"""``python -m memory.cli <subcommand> ...`` — subcommand dispatcher.

Cycle-1 surface: only ``promote``. The dispatch is hand-rolled (not
argparse subparsers) because each subcommand owns its own
:class:`argparse.ArgumentParser` — that keeps subcommand-specific
``--help`` clean and lets future subcommands be added without touching
this file beyond adding a row to :data:`_SUBCOMMANDS`.

Exit codes:

* ``0`` — subcommand succeeded.
* ``1`` — no subcommand given, or unknown subcommand. Usage printed.
* ``2`` — subcommand raised :class:`cli.promote.CLIError` (e.g. node
  not found, sidecar unreachable). Error message printed to stderr.
* Anything else — propagated from the subcommand.
"""

from __future__ import annotations

import sys
from typing import Callable

__all__ = ["main"]


# Lazy-import the subcommand entry points so ``python -m memory.cli``
# without a subcommand doesn't pay the import cost of every subcommand.
def _load_promote() -> Callable[[list[str] | None], int]:
    from cli.promote import main as promote_main

    return promote_main


_SUBCOMMANDS: dict[str, Callable[[], Callable[[list[str] | None], int]]] = {
    "promote": _load_promote,
}
"""Registry of subcommand name → loader function. The loader returns the
subcommand's ``main(argv) -> int``. Lazy so ``--help`` at the dispatcher
level doesn't import every submodule.

Adding a cycle-2 subcommand: write the module's ``main(argv)`` and
register a one-line loader here.
"""


def _print_usage(stream=None) -> None:
    """Print top-level usage. Lists registered subcommands.

    The leading ``python -m memory.cli`` is intentional — operators
    learn the invocation form from the help text, not from documentation
    elsewhere.

    ``stream`` defaults to :data:`sys.stderr` resolved at CALL time, not
    at function-definition time. This matters under pytest's capture
    fixtures, which rebind ``sys.stderr`` per-test — capturing the
    module-load-time stderr in a default argument would write to a
    closed stream after the second test.
    """
    if stream is None:
        stream = sys.stderr
    print(
        "usage: python -m memory.cli <subcommand> [args...]\n"
        "\nsubcommands:\n"
        + "\n".join(f"  {name}" for name in sorted(_SUBCOMMANDS))
        + "\n\nRun `python -m memory.cli <subcommand> --help` for subcommand details.",
        file=stream,
    )


def main(argv: list[str] | None = None) -> int:
    """Top-level dispatcher entry point.

    Args:
        argv: The argv tail AFTER ``python -m memory.cli``. When ``None``,
            we read from :data:`sys.argv` (so the module form works).
            The first element is the subcommand name; the rest are passed
            to the subcommand's own argv.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_usage()
        return 1

    name, *rest = argv

    # Treat top-level --help / -h as usage info, not as a subcommand.
    if name in {"-h", "--help"}:
        _print_usage(stream=sys.stdout)
        return 0

    loader = _SUBCOMMANDS.get(name)
    if loader is None:
        print(f"unknown subcommand: {name!r}", file=sys.stderr)
        _print_usage()
        return 1

    sub_main = loader()
    return sub_main(rest)


if __name__ == "__main__":
    sys.exit(main())
