"""Entry point for ``python -m memory.watcher``.

Delegates to :func:`memory.watcher.daemon.main`. Kept tiny so the
daemon module is import-safe even when watchdog is missing — the
ImportError-and-exit-1 path lives inside :func:`main`, not at the
top of any module here.
"""

from __future__ import annotations

import sys

from .daemon import main

if __name__ == "__main__":
    sys.exit(main())
