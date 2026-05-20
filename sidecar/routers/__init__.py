"""Router package for the FastAPI sidecar.

Intentionally empty: each router module exposes a ``router`` attribute
that :mod:`sidecar.app` imports directly via ``from sidecar.routers.X
import router`` and wires up with ``app.include_router``. We do NOT
re-export here because that would force :mod:`sidecar.app` to load every
router module at import time — including ones whose dependencies haven't
been ported yet — which makes parallel task lanes step on each other.

Routers shipped so far:

* :mod:`sidecar.routers.sessions` (Task 10) — SessionEnd utility scorer.

Later tasks add :mod:`sidecar.routers.nodes` (Task 8) and any others.
"""
