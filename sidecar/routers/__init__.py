"""Router package for the FastAPI sidecar.

Each router module exposes a ``router`` attribute that :mod:`sidecar.app`
imports directly (``from sidecar.routers.X import router``) and wires up
with ``app.include_router``. We deliberately do NOT re-export from this
package's ``__init__`` because that would force :mod:`sidecar.app` to load
every router module at import time — making parallel task lanes step on
each other.

Routers shipped:

* :mod:`sidecar.routers.nodes` (Task 8) — generic REST CRUD for the 12
  ontology labels, with dynamic per-label dispatch keyed off
  :data:`sidecar.models.LABEL_CREATE_MODELS`.
* :mod:`sidecar.routers.sessions` (Task 10) — SessionEnd utility scorer
  at ``POST /api/sessions/{id}/score``.
"""
