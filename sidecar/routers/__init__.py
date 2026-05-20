"""FastAPI routers for the sidecar HTTP surface.

Modules:

* :mod:`nodes` (Task 8) — generic REST CRUD for the 12 ontology labels.
  One router with dynamic per-label dispatch keyed off
  :data:`sidecar.models.LABEL_CREATE_MODELS`.

Later tasks add :mod:`sessions` (Task 10 — SessionEnd scoring) and
possibly :mod:`feedback` (Task 9 — retrieval-edge writer).
"""
