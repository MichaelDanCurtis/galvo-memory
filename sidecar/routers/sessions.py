"""SessionEnd scoring endpoint — wraps :mod:`sidecar.scoring`.

Single endpoint::

    POST /api/sessions/{session_id}/score

Body: :class:`sidecar.scoring.ScoringPayload`
Response: :class:`sidecar.scoring.ScoringReport`

The endpoint exists purely to give the SessionEnd hook (Task 15) an HTTP
seam — the actual scoring logic lives in :mod:`sidecar.scoring` and is
unit-tested independently of FastAPI. The router contributes only the
URL surface, path/body consistency check, and the dependency injection
needed to reach the live :class:`MemoryClient`.

The ``session_id`` appears in both the URL path *and* the request body;
we enforce equality with a 400 because a mismatch usually means the
client built the request wrong (e.g. typo in the path) and silently
scoring the body's id would be confusing during cycle-1 debugging.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from sidecar.deps import MemoryDep
from sidecar.scoring import ScoringPayload, ScoringReport, score_session


__all__ = ["router"]


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/{session_id}/score", response_model=ScoringReport)
async def score_session_endpoint(
    session_id: str,
    payload: ScoringPayload,
    memory: MemoryDep,
) -> ScoringReport:
    """Run the D5 cycle-1 utility scorer over a session's RETRIEVED_IN edges.

    Args:
        session_id: From the URL path. Must equal ``payload.session_id``
            or the endpoint returns 400 — guards against client typos
            silently scoring the wrong session.
        payload: SessionEnd signal bundle from the Task-15 hook.
        memory: Injected via :data:`sidecar.deps.MemoryDep`.

    Returns:
        :class:`ScoringReport` with per-edge breakdowns and write counts.

    Raises:
        HTTPException 400: When the path's ``session_id`` does not match
            the body's. The detail string names both values to make
            client-side debugging trivial.
    """
    if payload.session_id != session_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"session_id in path ({session_id!r}) must match body "
                f"({payload.session_id!r})"
            ),
        )
    return await score_session(memory, payload=payload)
