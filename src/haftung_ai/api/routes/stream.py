"""SSE progress streaming endpoint."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

# Simple progress store
_progress: dict[str, list[dict]] = {}


def emit_progress(analysis_id: str, step: str, status: str, detail: str = ""):
    """Emit a progress event (called by agents)."""
    if analysis_id not in _progress:
        _progress[analysis_id] = []
    _progress[analysis_id].append({"step": step, "status": status, "detail": detail})


@router.get("/{analysis_id}")
async def stream_progress(analysis_id: str):
    """Stream analysis progress via SSE."""

    async def event_generator():
        sent = 0
        while True:
            events = _progress.get(analysis_id, [])
            while sent < len(events):
                yield {"data": json.dumps(events[sent])}
                sent += 1
                if events[sent - 1].get("status") == "complete":
                    return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())
