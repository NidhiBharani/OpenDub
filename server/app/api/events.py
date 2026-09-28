"""Server-sent events: per-project stream of job + project updates."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from .. import store
from ..jobs import engine

router = APIRouter()

# How often we poll for client disconnect while waiting on the per-project queue. The
# EventSourceResponse `ping` keepalive (15s) is independent of this and always active.
_POLL_INTERVAL = 5.0


@router.get("/projects/{pid}/events")
async def project_events(pid: str, request: Request) -> EventSourceResponse:
    project = store.load(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")

    sub_id, queue = engine.bus.subscribe(pid)

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        try:
            yield {"event": "project", "data": project.model_dump_json()}
            for job in engine.list(pid):
                if job.status in ("queued", "running"):
                    yield {"event": "job", "data": job.model_dump_json()}

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event_name, data = await asyncio.wait_for(queue.get(), timeout=_POLL_INTERVAL)
                except TimeoutError:
                    continue
                yield {"event": event_name, "data": data}
        finally:
            engine.bus.unsubscribe(pid, sub_id)

    return EventSourceResponse(event_generator(), ping=15)
