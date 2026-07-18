"""Shared helpers for cloud providers that talk to job-based REST APIs.

Most audio/video cloud services follow the same shape: upload a file, kick off an async job, poll
until it finishes, then download result file(s). These helpers centralize the download and polling
so each provider only encodes its own endpoints and payload shapes.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx


async def download_file(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` (created parent dirs). Raises RuntimeError on HTTP error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread())[:300]
            raise RuntimeError(f"download failed ({resp.status_code}) for {url}: {body!r}")
        with dest.open("wb") as f:
            async for chunk in resp.aiter_bytes(65536):
                f.write(chunk)


async def poll_until(
    fetch: Callable[[], Awaitable[dict[str, Any]]],
    *,
    is_done: Callable[[dict[str, Any]], bool],
    is_failed: Callable[[dict[str, Any]], bool],
    on_poll: Callable[[dict[str, Any]], None] | None = None,
    interval: float = 3.0,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Poll ``fetch()`` every ``interval`` seconds until ``is_done`` (return payload) or
    ``is_failed`` (raise). Times out after ``timeout`` seconds.

    ``on_poll`` receives each payload — use it to report progress. Kept provider-agnostic: each
    API names its status fields differently, so the predicates are passed in.
    """
    waited = 0.0
    while True:
        payload = await fetch()
        if on_poll is not None:
            on_poll(payload)
        if is_failed(payload):
            raise RuntimeError(f"remote job failed: {str(payload)[:400]}")
        if is_done(payload):
            return payload
        if waited >= timeout:
            raise TimeoutError(f"remote job did not finish within {timeout:.0f}s")
        await asyncio.sleep(interval)
        waited += interval
