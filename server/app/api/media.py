"""Media file + waveform serving routes.

All paths come in relative to a project dir; `store.resolve` validates against path traversal.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import store

router = APIRouter()

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".json": "application/json",
}

WAVEFORM_ASSETS = {"original", "vocals", "dub_mix"}


@router.get("/media/{pid}/{path:path}")
async def get_media(pid: str, path: str) -> FileResponse:
    try:
        abs_path = store.resolve(pid, path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not abs_path.is_file():
        raise HTTPException(404, f"media '{path}' not found for project '{pid}'")
    media_type = _CONTENT_TYPES.get(abs_path.suffix.lower())
    return FileResponse(abs_path, media_type=media_type)


@router.get("/projects/{pid}/waveform/{asset}")
async def get_waveform(pid: str, asset: str) -> dict:
    if asset not in WAVEFORM_ASSETS:
        raise HTTPException(
            400, f"unknown waveform asset '{asset}'; expected one of {sorted(WAVEFORM_ASSETS)}"
        )
    try:
        abs_path = store.resolve(pid, f"waveforms/{asset}.json")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not abs_path.is_file():
        raise HTTPException(404, f"waveform '{asset}' not found for project '{pid}'")
    return json.loads(abs_path.read_text())
