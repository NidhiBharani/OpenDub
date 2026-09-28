"""System routes: which models are resident, and a manual way to free them."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..providers import _runtime
from ..providers._runtime import manager

router = APIRouter()


class UnloadBody(BaseModel):
    key: str | None = None  # a key as listed by GET /system/models; omitted = everything


@router.get("/system/models")
async def list_models() -> dict[str, Any]:
    gpu = await asyncio.to_thread(_runtime.gpu_usage)  # nvidia-smi; {} without a GPU
    budget = await asyncio.to_thread(_runtime.default_gpu_budget_gb)
    return {
        "models": manager.snapshot(),
        "idle_ttl_s": _runtime.default_idle_ttl_s(),
        "gpu_budget_gb": budget,
        "gpu": gpu,
    }


@router.post("/system/models/unload")
async def unload_models(body: UnloadBody | None = None) -> dict[str, Any]:
    """Unload one model (``key``) or all. Models in use finish their current work first: they
    unload the moment they are released, and are reported under ``pending``."""
    key = body.key if body else None
    if key:
        if key not in manager.loaded_keys():
            raise HTTPException(404, f"no loaded model '{key}'")
        unloaded = [key] if await asyncio.to_thread(manager.unload, key) else []
    else:
        unloaded = await asyncio.to_thread(manager.unload_all)
    pending = [m["key"] for m in manager.snapshot() if key in (None, m["key"])]
    return {"unloaded": unloaded, "pending": pending}
