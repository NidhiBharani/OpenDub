"""Provider registry introspection + settings routes.

Provider instantiation and `available()` calls may import optional ML dependencies lazily inside the
provider classes themselves (per providers/base.py contract); nothing here does heavy imports.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config
from ..providers.base import REGISTRY, ProviderMeta, get_provider_class

router = APIRouter()


class ProviderInfo(BaseModel):
    meta: ProviderMeta
    available: bool
    reason: str
    configured_options: dict[str, Any]


class ProviderOptionsBody(BaseModel):
    options: dict[str, Any]


def _secret_keys(meta: ProviderMeta) -> set[str]:
    return {f.key for f in meta.fields if f.type == "secret"}


def _build_info(provider_id: str) -> ProviderInfo:
    cls = get_provider_class(provider_id)
    options = config.provider_options(provider_id)
    try:
        instance = cls(options)
        available, reason = instance.available()
    except Exception as e:  # noqa: BLE001 - provider code is untrusted/optional
        available, reason = False, f"error initializing provider: {e}"
    configured_options = config.mask_options(options, _secret_keys(cls.meta))
    return ProviderInfo(
        meta=cls.meta,
        available=available,
        reason=reason,
        configured_options=configured_options,
    )


@router.get("/providers")
async def list_providers() -> list[ProviderInfo]:
    return [_build_info(pid) for pid in sorted(REGISTRY.keys())]


@router.put("/settings/providers/{provider_id}")
async def set_provider_options(provider_id: str, body: ProviderOptionsBody) -> ProviderInfo:
    try:
        cls = get_provider_class(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    config.set_provider_options(provider_id, body.options, _secret_keys(cls.meta))
    return _build_info(provider_id)


@router.post("/providers/{provider_id}/check")
async def check_provider(provider_id: str) -> dict[str, Any]:
    try:
        cls = get_provider_class(provider_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    options = config.provider_options(provider_id)
    try:
        instance = cls(options)
        available, reason = await asyncio.to_thread(instance.available, True)
    except Exception as e:  # noqa: BLE001 - provider code is untrusted/optional
        available, reason = False, f"error checking provider: {e}"
    return {"available": available, "reason": reason}
