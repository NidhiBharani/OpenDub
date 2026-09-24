"""Capability map introspection: the 41 capabilities, the presets, and what a project's
configuration resolves to."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import capabilities, store
from ..capabilities import Capability, ResolvedCapability

router = APIRouter()


class PresetInfo(BaseModel):
    id: str
    description: str
    capabilities: list[str]


class CapabilityMap(BaseModel):
    phases: dict[str, str]
    capabilities: list[Capability]
    presets: list[PresetInfo]


@router.get("/capabilities")
async def capability_map() -> CapabilityMap:
    return CapabilityMap(
        phases=capabilities.PHASES,
        capabilities=capabilities.CAPABILITIES,
        presets=[
            PresetInfo(id=k, description=capabilities.PRESET_LABELS[k], capabilities=v)
            for k, v in capabilities.PRESETS.items()
        ],
    )


@router.get("/projects/{pid}/capabilities")
async def resolved_capabilities(pid: str) -> dict[str, ResolvedCapability]:
    project = store.load(pid)
    if project is None:
        raise HTTPException(404, f"project '{pid}' not found")
    return capabilities.resolve(project.pipeline)
