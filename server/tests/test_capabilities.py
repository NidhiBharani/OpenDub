"""Capability map: registry integrity, presets, resolution rules, and the stage step loop."""
from __future__ import annotations

import pytest

from app import capabilities as caps
from app.models import CapabilityChoice, PipelineConfig, Project, ProviderChoice
from app.pipeline import stages
from app.providers.base import (
    REGISTRY,
    ProgressFn,
    ProviderMeta,
    StepContext,
    StepProvider,
    load_all,
    register,
)

load_all()


def test_registry_has_41_unique_capabilities_with_valid_deps():
    assert len(caps.CAPABILITIES) == 41
    assert len(caps.BY_ID) == 41 and len(caps.BY_KIND) == 41
    assert {c.phase for c in caps.CAPABILITIES} == set(caps.PHASES)
    for cap in caps.CAPABILITIES:
        for dep in (*cap.needs, *cap.uses):
            assert dep in caps.BY_ID, f"{cap.id} depends on unknown {dep}"
        assert cap.builtin in REGISTRY, f"{cap.id}: builtin '{cap.builtin}' is not registered"
        assert REGISTRY[cap.builtin].meta.kind == cap.kind


def test_default_config_resolves_to_the_minimal_set():
    resolved = caps.resolve(PipelineConfig())
    on = {k for k, v in resolved.items() if v.enabled}
    assert on == set(caps.PRESETS["minimal"])
    assert "E5" not in on  # provenance is deferred: in no preset
    for preset in caps.PRESETS.values():
        assert "E5" not in preset


def test_legacy_manifest_with_lipsync_provider_keeps_lip_sync_on():
    cfg = PipelineConfig.model_validate({"lipsync": {"provider_id": "lipsync.wav2lip"}})
    assert cfg.lipsync_enabled
    assert caps.resolve(cfg)["F1"].enabled
    assert not caps.resolve(PipelineConfig())["F1"].enabled


def test_dependency_closure_and_feature_gates():
    cfg = PipelineConfig()
    cfg.capabilities["G4"] = CapabilityChoice(enabled=True, provider_id="judge_emotion.fake")
    cfg.capabilities["A8"] = CapabilityChoice(enabled=False, provider_id="delivery.fake")
    resolved = caps.resolve(cfg)
    assert resolved["A8"].enabled and "G4" in resolved["A8"].reason

    # subtitles off wins over an explicit enable
    cfg.capabilities["C6"] = CapabilityChoice(enabled=True, provider_id="subtitles.fake")
    assert not caps.resolve(cfg)["C6"].enabled
    cfg.subtitles_enabled = True
    assert caps.resolve(cfg)["C6"].enabled

    # a core capability cannot be switched off
    cfg.capabilities["E1"] = CapabilityChoice(enabled=False)
    assert caps.resolve(cfg)["E1"].enabled


def test_apply_preset_falls_back_to_builtins_and_says_so():
    cfg = PipelineConfig()
    notes = caps.apply_preset(cfg, "balanced", "cloud", lipsync=False, subtitles=False)
    assert cfg.preset == "balanced" and cfg.runtime == "cloud"
    assert cfg.tts.provider_id != "tts.elevenlabs"  # no API keys in the test environment
    assert any(n.startswith("D1") and "no cloud provider ready" in n for n in notes)
    assert cfg.lipsync.provider_id == "lipsync.none"
    assert set(cfg.capabilities) == set(caps.BY_ID)
    with pytest.raises(ValueError):
        caps.apply_preset(cfg, "nope", "cloud")


@register
class _FakeShots(StepProvider):
    meta = ProviderMeta(id="shots.fake", kind="shots", name="Fake shots")

    async def run(self, ctx: StepContext, progress: ProgressFn) -> str:
        assert ctx.capability_id == "B2" and ctx.enabled["A1"]
        return "3 shots"


@register
class _BrokenOcr(StepProvider):
    meta = ProviderMeta(id="ocr.broken", kind="ocr", name="Broken OCR")

    async def run(self, ctx: StepContext, progress: ProgressFn) -> str:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_step_loop_runs_enabled_steps_and_degrades_on_failure():
    project = Project(name="steps")
    runner = stages.STAGE_RUNNERS["analyze"]

    with pytest.raises(stages.StageSkipped):  # nothing enabled → the stage is skipped
        await runner(project, None, lambda f, m="": None)

    project.pipeline.capabilities["B2"] = CapabilityChoice(enabled=True, provider_id="shots.fake")
    project.pipeline.capabilities["B3"] = CapabilityChoice(enabled=True, provider_id="ocr.broken")
    await runner(project, None, lambda f, m="": None)
    steps = project.stage("analyze").steps
    assert steps["B2"].status == "done" and steps["B2"].detail == "3 shots"
    assert steps["B3"].status == "error" and "boom" in steps["B3"].detail

    # switching one off again drops its stale step state
    project.pipeline.capabilities["B3"] = CapabilityChoice(enabled=False)
    await runner(project, None, lambda f, m="": None)
    assert "B3" not in project.stage("analyze").steps


def test_pipeline_choice_rejects_non_legacy_kinds():
    cfg = PipelineConfig(tts=ProviderChoice(provider_id="tts.mock"))
    assert cfg.choice("tts").provider_id == "tts.mock"
    with pytest.raises(KeyError):
        cfg.choice("shots")


@pytest.mark.asyncio
async def test_capability_api_preset_and_patch():
    import httpx

    from app import store
    from app.main import app

    project = Project(name="api")
    project.stage("translate").status = "done"
    store.save(project)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/capabilities")
        body = r.json()
        assert r.status_code == 200 and len(body["capabilities"]) == 41
        assert {p["id"] for p in body["presets"]} == {"minimal", "balanced", "max"}

        r = await client.post(
            f"/api/projects/{project.id}/pipeline/preset",
            json={"preset": "balanced", "runtime": "builtin", "subtitles": True},
        )
        assert r.status_code == 200, r.text
        pipeline = r.json()["project"]["pipeline"]
        assert pipeline["preset"] == "balanced" and pipeline["subtitles_enabled"]
        assert pipeline["capabilities"]["C4"]["enabled"]

        # an advanced edit flips the preset to custom and dirties the capability's stage
        r = await client.patch(
            f"/api/projects/{project.id}",
            json={"mode": "advanced", "capabilities": {"C4": {"enabled": False}}},
        )
        assert r.status_code == 200, r.text
        patched = r.json()
        assert patched["pipeline"]["preset"] == "custom"
        assert patched["pipeline"]["mode"] == "advanced"
        assert patched["stages"]["translate"]["status"] == "dirty"

        r = await client.get(f"/api/projects/{project.id}/capabilities")
        assert r.json()["C4"]["enabled"] is False

        r = await client.patch(f"/api/projects/{project.id}", json={"capabilities": {"Z9": {}}})
        assert r.status_code == 422
        r = await client.patch(
            f"/api/projects/{project.id}", json={"pipeline": {"shots": {"provider_id": "x"}}}
        )
        assert r.status_code == 422

        # partial patches merge: changing params keeps enabled/provider
        r = await client.patch(
            f"/api/projects/{project.id}", json={"capabilities": {"C5": {"params": {"x": 1}}}}
        )
        c5 = r.json()["pipeline"]["capabilities"]["C5"]
        assert c5["enabled"] is True and c5["params"] == {"x": 1}
