"""Every pipeline stage that uses a model must offer at least one cloud (API-key) provider, and
cloud providers must gate availability on their key. Guards against a stage silently having only
local options again.
"""
from __future__ import annotations

from app.providers.base import REGISTRY, get_provider_class, load_all

load_all()

# ingest/mix/render are pure ffmpeg (no provider); these six are the model-backed stages.
MODEL_KINDS = ["separation", "asr", "diarization", "translation", "tts", "lipsync"]
NEW_CLOUD = ["separation.lalalai", "diarization.pyannote_api", "lipsync.replicate"]


def test_every_model_stage_has_a_cloud_provider():
    cloud_kinds = {
        cls.meta.kind for cls in REGISTRY.values() if cls.meta.runtime == "cloud"
    }
    missing = [k for k in MODEL_KINDS if k not in cloud_kinds]
    assert not missing, f"stages without any cloud provider: {missing}"


def test_new_cloud_providers_registered_with_cloud_runtime():
    for pid in NEW_CLOUD:
        assert pid in REGISTRY, f"{pid} not registered"
        assert REGISTRY[pid].meta.runtime == "cloud"


def test_cloud_providers_require_api_key():
    for pid in NEW_CLOUD:
        cls = get_provider_class(pid)
        ok_without, reason = cls({}).available()
        assert ok_without is False and "api_key" in reason
        ok_with, _ = cls({"api_key": "test-key"}).available()
        assert ok_with is True


def test_cloud_providers_declare_an_api_key_field():
    for pid in NEW_CLOUD:
        keys = {f.key for f in REGISTRY[pid].meta.fields}
        assert "api_key" in keys, f"{pid} has no api_key config field"
        secret_fields = {f.key for f in REGISTRY[pid].meta.fields if f.type == "secret"}
        assert "api_key" in secret_fields, f"{pid} api_key must be a secret field"
