"""End-to-end acceptance test: full pipeline over a synthesized test video using only
always-available providers (mock ASR/translation/TTS, passthrough separation, no lipsync).

This is the definition of "the pipeline works" — no ML dependency, no GPU, no network.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest

TMP = Path(tempfile.mkdtemp(prefix="opendub-e2e-"))
os.environ["OPENDUB_DATA_DIR"] = str(TMP / "data")
os.environ["OPENDUB_CONFIGS_DIR"] = str(TMP / "configs")

from app.main import app  # noqa: E402  (env must be set before import)


def make_test_video(path: Path, duration: float = 8.0) -> None:
    """Color bars + speech-like beeps separated by silence (so silencedetect finds segments)."""
    beeps = (
        "sine=frequency=300:duration=1.2,volume=1.0 [a0]; "
        "anullsrc=r=48000:cl=stereo,atrim=duration=0.8 [s0]; "
        "sine=frequency=420:duration=1.5,volume=1.0 [a1]; "
        "anullsrc=r=48000:cl=stereo,atrim=duration=0.9 [s1]; "
        "sine=frequency=360:duration=1.4,volume=1.0 [a2]; "
        "anullsrc=r=48000:cl=stereo,atrim=duration=2.2 [s2]; "
        "[a0][s0][a1][s1][a2][s2] concat=n=6:v=0:a=1 [aout]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=24:duration={duration}",
            "-filter_complex", beeps,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
    )


async def wait_for_job(client: httpx.AsyncClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        r = await client.get("/api/jobs")
        r.raise_for_status()
        job = next((j for j in r.json() if j["id"] == job_id), None)
        assert job is not None, f"job {job_id} disappeared"
        if job["status"] in ("done", "error", "cancelled"):
            return job
        assert asyncio.get_event_loop().time() < deadline, (
            f"job timed out at stage={job.get('stage')} message={job.get('message')}"
        )
        await asyncio.sleep(0.25)


@pytest.mark.asyncio
async def test_full_pipeline_with_mock_providers():
    video = TMP / "input.mp4"
    make_test_video(video)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
        async with app.router.lifespan_context(app):
            # health
            r = await client.get("/api/health")
            assert r.status_code == 200 and r.json()["ok"] is True

            # providers listed, mocks available
            r = await client.get("/api/providers")
            providers = {p["meta"]["id"]: p for p in r.json()}
            for pid in ("asr.mock", "translation.mock", "tts.mock",
                        "separation.passthrough", "diarization.single_speaker", "lipsync.none"):
                assert pid in providers, f"{pid} not registered"
                assert providers[pid]["available"], f"{pid} unavailable: {providers[pid]['reason']}"

            # create project (upload)
            with video.open("rb") as f:
                r = await client.post(
                    "/api/projects",
                    files={"file": ("input.mp4", f, "video/mp4")},
                    data={"name": "E2E", "source_lang": "ja", "target_lang": "en"},
                )
            assert r.status_code == 200, r.text
            project = r.json()
            pid = project["id"]

            # ingest job auto-started
            r = await client.get(f"/api/jobs?project_id={pid}")
            jobs = r.json()
            assert jobs, "no ingest job created"
            ingest = await wait_for_job(client, jobs[0]["id"])
            assert ingest["status"] == "done", f"ingest failed: {ingest['error']}"

            r = await client.get(f"/api/projects/{pid}")
            project = r.json()
            assert project["media"] and project["media"]["duration"] > 6
            assert project["stages"]["ingest"]["status"] == "done"

            # media + waveform served
            r = await client.get(f"/api/media/{pid}/playback.mp4")
            assert r.status_code == 200
            r = await client.get(f"/api/projects/{pid}/waveform/original")
            assert r.status_code == 200
            wf = r.json()
            assert wf["peaks"] and wf["sample_rate"] > 0

            # run the remaining pipeline (mock providers are the defaults)
            r = await client.post(f"/api/projects/{pid}/pipeline/run", json={})
            assert r.status_code == 200, r.text
            job = await wait_for_job(client, r.json()["id"], timeout=180)
            assert job["status"] == "done", f"pipeline failed: {job['error']}"

            r = await client.get(f"/api/projects/{pid}")
            project = r.json()
            segs = project["segments"]
            assert len(segs) >= 2, f"expected ≥2 segments from beeps, got {len(segs)}"
            for s in segs:
                assert s["source_text"], "empty source text"
                assert s["translated_text"].startswith("(en)"), s["translated_text"]
                assert s["takes"], "segment has no take"
                assert s["active_take_id"]
            assert project["speakers"], "no speakers created"
            for key in ("separate", "transcribe", "translate", "synthesize", "mix", "render"):
                assert project["stages"][key]["status"] == "done", (key, project["stages"][key])
            assert project["stages"]["lipsync"]["status"] == "skipped"

            # artifacts exist and are served
            for rel in ("audio/dub_mix.wav", "render/dubbed.mp4"):
                r = await client.get(f"/api/media/{pid}/{rel}")
                assert r.status_code == 200, rel
            r = await client.get(f"/api/projects/{pid}/waveform/dub_mix")
            assert r.status_code == 200

            # edit a segment → dirty rules
            sid = segs[0]["id"]
            r = await client.patch(
                f"/api/projects/{pid}/segments/{sid}", json={"translated_text": "Hello there!"}
            )
            assert r.status_code == 200, r.text
            seg = r.json()
            assert seg["translated_text"] == "Hello there!"
            assert seg["synth_dirty"] is True and seg["translate_dirty"] is False

            r = await client.get(f"/api/projects/{pid}")
            assert r.json()["stages"]["mix"]["status"] == "dirty"

            # per-segment regeneration (re-voice)
            r = await client.post(
                f"/api/projects/{pid}/segments/{sid}/regenerate", json={"stages": ["synthesize"]}
            )
            assert r.status_code == 200, r.text
            job = await wait_for_job(client, r.json()["id"])
            assert job["status"] == "done", f"regen failed: {job['error']}"

            r = await client.get(f"/api/projects/{pid}")
            seg = next(s for s in r.json()["segments"] if s["id"] == sid)
            assert len(seg["takes"]) >= 2, "regeneration did not add a take"
            assert seg["synth_dirty"] is False

            # provider settings round-trip with secret masking
            r = await client.put(
                "/api/settings/providers/translation.anthropic",
                json={"options": {"api_key": "sk-test-123", "model": "claude-sonnet-4-5"}},
            )
            assert r.status_code == 200
            info = r.json()
            assert info["configured_options"]["api_key"] == "•••"
            assert info["configured_options"]["model"] == "claude-sonnet-4-5"
            # sending the mask back must not clobber the stored secret
            r = await client.put(
                "/api/settings/providers/translation.anthropic",
                json={"options": {"api_key": "•••", "model": "claude-sonnet-4-5"}},
            )
            assert r.status_code == 200
            from app import config as cfg
            assert cfg.provider_options("translation.anthropic")["api_key"] == "sk-test-123"

            # delete project
            r = await client.delete(f"/api/projects/{pid}")
            assert r.status_code == 200
            r = await client.get(f"/api/projects/{pid}")
            assert r.status_code == 404


@pytest.mark.asyncio
async def test_media_path_traversal_blocked():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.get("/api/media/nope/../../etc/passwd")
            assert r.status_code in (400, 404)
