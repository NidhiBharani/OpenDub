"""Editing features end to end over mock providers: skip ranges, anchors, transcript
structure edits, versions (auto snapshot, restore) and switching the target language."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest

from tests.test_e2e import make_test_video, wait_for_job

TMP = Path(tempfile.mkdtemp(prefix="opendub-editing-"))
os.environ["OPENDUB_DATA_DIR"] = str(TMP / "data")
os.environ["OPENDUB_CONFIGS_DIR"] = str(TMP / "configs")

from app.main import app


@pytest.mark.asyncio
async def test_editing_features():
    video = TMP / "input.mp4"
    make_test_video(video)
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as c,
        app.router.lifespan_context(app),
    ):
        with video.open("rb") as f:
            r = await c.post(
                "/api/projects",
                files={"file": ("input.mp4", f, "video/mp4")},
                data={"name": "Edit", "source_lang": "ja", "target_lang": "en"},
            )
        pid = r.json()["id"]
        jobs = (await c.get(f"/api/jobs?project_id={pid}")).json()
        assert (await wait_for_job(c, jobs[0]["id"]))["status"] == "done"

        # ---- anchors + filmstrip exist after ingest --------------------------------
        r = await c.get(f"/api/projects/{pid}/anchors")
        assert r.status_code == 200, r.text
        r = await c.get(f"/api/projects/{pid}/filmstrip")
        assert r.status_code == 200, r.text
        strip = r.json()
        assert strip["count"] >= 1 and strip["cols"] > 0
        r = await c.get(strip["url"])
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/")

        # ---- full run -> one auto version -----------------------------------------
        r = await c.post(f"/api/projects/{pid}/pipeline/run", json={})
        job = await wait_for_job(c, r.json()["id"], timeout=180)
        assert job["status"] == "done", job["error"]
        versions = (await c.get(f"/api/projects/{pid}/versions")).json()
        assert len(versions) == 1
        v1 = versions[0]
        assert v1["kind"] == "auto" and v1["has_mix"] and v1["has_render"]
        assert v1["target_lang"] == "en" and v1["voiced_count"] >= 1
        project = (await c.get(f"/api/projects/{pid}")).json()
        assert project["active_version_id"] == v1["id"]
        r = await c.get(f"/api/media/{pid}/{v1['outputs']['playback_dub']}")
        assert r.status_code == 200

        # anchors now include silences and segment edges
        anchors = (await c.get(f"/api/projects/{pid}/anchors")).json()["anchors"]
        kinds = {a["kind"] for a in anchors}
        assert "segment" in kinds and "silence" in kinds

        # ---- skip range: the covered line stops being voiced ---------------------
        segs = sorted(project["segments"], key=lambda s: s["start"])
        first = segs[0]
        r = await c.put(
            f"/api/projects/{pid}/skip-ranges",
            json={"ranges": [{"start": first["start"] - 0.05, "end": first["end"] + 0.05, "label": "intro"}]},
        )
        assert r.status_code == 200, r.text
        project = r.json()
        assert project["active_version_id"] is None
        assert len(project["skip_ranges"]) == 1 and project["skip_ranges"][0]["id"].startswith("rng_")
        by_id = {s["id"]: s for s in project["segments"]}
        assert by_id[first["id"]]["skipped"] is True
        assert project["stages"]["mix"]["status"] == "dirty"
        # regenerating a kept line is refused
        r = await c.post(f"/api/projects/{pid}/segments/{first['id']}/regenerate", json={"stages": ["synthesize"]})
        assert r.status_code == 409
        # mix again: the original is spliced in, and a second version is saved
        r = await c.post(f"/api/projects/{pid}/pipeline/run", json={"stages": ["mix", "render"]})
        job = await wait_for_job(c, r.json()["id"], timeout=180)
        assert job["status"] == "done", job["error"]
        project = (await c.get(f"/api/projects/{pid}")).json()
        assert "original kept in 1 range" in project["stages"]["mix"]["detail"]
        versions = (await c.get(f"/api/projects/{pid}/versions")).json()
        assert len(versions) == 2 and versions[0]["skipped_ranges"] == 1

        # ---- transcript structure: split / merge / insert / delete ----------------
        second = segs[1]
        mid = (second["start"] + second["end"]) / 2
        r = await c.post(f"/api/projects/{pid}/segments/{second['id']}/split", json={"at": mid})
        assert r.status_code == 200, r.text
        project = r.json()
        halves = [s for s in project["segments"] if abs(s["start"] - second["start"]) < 1e-3 or abs(s["start"] - mid) < 1e-3]
        assert len(halves) == 2
        assert all(s["translate_dirty"] and s["synth_dirty"] for s in halves)
        # stage statuses follow the existing convention: per-line flags select translate,
        # everything after it is marked dirty
        assert project["stages"]["synthesize"]["status"] == "dirty"
        r = await c.post(f"/api/projects/{pid}/segments/{second['id']}/merge-next")
        assert r.status_code == 200, r.text
        project = r.json()
        merged = next(s for s in project["segments"] if s["id"] == second["id"])
        assert abs(merged["end"] - second["end"]) < 1e-3
        assert len(project["segments"]) == len(segs)

        n = len(project["segments"])
        r = await c.post(f"/api/projects/{pid}/segments", json={"start": 6.0, "end": 7.0, "source_text": "extra"})
        assert r.status_code == 200, r.text
        project = r.json()
        assert len(project["segments"]) == n + 1
        new = next(s for s in project["segments"] if s["source_text"] == "extra")
        assert new["translate_dirty"] is True and new["speaker_id"]
        r = await c.delete(f"/api/projects/{pid}/segments/{new['id']}")
        assert r.status_code == 200 and len(r.json()["segments"]) == n

        # re-transcribe one line with the mock ASR (a segment job)
        r = await c.post(f"/api/projects/{pid}/segments/{second['id']}/transcribe")
        assert r.status_code == 200, r.text
        job = await wait_for_job(c, r.json()["id"])
        assert job["status"] == "done", job["error"]

        # an edit invalidates word timings unless the token count matches
        r = await c.patch(f"/api/projects/{pid}/segments/{second['id']}", json={"source_text": "hello there"})
        assert r.status_code == 200

        # ---- versions: rename, restore (with restore point), delete --------------
        r = await c.patch(f"/api/projects/{pid}/versions/{v1['id']}", json={"label": "first cut"})
        assert r.status_code == 200 and r.json()["label"] == "first cut"
        r = await c.post(f"/api/projects/{pid}/versions/{v1['id']}/restore")
        assert r.status_code == 200, r.text
        project = r.json()
        assert project["active_version_id"] == v1["id"]
        assert project["skip_ranges"] == []
        assert len(project["segments"]) == len(segs)
        assert all(s["active_take_id"] for s in project["segments"])
        assert project["stages"]["mix"]["status"] == "done"
        versions = (await c.get(f"/api/projects/{pid}/versions")).json()
        assert len(versions) == 3 and versions[0]["kind"] == "pre_restore"
        # restored take files are readable
        take = project["segments"][0]["takes"][0]
        assert (await c.get(f"/api/media/{pid}/{take['path']}")).status_code == 200
        r = await c.delete(f"/api/projects/{pid}/versions/{versions[0]['id']}")
        assert r.status_code == 200
        assert len((await c.get(f"/api/projects/{pid}/versions")).json()) == 2

        # ---- switching the target language starts a new lineage -----------------
        r = await c.patch(f"/api/projects/{pid}", json={"target_lang": "hi"})
        assert r.status_code == 200, r.text
        project = r.json()
        assert project["target_lang"] == "hi"
        assert all(s["translate_dirty"] for s in project["segments"] if s["source_text"].strip())
        assert project["stages"]["translate"]["status"] == "dirty"
        r = await c.post(f"/api/projects/{pid}/pipeline/run", json={})
        job = await wait_for_job(c, r.json()["id"], timeout=180)
        assert job["status"] == "done", job["error"]
        versions = (await c.get(f"/api/projects/{pid}/versions")).json()
        assert versions[0]["target_lang"] == "hi"
        assert {v["target_lang"] for v in versions} == {"en", "hi"}
        project = (await c.get(f"/api/projects/{pid}")).json()
        assert all(t["lang"] == "hi" for s in project["segments"] for t in s["takes"][-1:])
        summary = next(p for p in (await c.get("/api/projects")).json() if p["id"] == pid)
        assert set(summary["languages"]) == {"en", "hi"} and summary["version_count"] == len(versions)

        # restoring the English cut brings the language back
        en = next(v for v in versions if v["target_lang"] == "en")
        project = (await c.post(f"/api/projects/{pid}/versions/{en['id']}/restore")).json()
        assert project["target_lang"] == "en"
