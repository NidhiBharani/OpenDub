"""Run the real OpenDub pipeline on every SCENE clip × target language, then ladder-audit each.

For each pack item (e.g. the SPY×FAMILY ep1 clips from ``builders/spyfamily.py``) and each
target language, this creates a project in-process through the app's ASGI transport (the
``bench/runner.py`` pattern: no separate server, the real providers from
``configs/settings.yaml``), applies a pipeline config, runs it, and writes the stage ladder
(``bench.arena.ladder.write``) so every line can be audited next to the official dub.

    uv run --frozen python -m bench.arena.scene_projects path/to/pipeline.json \
        [--pack SCENE/ja] [--targets en hi] [--items spyfamily-ep1-s1 …] \
        [--asr-check] [--judge] [--timeout 5400]

Pipeline config (JSON; every key optional)::

    {
      "name": "local-balanced",                 # run label (default: the file stem)
      "preset": {"preset": "balanced", "runtime": "local", "lipsync": false},
      "pipeline": {"tts": {"provider_id": "tts.f5_tts", "options": {}}},   # legacy kinds
      "capabilities": {"C2": {"provider_id": "…", "options": {}}},         # by capability id
      "per_target": {"hi": {"pipeline": {…}, "capabilities": {…}}},        # merged on top
      "stages": null                             # subset of stages to run (default: all)
    }

The preset is applied first (``POST /pipeline/preset``), then ``pipeline``/``capabilities``
(``PATCH /projects/{id}``). Results are indexed in
``data/arena/scene/<name>/runs.jsonl`` with the project id, job status, ladder path and the
item's reference paths (official dub audio / transcripts) for side-by-side review.

This runs models (whatever the config selects). The user orchestrates it; nothing imports or
runs it implicitly, and importing this module does not import the app.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from . import paths
from .packs import Item, load_pack

CONFIG_KEYS = {"name", "preset", "pipeline", "capabilities", "per_target", "stages"}


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: the pipeline config must be a JSON object")
    unknown = set(cfg) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; allowed {sorted(CONFIG_KEYS)}")
    cfg.setdefault("name", Path(path).stem)
    return cfg


def config_for(cfg: dict[str, Any], target: str) -> dict[str, Any]:
    """The effective config for one target language (``per_target`` merged per kind/capability)."""
    over = (cfg.get("per_target") or {}).get(target, {})
    out = {"preset": over.get("preset", cfg.get("preset")),
           "stages": over.get("stages", cfg.get("stages"))}
    for key in ("pipeline", "capabilities"):
        merged = {**(cfg.get(key) or {}), **(over.get(key) or {})}
        out[key] = merged or None
    return out


async def _wait_for_job(client, pid: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = (await client.get(f"/api/jobs?project_id={pid}")).json()
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["status"] in ("done", "error", "cancelled"):
            return job
        await asyncio.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


async def _run_one(client, item: Item, target: str, cfg: dict[str, Any], name: str,
                   timeout: float) -> dict[str, Any]:
    video = Path(item.inputs["video"])
    data = await asyncio.to_thread(video.read_bytes)  # clips are small (1–3 min)
    resp = await client.post(
        "/api/projects", files={"file": (video.name, data, "video/mp4")},
        data={"name": f"scene:{name}:{item.id}:{target}", "source_lang": item.lang,
              "target_lang": target})
    resp.raise_for_status()
    pid = resp.json()["id"]
    for job in (await client.get(f"/api/jobs?project_id={pid}")).json():  # ingest
        await _wait_for_job(client, pid, job["id"], timeout)

    eff = config_for(cfg, target)
    if eff["preset"]:
        (await client.post(f"/api/projects/{pid}/pipeline/preset",
                           json=eff["preset"])).raise_for_status()
    patch = {k: eff[k] for k in ("pipeline", "capabilities") if eff[k]}
    if patch:
        (await client.patch(f"/api/projects/{pid}", json=patch)).raise_for_status()

    t0 = time.time()
    run = await client.post(f"/api/projects/{pid}/pipeline/run", json={"stages": eff["stages"]})
    run.raise_for_status()
    job = await _wait_for_job(client, pid, run.json()["id"], timeout)
    return {"project_id": pid, "status": job["status"], "error": job.get("error"),
            "seconds": round(time.time() - t0, 1)}


async def _run_all(items: list[Item], targets: list[str], cfg: dict[str, Any],
                   timeout: float, log) -> list[dict[str, Any]]:
    import httpx

    from app.main import app

    rows = []
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://scene") as client:
        for item in items:
            for target in targets:
                log(f"scene: {item.id} → {target}")
                base = {"item": item.id, "target": target, "config": cfg["name"],
                        "start_s": item.meta.get("start_s"), "end_s": item.meta.get("end_s"),
                        "refs": {k: v for k, v in item.refs.items()
                                 if k.startswith(f"dub_{target}_")}}
                try:
                    res = await _run_one(client, item, target, cfg, cfg["name"], timeout)
                except Exception as exc:  # noqa: BLE001 - keep going with the other runs
                    res = {"project_id": None, "status": "error", "error": repr(exc)}
                log(f"  {res.get('project_id')}: {res['status']}"
                    + (f" ({res['error']})" if res.get("error") else ""))
                rows.append({**base, **res})
    return rows


def run(config: Path, *, capability: str = "SCENE", lang: str = "ja",
        targets: list[str] | None = None, item_ids: list[str] | None = None,
        asr_check: bool = False, judge: bool = False, timeout: float = 5400.0,
        log=print) -> Path:
    """Create + run one project per (item, target), then write each stage ladder."""
    from . import ladder

    cfg = load_config(config)
    items = load_pack(capability, lang)
    if item_ids:
        items = [it for it in items if it.id in set(item_ids)]
    if not items:
        raise ValueError(f"no items selected from {capability}/{lang}")
    targets = targets or ["en", "hi"]
    rows = asyncio.run(_run_all(items, targets, cfg, timeout, log))
    for row in rows:  # after the event loop: ladder.build runs its own asyncio loop
        if not row.get("project_id"):
            continue
        try:
            row["ladder"] = str(ladder.write(row["project_id"], asr_check=asr_check,
                                             judge=judge))
            log(f"  ladder: {row['ladder']}")
        except Exception as exc:  # noqa: BLE001
            row["ladder_error"] = repr(exc)
    out = paths.arena_dir() / "scene" / cfg["name"] / "runs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for row in rows:
            f.write(json.dumps({**row, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                               ensure_ascii=False) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.arena.scene_projects",
                                description=__doc__.split("\n")[0])
    p.add_argument("config", type=Path, help="pipeline config JSON (see module docstring)")
    p.add_argument("--pack", default="SCENE/ja", help="<capability>/<source lang>")
    p.add_argument("--targets", nargs="+", default=["en", "hi"])
    p.add_argument("--items", nargs="+", help="only these item ids")
    p.add_argument("--asr-check", action="store_true",
                   help="ladder round-trip ASR of every take (loads Whisper on the GPU)")
    p.add_argument("--judge", action="store_true", help="ladder whole-pipeline stage metrics")
    p.add_argument("--timeout", type=float, default=5400.0, help="per job, seconds")
    a = p.parse_args(argv)
    cap, lang = a.pack.split("/")
    print(run(a.config, capability=cap, lang=lang, targets=a.targets, item_ids=a.items,
              asr_check=a.asr_check, judge=a.judge, timeout=a.timeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
