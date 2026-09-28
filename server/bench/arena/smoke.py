"""Load/unload smoke tests: does every model this machine can run actually load, produce an output,
and give all of its memory back?

Two layers:

- **Arena workers** (``arena smoke``): each runnable candidate and model judge runs one worker job
  on a single fixture item. Recorded: env ready, load time, run status, peak VRAM, the GPU memory
  left over after the worker exited (must return to the baseline), GPU processes left behind
  (e.g. a model server the worker forgot to stop) and Ollama models still resident.
- **App model manager** (``arena smoke --app``): the server's in-process models (faster-whisper,
  F5-TTS, …) are acquired through ``app.providers._runtime.manager`` and then unloaded; recorded:
  process GPU memory before, loaded, and after unload.

Results go to the ``smoke`` table of ``arena.sqlite`` and into ``docs/arena-models.md`` via
``arena docs``. Fixture items are one item of the capability's own pack when it exists, else a
generic superset fixture (``data/arena/smoke/_fixture``) built from local media. Nothing here runs
unless the user calls it; downloads happen inside the workers exactly as in a real run.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import db, envs, hardware, paths
from .packs import Item, load_pack, pack_dir
from .registry import Candidate, capabilities, load_candidates, select
from .runner import run_worker_job

SMOKE_SCHEMA = """
CREATE TABLE IF NOT EXISTS smoke (
  capability TEXT, candidate TEXT, cand_key TEXT, kind TEXT, host TEXT, profile TEXT, ts REAL,
  status TEXT,          -- ok | load_failed | run_failed | env_missing | env_setup_failed |
                        -- skipped | leak
  load_s REAL, run_s REAL, peak_vram_mib REAL, residual_vram_mib REAL, leaked TEXT,
  downloaded_gb REAL, error TEXT, job TEXT,
  PRIMARY KEY (capability, cand_key, host)
);
"""

Log = Callable[[str], None]


def _con():
    con = db.connect()
    con.executescript(SMOKE_SCHEMA)
    return con


def _gpu_used_mib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
        return float(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _gpu_pids() -> dict[int, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    pids = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2 and parts[0].isdigit():
            pids[int(parts[0])] = parts[1]
    return pids


def _ollama_loaded() -> list[str]:
    try:
        import httpx

        r = httpx.get("http://127.0.0.1:11434/api/ps", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001 - no Ollama on this machine is the normal case
        return []


def _dir_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 2**30


def _hf_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() /
                ".cache" / "huggingface")) / "hub")


# ---------------------------------------------------------------- fixtures

def fixture_dir() -> Path:
    return paths.arena_dir() / "smoke" / "_fixture"


def build_fixture() -> dict[str, Any]:
    """A superset item every worker family can read (extra keys are ignored by workers):
    4 s of real speech, a reference voice, a 5 s video with audio, and texts."""
    d = fixture_dir()
    d.mkdir(parents=True, exist_ok=True)
    fx = d / "fixture.json"
    if fx.exists():
        return json.loads(fx.read_text())
    speech = sorted((paths.eval_dir() / "A4" / "en" / "audio").glob("*.wav"))
    ja = sorted((paths.eval_dir() / "A4" / "ja" / "audio").glob("*.wav"))
    audio, ref, video = d / "speech_en.wav", d / "ref_ja.wav", d / "clip.mp4"

    def ff(*args: str) -> None:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)

    if speech:
        ff("-i", str(speech[0]), "-t", "4", "-ac", "1", "-ar", "16000", str(audio))
    else:
        ff("-f", "lavfi", "-i", "sine=frequency=220:duration=4", "-ar", "16000", str(audio))
    if ja:
        ff("-i", str(ja[0]), "-t", "6", "-ac", "1", "-ar", "24000", str(ref))
    else:
        shutil.copy(audio, ref)
    scene = sorted((paths.eval_dir() / "SCENE" / "ja").glob("**/*.mp4"))
    if scene:
        ff("-ss", "10", "-i", str(scene[0]), "-t", "5", "-c:v", "libx264", "-preset", "veryfast",
           "-c:a", "aac", str(video))
    else:
        ff("-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=5", "-i", str(audio),
           "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video))
    item = {"audio": str(audio), "ref_audio": str(ref), "ref_text": "",
            "video": str(video), "text": "Hello, this is a short test of the dubbing system.",
            "source": "これはテストです。", "hypothesis": "This is a test.",
            "reference": "This is a test.", "context": "", "slot_s": 4.0, "target_s": 3.5,
            "src_lang": "ja", "tgt_lang": "en", "emotion": "neutral"}
    fx.write_text(json.dumps(item, ensure_ascii=False, indent=1))
    return item


def fixture_item(capability: str, cand: Candidate) -> tuple[Item, str]:
    """One item from the capability's own pack (preferring en/ja/hi), else the superset."""
    for lang in ("en", "ja", "hi", "ja-en", "ja-hi", "en-hi"):
        if cand.supports(lang) and (pack_dir(capability, lang) / "manifest.jsonl").exists():
            try:
                items = load_pack(capability, lang, split=None, limit=1)
            except (OSError, ValueError):
                continue
            if items:
                return items[0], lang
    lang = "en" if cand.supports("en") else (cand.languages[0] if isinstance(
        cand.languages, list) and cand.languages else "en")
    return Item(id="smoke", lang=lang, inputs=build_fixture(), meta={"duration_s": 4.0}), lang


# ---------------------------------------------------------------- worker smoke

def _hf_repos() -> set[str]:
    hub = _hf_cache()
    return {p.name for p in hub.iterdir() if p.is_dir()} if hub.exists() else set()


def smoke_candidate(capability: str, cand: Candidate, *, setup_envs: bool = False,
                    include_api: bool = False, timeout: float = 1800, settle_s: float = 15.0,
                    evict_weights: bool = False, log: Log = print) -> dict[str, Any]:
    hw = hardware.detect()
    rec: dict[str, Any] = {"capability": capability, "candidate": cand.id, "cand_key": cand.key,
                           "kind": cand.kind, "host": socket.gethostname(),
                           "profile": hw.profile, "ts": time.time()}
    blockers = cand.blockers(hw)
    env_missing = [b for b in blockers if b.startswith("env ")]
    other = [b for b in blockers if not b.startswith("env ")]
    if cand.kind == "api" and not include_api:
        other.append("API candidate (pass --include-api; costs money)")
    if other:
        return {**rec, "status": "skipped", "error": "; ".join(other)}
    if env_missing:
        if not setup_envs:
            return {**rec, "status": "env_missing", "error": "; ".join(env_missing)}
        log(f"  · setting up env {cand.env}")
        if envs.setup(envs.load_env(cand.env)) != 0:
            return {**rec, "status": "env_setup_failed", "error": f"env {cand.env} setup failed"}

    item, lang = fixture_item(capability, cand)
    out = paths.arena_dir() / "smoke" / capability / cand.key / "item"
    baseline = _gpu_used_mib()
    pids_before, ollama_before = set(_gpu_pids()), set(_ollama_loaded())
    hf_before = _dir_gb(_hf_cache())
    repos_before = _hf_repos()
    res = run_worker_job(worker=cand.worker, env=cand.env, params=cand.params, lang=lang,
                         items=[{"id": item.id, "inputs": item.inputs, "meta": item.meta,
                                 "out": str(out)}],
                         label=f"smoke-{capability}-{cand.id}", timeout=timeout, log=log)
    # The worker has exited; its memory must come back (allow the driver a moment).
    residual = None
    deadline = time.time() + settle_s
    while True:
        now = _gpu_used_mib()
        residual = None if now is None or baseline is None else max(0.0, now - baseline)
        if residual is None or residual < 64 or time.time() > deadline:
            break
        time.sleep(1)
    leaked_pids = {p: n for p, n in _gpu_pids().items() if p not in pids_before}
    leaked_ollama = sorted(set(_ollama_loaded()) - ollama_before)
    leaked = [f"gpu pid {p} ({n})" for p, n in leaked_pids.items()] + \
             [f"ollama {m}" for m in leaked_ollama]
    downloaded = round(max(0.0, _dir_gb(_hf_cache()) - hf_before), 2)
    if evict_weights:
        # Disk is the constraint when smoke-testing hundreds of models: drop what this test
        # downloaded (repos cached before the test are never touched).
        for name in _hf_repos() - repos_before:
            shutil.rmtree(_hf_cache() / name, ignore_errors=True)
    r = res.records.get(item.id, {})
    status = r.get("status")
    if res.info.get("load_seconds") is None:
        verdict = "load_failed"
    elif status != "ok":
        verdict = "run_failed" if status == "error" else (status or "run_failed")
    elif leaked or (residual is not None and residual >= 256):
        verdict = "leak"
    else:
        verdict = "ok"
    return {**rec, "status": verdict, "load_s": res.info.get("load_seconds"),
            "run_s": r.get("seconds"), "peak_vram_mib": res.info.get("peak_vram_mib"),
            "residual_vram_mib": residual, "leaked": "; ".join(leaked) or None,
            "downloaded_gb": downloaded,
            "error": r.get("error") or (None if res.returncode == 0 else
                                        f"worker exit {res.returncode}"),
            "job": res.job}


def save(rec: dict[str, Any]) -> None:
    con = _con()
    cols = ["capability", "candidate", "cand_key", "kind", "host", "profile", "ts", "status",
            "load_s", "run_s", "peak_vram_mib", "residual_vram_mib", "leaked", "downloaded_gb",
            "error", "job"]
    con.execute(f"INSERT OR REPLACE INTO smoke ({','.join(cols)}) VALUES "
                f"({','.join('?' * len(cols))})", [rec.get(c) for c in cols])
    con.commit()
    con.close()


def results() -> list[dict[str, Any]]:
    con = _con()
    rows = [dict(r) for r in con.execute("SELECT * FROM smoke ORDER BY capability, candidate")]
    con.close()
    return rows


def smoke_all(caps: list[str], ids: list[str] | None = None, *, setup_envs: bool = False,
              include_api: bool = False, min_free_disk_gb: float = 15.0, rerun: bool = False,
              evict_weights: bool = False, log: Log = print) -> list[dict[str, Any]]:
    done = {(r["capability"], r["cand_key"]) for r in results()
            if r["host"] == socket.gethostname() and r["status"] == "ok"}
    out = []
    for cap in caps:
        for cand in select(load_candidates(cap), ids) if ids else load_candidates(cap):
            if not rerun and (cap, cand.key) in done:
                continue
            if shutil.disk_usage(paths.data_dir()).free / 2**30 < min_free_disk_gb:
                log(f"stopping: less than {min_free_disk_gb:g} GB disk free")
                return out
            log(f"▶ smoke {cap} {cand.id}")
            rec = smoke_candidate(cap, cand, setup_envs=setup_envs, include_api=include_api,
                                  evict_weights=evict_weights, log=log)
            save(rec)
            log(f"  = {rec['status']}"
                + (f" · load {rec['load_s']}s" if rec.get("load_s") is not None else "")
                + (f" · peak {rec['peak_vram_mib'] / 1024:.1f} GB" if rec.get("peak_vram_mib")
                   else "")
                + (f" · {rec['error'][:160]}" if rec.get("error") else ""))
            out.append(rec)
    return out


# ---------------------------------------------------------------- app model manager smoke

def _proc_gpu_mib() -> float:
    pid = os.getpid()
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    for line in out.splitlines():
        p, _, m = line.partition(",")
        if p.strip() == str(pid):
            return float(m)
    return 0.0


def smoke_app(log: Log = print) -> list[dict[str, Any]]:
    """Acquire each in-process app model through the manager, then unload it, measuring this
    process's GPU memory. A CUDA context (~0.3–0.6 GB) legitimately stays once CUDA is
    initialised; ``residual_vram_mib`` is what remains above the pre-load level."""
    import asyncio

    from app.providers._runtime import manager

    cases: list[tuple[str, Callable[[], Any]]] = []

    def whisper_case(name: str):
        def run():
            from app.providers.asr.faster_whisper import acquire_whisper

            with acquire_whisper(name, device="cuda", compute_type="float16"):
                return _proc_gpu_mib()
        return run

    def f5_case(model: str, ckpt: str = "", vocab: str = ""):
        def run():
            from app.providers.tts import f5_tts

            async def go():
                async with f5_tts._acquire_model(model, "cuda", ckpt, vocab):
                    return _proc_gpu_mib()
            return asyncio.run(go())
        return run

    cases.append(("asr.faster_whisper large-v3-turbo", whisper_case("large-v3-turbo")))
    cases.append(("asr.faster_whisper large-v3", whisper_case("large-v3")))
    cases.append(("tts.f5_tts F5TTS_v1_Base", f5_case("F5TTS_v1_Base")))
    cases.append(("tts.f5_tts F5-Hindi", f5_case(
        "F5TTS_Small", "hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors",
        "hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt")))

    # Create the CUDA context first so the per-model numbers exclude it (the context stays for
    # the life of the process; the manager's job is to free the model, not the context).
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
    except ImportError:
        pass
    out = []
    for name, run in cases:
        before = _proc_gpu_mib()
        rec: dict[str, Any] = {"capability": "APP", "candidate": name, "cand_key": name,
                               "kind": "app", "host": socket.gethostname(),
                               "profile": hardware.detect().profile, "ts": time.time()}
        t0 = time.time()
        try:
            loaded = run()
        except Exception as exc:  # noqa: BLE001 - a failing model is a result, not a crash
            rec.update(status="load_failed", error=f"{type(exc).__name__}: {exc}"[:400])
            manager.unload_all()
            save(rec)
            out.append(rec)
            log(f"  {name}: load_failed ({rec['error'][:120]})")
            continue
        load_s = time.time() - t0
        still = manager.loaded_keys()  # released but warm (idle TTL): now force the unload
        manager.unload_all()
        time.sleep(2)
        after = _proc_gpu_mib()
        residual = max(0.0, after - before)
        rec.update(status="ok" if residual < 256 else "leak", load_s=round(load_s, 2),
                   peak_vram_mib=loaded, residual_vram_mib=residual,
                   error=None if still else "model was not kept warm after release")
        save(rec)
        out.append(rec)
        log(f"  {name}: {rec['status']} · loaded {loaded:.0f} MiB · after unload +{residual:.0f}"
            f" MiB over the pre-load level")
    return out


def all_capabilities() -> list[str]:
    return capabilities()
