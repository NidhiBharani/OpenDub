"""Generate outputs: run each candidate as an isolated worker over the items it has not done yet.

Outputs are content-addressed by (candidate key, item input hash), so re-running a capability
only generates what is missing. They live under ``data/arena/outputs`` permanently: they are
both the cache and the audit trail the user inspects. Model-based judges (``judges.py``) reuse
:func:`run_worker_job`, so every model — generator or judge — runs in its own process and frees
its GPU memory the moment the job ends.
"""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import db, paths
from .hardware import detect
from .packs import Item
from .registry import Candidate, env_python

Log = Callable[[str], None]


def output_key(cand: Candidate, item: Item) -> str:
    return hashlib.sha256(f"{cand.key}|{item.input_hash()}".encode()).hexdigest()[:24]


def _safe(name: str) -> str:
    """Filesystem-safe name with no dots: output prefixes get suffixes appended with
    ``with_suffix``, which would otherwise eat everything after a dot in an item id and make
    ``x.1`` and ``x.2`` collide."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def output_prefix(cand: Candidate, item: Item) -> Path:
    return paths.outputs_dir() / cand.capability / item.lang / cand.key / _safe(item.id)


def gpu_free_mib() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
        return float(out.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def ensure_gpu_room(label: str, vram_gb: float) -> None:
    """Refuse to start a GPU job that cannot fit next to whatever else is resident."""
    if vram_gb <= 0:
        return
    free = gpu_free_mib()
    need = (vram_gb + 0.5) * 1024
    if free is not None and free < need:
        raise RuntimeError(f"{label} needs ~{vram_gb:g} GB VRAM but only {free / 1024:.1f} GB is "
                           f"free; stop other GPU jobs (Ollama keeps models resident: "
                           f"`ollama stop <model>`) or pass --no-gpu-check")


@dataclass
class JobResult:
    job: str
    records: dict[str, dict[str, Any]] = field(default_factory=dict)  # item id -> result record
    info: dict[str, Any] = field(default_factory=dict)                # worker.json + exit code
    returncode: int = 0


def run_worker_job(*, worker: str, env: str, params: dict[str, Any], lang: str,
                   items: list[dict[str, Any]], label: str, timeout: float = 6 * 3600,
                   budget_usd: float | None = None, log: Log = print) -> JobResult:
    """Run one worker process over ``items`` ({id, inputs, meta, out}) and collect its results."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    job_dir = paths.arena_dir() / "jobs" / f"{stamp}-{_safe(label)}"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    job: dict[str, Any] = {"params": params, "lang": lang, "items": items}
    if budget_usd is not None:
        job["budget_usd"] = budget_usd
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=1))

    python = env_python(env)
    script = paths.WORKERS_DIR / f"{worker}.py"
    log(f"  ▶ {label}: {len(items)} items ({script.name}, {python})")
    t0 = time.time()
    with (job_dir / "worker.log").open("w") as logf:
        try:
            proc = subprocess.run([python, str(script), "--job", str(job_path)],
                                  cwd=paths.WORKERS_DIR, stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            logf.write(f"\n[arena] killed after {timeout}s timeout\n")

    res = JobResult(job=job_dir.name, returncode=rc)
    res_path = job_path.with_suffix(".results.jsonl")
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                res.records[rec["id"]] = rec
    info_path = job_path.with_suffix(".worker.json")
    res.info = json.loads(info_path.read_text()) if info_path.exists() else {}
    res.info.update(returncode=rc, wall_seconds=round(time.time() - t0, 2),
                    host=socket.gethostname(), hardware=detect().profile)
    ok = sum(r.get("status") == "ok" for r in res.records.values())
    status = "ok" if rc == 0 else f"exit {rc}, see {job_dir}/worker.log"
    log(f"    {ok}/{len(items)} ok in {res.info['wall_seconds']}s ({status})")
    return res


def generate(capability: str, lang: str, candidates: list[Candidate], items: list[Item], *,
             force: bool = False, check_gpu: bool = True, check_hardware: bool = True,
             timeout: float = 6 * 3600, budget_usd: float | None = None,
             log: Log = print) -> dict[str, int]:
    """Returns {cand_key: number of newly generated ok outputs}.

    Skips candidates that are disabled, cannot run on this hardware (``requires``), or do not
    declare the language; ``check_hardware=False`` only relaxes the hardware part.
    """
    con = db.connect()
    done: dict[str, int] = {}
    host = socket.gethostname()
    for cand in candidates:
        if not cand.supports(lang):
            log(f"  - {cand.id}: does not declare {lang}; skipped")
            continue
        blockers = cand.blockers()
        if not check_hardware:
            blockers = [b for b in blockers if b.startswith(("disabled", "env ", "worker "))]
        if blockers:
            log(f"  - {cand.id}: skipped ({'; '.join(blockers)})")
            continue
        pending = []
        for it in items:
            row = db.get_output(con, output_key(cand, it))
            ok = (row is not None and row["status"] == "ok"
                  and db.output_path(row).with_suffix(".json").exists())
            if force or not ok:
                pending.append(it)
        if not pending:
            log(f"  = {cand.id}: all {len(items)} items cached")
            done[cand.key] = 0
            continue
        if check_gpu:
            ensure_gpu_room(cand.id, cand.requires.vram_gb)
        res = run_worker_job(
            worker=cand.worker, env=cand.env, params=cand.params, lang=lang,
            items=[{"id": it.id, "inputs": it.inputs, "meta": it.meta,
                    "out": str(output_prefix(cand, it))} for it in pending],
            label=f"{capability}-{lang}-{cand.id}", timeout=timeout, budget_usd=budget_usd,
            log=log)
        ok = 0
        for it in pending:
            rec = res.records.get(it.id, {"status": "error", "error":
                                          f"worker exited {res.returncode} before this item"})
            ok += rec["status"] == "ok"
            db.put_output(con, key=output_key(cand, it), capability=capability, lang=lang,
                          candidate=cand.id, cand_key=cand.key, item=it.id,
                          input_hash=it.input_hash(), path=db.rel_output(output_prefix(cand, it)),
                          status=rec["status"], error=rec.get("error"),
                          seconds=rec.get("seconds"), job=res.job,
                          cost_usd=rec.get("cost_usd"), host=host)
        db.put_job(con, res.job, res.info, capability=capability, lang=lang, candidate=cand.id,
                   cand_key=cand.key, items=len(pending), ok=ok)
        con.commit()
        done[cand.key] = ok
    con.close()
    return done
