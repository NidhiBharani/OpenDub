"""Worker protocol, stdlib only, so it runs inside any isolated candidate env.

The runner launches ``<env python> workers/<name>.py --job job.json``. ``job.json``::

    {"params": {...}, "lang": "hi", "items": [{"id", "inputs": {...}, "out": "/abs/prefix"}]}

For every item the worker writes its outputs next to ``out`` (``<out>.json`` at minimum) and
appends one line to ``<job>.results.jsonl``: ``{"id", "status": "ok"|"error"|"unsupported"|
"skipped", "seconds", "error"?, "cost_usd"?}``. When done it writes ``<job>.worker.json`` with the
model identity and the measured peak VRAM. A crash on one item never aborts the rest, and the
process exits when the job ends, so the model's GPU memory is always released.

API workers put the item's spend in the payload as ``"_cost_usd"``; when the job carries
``budget_usd`` the remaining items are marked ``skipped`` once the running total exceeds it.
Raise :class:`Unsupported` for an item the model cannot handle (language, length, modality).

A worker module defines ``load(params, lang) -> state`` and ``run(state, item, out) -> dict`` and
calls ``serve(load, run)``. ``run`` returns the JSON written to ``<out>.json``; it may also write
extra files (``<out>.wav`` …). Optional ``describe(state) -> dict`` adds identity fields (model
revision, device) to worker.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


class Unsupported(Exception):
    """This model cannot handle this item (not a failure of the model): recorded, not scored."""


def api_key(*names: str) -> str:
    """First set env var among ``names``; raises with the names so the log says what to export."""
    import os

    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise RuntimeError(f"set one of: {', '.join(names)}")


def preload_pip_cuda_libs(*rels: str) -> None:
    """dlopen pip-installed CUDA libs (``nvidia/<rel>``) globally before a CUDA-12 extension needs
    them. CTranslate2 and onnxruntime-gpu dlopen ``libcublas.so.12`` by bare name, which the loader
    cannot find inside site-packages (a CUDA 13 torch ships only ``.so.13``). Same fix as
    ``app/providers/asr/faster_whisper.py``; missing files are skipped."""
    import ctypes
    import sysconfig

    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for rel in rels:
        lib = nvidia / rel
        if lib.exists():
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


CUDA12_LIBS = ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
               "cudnn/lib/libcudnn.so.9")


class _GpuSampler:
    """Peak GPU memory of this process from nvidia-smi, for frameworks torch cannot see
    (CTranslate2, onnxruntime, llama.cpp). Samples every 0.5 s on a daemon thread."""

    def __init__(self) -> None:
        import os
        import threading

        self.pid = os.getpid()
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import subprocess

        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5, check=False).stdout
                for line in out.splitlines():
                    pid, mib = (x.strip() for x in line.split(","))
                    if int(pid) == self.pid:
                        self.peak = max(self.peak, float(mib))
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self._stop.wait(0.5)

    def stop(self) -> float | None:
        self._stop.set()
        self._thread.join(timeout=6)
        return self.peak or None


def _peak_vram_mib() -> float | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
    except Exception:  # noqa: BLE001 - probing must never fail a job
        return None
    return None


def serve(load: Callable[[dict, str], Any], run: Callable[[Any, dict, Path], dict],
          describe: Callable[[Any], dict] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job_path = Path(ap.parse_args().job)
    job = json.loads(job_path.read_text())
    results = job_path.with_suffix(".results.jsonl")

    sampler = _GpuSampler()
    t0 = time.perf_counter()
    state = load(job.get("params", {}), job.get("lang", ""))
    load_s = time.perf_counter() - t0

    budget = job.get("budget_usd")
    spent = 0.0
    with results.open("a") as log:
        for item in job["items"]:
            out = Path(item["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            rec: dict[str, Any] = {"id": item["id"]}
            if budget is not None and spent >= budget:
                rec.update(status="skipped", error=f"budget ${budget:g} reached", seconds=0.0)
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            try:
                payload = run(state, item, out)
                cost = payload.pop("_cost_usd", None) if isinstance(payload, dict) else None
                if cost is not None:
                    rec["cost_usd"] = float(cost)
                    spent += float(cost)
                out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False,
                                                               indent=1))
                rec["status"] = "ok"
            except Unsupported as exc:
                rec["status"] = "unsupported"
                rec["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - one bad item must not sink the batch
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(file=sys.stderr)
            rec["seconds"] = round(time.perf_counter() - start, 4)
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()

    process_peak = sampler.stop()
    info = {"load_seconds": round(load_s, 3),
            # nvidia-smi sees the whole process (incl. CUDA context); torch sees only its tensors
            "peak_vram_mib": process_peak or _peak_vram_mib(),
            "torch_peak_allocated_mib": _peak_vram_mib(),
            "python": sys.version.split()[0], "spent_usd": round(spent, 6)}
    torch = sys.modules.get("torch")
    if torch is not None:
        info["torch"] = getattr(torch, "__version__", None)
    if describe is not None:
        try:
            info.update(describe(state))
        except Exception as exc:  # noqa: BLE001 - identity is best-effort
            info["describe_error"] = repr(exc)
    job_path.with_suffix(".worker.json").write_text(json.dumps(info, indent=1))
