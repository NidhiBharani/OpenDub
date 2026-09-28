"""A2 worker: UniPASE (NJU, TASLP 2026; code MIT, HF Xiaobin-Rong/unipase card Apache-2.0).
Runs the repo's CLI ``python -m inference.inference -I <in_dir> -O <out_dir> -D cuda:0
--sr_out <rate>`` from the cloned repo (github.com/xiaobin-rong/unipase, README checked
2026-09-28). Core model is 16 kHz; ``sr_out`` > 16 k goes through the PostNet bandwidth
extension (24/44.1/48 kHz tested upstream). Checkpoints auto-download from HF on first use.
params: sr_out (16000 | 48000), repo (default ~/.opendub/src/unipase), long (use
inference_long for > 30 s clips).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    repo = Path(params.get("repo", Path.home() / ".opendub" / "src" / "unipase")).expanduser()
    if not repo.exists():
        raise FileNotFoundError(f"UniPASE repo missing at {repo} (run the env setup)")
    return {"repo": repo, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    with tempfile.TemporaryDirectory(prefix="unipase-") as tmp:
        t = Path(tmp)
        (t / "in").mkdir()
        (t / "out").mkdir()
        src = Path(item["inputs"]["audio"])
        shutil.copy(src, t / "in" / f"clip{src.suffix}")
        mod = "inference.inference_long" if p.get("long") else "inference.inference"
        cmd = [sys.executable, "-m", mod, "-I", str(t / "in"), "-O", str(t / "out"),
               "-D", "cuda:0", "--sr_out", str(int(p.get("sr_out", 16000)))]
        proc = subprocess.run(cmd, cwd=state["repo"], capture_output=True, text=True,
                              check=False)
        produced = sorted((t / "out").rglob("*.wav"))
        if proc.returncode or not produced:
            raise RuntimeError(f"UniPASE failed ({proc.returncode}): {proc.stderr[-800:]}")
        dst = out.with_suffix(".wav")
        shutil.copy(produced[0], dst)
    return {"files": {"audio": str(dst)}, "sample_rate": int(p.get("sr_out", 16000))}


if __name__ == "__main__":
    serve(load, run)
