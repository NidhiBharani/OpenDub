"""D6 worker: Seed-VC (Plachtaa/seed-vc, GPL-3.0 → ship_ok false) from its repo
(``~/.opendub/src/seed-vc``). The repo's documented entry point is the ``inference.py`` CLI
(``--source --target --output --diffusion-steps --length-adjust --inference-cfg-rate
--f0-condition --auto-f0-adjust --semi-tone-shift --fp16``), so each item runs it once; the model
reloads per item (a few seconds) — acceptable for an evaluate-only comparator.

params: diffusion_steps (25; README suggests 30–50 for quality), inference_cfg_rate (0.7),
        f0_condition (false; true selects the 44.1 kHz singing model), fp16 (true),
        script (inference.py | inference_v2.py) and extra_args for V2.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    repo = dv.src_dir("seed-vc")
    if not (repo / params.get("script", "inference.py")).exists():
        raise RuntimeError(f"{repo} missing: run the d6_seed_vc env setup")
    return {"repo": repo, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src = (item.get("inputs") or {}).get("audio")
    if not src:
        raise Unsupported("VC item has no inputs.audio")
    ref, _ = dv.ref_of(item)
    tmp = Path(out).with_name(Path(out).name + "-seedvc")
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, p.get("script", "inference.py"), "--source", src, "--target", ref,
           "--output", str(tmp), "--diffusion-steps", str(p.get("diffusion_steps", 25)),
           "--length-adjust", "1.0", "--inference-cfg-rate", str(p.get("inference_cfg_rate", 0.7)),
           "--f0-condition", str(bool(p.get("f0_condition", False))),
           "--auto-f0-adjust", "False", "--semi-tone-shift", "0",
           "--fp16", str(bool(p.get("fp16", True))), *p.get("extra_args", [])]
    subprocess.run(cmd, cwd=state["repo"], check=True)
    made = sorted(tmp.glob("*.wav"), key=lambda f: f.stat().st_mtime)
    if not made:
        raise RuntimeError("seed-vc wrote no wav")
    wav = dv.wav_path(out)
    made[-1].replace(wav)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return dv.audio_payload(wav)


def describe(state: dict) -> dict:
    return {"script": state["params"].get("script", "inference.py")}


if __name__ == "__main__":
    serve(load, run, describe)
