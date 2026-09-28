"""E4 worker: ffmpeg ``loudnorm`` two-pass *linear* normalisation of the premix (the classical
pipeline tool; pass 1 measures, pass 2 applies a static gain with ``linear=true``). Programme
gating only; loudnorm falls back to dynamic mode when a linear gain would break the true-peak
target — that behaviour is part of what is ranked. No dialogue levelling (uses ``inputs.audio``).

params: lra (20: wide, so linear mode is honoured), tp_margin (0.0).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH")
    return {"p": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p, inp = state["p"], item["inputs"]
    d = inp["delivery"]
    tp = float(d["true_peak_dbtp"]) - float(p.get("tp_margin", 0.0))
    base = f"loudnorm=I={float(d['target_lufs']):.1f}:TP={tp:.1f}:LRA={float(p.get('lra', 20)):.1f}"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", inp["audio"], "-af",
                        base + ":print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    filt = (f"{base}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
            f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
            f":offset={m['target_offset']}:linear=true:print_format=json")
    wav = out.with_suffix(".wav")
    r2 = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", inp["audio"], "-af",
                         filt, "-ar", "48000", "-c:a", "pcm_s24le", str(wav)],
                        capture_output=True, text=True, check=True)
    m2 = json.loads(re.findall(r"\{[^{}]*\}", r2.stderr)[-1])
    return {"files": {"audio": str(wav)},
            "params": {"pass1": m, "normalization_type": m2.get("normalization_type")}}


if __name__ == "__main__":
    serve(load, run)
