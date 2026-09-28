"""B2 worker: TransVLM (HeyGen, ECCV 2026; Apache-2.0 code and weights, Qwen3-VL-4B + NeuFlow v2
optical flow as a 6-channel input). Shot *transition* detection: it returns transition spans.

The checkpoint only works through the repo's own ``inference/infer_video.py`` (a plain
transformers Qwen3-VL pipeline gives wrong results), so this worker runs inside the repo's venv
(``bench/envs/b2_transvlm.yaml``) and shells out to that script per item. The model therefore
reloads for every item: rtfx for this candidate includes model load time, so compare its speed
with care.

params: repo (~/.opendub/src/TransVLM), ckpt_dir (<repo>/inference/pretrained/TransVLM-v1),
backend (hf: the only backend matching the paper's scores), window_size (10), stride (9),
cut_max_s (0.08: a span shorter than this is reported as a hard cut).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _sdk import serve
from b_common import probe

HOME = Path.home() / ".opendub"


def load(params: dict, lang: str) -> dict:
    repo = Path(params.get("repo", HOME / "src" / "TransVLM")).expanduser()
    ckpt = Path(params.get("ckpt_dir", repo / "inference" / "pretrained" / "TransVLM-v1"))
    script = repo / "inference" / "infer_video.py"
    if not script.exists() or not ckpt.expanduser().exists():
        raise FileNotFoundError(f"TransVLM not set up: {script} / {ckpt} "
                                "(run `python -m bench arena env setup b2_transvlm`)")
    return {"script": script, "ckpt": ckpt.expanduser(), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    video = item["inputs"]["video"]
    info = probe(video)
    jsonl = out.parent / f"{out.name}.transvlm.jsonl"
    jsonl.unlink(missing_ok=True)
    cmd = [sys.executable, str(state["script"]), "--video", video, "--ckpt-dir",
           str(state["ckpt"]), "--output-jsonl", str(jsonl),
           "--window-size", str(p.get("window_size", 10.0)),
           "--stride", str(p.get("stride", 9.0))]
    if p.get("backend"):
        cmd += ["--backend", str(p["backend"])]
    subprocess.run(cmd, check=True, cwd=str(state["script"].parent))
    lines = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
    if not lines:
        raise RuntimeError("TransVLM wrote no record")
    cut_max = float(p.get("cut_max_s", 0.08))
    transitions = []
    for seg in lines[-1].get("segments", []):
        s, e = float(seg["start_time"]), float(seg["end_time"])
        if e - s <= cut_max:
            transitions.append({"start": e, "end": e, "type": "cut"})
        else:
            transitions.append({"start": s, "end": e, "type": "gradual"})
    return {"transitions": transitions, "fps": info["fps"], "duration_s": info["duration_s"]}


def describe(state: dict) -> dict:
    return {"ckpt_dir": str(state["ckpt"]), "backend": state["params"].get("backend", "hf")}


if __name__ == "__main__":
    serve(load, run, describe)
