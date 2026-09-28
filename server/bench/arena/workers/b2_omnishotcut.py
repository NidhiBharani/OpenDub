"""B2 worker: OmniShotCut (UVA CV Lab, MIT code *and* weights per the repo LICENSE; the arXiv
paper itself is CC BY-NC-ND). ``pip install git+https://github.com/UVA-Computer-Vision-Lab/
OmniShotCut``; weights from HF ``uva-cv-lab/OmniShotCut_v1.5`` (or ``_paper``).

``inference(video, mode="default")`` returns shot ranges ``[[start_frame, end_frame], ...]`` plus
intra/inter labels; ranges whose intra label names a transition (dissolve / fade / wipe /
transition) are reported as gradual transitions, every other boundary between consecutive
ranges as a cut. ``mode: clean_shot`` returns shots only (hard cuts between them).

params: repo_id (uva-cv-lab/OmniShotCut_v1.5), filename (OmniShotCut_ckpt.pth), mode (default).
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from _sdk import serve
from b_common import probe

GRADUAL = ("dissolve", "fade", "wipe", "transition", "gradual")


def load(params: dict, lang: str) -> dict:
    import omnishotcut

    model = omnishotcut.load(params.get("repo_id", "uva-cv-lab/OmniShotCut_v1.5"),
                             filename=params.get("filename", "OmniShotCut_ckpt.pth"))
    return {"model": model, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video = item["inputs"]["video"]
    info = probe(video)
    fps = info["fps"]
    mode = state["params"].get("mode", "default")
    res = state["model"].inference(video, mode=mode)
    ranges, intra = (res[0], res[1]) if isinstance(res, tuple) else (res, None)
    shots, grads = [], []
    for k, (a, b) in enumerate(ranges):
        label = str(intra[k]).lower() if intra is not None and k < len(intra) else ""
        if any(g in label for g in GRADUAL):
            grads.append({"start": a / fps, "end": (b + 1) / fps, "type": label or "gradual"})
        else:
            shots.append((int(a), int(b), label))
    shots.sort()
    transitions = list(grads)
    for (_a0, b0, _l0), (a1, _b1, l1) in pairwise(shots):
        covered = any(g["start"] <= a1 / fps + 1e-6 and g["end"] >= (b0 + 1) / fps - 1e-6
                      for g in grads)
        if covered:
            continue
        if a1 - b0 > 2:
            transitions.append({"start": (b0 + 1) / fps, "end": a1 / fps, "type": "gradual"})
        else:
            transitions.append({"start": a1 / fps, "end": a1 / fps,
                                "type": "sudden_jump" if "jump" in l1 else "cut"})
    transitions.sort(key=lambda t: t["start"])
    return {"transitions": transitions,
            "shots": [{"start": a / fps, "end": (b + 1) / fps, "label": lab} for a, b, lab in shots],
            "fps": fps, "duration_s": info["duration_s"]}


def describe(state: dict) -> dict:
    p = state["params"]
    return {"repo_id": p.get("repo_id", "uva-cv-lab/OmniShotCut_v1.5"),
            "mode": p.get("mode", "default")}


if __name__ == "__main__":
    serve(load, run, describe)
