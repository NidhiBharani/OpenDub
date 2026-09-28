"""B4 worker: a Hugging Face image classifier with its own head (e.g.
prithivMLmods/Anime-Classification-v1.0, a SigLIP 2 fine-tune over 3D / Bangumi / Comic /
Illustration).

params: model, revision, dtype, label_map ({model label: live_action|2d|3d}, case-insensitive;
unmapped labels are dropped), n_frames (32), frame_side (512).

A classifier without a live-action class can never say live_action; that is the honest result
for it (research note: "cannot make the gate decision alone").
"""
from __future__ import annotations

import io
from pathlib import Path

from _sdk import serve
from b_common import B4_CLASSES, aggregate_title, grab_jpeg, probe, sample_times


def load(params: dict, lang: str) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(
        params["model"], revision=params.get("revision"),
        dtype=getattr(torch, params.get("dtype", "float32"))).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(params["model"], revision=params.get("revision"))
    lmap = {str(k).lower(): v for k, v in (params.get("label_map") or {}).items()}
    return {"model": model, "proc": proc, "device": device, "params": params, "lmap": lmap,
            "torch": torch}


def run(state: dict, item: dict, out: Path) -> dict:
    from PIL import Image

    torch, p = state["torch"], state["params"]
    video = item["inputs"]["video"]
    dur = probe(video)["duration_s"]
    frames = []
    for t in sample_times(dur, int(p.get("n_frames", 32))):
        im = Image.open(io.BytesIO(grab_jpeg(video, t, max_side=int(p.get("frame_side", 512)))))
        inp = state["proc"](images=im.convert("RGB"), return_tensors="pt").to(state["device"])
        with torch.inference_mode():
            probs = torch.softmax(state["model"](**inp).logits.float(), dim=-1)[0].tolist()
        id2label = state["model"].config.id2label
        mapped = {c: 0.0 for c in B4_CLASSES}
        raw = {}
        for k, v in enumerate(probs):
            name = str(id2label.get(k, k))
            raw[name] = v
            target = state["lmap"].get(name.lower())
            if target in mapped:
                mapped[target] += v
        frames.append({"t": round(t, 3), "probs": mapped, "raw": raw})
    verdict = aggregate_title([f["probs"] for f in frames])
    verdict["frames"] = frames
    return verdict


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision")}


if __name__ == "__main__":
    serve(load, run, describe)
