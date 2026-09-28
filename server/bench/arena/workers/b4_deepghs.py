"""B4 worker: deepghs anime classifiers via ``dghs-imgutils`` (ONNX, CPU-capable).

- ``imgutils.validate.anime_classify_score`` — 5 classes: 3d, bangumi, comic, illustration,
  not_painting (default model mobilenetv3_v1.5_dist; caformer_s36_v1.3_focal is the accurate one).
- ``imgutils.validate.anime_real_score`` — 2 classes: anime, real (deepghs/anime_real_cls).

Per frame: p(live_action) = anime_real "real" when ``real_model`` is set, else not_painting; the
remaining mass is split between 3d and 2d (bangumi + comic + illustration) in proportion to the
classifier scores. Frames are aggregated per title (``b_common.aggregate_title``).

params: model (anime_classify model name), real_model (anime_real model name or null),
n_frames (32), frame_side (512).
"""
from __future__ import annotations

import io
from pathlib import Path

from _sdk import serve
from b_common import aggregate_title, grab_jpeg, probe, sample_times


def load(params: dict, lang: str) -> dict:
    from imgutils.validate import anime_classify_score, anime_real_score

    return {"params": params, "cls": anime_classify_score, "real": anime_real_score}


def frame_probs(scores: dict[str, float], real: dict[str, float] | None) -> dict[str, float]:
    s3d = float(scores.get("3d", 0.0))
    s2d = sum(float(scores.get(k, 0.0)) for k in ("bangumi", "comic", "illustration"))
    live = float(real.get("real", 0.0)) if real else float(scores.get("not_painting", 0.0))
    rest = max(0.0, 1.0 - live)
    denom = (s3d + s2d) or 1.0
    return {"live_action": live, "2d": rest * s2d / denom, "3d": rest * s3d / denom}


def run(state: dict, item: dict, out: Path) -> dict:
    from PIL import Image

    p = state["params"]
    video = item["inputs"]["video"]
    dur = probe(video)["duration_s"]
    frames = []
    for t in sample_times(dur, int(p.get("n_frames", 32))):
        im = Image.open(io.BytesIO(grab_jpeg(video, t, max_side=int(p.get("frame_side", 512)))))
        im = im.convert("RGB")
        scores = state["cls"](im, model_name=p.get("model", "caformer_s36_v1.3_focal"))
        real = (state["real"](im, model_name=p["real_model"]) if p.get("real_model") else None)
        frames.append({"t": round(t, 3), "probs": frame_probs(scores, real), "raw": scores,
                       "real": real})
    verdict = aggregate_title([f["probs"] for f in frames])
    verdict["frames"] = frames
    return verdict


def describe(state: dict) -> dict:
    from importlib.metadata import version

    p = state["params"]
    return {"model": p.get("model"), "real_model": p.get("real_model"),
            "dghs_imgutils": version("dghs-imgutils")}


if __name__ == "__main__":
    serve(load, run, describe)
