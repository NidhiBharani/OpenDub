"""B4 worker: frozen image encoder (SigLIP 2, or an open_clip model such as PE-Core) with either a
zero-shot text head or a linear probe fitted on the pack's ``dev`` titles, plus optional split
conformal abstention.

params:
  backbone: siglip2 (transformers AutoModel) | open_clip
  model: HF repo id (siglip2) or open_clip name (e.g. ``hf-hub:timm/PE-Core-L-14-336``)
  revision, dtype (bfloat16), n_frames (32), frame_side (512)
  head: zeroshot | probe
  train_manifest: dev-split source for ``probe`` — default ``{eval}/B4/{lang}/manifest.jsonl``
    (``{eval}`` = $OPENDUB_ARENA_DATA/eval or <repo>/data/eval); only ``split: dev`` rows are used,
    so the probe never sees a test title. Needs ≥ 2 classes among the dev titles.
  l2 (1e-2), epochs (300): multinomial logistic regression on L2-normalised embeddings (numpy).
  conformal_alpha: e.g. 0.1 → dev titles are split in half (fit / calibrate) and a title abstains
    unless its conformal prediction set is a single class.

Frames are sampled evenly over the whole title (``b_common.sample_times``), classified into
live_action / 2d / 3d, and aggregated per title (``b_common.aggregate_title``: mixed when two
classes each hold a real share of the frames).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from _sdk import serve
from b_common import B4_CLASSES, aggregate_title, grab_jpeg, probe, sample_times

PROMPTS = {
    "live_action": ["a photograph", "a frame from a live-action film shot with a camera",
                    "a still from a TV show with real actors"],
    "2d": ["a frame from a 2D hand-drawn cartoon", "an anime screenshot",
           "a frame of cel animation"],
    "3d": ["a frame from a 3D computer-animated film", "a CGI render",
           "a still from a Pixar-style 3D animation"],
}


def _eval_dir() -> Path:
    env = os.environ.get("OPENDUB_ARENA_DATA")
    return (Path(env) if env else Path(__file__).resolve().parents[4] / "data") / "eval"


class Encoder:
    def __init__(self, params: dict):
        import torch

        self.torch = torch
        self.backbone = params.get("backbone", "siglip2")
        self.device = params.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, params.get("dtype", "bfloat16"))
        if self.backbone == "siglip2":
            from transformers import AutoModel, AutoProcessor

            self.model = AutoModel.from_pretrained(params["model"],
                                                   revision=params.get("revision"),
                                                   dtype=dtype).to(self.device).eval()
            self.proc = AutoProcessor.from_pretrained(params["model"],
                                                      revision=params.get("revision"))
        else:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                params["model"])
            self.model = self.model.to(self.device, dtype=dtype).eval()
            self.tokenizer = open_clip.get_tokenizer(params["model"])
        self.dtype = dtype

    def images(self, pil: list) -> list[list[float]]:
        torch = self.torch
        with torch.inference_mode():
            if self.backbone == "siglip2":
                inp = self.proc(images=pil, return_tensors="pt").to(self.device)
                # naflex checkpoints add pixel_attention_mask / spatial_shapes: pass everything
                feats = self.model.get_image_features(**{
                    k: (v.to(self.dtype) if k == "pixel_values" else v) for k, v in inp.items()})
            else:
                x = torch.stack([self.preprocess(im) for im in pil]).to(self.device, self.dtype)
                feats = self.model.encode_image(x)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().tolist()

    def texts(self, texts: list[str]) -> list[list[float]]:
        torch = self.torch
        with torch.inference_mode():
            if self.backbone == "siglip2":
                inp = self.proc(text=texts, padding="max_length", max_length=64,
                                return_tensors="pt").to(self.device)
                feats = self.model.get_text_features(**inp)
            else:
                feats = self.model.encode_text(self.tokenizer(texts).to(self.device))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().tolist()


def _frames(video: str, n: int, side: int) -> list:
    import io

    from PIL import Image

    dur = probe(video)["duration_s"]
    return [Image.open(io.BytesIO(grab_jpeg(video, t, max_side=side))).convert("RGB")
            for t in sample_times(dur, n)]


def _softmax(z):
    import numpy as np

    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_probe(X, y, n_classes: int, l2: float, epochs: int):
    """Multinomial logistic regression by full-batch gradient descent (tiny data, no sklearn)."""
    import numpy as np

    W = np.zeros((X.shape[1], n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    lr = 1.0
    for _ in range(epochs):
        P = _softmax(X @ W + b)
        G = (P - Y) / len(X)
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b


def _dev_titles(params: dict, lang: str) -> list[tuple[str, str, str]]:
    """[(title id, video path, label)] of the dev split."""
    src = params.get("train_manifest", "{eval}/B4/{lang}/manifest.jsonl")
    path = Path(src.format(eval=_eval_dir(), lang=lang))
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "dev":
            continue
        video = r["inputs"]["video"]
        video = video if video.startswith("/") else str(path.parent / video)
        rows.append((r["id"], video, r["refs"]["label"]))
    return rows


def load(params: dict, lang: str) -> dict:
    import numpy as np

    enc = Encoder(params)
    state = {"enc": enc, "params": params, "head": params.get("head", "zeroshot"),
             "qhat": None}
    n, side = int(params.get("n_frames", 32)), int(params.get("frame_side", 512))
    if state["head"] == "zeroshot":
        T = []
        for c in B4_CLASSES:
            t = np.array(enc.texts(PROMPTS[c])).mean(axis=0)
            T.append(t / np.linalg.norm(t))
        state["T"] = np.array(T)
        state["scale"] = float(params.get("logit_scale", 100.0))
        return state

    dev = [d for d in _dev_titles(params, lang) if d[2] in B4_CLASSES]
    if len({d[2] for d in dev}) < 2:
        raise RuntimeError("probe head needs dev titles of ≥ 2 pure classes in the B4 pack")
    alpha = params.get("conformal_alpha")
    fit, cal = (dev[0::2], dev[1::2]) if alpha else (dev, [])
    feats = {d[0]: np.array(enc.images(_frames(d[1], n, side))) for d in dev}
    X = np.concatenate([feats[d[0]] for d in fit])
    y = np.concatenate([[B4_CLASSES.index(d[2])] * len(feats[d[0]]) for d in fit])
    state["W"], state["b"] = _fit_probe(X, y, len(B4_CLASSES), float(params.get("l2", 1e-2)),
                                        int(params.get("epochs", 300)))
    if alpha and cal:
        scores = []
        for d in cal:
            probs = _title_probs(state, feats[d[0]])
            scores.append(1.0 - probs.get(d[2], 0.0))
        k = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - float(alpha)))))
        state["qhat"] = sorted(scores)[k - 1]
    return state


def _frame_probs(state: dict, F) -> list[dict[str, float]]:
    import numpy as np

    if state["head"] == "zeroshot":
        P = _softmax(state["scale"] * (np.asarray(F) @ state["T"].T))
    else:
        P = _softmax(np.asarray(F) @ state["W"] + state["b"])
    return [{c: float(p[k]) for k, c in enumerate(B4_CLASSES)} for p in P]


def _title_probs(state: dict, F) -> dict[str, float]:
    return aggregate_title(_frame_probs(state, F))["probs"]


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    F = state["enc"].images(_frames(item["inputs"]["video"], int(p.get("n_frames", 32)),
                                    int(p.get("frame_side", 512))))
    fps = _frame_probs(state, F)
    verdict = aggregate_title(fps)
    if state["qhat"] is not None:
        pset = [c for c, v in verdict["probs"].items() if 1.0 - v <= state["qhat"]]
        verdict["prediction_set"] = pset
        if len(pset) != 1:
            verdict["abstain"] = True
    verdict["frames"] = [{"probs": f} for f in fps]
    return verdict


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"), "head": state["head"],
            "backbone": p.get("backbone", "siglip2"), "qhat": state["qhat"]}


if __name__ == "__main__":
    serve(load, run, describe)
