"""Judge worker (phase F): face identity and mouth-region fidelity.

item.inputs: {video (candidate output), ref_video (identity reference: the untouched source),
gt_video? (ground truth, self-reenactment only), face_box? [x0, y0, x1, y1] to prefer}.

InsightFace ``buffalo_l`` (SCRFD-10G detector, ArcFace w600k-r50 recognition, 2d106 landmarks;
the model pack is non-commercial research — fine for a judge) on frames sampled at ``fps``
(default 5) from t = 0 in every stream, so the streams stay time-aligned.

Metrics (weights = frames used):
- ``csim_arcface``: mean cosine between each output face embedding and the reference identity
  (mean of the reference frames' normalised embeddings). Largest face per frame.
- ``face_rate``: output frames with a face / reference frames with a face (capped at 1).
- with ``gt_video``: ``mouth_ssim``, ``mouth_psnr`` (luma, 96x96 crops) and ``mouth_lpips``
  (AlexNet LPIPS on RGB crops) in a square around the ground-truth mouth corners;
  ``lmd_lower``: mean distance between output and ground-truth 106-point landmarks below the
  nose tip, in % of the inter-ocular distance (index-order free, so no 106-layout assumption).
params: model (buffalo_l), fps (5), det_size (640), lpips_net (alex), max_frames (60).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import lpips
    import onnxruntime as ort
    import torch
    from insightface.app import FaceAnalysis

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    app = FaceAnalysis(name=params.get("model", "buffalo_l"), providers=providers,
                       allowed_modules=["detection", "recognition", "landmark_2d_106"])
    det = int(params.get("det_size", 640))
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(det, det))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net=params.get("lpips_net", "alex"), verbose=False).to(device).eval()
    return {"app": app, "lpips": net, "device": device, "params": params,
            "providers": providers}


def _largest(app, frame_rgb):
    faces = app.get(frame_rgb[..., ::-1].copy())  # insightface wants BGR
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _mouth_box(face, w: int, h: int, side_scale: float = 1.9):
    import numpy as np

    kps = face.kps
    c = (kps[3] + kps[4]) / 2
    side = max(np.linalg.norm(kps[3] - kps[4]) * side_scale,
               0.3 * (face.bbox[2] - face.bbox[0]))
    x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    return x0, y0, int(min(w, x0 + side)), int(min(h, y0 + side))


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import torch
    from f_frames import psnr, read_frames, ssim
    from PIL import Image

    p, app = state["params"], state["app"]
    fps, cap = float(p.get("fps", 5)), int(p.get("max_frames", 60))
    inp = item["inputs"]
    outv, _ = read_frames(inp["video"], fps=fps, max_frames=cap)
    if len(outv) == 0:
        return {"metrics": {"face_rate": 0.0}}
    h, w = outv.shape[1:3]
    refv, _ = read_frames(inp["ref_video"], fps=fps, size=(w, h), max_frames=cap)
    gtv = None
    if inp.get("gt_video"):
        gtv, _ = read_frames(inp["gt_video"], fps=fps, size=(w, h), max_frames=cap)

    ref_faces = [_largest(app, f) for f in refv]
    ref_emb = [f.normed_embedding for f in ref_faces if f is not None]
    if not ref_emb:
        from _sdk import Unsupported

        raise Unsupported("no face in the reference video")
    ident = np.mean(ref_emb, axis=0)
    ident /= np.linalg.norm(ident) + 1e-9

    out_faces = [_largest(app, f) for f in outv]
    sims = [float(np.dot(f.normed_embedding, ident)) for f in out_faces if f is not None]
    n_ref = sum(f is not None for f in ref_faces[: len(outv)]) or 1
    metrics = {"face_rate": min(1.0, len(sims) / n_ref)}
    weights = {"face_rate": float(n_ref)}
    if sims:
        metrics["csim_arcface"], weights["csim_arcface"] = float(np.mean(sims)), float(len(sims))

    if gtv is not None:
        ss, ps, lp, lmd = [], [], [], []
        for i in range(min(len(outv), len(gtv))):
            g = _largest(app, gtv[i])
            if g is None:
                continue
            x0, y0, x1, y1 = _mouth_box(g, w, h)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            a = np.asarray(Image.fromarray(gtv[i][y0:y1, x0:x1]).resize((96, 96)), np.float32)
            b = np.asarray(Image.fromarray(outv[i][y0:y1, x0:x1]).resize((96, 96)), np.float32)
            ga, gb = a @ [0.299, 0.587, 0.114], b @ [0.299, 0.587, 0.114]
            ss.append(ssim(ga[None], gb[None]))
            ps.append(psnr(ga, gb))
            with torch.no_grad():
                ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(state["device"]) / 127.5 - 1
                tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(state["device"]) / 127.5 - 1
                lp.append(float(state["lpips"](ta, tb).item()))
            o = out_faces[i]
            if o is not None and getattr(g, "landmark_2d_106", None) is not None \
                    and getattr(o, "landmark_2d_106", None) is not None:
                below = g.landmark_2d_106[:, 1] > g.kps[2][1]
                iod = float(np.linalg.norm(g.kps[0] - g.kps[1])) or 1.0
                d = np.linalg.norm(o.landmark_2d_106[below] - g.landmark_2d_106[below], axis=1)
                lmd.append(float(d.mean()) / iod * 100.0)
        for name, vals in (("mouth_ssim", ss), ("mouth_psnr", ps), ("mouth_lpips", lp),
                           ("lmd_lower", lmd)):
            if vals:
                metrics[name], weights[name] = float(np.mean(vals)), float(len(vals))
    return {"metrics": metrics, "weights": weights}


def describe(state: dict) -> dict:
    import insightface

    return {"model": state["params"].get("model", "buffalo_l"),
            "insightface": insightface.__version__, "providers": state["providers"]}


if __name__ == "__main__":
    serve(load, run, describe)
