"""Judge worker (phase F): full-frame LPIPS and a per-clip I3D Fréchet distance vs ground truth.

item.inputs: {video (candidate output), gt_video (the original, self-reenactment packs only)}.
Both streams are decoded at 25 fps, ``height`` px (default 256), from t = 0, and compared over
their common length (≤ ``max_frames``).

- ``lpips``: AlexNet LPIPS (lpips 0.1.4) averaged over frames sampled every ``lpips_stride``.
- ``fvd_clip``: Fréchet distance between the I3D (Kinetics-400, the StyleGAN-V TorchScript port
  used by most FVD code: ``i3d_torchscript.pt``) features of the sliding 16-frame windows of the
  output and of the ground truth. One clip gives few windows, so this is a per-item diagnostic
  (not comparable to the corpus-level FVD in papers); ``fvd_feat_l2`` (mean-feature distance) is
  reported alongside because it is stable with few windows. The detector is called with
  ``rescale=True, resize=True`` on [0, 255] float input (as StyleGAN-V's metric code).
params: lpips_net (alex), height (256), fvd_window (16), fvd_stride (4), lpips_stride (2),
max_frames (250), i3d_path (default ~/.opendub/weights/i3d_torchscript.pt; the env recipe
downloads it).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

I3D_DEFAULT = Path.home() / ".opendub" / "weights" / "i3d_torchscript.pt"


def load(params: dict, lang: str):
    import lpips
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net=params.get("lpips_net", "alex"), verbose=False).to(device).eval()
    i3d_path = Path(params.get("i3d_path") or I3D_DEFAULT).expanduser()
    i3d = torch.jit.load(str(i3d_path), map_location=device).eval() if i3d_path.exists() else None
    return {"lpips": net, "i3d": i3d, "device": device, "params": params}


def _frechet(a, b) -> float:
    import numpy as np
    from scipy import linalg

    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False) if len(a) > 1 else np.zeros((a.shape[1], a.shape[1]))
    cb = np.cov(b, rowvar=False) if len(b) > 1 else np.zeros((b.shape[1], b.shape[1]))
    covmean, _ = linalg.sqrtm(ca @ cb, disp=False)
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(ca + cb - 2 * np.real(covmean)))


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import torch
    from f_frames import probe_size, read_frames

    p, dev = state["params"], state["device"]
    height = int(p.get("height", 256))
    gw, gh, _ = probe_size(item["inputs"]["gt_video"])
    size = (max(2, round(gw * height / gh / 2) * 2), height)
    cap = int(p.get("max_frames", 250))
    gt, _ = read_frames(item["inputs"]["gt_video"], fps=25, size=size, max_frames=cap)
    ov, _ = read_frames(item["inputs"]["video"], fps=25, size=size, max_frames=cap)
    n = min(len(gt), len(ov))
    if n == 0:
        return {"metrics": {}}
    metrics, weights = {}, {}
    idx = list(range(0, n, int(p.get("lpips_stride", 2))))
    vals = []
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            sl = idx[i:i + 16]
            a = torch.from_numpy(gt[sl]).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            b = torch.from_numpy(ov[sl]).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            vals += state["lpips"](a, b).flatten().tolist()
    metrics["lpips"], weights["lpips"] = float(np.mean(vals)), float(len(vals))

    win, stride = int(p.get("fvd_window", 16)), int(p.get("fvd_stride", 4))
    if state["i3d"] is not None and n >= win:
        def feats(frames):
            starts = list(range(0, n - win + 1, stride))
            clips = np.stack([frames[s:s + win] for s in starts])  # (k, t, h, w, c)
            x = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).float().to(dev)
            out_f = []
            with torch.no_grad():
                for j in range(0, len(x), 8):
                    out_f.append(state["i3d"](x[j:j + 8], rescale=True, resize=True,
                                              return_features=True).cpu().numpy())
            return np.concatenate(out_f)

        fa, fb = feats(gt), feats(ov)
        metrics["fvd_clip"], weights["fvd_clip"] = _frechet(fa, fb), float(len(fa))
        metrics["fvd_feat_l2"] = float(np.linalg.norm(fa.mean(0) - fb.mean(0)))
    return {"metrics": metrics, "weights": weights}


def describe(state: dict) -> dict:
    return {"lpips_net": state["params"].get("lpips_net", "alex"),
            "i3d": state["i3d"] is not None}


if __name__ == "__main__":
    serve(load, run, describe)
