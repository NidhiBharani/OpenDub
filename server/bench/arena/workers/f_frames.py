"""Numpy frame helpers for phase-F workers. Not a worker; module level is stdlib-only (numpy is
imported inside each function, and every F env has it).

- ``read_frames`` / ``write_frames``: RGB uint8 arrays through ffmpeg pipes.
- ``edit_mask``: where a lip-synced video differs from its untouched source (the F1 mask),
  feathered — so F2 restorers only touch what F1 changed.
- ``mask_composite``: blend a restored video back into the lip-synced one inside that mask.
- ``annulus_match``: the classical F2 path (colour/luma transfer + grain from the untouched
  surround, feathered seam), which uses statistics outside the mask only.
- ``paste_keyframe``: F4 image editors edit one keyframe; paste the edited box region into every
  frame of the text's span with a feathered edge.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe_size(path: str | Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, _, den = s["r_frame_rate"].partition("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den or 1)


def read_frames(path: str | Path, *, fps: float | None = None, size: tuple[int, int] | None = None,
                max_frames: int | None = None):
    import numpy as np

    w, h, native = probe_size(path)
    if size:
        w, h = size
    vf = []
    if fps:
        vf.append(f"fps={fps:g}")
    if size:
        vf.append(f"scale={w}:{h}")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    raw = subprocess.run(cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True,
                         check=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).copy(), fps or native


def write_frames(frames, fps: float, dst: Path, audio: str | Path | None = None) -> Path:
    _, h, w, _ = frames.shape
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:g}", "-i", "-"]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.run(cmd, input=frames.astype("uint8").tobytes(), capture_output=True,
                          check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])
    return dst


def _blur(x, k: int):
    """Separable box blur (k odd) over the last two axes, same size."""
    import numpy as np

    if k <= 1:
        return x
    pad = k // 2
    for axis in (-1, -2):
        xp = np.pad(x, [(0, 0)] * (x.ndim - 2) + ([(0, 0), (pad, pad)] if axis == -1
                                                  else [(pad, pad), (0, 0)]), mode="edge")
        c = np.cumsum(xp, axis=axis, dtype=np.float32)
        c = np.concatenate([np.zeros_like(c.take([0], axis=axis)), c], axis=axis)
        n = x.shape[axis]
        x = (c.take(range(k, n + k), axis=axis) - c.take(range(n), axis=axis)) / k
    return x


def edit_mask(edited, source, *, thresh: float = 6.0, grow: int = 15, feather: int = 21):
    """Per-frame soft mask (N, H, W) in [0, 1] of where ``edited`` differs from ``source``."""
    import numpy as np

    n = min(len(edited), len(source))
    diff = np.abs(edited[:n].astype(np.float32) - source[:n].astype(np.float32)).mean(axis=-1)
    hard = (diff > thresh).astype(np.float32)
    # union over a short window: the mouth patch is a stable region, per-frame noise is not
    hard = np.maximum(hard, np.roll(hard, 1, 0))
    grown = (_blur(hard, grow | 1) > 0.02).astype(np.float32)
    return np.clip(_blur(grown, feather | 1), 0.0, 1.0)


def mask_composite(restored, edited, source):
    """Restored pixels inside the F1 edit mask, the lip-synced frame elsewhere."""
    import numpy as np

    n = min(len(restored), len(edited), len(source))
    a = edit_mask(edited[:n], source[:n])[..., None]
    out = edited[:n].astype(np.float32) * (1 - a) + restored[:n].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def annulus_match(edited, source, *, ring: int = 25, grain: bool = True, seed: int = 0):
    """Classical F2: per frame, match the patch's colour mean/std and high-frequency grain to a
    ring of untouched pixels around the mask, then feather. Never reads source pixels inside the
    mask, so on self-reenactment packs it cannot leak ground truth."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = min(len(edited), len(source))
    soft = edit_mask(edited[:n], source[:n])
    out = edited[:n].astype(np.float32).copy()
    for i in range(n):
        inside = soft[i] > 0.5
        if inside.sum() < 64:
            continue
        ring_m = (_blur(inside[None].astype(np.float32), 2 * ring + 1)[0] > 0.01) & ~inside \
            & (soft[i] < 0.05)
        if ring_m.sum() < 64:
            continue
        src_ring = source[i][ring_m].astype(np.float32)
        edt_ring = out[i][ring_m]
        patch = out[i][inside]
        # colour transfer: align the patch's surround statistics to the source's surround
        mu_s, sd_s = src_ring.mean(0), src_ring.std(0) + 1e-3
        mu_e, sd_e = edt_ring.mean(0), edt_ring.std(0) + 1e-3
        patch = (patch - mu_e) / sd_e * sd_s + mu_s
        if grain:
            hp_src = source[i].astype(np.float32) - _blur(source[i].astype(np.float32)
                                                          .transpose(2, 0, 1), 3).transpose(1, 2, 0)
            hp_edt = out[i] - _blur(out[i].transpose(2, 0, 1), 3).transpose(1, 2, 0)
            g_src, g_edt = hp_src[ring_m].std(0), hp_edt[inside].std(0)
            extra = np.sqrt(np.clip(g_src ** 2 - g_edt ** 2, 0, None))
            patch = patch + rng.normal(0, 1, patch.shape) * extra
        frame = out[i].copy()
        frame[inside] = patch
        a = soft[i][..., None]
        out[i] = out[i] * (1 - a) + frame * a
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_keyframe(frames, fps: float, edited_key, box, start: float, end: float, *,
                   margin: int = 12, feather: int = 9):
    """Paste ``edited_key``'s box region (+margin, feathered) into frames within [start, end]."""
    import numpy as np

    n, h, w, _ = frames.shape
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
    alpha = np.zeros((h, w), np.float32)
    alpha[y0:y1, x0:x1] = 1.0
    alpha = np.clip(_blur(alpha[None], feather | 1)[0] * 1.0, 0, 1)[..., None]
    key = edited_key.astype(np.float32)
    out = frames.copy()
    for i in range(int(start * fps), min(n, int(np.ceil(end * fps)))):
        out[i] = np.clip(frames[i] * (1 - alpha) + key * alpha, 0, 255).astype(np.uint8)
    return out


def mask_bbox(edited, source, *, square: bool = True, pad: float = 0.35):
    """Union box (x0, y0, x1, y1) of the F1 edit mask over all frames, padded (and squared)."""
    import numpy as np

    m = edit_mask(edited, source).max(axis=0) > 0.5
    _, h, w, _ = edited.shape
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * (1 + pad), (y1 - y0) * (1 + pad)
    if square:
        bw = bh = max(bw, bh)
    bw, bh = min(bw, w), min(bh, h)
    x0 = int(min(max(0, cx - bw / 2), w - bw))
    y0 = int(min(max(0, cy - bh / 2), h - bh))
    return x0, y0, int(x0 + bw), int(y0 + bh)


def restore_finish(restored_path: str | Path, item: dict, dst: Path, *, mask_only: bool = True,
                   box: tuple[int, int, int, int] | None = None) -> Path:
    """Bring a restorer's output back to the lip-synced frame: resize (or paste the crop ``box``),
    keep only the F1 edit region when ``mask_only`` (restore inside the mask, per the research
    methodology), and mux the driving audio."""
    import numpy as np

    edited, fps = read_frames(item["inputs"]["video"])
    n, h, w, _ = edited.shape
    if box is None:
        restored, _ = read_frames(restored_path, fps=fps, size=(w, h))
    else:
        x0, y0, x1, y1 = box
        crop, _ = read_frames(restored_path, fps=fps, size=(x1 - x0, y1 - y0))
        restored = edited.copy()
        m = min(len(crop), n)
        restored[:m, y0:y1, x0:x1] = crop[:m]
    m = min(len(restored), n)
    if mask_only and item["inputs"].get("ref_video"):
        source, _ = read_frames(item["inputs"]["ref_video"], fps=fps, size=(w, h))
        result = mask_composite(restored[:m], edited[:m], source)
    else:
        result = restored[:m]
    if len(result) < n:  # a restorer that dropped trailing frames: keep the raw tail
        result = np.concatenate([result, edited[len(result):]], axis=0)
    return write_frames(result, fps, dst, audio=item["inputs"].get("audio"))


def _box_mean(x, k: int):
    import numpy as np

    c = np.cumsum(np.cumsum(np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)]), -1), -2)
    return (c[..., k:, k:] - c[..., :-k, k:] - c[..., k:, :-k] + c[..., :-k, :-k]) / (k * k)


def ssim(a, b, k: int = 7, data_range: float = 255.0) -> float:
    """Mean SSIM, uniform k×k window (same as ``bench.arena.specs.f1.ssim``)."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = _box_mean(a, k), _box_mean(b, k)
    cov = k * k / (k * k - 1)
    va = (_box_mean(a * a, k) - mu_a ** 2) * cov
    vb = (_box_mean(b * b, k) - mu_b ** 2) * cov
    vab = (_box_mean(a * b, k) - mu_a * mu_b) * cov
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def psnr(a, b, data_range: float = 255.0) -> float:
    import numpy as np

    mse = float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))
    return 99.0 if mse <= 1e-10 else float(10 * np.log10(data_range ** 2 / mse))
