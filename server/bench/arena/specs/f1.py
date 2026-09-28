"""F1 — live-action lip sync; also the phase-F judge toolkit that f2–f4 import.

Two kinds of items share the F1 packs:

- **Self-reenactment** (HDTF, VoxCeleb2 test, LRS3 test, user media): the original video is
  re-driven with its own audio, so the original is ground truth (``refs.video``). Scored by
  full-frame PSNR/SSIM (function judges), mouth-region SSIM/PSNR/LPIPS and lower-face landmark
  distance (``face.arcface``), full-frame LPIPS and a per-clip I3D FVD (``video.lpips_fvd``).
- **Cross-lingual** (user-supplied live-action clips with a dub track): no ground truth, so
  only the sync panel, identity and face-detection rate apply.

Primary is the Synchformer |offset| (a sync judge from a lineage disjoint from every generator;
SyncNet LSE is reported for comparability only — LatentSync and Wav2Lip are trained against
SyncNet, docs/plans/model-ranking.md §4.7). ArcFace CSIM against the source is a gate: a model
that changes the actor cannot win, whatever its sync score.

SPY×FAMILY (the ``scene_dub`` pack) is 2D animation: its rows land in F1 only under split
``control`` (negative control for B4 gating failures), never in the ranked ``test`` split.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..hardware import Requires
from ..judgelib import _metrics_rows, lipsync_score
from ..judges import ModelJudge, Row, Spec, cost, output_file, speed
from ..packs import Item

# ---------------------------------------------------------------- frame access (numpy + ffmpeg)

JUDGE_FPS = 25.0        # every comparison resamples both videos to this rate
MAX_FRAMES = 250        # 10 s at 25 fps per item keeps function judges cheap


def read_gray(path: str | Path, *, width: int, height: int, fps: float = JUDGE_FPS,
              max_frames: int = MAX_FRAMES, start: float = 0.0) -> np.ndarray:
    """Decode to an (N, H, W) float32 luma array at a fixed size and frame rate."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-vf",
           f"fps={fps:g},scale={width}:{height}:flags=area", "-frames:v", str(max_frames),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout
    n = len(raw) // (width * height)
    return np.frombuffer(raw[: n * width * height], np.uint8).reshape(n, height, width) \
        .astype(np.float32)


def video_size(path: str | Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True, timeout=60).stdout
    w, h = (int(x) for x in out.strip().split(",")[:2])
    return w, h


def judge_size(path: str | Path, height: int = 256) -> tuple[int, int]:
    """(width, height) with the given height, even width, source aspect kept."""
    w, h = video_size(path)
    return max(2, round(w * height / h / 2) * 2), height


def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    """k×k mean filter (valid region) over the last two axes via summed-area tables."""
    c = np.cumsum(np.cumsum(np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)]), -1), -2)
    return (c[..., k:, k:] - c[..., :-k, k:] - c[..., k:, :-k] + c[..., :-k, :-k]) / (k * k)


def ssim(a: np.ndarray, b: np.ndarray, k: int = 7, data_range: float = 255.0) -> float:
    """Mean SSIM with a uniform k×k window (skimage's default), frames on axis 0."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = _box_mean(a, k), _box_mean(b, k)
    n = k * k
    cov = n / (n - 1)  # sample covariance, as skimage
    va = (_box_mean(a * a, k) - mu_a ** 2) * cov
    vb = (_box_mean(b * b, k) - mu_b ** 2) * cov
    vab = (_box_mean(a * b, k) - mu_a * mu_b) * cov
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-10 else float(10 * np.log10(data_range ** 2 / mse))


def gt_video(item: Item) -> str | None:
    p = item.refs.get("video")
    return p if isinstance(p, str) and Path(p).exists() else None


def fullframe_vs_gt(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """Self-reenactment only: luma PSNR/SSIM of the output against the original at 256p/25 fps.

    Weight = frames compared, so the pack mean is per frame. LatentSync-style models resample
    to 25 fps; both streams are resampled here, and only the common length is compared.
    """
    gt, video = gt_video(item), output_file(out, "video")
    if not gt or not video:
        return []
    w, h = judge_size(gt)
    a = read_gray(gt, width=w, height=h)
    b = read_gray(video, width=w, height=h)
    n = min(len(a), len(b))
    if n == 0:
        return [("frames_missing", 1.0, 1.0)]
    rows: list[Row] = [("psnr", psnr(a[:n], b[:n]), float(n)), ("ssim", ssim(a[:n], b[:n]), float(n)),
                       ("frames_missing", float(max(0, len(a) - len(b)) / len(a)), 1.0)]
    return rows


def duration_match(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """|output duration − driving audio duration| in seconds: truncated or padded output."""
    want = item.meta.get("audio_duration_s") or item.meta.get("duration_s")
    got = out.get("duration_s")
    if not want or got is None:
        return []
    return [("duration_err_s", abs(float(got) - float(want)), 1.0)]


# ---------------------------------------------------------------- model judges owned by phase F

FACE_ENV = "f_judge_face"
VIDEO_ENV = "f_judge_video"


def face_judge(*, version: int = 1, fps: float = 5.0, ref_key: str = "video") -> ModelJudge:
    """ArcFace identity (CSIM vs the source), face-detection rate and, where ground truth exists,
    mouth-region SSIM/PSNR/LPIPS and lower-face landmark distance (``f_judge_face.py``).

    Metrics: ``csim_arcface`` (↑), ``face_rate`` (↑, output frames with a face / source frames
    with a face), ``mouth_ssim`` (↑), ``mouth_psnr`` (↑), ``mouth_lpips`` (↓), ``lmd_lower`` (↓,
    lower-face landmark distance in % of the inter-ocular distance).
    """
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        video = output_file(out, "video")
        ref = item.inputs.get(ref_key)
        if not video or not ref:
            return None
        d = {"video": video, "ref_video": ref}
        if gt_video(item):
            d["gt_video"] = gt_video(item)
        if item.meta.get("face_box"):
            d["face_box"] = item.meta["face_box"]
        return d

    return ModelJudge(
        id=f"face.arcface@{version}", worker="f_judge_face", env=FACE_ENV,
        params={"model": "buffalo_l", "fps": fps, "lpips_net": "alex"},
        requires=Requires.parse({"gpu": True, "vram_gb": 2}), lang_param=False,
        inputs=inputs, to_rows=lambda item, payload, out: _metrics_rows(payload))


def video_judge(*, version: int = 1) -> ModelJudge:
    """Full-frame LPIPS and a per-clip I3D Fréchet distance against ground truth
    (``f_judge_video.py``). ``fvd_clip`` compares the sets of 16-frame window features of one
    clip, so it is a per-item diagnostic, not the corpus FVD papers report."""
    def inputs(item: Item, out: dict[str, Any], prefix: Path):
        video, gt = output_file(out, "video"), gt_video(item)
        return {"video": video, "gt_video": gt} if video and gt else None

    return ModelJudge(
        id=f"video.lpips_fvd@{version}", worker="f_judge_video", env=VIDEO_ENV,
        params={"lpips_net": "alex", "fvd_window": 16, "fvd_stride": 4, "height": 256},
        requires=Requires.parse({"gpu": True, "vram_gb": 3}), lang_param=False,
        inputs=inputs, to_rows=lambda item, payload, out: _metrics_rows(payload))


def abs_metric(name: str):
    """Derived |x| of a signed metric (e.g. the sync offset), so ranking rewards ~0."""
    def fn(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
        if name not in values:
            return None
        v, w = values[name]
        return abs(v), w
    return fn


SYNC_PRIMARY = "sync_abs_offset_ms_synchformer"

# Direction of every metric an F spec may show (judge_lipsync names from judgelib).
F_DIRECTIONS: dict[str, bool] = {
    SYNC_PRIMARY: False, "sync_offset_ms_synchformer": False, "sync_conf_synchformer": True,
    "sync_offset_argmax_ms_synchformer": False, "sync_p0_synchformer": True,
    "sync_abs_offset_ms_syncnet": False, "sync_offset_ms_syncnet": False,
    "sync_conf_syncnet": True, "lse_c": True, "lse_d": False, "peavs": True,
    "sync_conf_peavs": True,
    "csim_arcface": True, "face_rate": True, "mouth_ssim": True, "mouth_psnr": True,
    "mouth_lpips": False, "lmd_lower": False, "psnr": True, "ssim": True, "lpips": False,
    "fvd_clip": False, "frames_missing": False, "duration_err_s": False, "rtfx": True,
    "cost_usd": False,
}


def sync_panel(lang: str) -> list[ModelJudge]:
    """G5 panel (``judge_lipsync``): Synchformer ranks; PEAVS is supplementary (it sees no
    face track); SyncNet LSE-C/D is for comparability with papers only — never rank
    SyncNet-supervised generators (Wav2Lip, LatentSync) by it."""
    return [lipsync_score("synchformer"), lipsync_score("peavs"), lipsync_score("syncnet")]


SPEC = Spec(
    id="F1",
    title="Live-action lip sync",
    judges={"fullframe_vs_gt@1": fullframe_vs_gt, "duration@1": duration_match,
            "speed@1": speed, "cost@1": cost},
    primary={"*": SYNC_PRIMARY},
    higher_is_better=F_DIRECTIONS,
    threshold={SYNC_PRIMARY: 40.0, "mouth_lpips": 0.01, "csim_arcface": 0.01},
    secondary=["sync_p0_synchformer", "csim_arcface", "mouth_lpips", "mouth_ssim",
               "lmd_lower", "lpips", "fvd_clip", "psnr", "ssim", "face_rate",
               "peavs", "lse_c", "lse_d", "duration_err_s", "rtfx", "cost_usd"],
    model_judges=lambda lang: [*sync_panel(lang), face_judge(), video_judge()],
    derived={SYNC_PRIMARY: abs_metric("sync_offset_ms_synchformer"),
             "sync_abs_offset_ms_syncnet": abs_metric("sync_offset_ms_syncnet")},
    # Research proposes CSIM ≥ 0.90 against the untouched source; the absolute level depends on
    # the ArcFace pack and sampling, so the gate starts looser — tighten once the baseline is
    # measured (LatentSync ≈ 0.84–0.93 in published tables).
    gates={"csim_arcface": (">=", 0.80), "face_rate": (">=", 0.90)},
    packs=["f1_selfreenact", "f1_crossling", "scene_dub"],
    io="""item.inputs: {video (25 fps CFR mp4), audio? (driving audio wav; absent = the video's
own track, i.e. self-reenactment)}; item.refs: {video?: ground-truth original};
item.meta: {duration_s, audio_duration_s, kind: self|cross, source, face_box?}.
payload: {files: {video: <out>.mp4 with the driving audio muxed, audio: driving wav},
duration_s, fps, width, height}""",
)
