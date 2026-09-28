"""F2 — mouth-region restoration of lip-synced video.

Pack (``f2_from_f1``): cached F1 self-reenactment outputs of one generator (LatentSync 1.6 by
default) become the inputs; the untouched original is ground truth. So a restorer is ranked by
how far it moves the generated mouth back toward the real one (mouth LPIPS vs ground truth),
never by how sharp it looks — restorers that invent detail change identity, which the ArcFace
gate catches. The ``passthrough`` baseline (raw F1 output) is the bar every model must clear.
"""
from __future__ import annotations

from ..judges import Spec, cost, speed
from .f1 import (
    F_DIRECTIONS,
    SYNC_PRIMARY,
    abs_metric,
    face_judge,
    fullframe_vs_gt,
    sync_panel,
    video_judge,
)

SPEC = Spec(
    id="F2",
    title="Mouth-region restoration",
    judges={"fullframe_vs_gt@1": fullframe_vs_gt, "speed@1": speed, "cost@1": cost},
    primary={"*": "mouth_lpips"},
    higher_is_better=F_DIRECTIONS,
    threshold={"mouth_lpips": 0.005, "csim_arcface": 0.01},
    secondary=["csim_arcface", "mouth_ssim", "mouth_psnr", "lmd_lower", "lpips", "fvd_clip",
               "psnr", "ssim", SYNC_PRIMARY, "sync_p0_synchformer", "face_rate", "rtfx",
               "cost_usd"],
    # Identity reference = the untouched source picture (what the actor really looks like).
    model_judges=lambda lang: [face_judge(ref_key="ref_video"), video_judge(),
                               *sync_panel(lang)],
    derived={SYNC_PRIMARY: abs_metric("sync_offset_ms_synchformer"),
             "sync_abs_offset_ms_syncnet": abs_metric("sync_offset_ms_syncnet")},
    # No-regression on the F1 sync panel is catastrophic-only here (restoration must not break
    # sync); identity drift is the real veto.
    gates={"csim_arcface": (">=", 0.80), SYNC_PRIMARY: ("<=", 200.0)},
    packs=["f2_from_f1"],
    io="""item.inputs: {video: lip-synced video (an F1 output), ref_video: untouched source,
audio: driving audio}; item.refs: {video: ground-truth original};
item.meta: {f1_candidate, source, duration_s}.
payload: {files: {video: <out>.mp4 (audio kept), audio}, duration_s, fps, width, height}""",
)
