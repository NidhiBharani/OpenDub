"""G5 pack: lip-sync scorers on talking-face clips with injected audio offsets (no model).

Each source clip is cut to ``clip_s`` seconds; its audio is shifted by every offset in
``offsets_ms`` (default −200…+200 ms in 40 ms steps, 0 included twice so in-sync and out-of-sync
are balanced) with numpy and re-muxed over the untouched picture. **Positive offset = audio
delayed (lags the picture)** — the arena-wide convention the lip-sync worker reports in.

refs.offset_ms is ground truth by construction. G5 ranks scorers by the monotonic response of
their desync score to |offset| (SRCC), detection AUC at ±80 ms (a viewer-noticeable lag), and,
for offset estimators, signed SRCC and absolute error. Groups: source clip.

Sources: the user's talking-face clips (live action, frontal, one visible speaker; e.g. HDTF /
VoxCeleb2 test / LRS3 files they are entitled to use, or their own footage), as a folder of video
files passed with ``src=`` (default data/eval/_sources/lipsync/clips). Animation is out of the
scorers' training domain; keep it out of this pack or tag it.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from ..packs import pack_dir
from . import builder
from ._gcommon import (
    SR,
    cut_clip,
    media_duration,
    merge_pack,
    mux,
    read_audio,
    shift_audio,
    write_audio,
)

DEFAULT_OFFSETS = (-200, -160, -120, -80, -40, 0, 0, 40, 80, 120, 160, 200)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

LICENSE = """# Lip-sync offsets pack ({lang})

Source: user-supplied talking-face clips from {src} (their own terms apply; record them here).
{clips} clips × {k} offsets ({offsets} ms; positive = audio lags picture) = {n} items. Offsets
injected with numpy on the 16 kHz audio and re-muxed with ffmpeg over the original picture.
"""


def offset_rows(clip_id: str, video: Path, wav: Path, out_dir: Path,
                offsets: tuple[int, ...] | list[int]) -> list[dict]:
    """Shifted copies of one clip; returns manifest rows (paths relative to ``out_dir``)."""
    x = read_audio(wav)
    rows, seen = [], {}
    for off in offsets:
        rep = seen.get(off, 0)
        seen[off] = rep + 1
        tag = f"{clip_id}_off{off:+d}" + (f"_r{rep}" if rep else "")
        shifted = write_audio(out_dir / "audio" / f"{tag}.wav", shift_audio(x, off, SR))
        muxed = mux(video, shifted, out_dir / "video" / f"{tag}.mp4")
        rows.append({"id": tag, "group": clip_id, "split": "test",
                     "inputs": {"video": str(muxed.relative_to(out_dir)),
                                "audio": str(shifted.relative_to(out_dir))},
                     "refs": {"offset_ms": float(off)},
                     "meta": {"duration_s": round(len(x) / SR, 3), "clip": clip_id}})
    return rows


@builder("lipsync_offsets", capabilities=["G5"], langs=["en", "hi", "ja"],
         license="user-supplied clips")
def lipsync_offsets(lang: str, n: int | None = 20, src: str | None = None, clip_s: float = 6.0,
                    start_s: float = 0.5, offsets_ms: str | None = None) -> Path:
    """G5 pack: ``n`` talking-face clips × injected audio offsets (−200…+200 ms)."""
    folder = Path(src).expanduser() if src else paths.eval_dir() / "_sources" / "lipsync" / "clips"
    vids = sorted(p for p in folder.glob("**/*") if p.suffix.lower() in VIDEO_EXT) \
        if folder.exists() else []
    if not vids:
        raise FileNotFoundError(f"no talking-face clips under {folder}: put frontal live-action "
                                "clips there (≥ 7 s each) or pass src=/path/to/clips")
    offsets = tuple(int(x) for x in offsets_ms.split(",")) if offsets_ms else DEFAULT_OFFSETS
    out = pack_dir("G5", lang)
    rows, used = [], 0
    for v in vids:
        if n and used >= n:
            break
        if media_duration(v) < start_s + clip_s:
            continue
        cid = f"clip{used:03d}"
        video, wav = out / "src" / f"{cid}.mp4", out / "src" / f"{cid}.wav"
        cut_clip(v, video, wav, start_s, clip_s)
        rows += offset_rows(cid, video, wav, out, offsets)
        used += 1
    return merge_pack("G5", lang, rows, "offsets", LICENSE.format(
        lang=lang, src=folder, clips=used, k=len(offsets),
        offsets=", ".join(f"{o:+d}" for o in offsets), n=len(rows)))
