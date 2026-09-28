"""F1 cross-lingual pack: live-action clips driven by a dub track in another language.

No ground truth exists, so these items are scored by the sync panel, ArcFace identity and the
face-detection rate only. Media come from the user (``media_dir`` opt, ``OPENDUB_F1_MEDIA`` env,
default ``data/eval/_sources/f1_selfreenact``), laid out as::

    crossling/<lang>/<name>.mp4               # live-action picture (its own audio is ignored)
    crossling/<lang>/<name>.wav | .flac | …   # dub audio in <lang>, time-aligned to the picture
    crossling/<lang>/<name>.txt               # optional: the dub line (text-driven generators)

``<lang>`` is the dub (target) language: the pack lands in ``data/eval/F1/<lang>``. Eval-only,
private; nothing is downloaded. The SPY×FAMILY official dubs live in ``scene_dub`` instead
(2D animation is not live action).
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import paths
from . import builder
from ._fvideo import AUDIO_EXT, VIDEO_EXT, extract_audio, merge_pack, normalize_clip, probe, sample
from .f1_selfreenact import media_root, sidecar_text

SOURCE = "f1_crossling"

LICENSE = """# F1 cross-lingual ({lang})

Built by `bench/arena/builders/f1_crossling.py` from user-supplied live-action clips and dub
tracks under `{media}/crossling/{lang}`. Eval-only, private, never redistributed.
No ground truth: ranked by the sync panel with the ArcFace identity gate.
Groups: clip name prefix before the first `_` (one source video or speaker).
"""


def pairs(folder: Path) -> list[tuple[Path, Path]]:
    out = []
    for v in sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXT):
        audio = next((v.with_suffix(ext) for ext in sorted(AUDIO_EXT)
                      if v.with_suffix(ext).exists()), None)
        if audio is not None:
            out.append((v, audio))
    return out


@builder(SOURCE, capabilities=["F1"], langs=["en", "hi", "ja"], license="user media (eval-only)")
def f1_crossling(lang: str, n: int | None = None, seed: int = 0, media_dir: str | None = None,
                 max_s: float = 20.0) -> Path:
    """F1 cross-lingual pack from user-supplied live-action clips + dub tracks."""
    root = media_root(media_dir)
    folder = root / "crossling" / lang
    found = pairs(folder) if folder.exists() else []
    if not found:
        raise FileNotFoundError(f"no <name>.mp4 + <name>.<audio> pairs under {folder}")
    out = paths.eval_dir() / "F1" / lang
    rows = []
    for v_src, a_src in sample(found, n, seed):
        vi, ai = probe(v_src), probe(a_src)
        dur = min(vi["duration"], ai["duration"], max_s)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", v_src.stem)
        v, a = out / SOURCE / f"{stem}.mp4", out / SOURCE / f"{stem}.wav"
        if not v.exists():
            normalize_clip(v_src, v, None, duration=dur)
        if not a.exists():
            extract_audio(a_src, a, duration=dur)
        meta = {"kind": "cross", "source": "user", "duration_s": round(dur, 3),
                "audio_duration_s": round(dur, 3), "fps": 25, "orig": str(v_src)}
        text = sidecar_text(v_src)
        if text:
            meta["text"] = text
        inputs = {"video": str(v.relative_to(out)), "audio": str(a.relative_to(out))}
        if vi["has_audio"]:  # the original speech: voice reference for text-driven generators
            src_a = out / SOURCE / f"{stem}.src.wav"
            if not src_a.exists():
                extract_audio(v_src, src_a, duration=dur)
            inputs["src_audio"] = str(src_a.relative_to(out))
        rows.append({"id": f"cross-{stem}", "group": f"user:{stem.split('_')[0]}",
                     "split": "test", "inputs": inputs, "refs": {}, "meta": meta})
    return merge_pack("F1", lang, SOURCE, rows, LICENSE.format(lang=lang, media=root))
