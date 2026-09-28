"""SPY×FAMILY picture + official dub audio → F3 (``test``) and F1 (``control``) packs.

Reads the ``SCENE`` pack (``data/eval/SCENE/ja/manifest.jsonl``, built by the scene builder
from the official Muse Asia uploads) and pairs each item's Japanese picture with the official
``<lang>`` dub audio as the driving audio. Pack language = the dub language (en, hi).

The SCENE manifest (``spyfamily.py``) stores pre-cut, dub-aligned clips: ``inputs.video``,
``inputs.audio`` (Japanese), ``refs.dub_<lang>_audio`` and, after the opt-in transcription,
``refs.dub_<lang>_asr`` (timed ASR segments of the official dub → ``refs.dub_speech``). Those
keys are read first; for other layouts the builder falls back to defensive discovery:

- picture: ``inputs.video`` (else any ``*video*`` path in inputs/refs/meta);
- window: ``meta.start_s/end_s`` (or ``start/end``, ``t0/t1``) only when the video is longer
  than the window end (a whole episode); pre-cut clips are used from t = 0;
- dub audio: any path whose key mentions ``dub`` and the language (``dub_audio_en``,
  ``{"dub_audio": {"en": …}}``, ``refs.dub.en.audio`` …), audio or video file; cut to the
  same window plus ``dub_offset_s`` (opt, or ``meta.dub_offset_s.<lang>``);
- timings: dub line intervals (key mentions ``dub``, the language and segments/lines/speech)
  → ``refs.dub_speech``; original line intervals (``segments``/``lines`` without ``dub``) →
  ``inputs.src_speech``. Explicit ``video_key`` / ``dub_key`` opts override discovery.

2D animation is not live action: in F1 the rows go to split ``control`` (a negative control for
B4 gating failures, run with ``--split control``) and never into the ranked ``test`` split.
Official uploads: eval-only, private, never redistributed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .. import paths
from . import builder
from ._fvideo import AUDIO_EXT, VIDEO_EXT, extract_audio, merge_pack, normalize_clip, probe

SOURCE = "scene_dub"
MEDIA_EXT = AUDIO_EXT | VIDEO_EXT

LICENSE = """# SPY×FAMILY picture + official {lang} dub ({capability})

Built by `bench/arena/builders/scene_dub.py` from the SCENE pack (official Muse Asia uploads of
SPY×FAMILY ep. 1: Japanese original and the official {lang} dub). Eval-only, private, never
committed or redistributed. The official dub audio drives the picture; its line timings (when
the SCENE pack has them) are the reference speech track.
Groups: scene / shot id from the SCENE manifest.
"""


def _walk(obj: Any, path: tuple[str, ...] = ()):
    """Yield (key path, value) for every leaf and list in a nested manifest row."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, (*path, str(k)))
    else:
        yield path, obj


def _is_intervals(v: Any) -> bool:
    if not isinstance(v, list) or not v:
        return False
    first = v[0]
    return (isinstance(first, dict) and "start" in first and "end" in first) or \
        (isinstance(first, list | tuple) and len(first) >= 2
         and all(isinstance(x, int | float) for x in first[:2]))


def _intervals(v: list, t0: float, t1: float | None) -> list[list[float]]:
    out = []
    for iv in v:
        s, e = (iv["start"], iv["end"]) if isinstance(iv, dict) else (iv[0], iv[1])
        s, e = float(s) - t0, float(e) - t0
        if t1 is not None:
            e = min(e, t1 - t0)
        if e > 0 and (t1 is None or s < t1 - t0):
            out.append([round(max(0.0, s), 3), round(e, 3)])
    return out


def as_json(v: Any, root: Path) -> dict | None:
    if not isinstance(v, str) or not v.endswith(".json"):
        return None
    p = Path(v) if v.startswith("/") else root / v
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _tokens(path: tuple[str, ...]) -> set[str]:
    return {t for p in path for t in re.split(r"[^a-z0-9]+", p.lower()) if t}


def resolve(row: dict, lang: str, root: Path, *, video_key: str | None = None,
            dub_key: str | None = None) -> dict[str, Any] | None:
    """Pick picture, window, dub audio and timings out of one SCENE row."""
    leaves = list(_walk({k: row.get(k) for k in ("inputs", "refs", "meta")}))

    def as_path(v: Any) -> Path | None:
        if not isinstance(v, str) or len(v) > 400:
            return None
        p = Path(v) if v.startswith("/") else root / v
        return p if p.suffix.lower() in MEDIA_EXT and p.exists() else None

    def by_key(dotted: str) -> Any:
        cur: Any = row
        for part in dotted.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        return cur

    video = as_path(by_key(video_key)) if video_key else None
    video = video or as_path((row.get("inputs") or {}).get("video"))
    if video is None:
        video = next((as_path(v) for pth, v in leaves if "video" in _tokens(pth)
                      and "dub" not in _tokens(pth) and as_path(v)), None)
    dub = as_path(by_key(dub_key)) if dub_key else None
    dub = dub or as_path((row.get("refs") or {}).get(f"dub_{lang}_audio"))
    if dub is None:
        dub = next((as_path(v) for pth, v in leaves if {"dub", lang} <= _tokens(pth)
                    and as_path(v)), None)
    if video is None or dub is None:
        return None
    meta = row.get("meta") or {}
    t0 = next((float(meta[k]) for k in ("start_s", "start", "t0") if k in meta), 0.0)
    t1 = next((float(meta[k]) for k in ("end_s", "end", "t1") if k in meta), None)
    if t1 is not None and probe(video)["duration"] < t1 - 0.5:
        # a pre-cut clip (meta holds its window in the episode): its own timeline starts at 0
        t0, t1 = 0.0, None
    dub_speech = next((v for pth, v in leaves if {"dub", lang} <= _tokens(pth)
                       and _tokens(pth) & {"segments", "lines", "speech", "turns"}
                       and _is_intervals(v)), None)
    asr = as_json((row.get("refs") or {}).get(f"dub_{lang}_asr"), root)
    if dub_speech is None and asr is not None:
        segs = asr.get("segments") or []
        dub_speech = [[s["start"], s["end"]] for s in segs if "start" in s and "end" in s] or None
    src_audio = as_path((row.get("inputs") or {}).get("audio"))
    src_speech = next((v for pth, v in leaves if "dub" not in _tokens(pth)
                       and _tokens(pth) & {"segments", "lines", "speech", "turns"}
                       and _is_intervals(v)), None)
    offset = meta.get("dub_offset_s")
    offset = float(offset.get(lang, 0.0)) if isinstance(offset, dict) else float(offset or 0.0)
    return {"video": video, "dub": dub, "t0": t0, "t1": t1, "offset": offset,
            "src_audio": src_audio,
            "dub_speech": _intervals(dub_speech, t0, t1) if dub_speech else None,
            "src_speech": _intervals(src_speech, t0, t1) if src_speech else None}


def build_rows(lang: str, capability: str, *, n: int | None, max_s: float,
               video_key: str | None, dub_key: str | None, dub_offset_s: float | None) -> list:
    scene = paths.eval_dir() / "SCENE" / "ja"
    manifest = scene / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"no SCENE pack at {manifest} (build the SCENE pack first)")
    out = paths.eval_dir() / capability / lang
    split = "test" if capability == "F3" else "control"
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        src = json.loads(line)
        r = resolve(src, lang, scene, video_key=video_key, dub_key=dub_key)
        if r is None:
            continue
        dur = (r["t1"] - r["t0"]) if r["t1"] is not None else probe(r["video"])["duration"]
        dur = min(dur, max_s)
        if dur <= 0.5:
            continue
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", str(src["id"]))
        v, a = out / SOURCE / f"{stem}.mp4", out / SOURCE / f"{stem}.{lang}.wav"
        if not v.exists():
            normalize_clip(r["video"], v, None, start=r["t0"], duration=dur)
        if not a.exists():
            off = r["offset"] if dub_offset_s is None else dub_offset_s
            extract_audio(r["dub"], a, start=max(0.0, r["t0"] + off), duration=dur)
        inputs: dict[str, Any] = {"video": str(v.relative_to(out)), "audio": str(a.relative_to(out))}
        if r["src_audio"] is not None:  # original (ja) speech: voice reference / flap proxy
            sa = out / SOURCE / f"{stem}.ja.wav"
            if not sa.exists():
                extract_audio(r["src_audio"], sa, start=r["t0"], duration=dur)
            inputs["src_audio"] = str(sa.relative_to(out))
        if r["src_speech"] is not None:
            inputs["src_speech"] = [iv for iv in r["src_speech"] if iv[0] < dur]
        refs: dict[str, Any] = {}
        if r["dub_speech"] is not None:
            refs["dub_speech"] = [iv for iv in r["dub_speech"] if iv[0] < dur]
        rows.append({"id": f"spyfam-{stem}", "group": str(src.get("group") or src["id"]),
                     "split": split, "inputs": inputs, "refs": refs,
                     "meta": {"kind": "cross", "content": "2d_animation", "source": "spy_family",
                              "src_lang": "ja", "duration_s": round(dur, 3),
                              "audio_duration_s": round(dur, 3), "fps": 25}})
        if n and len(rows) >= n:
            break
    return rows


@builder(SOURCE, capabilities=["F3", "F1"], langs=["en", "hi"],
         license="official uploads (eval-only, private)")
def scene_dub(lang: str, n: int | None = None, capability: str = "F3", max_s: float = 30.0,
              video_key: str | None = None, dub_key: str | None = None,
              dub_offset_s: float | None = None, both: bool = True) -> list[Path]:
    """SPY×FAMILY picture driven by the official dub: F3 test split + F1 control split."""
    caps = ["F3", "F1"] if both else [capability]
    roots = []
    for cap in caps:
        rows = build_rows(lang, cap, n=n, max_s=max_s, video_key=video_key, dub_key=dub_key,
                          dub_offset_s=dub_offset_s)
        if not rows:
            raise ValueError(f"SCENE rows had no picture + {lang} dub pair; pass video_key/"
                             f"dub_key (dotted paths into the SCENE manifest rows)")
        roots.append(merge_pack(cap, lang, SOURCE, rows,
                                LICENSE.format(lang=lang, capability=cap)))
    return roots
