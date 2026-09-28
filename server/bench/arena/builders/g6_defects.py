"""G6 pack: multimodal reviewers on an injected-defect set (wrong speaker, off-sync, clipping,
untranslated line), built with numpy + ffmpeg from FLEURS (parallel across languages) and,
for off-sync, the user's talking-face clips.

Each item is one dubbed line: inputs {audio: the take, ref_audio: a clip of the intended voice,
text: the target-language line, src_text: the source line, video?: the take in picture}.
refs.defects lists the injected defects ([] = clean); refs.weight balances the classes so the
weighted mean of per-item correctness is the macro recall over {clean, each defect}.

- clean: a FLEURS reading in the target language; ref_audio = its own first 3 s (same voice).
- wrong_speaker: the same take, ref_audio = the first 3 s of a *different* reader (half the
  cases the other gender, half the same gender: the hard ones).
- clipping: the take boosted +18 dB and hard-clipped at half scale.
- untranslated: the source-language FLEURS reading of the *same* FLoRes sentence (so the content
  is right but the language is not), with the target-language text as the intended line.
- off_sync (needs ``video_src``): a talking-face clip with its audio shifted ±240–400 ms; clean
  video items (0 ms) are added in equal number so the picture itself is not a tell.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..packs import pack_dir
from . import builder
from ._gcommon import (
    SR,
    class_weights,
    cut_clip,
    fleurs_extract,
    fleurs_rows,
    hard_clip,
    media_duration,
    merge_pack,
    mux,
    read_audio,
    shift_audio,
    write_audio,
)
from .g5_offsets import VIDEO_EXT

SOURCE_LANG = {"en": "ja", "hi": "en", "ja": "en"}
LICENSE = """# G6 injected-defect pack ({lang})

Audio: FLEURS test ({lang} takes, {src} source lines), https://huggingface.co/datasets/google/fleurs,
CC BY 4.0. Video: {video}. Defects injected with numpy/ffmpeg: {counts}.
Labels by construction (refs.defects). Groups: FLoRes sentence id (video items: source clip).
"""


def _item(uid: str, group: str, take: str, ref: str, text: str, src_text: str,
          defects: list[str], **extra) -> dict:
    inputs = {"audio": take, "ref_audio": ref, "text": text, "src_text": src_text}
    if extra.get("video"):
        inputs["video"] = extra.pop("video")
    return {"id": uid, "group": group, "split": "test", "inputs": inputs,
            "refs": {"defects": defects, "has_video": "video" in inputs, "has_ref": True},
            "meta": extra}


def audio_items(tgt: list[dict], src_by_sid: dict[str, dict], tgt_dir: Path, src_dir: Path,
                out: Path, n: int, seed: int = 0) -> list[dict]:
    """Clean / wrong_speaker / clipping / untranslated items from FLEURS rows."""
    rng = random.Random(seed)
    usable = [r for r in tgt if r["sid"] in src_by_sid]
    rng.shuffle(usable)
    kinds = ["clean", "wrong_speaker", "clipping", "untranslated"]
    items = []
    for i, r in enumerate(usable[:n]):
        kind = kinds[i % len(kinds)]
        x = read_audio(tgt_dir / r["file"])
        stem = Path(r["file"]).stem
        head = write_audio(out / "ref" / f"{stem}_head.wav", x[: 3 * SR])
        take = tgt_dir / r["file"]
        ref = head
        text, src_text = r["text"], src_by_sid[r["sid"]]["text"]
        if kind == "wrong_speaker":
            same_gender = i % 8 < 4
            others = [o for o in usable if o["sid"] != r["sid"] and
                      ((o["gender"] == r["gender"]) == same_gender)] or \
                     [o for o in usable if o["sid"] != r["sid"]]
            o = rng.choice(others)
            ox = read_audio(tgt_dir / o["file"])
            ref = write_audio(out / "ref" / f"{Path(o['file']).stem}_head.wav", ox[: 3 * SR])
        elif kind == "clipping":
            take = write_audio(out / "audio" / f"{stem}_clipped.wav", hard_clip(x))
        elif kind == "untranslated":
            s = src_by_sid[r["sid"]]
            take = src_dir / s["file"]
            sx = read_audio(take)
            ref = write_audio(out / "ref" / f"{Path(s['file']).stem}_head.wav", sx[: 3 * SR])
        items.append(_item(f"g6-{kind}-{stem}", r["sid"], str(take), str(ref), text, src_text,
                           [] if kind == "clean" else [kind], kind=kind,
                           duration_s=round(len(x) / SR, 3)))
    return items


def video_items(clips: list[Path], out: Path, n: int, seed: int = 0, clip_s: float = 6.0
                ) -> list[dict]:
    """Off-sync and clean video items from talking-face clips (half each)."""
    rng = random.Random(seed)
    items, used = [], 0
    for v in clips:
        if used >= n:
            break
        if media_duration(v) < clip_s + 0.5:
            continue
        cid = f"vid{used:03d}"
        video, wav = out / "src" / f"{cid}.mp4", out / "src" / f"{cid}.wav"
        cut_clip(v, video, wav, 0.5, clip_s)
        x = read_audio(wav)
        off = 0 if used % 2 == 0 else rng.choice([-1, 1]) * rng.randrange(240, 401, 40)
        shifted = write_audio(out / "audio" / f"{cid}_off{off:+d}.wav", shift_audio(x, off))
        muxed = mux(video, shifted, out / "video" / f"{cid}_off{off:+d}.mp4")
        head = write_audio(out / "ref" / f"{cid}_head.wav", x[: 3 * SR])
        items.append(_item(f"g6-video-{cid}", cid, str(shifted), str(head), "", "",
                           ["off_sync"] if off else [], video=str(muxed),
                           kind="off_sync" if off else "clean_video", offset_ms=off))
        used += 1
    return items


def add_weights(items: list[dict]) -> list[dict]:
    labels = [(it["refs"]["defects"] or ["clean"])[0] for it in items]
    w = class_weights(labels)
    for it, lab in zip(items, labels):
        it["refs"]["weight"] = round(w[lab], 6)
    return items


@builder("g6_defects", capabilities=["G6"], langs=["en", "hi", "ja"],
         license="CC BY 4.0 (FLEURS) + user clips")
def g6_defects(lang: str, n: int | None = 200, seed: int = 0, src_lang: str | None = None,
               video_src: str | None = None, n_video: int = 40) -> Path:
    """G6 pack: FLEURS lines with injected wrong-speaker / clipping / untranslated defects,
    plus off-sync talking-face clips when ``video_src`` is given."""
    src_lang = src_lang or SOURCE_LANG[lang]
    tgt, src = fleurs_rows(lang), fleurs_rows(src_lang)
    src_by_sid: dict[str, dict] = {}
    for s in src:
        src_by_sid.setdefault(s["sid"], s)
    rng = random.Random(seed)
    tgt = [r for r in tgt if r["sid"] in src_by_sid]
    rng.shuffle(tgt)
    tgt = tgt[: (n or 200) * 2]  # pool for wrong-speaker donors
    tgt_dir = fleurs_extract(lang, {r["file"] for r in tgt})
    src_dir = fleurs_extract(src_lang, {src_by_sid[r["sid"]]["file"] for r in tgt})
    out = pack_dir("G6", lang)
    items = audio_items(tgt, src_by_sid, tgt_dir, src_dir, out, n or 200, seed)
    video_note = "none (off_sync not covered: pass video_src=)"
    if video_src:
        folder = Path(video_src).expanduser()
        clips = sorted(p for p in folder.glob("**/*") if p.suffix.lower() in VIDEO_EXT)
        items += video_items(clips, out, n_video, seed)
        video_note = f"user-supplied talking-face clips from {folder}"
    add_weights(items)
    counts: dict[str, int] = {}
    for it in items:
        k = (it["refs"]["defects"] or ["clean"])[0]
        counts[k] = counts.get(k, 0) + 1
    return merge_pack("G6", lang, items, "defects", LICENSE.format(
        lang=lang, src=src_lang, video=video_note,
        counts=", ".join(f"{k} {v}" for k, v in sorted(counts.items()))))
