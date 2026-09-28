"""A3 packs: speech / music / singing regions.

``a3-regions-synth``: ~60 s timelines assembled from FLEURS speech, music (DnR ``music*`` stems
from ``music_source``, or a generated tonal bed) and optional singing clips (``singing_source``:
any folder of a-cappella / song-vocal files the user has rights to), with silence, speech-only,
music-only, speech-over-music and singing(-over-music) sections. Labels are exact by
construction. Without a singing source the pack scores speech and music only
(``meta.classes``).

``a3-regions-ih``: the in-house song-heavy set — user audio in
``data/eval/_sources/a3_ih/<lang>/`` with a same-stem label file (Audacity ``.txt`` or Praat
``.TextGrid``, first tier) whose labels are speech / music / singing (overlaps allowed).
"""
from __future__ import annotations

import random
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import (
    FLEURS_NOTE,
    audio_files,
    duration,
    fleurs_clips,
    need,
    read_labels,
    read_mono,
    set_loudness,
    source_dir,
    synth_music,
    write,
    write_merged,
)

SR = 16000


@builder("a3-regions-synth", capabilities=["A3"], langs=["en", "hi", "ja"],
         license="CC BY 4.0 speech + source beds")
def a3_regions_synth(lang: str, n: int | None = 60, seed: int = 0,
                     music_source: str | None = None, singing_source: str | None = None,
                     seconds: float = 60.0) -> Path:
    """A3 pack: synthetic timelines with exact speech / music / singing labels."""
    import numpy as np

    rng, pyr = np.random.default_rng(seed), random.Random(seed)
    n = n or 60
    speech = fleurs_clips(lang, n * 8, source_dir("fleurs_clips") / lang, seed=seed)
    if music_source is None and source_dir("dnr_v3").exists():
        music_source = str(source_dir("dnr_v3"))
    musics = [f for f in audio_files(Path(music_source)) if f.stem.lower().startswith("music")] \
        if music_source else []
    singing = audio_files(Path(singing_source)) if singing_source else []
    classes = ["speech", "music", "singing"] if singing else ["speech", "music"]
    out = paths.eval_dir() / "A3" / lang
    rows, si = [], 0
    for i in range(n):
        total = int(seconds * SR)
        spk, mus, sng = np.zeros(total), np.zeros(total), np.zeros(total)
        segs: list[dict] = []
        t = 0
        while t < total - SR:
            kind = pyr.choice(["speech", "speech", "music", "speech+music", "silence"]
                              + (["singing", "singing+music"] if singing else []))
            if kind == "silence":
                t += int(pyr.uniform(0.5, 2.0) * SR)
                continue
            if "speech" in kind:
                x = read_mono(speech[si % len(speech)]["path"], SR)
                si += 1
                dst = spk
            elif "singing" in kind:
                x = read_mono(pyr.choice(singing), SR)
                x = x[: int(pyr.uniform(4, 12) * SR)]
                dst = sng
            else:
                x = np.zeros(int(pyr.uniform(3, 10) * SR))
                dst = None
            ln = min(len(x), total - t)
            if dst is not None:
                dst[t:t + ln] += set_loudness(x[:ln], SR, -20.0)
                label = "speech" if "speech" in kind else "singing"
                segs.append({"start": t / SR, "end": (t + ln) / SR, "label": label})
            if "music" in kind:
                if musics:
                    m = read_mono(pyr.choice(musics), SR)
                    m = np.tile(m, ln // max(1, len(m)) + 1)[:ln]
                else:
                    m = synth_music(ln, SR, rng)
                lvl = -30.0 if dst is not None else -22.0  # a bed under voice sits lower
                mus[t:t + ln] += set_loudness(m, SR, lvl + pyr.uniform(-3, 3))
                segs.append({"start": t / SR, "end": (t + ln) / SR, "label": "music"})
            t += ln + int(pyr.uniform(0.2, 1.5) * SR)
        mix = spk + mus + sng
        peak = np.abs(mix).max()
        if peak > 0.99:
            mix *= 0.99 / peak
        rel = Path(write(out / "audio" / f"regions{i:04d}.wav", mix, SR)).relative_to(out)
        rows.append({"id": f"regions-{lang}-{i:04d}", "group": f"regions{i:04d}",
                     "split": "test", "inputs": {"audio": str(rel)},
                     "refs": {"segments": [{**s, "start": round(s["start"], 3),
                                            "end": round(s["end"], 3)} for s in segs]},
                     "meta": {"duration_s": round(total / SR, 3), "classes": classes,
                              "source": "synthetic"}})
    lic = (f"# A3 synthetic regions ({lang})\n\nSpeech: {FLEURS_NOTE}\n"
           f"Music: {'DnR v3 music stems (CC BY-SA 4.0)' if musics else 'generated tonal bed'}"
           f".\nSinging: {singing_source or 'none (speech/music only)'}.\n"
           "Labels are exact by construction. Groups: one per timeline.\n")
    return write_merged("A3", lang, rows, lic, "synthetic")


@builder("a3-regions-ih", capabilities=["A3"], langs=["en", "hi", "ja", "ta", "ko"],
         license="in-house (user media)")
def a3_regions_ih(lang: str, n: int | None = None, source: str | None = None) -> Path:
    """A3 pack from user-annotated song-heavy clips (audio + Audacity/TextGrid labels)."""
    import shutil

    root = need(Path(source) if source else source_dir("a3_ih") / lang, f"A3 in-house ({lang})",
                "Put audio files there with same-stem .txt (Audacity labels) or .TextGrid.")
    out = paths.eval_dir() / "A3" / lang
    rows = []
    for audio in audio_files(root):
        lab = next((audio.with_suffix(s) for s in (".txt", ".TextGrid", ".textgrid")
                    if audio.with_suffix(s).exists()), None)
        if lab is None:
            continue
        dst = out / "audio" / audio.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(audio, dst)
        rows.append({"id": f"ih-{audio.stem}", "group": audio.stem.split("_")[0],
                     "split": "test", "inputs": {"audio": f"audio/{audio.name}"},
                     "refs": {"segments": read_labels(lab)},
                     "meta": {"duration_s": round(duration(audio), 3), "source": "in-house"}})
        if n and len(rows) >= n:
            break
    return write_merged("A3", lang, rows, f"# A3 in-house regions ({lang})\n\nUser media from "
                      f"{root}; private, evaluation only. Group = file-name prefix before '_'.\n",
                        "in-house")
