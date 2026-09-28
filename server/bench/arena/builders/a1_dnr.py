"""A1 packs: Divide and Remaster v3 (native) and a DnR-recipe remix of FLEURS speech.

``dnr_v3`` reads an extracted DnR v3 test set (Zenodo, CC BY-SA 4.0; 30+ GB per subset, so it is
downloaded by the user, not here) from ``data/eval/_sources/dnr_v3/<lang>/`` (or ``source=``)
and discovers tracks by file name: a mixture (``mix*``) plus ``speech*``/``dialog*``,
``music*``, ``sfx*``/``effects*`` stems in the same folder. A transcripts CSV
(``transcripts=``, columns ``track,text``) enables the ghost-dialogue primary metric; without it
only SI-SDR is scored.

``dnr_remix`` builds the same kind of item for any FLEURS language (the only way to get hi, and
transcripts for the ghost-dialogue metric in every language): 2–4 FLEURS utterances placed with
gaps on a ~20–30 s timeline, over music and effects taken from DnR stems (``bed_source=``,
default ``data/eval/_sources/dnr_v3``; any folder with ``music*``/``sfx*`` files works) or, with
``bed_source="synthetic"``, over a generated tonal bed + pink-noise effects. Loudness follows
the DnR recipe (dialogue −17, music −24, effects −21 LUFS, each jittered ±2 LU; DnR v3 re-fit
these to cinematic distributions — override with ``levels=``).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import (
    FLEURS_NOTE,
    audio_files,
    fleurs_clips,
    need,
    pink_noise,
    read_mono,
    set_loudness,
    source_dir,
    synth_music,
    write,
    write_merged,
)

SR = 44100
DNR_LICENSE = """# A1 {kind} ({lang})

{body}
Stems: dialogue, background (= music + effects), music, effects; mixture = their sum.
Groups: one per item (independent tracks).
"""


def _find(folder: Path, *prefixes: str) -> Path | None:
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".wav", ".flac") and f.stem.lower().startswith(prefixes):
            return f
    return None


def dnr_tracks(root: Path) -> list[dict[str, Path]]:
    tracks = []
    for mix in sorted(p for p in root.rglob("mix*") if p.suffix.lower() in (".wav", ".flac")):
        d = mix.parent
        speech, music, sfx = _find(d, "speech", "dialog"), _find(d, "music"), \
            _find(d, "sfx", "effect")
        if speech and music and sfx:
            tracks.append({"id": d.name, "mix": mix, "dialogue": speech, "music": music,
                           "effects": sfx})
    return tracks


@builder("a1-dnr-v3", capabilities=["A1"], langs=["en", "ja", "de", "fr", "es", "zh"],
         license="CC BY-SA 4.0")
def dnr_v3(lang: str, n: int | None = 100, seed: int = 0, source: str | None = None,
           transcripts: str | None = None) -> Path:
    """A1 pack from an extracted DnR v3 test subset (user download)."""
    import numpy as np

    root = need(Path(source) if source else source_dir("dnr_v3") / lang,
                f"DnR v3 ({lang})", "Download the matching test subset from Zenodo "
                "(github.com/kwatcharasupat/divide-and-remaster-v3), extract it there, or pass "
                "source=<dir>.")
    tracks = dnr_tracks(root)
    if not tracks:
        raise FileNotFoundError(f"no mix/speech/music/sfx track folders under {root}")
    texts: dict[str, str] = {}
    if transcripts:
        with open(transcripts, newline="", encoding="utf-8") as f:
            texts = {r["track"]: r["text"] for r in csv.DictReader(f)}
    random.Random(seed).shuffle(tracks)
    tracks = tracks[:n] if n else tracks
    out = paths.eval_dir() / "A1" / lang
    rows = []
    for t in tracks:
        d = out / "audio" / t["id"]
        stems = {k: read_mono(t[k], SR) for k in ("dialogue", "music", "effects")}
        m = min(len(v) for v in stems.values())
        stems = {k: v[:m] for k, v in stems.items()}
        bg = stems["music"] + stems["effects"]
        mix = stems["dialogue"] + bg
        rel = {k: write(d / f"{k}.wav", v, SR) for k, v in
               {**stems, "background": bg, "mix": mix}.items()}
        refs = {k: str(Path(v).relative_to(out)) for k, v in rel.items() if k != "mix"}
        if t["id"] in texts:
            refs["text"] = texts[t["id"]]
        rows.append({"id": f"dnr-{t['id']}", "group": t["id"], "split": "test",
                     "inputs": {"audio": str(Path(rel["mix"]).relative_to(out))}, "refs": refs,
                     "meta": {"duration_s": round(m / SR, 3), "source": "dnr_v3",
                              "peak": float(np.abs(mix).max())}})
    body = ("Source: Divide and Remaster v3 (Watcharasupat et al.), Zenodo, CC BY-SA 4.0. "
            "Mono downmix at 44.1 kHz; the mixture is re-summed from the stems.")
    return write_merged("A1", lang, rows,
                        DNR_LICENSE.format(kind="DnR v3", lang=lang, body=body), "dnr_v3")


@builder("a1-dnr-remix", capabilities=["A1"], langs=["en", "hi", "ja"],
         license="CC BY 4.0 speech + CC BY-SA 4.0 beds")
def dnr_remix(lang: str, n: int | None = 100, seed: int = 0, bed_source: str | None = None,
              levels: dict | None = None, seconds: float = 25.0) -> Path:
    """A1 pack: FLEURS dialogue over DnR (or synthetic) music + effects, DnR loudness recipe."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    lv = {"dialogue": -17.0, "music": -24.0, "effects": -21.0, **(levels or {})}
    n = n or 100
    clips = fleurs_clips(lang, n * 4, source_dir("fleurs_clips") / lang, seed=seed)
    synthetic = bed_source == "synthetic"
    musics = sfxs = []
    if not synthetic:
        root = need(Path(bed_source) if bed_source else source_dir("dnr_v3"),
                    "DnR music/effects stems", "Extract any DnR v3 subset there, pass "
                    "bed_source=<dir with music*/sfx* files>, or bed_source='synthetic'.")
        files = audio_files(root)
        musics = [f for f in files if f.stem.lower().startswith("music")]
        sfxs = [f for f in files if f.stem.lower().startswith(("sfx", "effect"))]
        if not musics or not sfxs:
            raise FileNotFoundError(f"no music*/sfx* stems under {root}")
    out = paths.eval_dir() / "A1" / lang
    rows = []
    total = int(seconds * SR)
    for i in range(n):
        picks = clips[i * 4: i * 4 + pyr.randint(2, 4)]
        if not picks:
            break
        dia = np.zeros(total)
        t = int(pyr.uniform(0.5, 2.0) * SR)
        used = []
        for c in picks:
            x = read_mono(c["path"], SR)
            if t + len(x) > total:
                break
            dia[t:t + len(x)] += x
            used.append(c)
            t += len(x) + int(pyr.uniform(0.3, 2.0) * SR)
        if not used:
            continue
        dia = set_loudness(dia, SR, lv["dialogue"] + pyr.uniform(-2, 2))

        def excerpt(pool, gen):
            if synthetic:
                return gen()
            src = read_mono(pyr.choice(pool), SR)
            if len(src) < total:
                src = np.tile(src, total // max(1, len(src)) + 1)
            off = pyr.randint(0, len(src) - total)
            return src[off:off + total]

        mus = set_loudness(excerpt(musics, lambda: synth_music(total, SR, rng)), SR,
                           lv["music"] + pyr.uniform(-2, 2))
        sfx = set_loudness(excerpt(sfxs, lambda: 0.3 * pink_noise(total, rng)), SR,
                           lv["effects"] + pyr.uniform(-2, 2))
        mix = dia + mus + sfx
        peak = np.abs(mix).max()
        if peak > 0.99:  # keep headroom; scale every stem so the sum stays exact
            g = 0.99 / peak
            dia, mus, sfx, mix = dia * g, mus * g, sfx * g, mix * g
        d = out / "audio" / f"remix{i:04d}"
        stems = {"dialogue": dia, "music": mus, "effects": sfx, "background": mus + sfx}
        refs = {k: str(Path(write(d / f"{k}.wav", v, SR)).relative_to(out))
                for k, v in stems.items()}
        refs["text"] = " ".join(c["text"] for c in used)
        rows.append({"id": f"remix-{lang}-{i:04d}", "group": f"remix{i:04d}", "split": "test",
                     "inputs": {"audio": str(Path(write(d / "mix.wav", mix, SR))
                                            .relative_to(out))},
                     "refs": refs,
                     "meta": {"duration_s": round(total / SR, 3), "source": "dnr_remix",
                              "fleurs_ids": [c["id"] for c in used],
                              "bed": "synthetic" if synthetic else "dnr"}})
    body = (f"Dialogue: {FLEURS_NOTE}\nBeds: "
            + ("synthetic tonal music + pink-noise effects (generated)." if synthetic else
               "music and effects stems from Divide and Remaster v3 (CC BY-SA 4.0).")
            + "\nRecipe: DnR loudness targets (dialogue −17, music −24, effects −21 LUFS ±2).")
    return write_merged("A1", lang, rows,
                        DNR_LICENSE.format(kind="DnR-recipe remix", lang=lang, body=body),
                        "dnr_remix")
