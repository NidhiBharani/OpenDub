"""A2 pack: clean FLEURS speech with simulated degradations (numpy only).

Each item draws a seeded combination of (always at least one):

- additive noise at 0–20 dB SNR: babble (3–6 other FLEURS utterances), pink noise, or effects
  from ``noise_source`` (any folder of wav/flac, e.g. DnR v3 ``sfx*`` stems);
- reverberation: a synthetic exponentially-decaying stereo-free RIR, T60 0.3–1.2 s, with a
  direct-path spike and a pre-delay (Polack model);
- clipping at the 90–99th percentile of |x|;
- band-limiting to 3.4–8 kHz (telephone / low-bitrate style) with an FFT brick-wall.

The clean clip is ``refs.clean`` (the speaker-similarity reference), the degraded clip is the
input, ``meta.degradations`` records what was applied. 16 kHz (FLEURS native), so 48 kHz
enhancers are judged on content and identity, not bandwidth extension (that is E2).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import (
    FLEURS_NOTE,
    audio_files,
    fleurs_clips,
    pink_noise,
    read_mono,
    source_dir,
    write,
    write_merged,
)

SR = 16000
LICENSE = """# A2 simulated degradations ({lang})

Clean speech: {fleurs}
Degradations (noise / reverb / clipping / band-limit) generated with a fixed seed ({seed}); see
each item's meta.degradations. Noise source: {noise}.
Groups: FLoRes sentence id of the clean clip.
"""


def rir(sr: int, t60: float, rng):
    import numpy as np

    n = int(sr * t60 * 1.2)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / t60)
    pre = int(sr * rng.uniform(0.002, 0.02))
    h[:pre] = 0
    h[pre] = 1.0 + np.abs(h).max()
    return h / np.sqrt((h ** 2).sum())


def degrade(clean, rng, pyr: random.Random, babble_pool, noise_files):
    import numpy as np
    from scipy.signal import fftconvolve

    ops = [op for op in ("noise", "reverb", "clip", "band") if pyr.random() < 0.5]
    if not ops:
        ops = [pyr.choice(["noise", "reverb"])]
    x, info = clean.copy(), {}
    if "reverb" in ops:
        t60 = pyr.uniform(0.3, 1.2)
        x = fftconvolve(x, rir(SR, t60, rng))[: len(x)]
        info["reverb_t60_s"] = round(t60, 3)
    if "noise" in ops:
        kind = pyr.choice(["babble", "pink"] + (["effects"] if noise_files else []))
        n = len(x)
        if kind == "babble":
            noise = np.zeros(n)
            for p in pyr.sample(babble_pool, min(len(babble_pool), pyr.randint(3, 6))):
                s = read_mono(p, SR)
                s = np.tile(s, n // max(1, len(s)) + 1)[:n]
                noise += s / (np.sqrt((s ** 2).mean()) + 1e-9)
        elif kind == "pink":
            noise = pink_noise(n, rng)
        else:
            s = read_mono(pyr.choice(noise_files), SR)
            s = np.tile(s, n // max(1, len(s)) + 1)
            off = pyr.randint(0, len(s) - n)
            noise = s[off:off + n]
        snr = pyr.uniform(0, 20)
        ps, pn = (x ** 2).mean(), (noise ** 2).mean() + 1e-12
        x = x + noise * np.sqrt(ps / (pn * 10 ** (snr / 10)))
        info.update(noise=kind, snr_db=round(snr, 2))
    if "band" in ops:
        cut = pyr.uniform(3400, 7900)
        spec = np.fft.rfft(x)
        spec[int(cut / (SR / 2) * (len(spec) - 1)):] = 0
        x = np.fft.irfft(spec, len(x))
        info["lowpass_hz"] = round(cut)
    peak = np.abs(x).max()
    if peak > 0.95:
        x *= 0.95 / peak
    if "clip" in ops:
        q = pyr.uniform(90, 99)
        lim = np.percentile(np.abs(x), q)
        x = np.clip(x, -lim, lim) / (lim + 1e-9) * 0.9
        info["clip_percentile"] = round(q, 1)
    return x, info


@builder("a2-degrade", capabilities=["A2"], langs=["en", "hi", "ja"], license="CC BY 4.0")
def a2_degrade(lang: str, n: int | None = 200, seed: int = 0,
               noise_source: str | None = None) -> Path:
    """A2 pack: FLEURS clean speech + seeded noise / reverb / clipping / band-limit."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    n = n or 200
    clips = fleurs_clips(lang, n + 60, source_dir("fleurs_clips") / lang, seed=seed)
    targets, babble = clips[:n], [c["path"] for c in clips[n:]] or [c["path"] for c in clips]
    noise_files = audio_files(Path(noise_source)) if noise_source else []
    out = paths.eval_dir() / "A2" / lang
    rows = []
    for c in targets:
        clean = read_mono(c["path"], SR)
        x, info = degrade(clean, rng, pyr, [b for b in babble if b != c["path"]], noise_files)
        rel_clean = Path(write(out / "clean" / f"{c['id']}.wav", clean, SR)).relative_to(out)
        rel_in = Path(write(out / "degraded" / f"{c['id']}.wav", x, SR)).relative_to(out)
        rows.append({"id": f"deg-{c['id']}", "group": c["group"], "split": "test",
                     "inputs": {"audio": str(rel_in)},
                     "refs": {"clean": str(rel_clean), "text": c["text"]},
                     "meta": {"duration_s": round(len(clean) / SR, 3), "degradations": info,
                              "gender": c["gender"], "source": "fleurs+sim"}})
    return write_merged("A2", lang, rows, LICENSE.format(
        lang=lang, fleurs=FLEURS_NOTE, seed=seed,
        noise=json.dumps(str(noise_source)) if noise_source else "babble / pink (generated)"),
        "fleurs+sim")
