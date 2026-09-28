"""E1 packs: takes to fit into slots, with the source pause structure as ground truth.

- ``e1-fleurs-synth``: FLEURS read speech is the *source line* (its slot = its duration, its pauses
  = the reference). The *take* is the same speech with every speech run time-scaled by a known
  factor (ffmpeg atempo) and its pauses flattened, stretched or dropped — the TTS failure E1
  exists to fix — so the exact inverse time map is known.
- ``e1-arena-takes``: real takes from the arena outputs of a voice capability (D1 by default) with
  the slot and source pauses from that pack's item metadata (``inputs.target_s`` or
  ``meta.slot_s``; ``meta.src_pauses``/``meta.src_speech``, or detected on ``meta.src_audio``).
"""
from __future__ import annotations

import random
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

from .. import paths
from ..specs.e1 import internal_pauses, speech_intervals
from . import builder
from .e_common import arena_outputs, atempo_chain, ffmpeg, fleurs_utterances, merge_pack, read_mono

SYNTH_LICENSE = """# E1 synthetic takes from FLEURS ({lang})

Source: google/fleurs test split (CC BY 4.0), https://huggingface.co/datasets/google/fleurs.
Derived: speech runs time-scaled by a known factor with ffmpeg atempo; pauses rescaled
(flattened/stretched/dropped); silence filled with −70 dBFS noise. Eval use; attribution to
Google FLEURS (Conneau et al., 2022). Groups: FLoRes sentence id.
"""

ARENA_LICENSE = """# E1 takes from arena outputs ({capability}, {lang})

Takes are outputs of arena candidates for {capability} (see data/arena/outputs/{capability});
their licences follow the generating model and the {capability} pack's own LICENSE.md.
Eval-only, never redistributed.
"""


def make_take(x, sr: int, speech: list[list[float]], factor: float, pause_scales: list[float],
              tmp: Path, lead: float = 0.08, tail: float = 0.10):
    """Concatenate time-scaled speech runs with rescaled gaps. Returns (take, take_speech)."""
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(len(x))

    def silence(sec: float):
        return rng.normal(0.0, 10 ** (-70 / 20), round(sec * sr))

    parts, take_speech, t = [silence(lead)], [], lead
    for i, (s, e) in enumerate(speech):
        seg = x[int(s * sr):int(e * sr)]
        src, dst = tmp / f"run{i}.wav", tmp / f"run{i}.out.wav"
        sf.write(src, seg, sr, subtype="FLOAT")
        ffmpeg("-i", str(src), "-filter:a", atempo_chain(1.0 / factor), "-ar", str(sr),
               "-ac", "1", "-c:a", "pcm_f32le", str(dst))
        y, _ = sf.read(dst, dtype="float64")
        n = min(len(y), int(0.005 * sr))
        if n:
            ramp = np.linspace(0.0, 1.0, n)
            y[:n] *= ramp
            y[-n:] *= ramp[::-1]
        parts.append(y)
        take_speech.append([round(t, 4), round(t + len(y) / sr, 4)])
        t += len(y) / sr
        if i < len(speech) - 1:
            gap = (speech[i + 1][0] - e) * pause_scales[i]
            parts.append(silence(gap))
            t += gap
    parts.append(silence(tail))
    return np.concatenate(parts), take_speech


@builder("e1-fleurs-synth", capabilities=["E1"], langs=["en", "hi", "ja"], license="CC BY 4.0")
def e1_fleurs_synth(lang: str, n: int | None = 200, seed: int = 0) -> Path:
    """E1 synthetic takes: FLEURS speech runs time-scaled by known factors, pauses rescaled."""
    import soundfile as sf

    rng = random.Random(seed)
    root = paths.eval_dir() / "E1" / lang
    (root / "takes").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    # Oversample: only lines with at least one internal pause are useful.
    for utt in fleurs_utterances(lang, (n or 200) * 3, seed=seed):
        if n and len(rows) >= n:
            break
        x, sr = read_mono(utt["path"])
        speech = speech_intervals(x, sr)
        pauses = internal_pauses(speech)
        if not pauses or len(speech) > 12:
            continue
        factor = round(rng.uniform(0.85, 1.25), 3)          # take speech is factor× longer
        scales = []
        for s0, s1 in pairwise(speech):
            r = rng.random()
            gap = s1[0] - s0[1]
            if r < 0.15:
                scales.append(0.05 / max(gap, 1e-3))          # pause dropped (≈50 ms)
            elif r < 0.85:
                scales.append(max(rng.uniform(0.3, 0.7), 0.16 / max(gap, 1e-3)))  # flattened
            else:
                scales.append(rng.uniform(1.3, 2.0))          # overlong
        with tempfile.TemporaryDirectory() as td:
            take, take_speech = make_take(x, sr, speech, factor, scales, Path(td))
        name = f"takes/{utt['id']}.wav"
        sf.write(root / name, take, sr, subtype="PCM_16")
        target = round(len(x) / sr, 4)
        rows.append({
            "id": f"e1s-{utt['id']}", "group": utt["group"], "split": "test",
            "inputs": {"audio": name, "target_s": target, "src_pauses": pauses,
                       "src_speech": speech},
            "refs": {"text": utt["text"], "pauses": pauses, "speech": speech, "target_s": target},
            "meta": {"duration_s": round(len(take) / sr, 4), "factor": factor,
                     "pause_scales": [round(s, 3) for s in scales], "take_speech": take_speech,
                     "slot_ratio": round(len(take) / sr / target, 4)}})
    return merge_pack("E1", lang, rows, SYNTH_LICENSE.format(lang=lang), "fleurs-synth")


def _slot(item) -> float | None:
    for v in (item.inputs.get("target_s"), item.meta.get("slot_s"), item.meta.get("target_s")):
        if v:
            return float(v)
    return None


def _source_structure(item) -> tuple[list[list[float]] | None, list[list[float]] | None]:
    if item.meta.get("src_speech") is not None:
        speech = item.meta["src_speech"]
        return item.meta.get("src_pauses", internal_pauses(speech)), speech
    src = item.meta.get("src_audio") or (item.inputs.get("ref_audio")
                                         if item.meta.get("ref_is_source_line") else None)
    if src and Path(str(src)).exists():
        x, sr = read_mono(src)
        speech = speech_intervals(x, sr)
        return internal_pauses(speech), speech
    if item.meta.get("src_pauses") is not None:
        return item.meta["src_pauses"], None
    return None, None


@builder("e1-arena-takes", capabilities=["E1"], langs=["en", "hi", "ja"],
         license="per generating model")
def e1_arena_takes(lang: str, n: int | None = None, capability: str = "D1",
                   candidates: list[str] | None = None) -> Path:
    """E1 pack from arena voice outputs (default D1), slots/pauses from that pack's metadata."""
    import shutil

    root = paths.eval_dir() / "E1" / lang
    rows: list[dict[str, Any]] = []
    for o in arena_outputs(capability, lang, candidates):
        it = o["item"]
        target = _slot(it)
        if not target:
            continue
        pauses, speech = _source_structure(it)
        dst = root / "arena" / capability / o["candidate"] / f"{it.id}{Path(o['audio']).suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(o["audio"], dst)
        x, sr = read_mono(dst)
        inputs: dict[str, Any] = {"audio": str(dst.relative_to(root)), "target_s": target}
        refs: dict[str, Any] = {"text": it.inputs.get("text") or it.refs.get("text", ""),
                                "target_s": target}
        if pauses is not None:
            inputs["src_pauses"] = refs["pauses"] = pauses
        if speech is not None:
            inputs["src_speech"] = refs["speech"] = speech
        rows.append({"id": f"e1a-{capability}-{o['candidate']}-{it.id}", "group": it.group,
                     "split": it.split, "inputs": inputs, "refs": refs,
                     "meta": {"duration_s": round(len(x) / sr, 4), "from": capability,
                              "candidate": o["candidate"],
                              "slot_ratio": round(len(x) / sr / target, 4)}})
        if n and len(rows) >= n:
            break
    return merge_pack("E1", lang, rows, ARENA_LICENSE.format(capability=capability, lang=lang),
                      f"arena-{capability}")
