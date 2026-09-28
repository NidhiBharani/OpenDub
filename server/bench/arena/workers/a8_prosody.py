"""A8 worker: measured prosody — OpenDub's builtin tier-one delivery analysis
(``app.pipeline.emotion.acoustic_prosody``: loudness, pause profile, energy variation, F0
median/spread), plus a transparent rule mapping it to an arousal proxy and a coarse label so
the DSP baseline can be scored on the same axes as the models. Server env, CPU.

params: high_arousal_db (energy variation above which the label is "angry"/"happy" by pitch
spread), low_energy_db. The rules are deliberately simple; the point is a floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import serve

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # server/ → import app.*


def load(params: dict, lang: str):
    from app.pipeline.emotion import acoustic_prosody

    return {"fn": acoustic_prosody, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import soundfile as sf

    p = state["params"]
    wav = out.with_suffix(".16k.wav")
    y, _ = librosa.load(item["inputs"]["audio"], sr=16000, mono=True)
    sf.write(str(wav), y, 16000, subtype="PCM_16")
    pro = state["fn"](wav)
    wav.unlink(missing_ok=True)
    ev = float(pro.get("energy_variation_db") or 0.0)
    spread = float(pro.get("pitch_spread_semitones") or 0.0)
    arousal = max(0.0, min(1.0, (ev - 3.0) / 9.0 * 0.6 + spread / 12.0 * 0.4))
    if ev >= float(p.get("high_arousal_db", 8.0)):
        label = "happy" if spread >= 6 else "angry"
    elif (pro.get("rms_dbfs") or 0) < float(p.get("low_energy_db", -35.0)):
        label = "sad"
    else:
        label = "neutral"
    dur = float(pro["duration_seconds"])
    return {"segments": [{"start": 0.0, "end": dur, "label": label, "score": 0.5}],
            "emotion": label, "dims": {"arousal": round(arousal, 3)}, "prosody": pro}


if __name__ == "__main__":
    serve(load, run)
