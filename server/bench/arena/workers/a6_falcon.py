"""A6 worker: Picovoice Falcon (on-device proprietary engine, Apache-2.0 SDK; ``pvfalcon``).
``pvfalcon.create(access_key=…)``; ``process_file(path)`` → segments with ``speaker_tag,
start_sec, end_sec``; ``delete()``. 16 kHz 16-bit input. Platforms: Linux x86_64 (the aarch64
library targets Raspberry Pi and may reject GB10). Needs PICOVOICE_ACCESS_KEY (online check).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve


def load(params: dict, lang: str):
    import pvfalcon

    return {"falcon": pvfalcon.create(access_key=api_key("PICOVOICE_ACCESS_KEY"))}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import soundfile as sf

    wav = out.with_suffix(".16k.wav")
    y, _ = librosa.load(item["inputs"]["audio"], sr=16000, mono=True)
    sf.write(str(wav), y, 16000, subtype="PCM_16")
    segs = state["falcon"].process_file(str(wav))
    wav.unlink(missing_ok=True)
    return {"turns": [{"start": s.start_sec, "end": s.end_sec, "speaker": str(s.speaker_tag)}
                      for s in segs]}


if __name__ == "__main__":
    serve(load, run)
