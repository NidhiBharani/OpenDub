"""F3 worker: decline to lip-sync animation + export a cue track (the research's top answer).

The picture is delivered unchanged; ``mouth_open`` is therefore the *original* flap track,
approximated by the original line timings (``inputs.src_speech``) or, without them, by an energy
VAD of the original audio (``inputs.src_audio``, else the source video's own track). The dub's own speech track is exported as ``cues``
(``{start, end, shape: "open"|"closed"}``) for C3 flap-aware script fitting or a finisher.
params: none. Server env (numpy + soundfile + ffmpeg).
"""
from __future__ import annotations

import json
from pathlib import Path

from _sdk import serve
from f_common import cleanup, item_io, to_wav, video_payload, workdir


def _vad(wav: Path, fps: float = 25.0) -> list[list[float]]:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    hop = int(sr / fps)
    n = len(x) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12))
    thr = max(float(np.percentile(db, 10)) + 15.0, float(db.max()) - 35.0)
    out, start = [], None
    for i, v in enumerate(db > thr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start / fps, i / fps])
            start = None
    if start is not None:
        out.append([start / fps, n / fps])
    return out


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, audio = item_io(item)
    work = workdir(out)
    src = item["inputs"].get("src_speech")
    if src is None:
        src = _src_vad(item, video, work)
    dub = _vad(to_wav(audio, work / "dub.wav")) if audio else []
    cues = [{"start": s, "end": e, "shape": "open"} for s, e in dub]
    cue_file = out.with_suffix(".cues.json")
    cue_file.write_text(json.dumps(cues))
    payload = video_payload(out, video, audio, mouth_open=src, cues=cues)
    payload["files"]["cues"] = str(cue_file)
    cleanup(work)
    return payload


def _src_vad(item: dict, video: Path, work: Path) -> list[list[float]]:
    """Original speech intervals: the pack's original audio track, else the video's own."""
    from f_common import probe

    src_audio = item["inputs"].get("src_audio")
    if src_audio:
        return _vad(to_wav(src_audio, work / "src.wav"))
    return _vad(to_wav(video, work / "src.wav")) if probe(video)["has_audio"] else []


if __name__ == "__main__":
    serve(load, run)
