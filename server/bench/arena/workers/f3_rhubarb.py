"""F3 worker: Rhubarb Lip Sync (DanielSWolf/rhubarb-lip-sync v1.14.0, MIT) as a cue exporter.

Rhubarb turns the dub audio into a 2D mouth-shape track (A–F basic, G/H/X extended) for rigged
characters or as a flap-count oracle for C3; it cannot edit finished animation, so the picture is
delivered unchanged (``mouth_open`` = the original flap proxy, as ``f3_decline``) and the track
is ``cues``. params: recognizer ("phonetic" for non-English, "pocketSphinx" for en;
default: pocketSphinx if lang == en else phonetic), extended_shapes ("GHX"), dialog (use
``item.meta.text`` as the dialog hint, default true), binary (default
~/.opendub/envs/f3_rhubarb/rhubarb/rhubarb).
"""
from __future__ import annotations

import json
from pathlib import Path

from _sdk import serve
from f3_decline import _src_vad
from f_common import cleanup, item_io, to_wav, video_payload, workdir
from f_common import run as sh

DEFAULT_BIN = Path.home() / ".opendub" / "envs" / "f3_rhubarb" / "rhubarb" / "rhubarb"


def load(params: dict, lang: str):
    binary = Path(params.get("binary") or DEFAULT_BIN).expanduser()
    if not binary.exists():
        raise FileNotFoundError(f"rhubarb binary not found at {binary}")
    rec = params.get("recognizer") or ("pocketSphinx" if lang == "en" else "phonetic")
    return {"bin": binary, "rec": rec, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, audio = item_io(item)
    work = workdir(out)
    wav = to_wav(audio or video, work / "dub.wav", sr=16000)
    cmd = [state["bin"], "-q", "-f", "json", "-r", state["rec"], "--extendedShapes",
           state["params"].get("extended_shapes", "GHX"), "-o", work / "cues.json"]
    text = (item.get("meta") or {}).get("text")
    if text and state["params"].get("dialog", True):
        (work / "dialog.txt").write_text(text)
        cmd += ["-d", work / "dialog.txt"]
    sh([*cmd, wav])
    doc = json.loads((work / "cues.json").read_text())
    cues = [{"start": float(c["start"]), "end": float(c["end"]), "shape": c["value"]}
            for c in doc.get("mouthCues", [])]
    cue_file = out.with_suffix(".cues.json")
    cue_file.write_text(json.dumps(cues))
    src = item["inputs"].get("src_speech")
    if src is None:
        src = _src_vad(item, video, work)
    payload = video_payload(out, video, audio, mouth_open=src, cues=cues,
                            recognizer=state["rec"])
    payload["files"]["cues"] = str(cue_file)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    return {"model": "rhubarb-lip-sync", "recognizer": state["rec"]}


if __name__ == "__main__":
    serve(load, run, describe)
