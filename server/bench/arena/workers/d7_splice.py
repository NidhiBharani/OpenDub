"""Method worker (D7): pass-through splicing of the ORIGINAL vocalisation around TTS speech — the
research "primary" for non-verbals (detector-gated splice, synthesis only as a fallback).

Item: inputs.text with inline tags (``"[laugh] そうなの? [sigh]"`` → speech segments between
tags), inputs.ref_audio (voice for the inner TTS), inputs.events = [tag, …] and
inputs.event_audio_<j> = the clip for events[j] — source vocalisations cut by the pack builder (the A1/A8
stages in the app). Each tag is replaced by its clip, loudness-matched (RMS) to the neighbouring
speech, with short cross-fades. A tag with no clip is passed through to the inner model in the
text (models with native tags may render it) unless ``drop_unmatched``.

Runs in the inner candidate's env. params: inner {worker, params}, fade_ms (30),
drop_unmatched (false), gap_ms (80 silence around a clip).
"""
from __future__ import annotations

import re
from pathlib import Path

import d_voice as dv
from _sdk import serve

TAG = re.compile(r"\[([^\]]+)\]")


def load(params: dict, lang: str):
    mod = dv.import_worker(params["inner"]["worker"])
    return {"mod": mod, "inner": mod.load(params["inner"].get("params", {}), lang),
            "params": params, "lang": lang}


def _read(path: str, sr: int):
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np

    p = state["params"]
    text = dv.text_of(item)
    inp = item.get("inputs") or {}
    clips: dict[str, list[str]] = {}
    for j, tag in enumerate(inp.get("events") or []):
        if inp.get(f"event_audio_{j}"):
            clips.setdefault(str(tag).lower(), []).append(inp[f"event_audio_{j}"])
    # split into [("speech", str) | ("event", tag)]
    parts, pos = [], 0
    for m in TAG.finditer(text):
        if text[pos:m.start()].strip():
            parts.append(("speech", text[pos:m.start()].strip()))
        tag = m.group(1).strip().lower()
        if clips.get(tag):
            parts.append(("event", tag))
        elif not p.get("drop_unmatched"):
            parts.append(("speech", m.group(0)))
        pos = m.end()
    if text[pos:].strip():
        parts.append(("speech", text[pos:].strip()))
    # merge adjacent speech parts (unmatched tags stay inline)
    merged: list[tuple[str, str]] = []
    for kind, val in parts:
        if merged and kind == "speech" and merged[-1][0] == "speech":
            merged[-1] = ("speech", f"{merged[-1][1]} {val}")
        else:
            merged.append((kind, val))
    sr, speech, pieces, used = None, [], [], 0
    seed = dv.item_seed(item, p.get("inner", {}).get("params", {}))
    for k, (kind, val) in enumerate(merged):
        if kind != "speech":
            continue
        sub = {**item, "seed": seed, "inputs": {**item["inputs"], "text": val}}
        payload = state["mod"].run(state["inner"], sub, Path(out).with_name(f"{Path(out).name}-s{k}"))
        sr = sr or int(payload["sample_rate"])
        speech.append((k, _read(payload["files"]["audio"], sr)))
    if sr is None:  # only events: keep the source vocalisations
        sr = 24000
    by_k = dict(speech)
    ref_rms = float(np.sqrt(np.mean(np.concatenate([a for _, a in speech]) ** 2))) if speech \
        else 0.05
    fade = int(sr * float(p.get("fade_ms", 30)) / 1000)
    gap = np.zeros(int(sr * float(p.get("gap_ms", 80)) / 1000), dtype=np.float32)
    for k, (kind, val) in enumerate(merged):
        if kind == "speech":
            pieces.append(by_k[k].astype(np.float32))
            continue
        clip = _read(clips[val][min(used, len(clips[val]) - 1)], sr).astype(np.float32)
        used += 1
        rms = float(np.sqrt(np.mean(clip ** 2))) or 1.0
        clip *= min(4.0, ref_rms / rms)
        if fade and len(clip) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        pieces += [gap, clip, gap]
    wav = dv.write_wav(dv.wav_path(out), np.concatenate(pieces) if pieces else gap, sr)
    return dv.audio_payload(wav, events_spliced=used,
                            segments=[kind for kind, _ in merged])


if __name__ == "__main__":
    serve(load, run)
