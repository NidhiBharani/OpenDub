"""B2 worker: a video-native VLM (Gemini) lists shot transitions. A semantic fallback in the
research ("act breaks and scene structure rather than frame-accurate cuts"); the arena measures
how far from frame-accurate it is.

params: provider: gemini, model, pricing, media_resolution, thinking_level, max_clip_s (600:
longer items raise Unsupported — a whole feature does not fit a sensible budget).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from b_common import make_vlm, parse_json, probe

PROMPT = """List every shot transition in this video: each hard cut, and each gradual
transition (dissolve, fade, wipe). Give times in seconds from the start with millisecond
precision. For a hard cut give the time of the first frame of the new shot as both start and end;
for a gradual transition give its start and end.
Return JSON only: {"transitions": [{"start": 12.345, "end": 12.345, "type": "cut"}]}"""


def load(params: dict, lang: str) -> dict:
    if params.get("provider") != "gemini":
        raise ValueError("b2_vlm needs a video-input model (provider: gemini)")
    return {"vlm": make_vlm(params), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video = item["inputs"]["video"]
    info = probe(video)
    if info["duration_s"] > float(state["params"].get("max_clip_s", 600)):
        raise Unsupported(f"clip longer than max_clip_s ({info['duration_s']:.0f}s)")
    reply = state["vlm"].ask(PROMPT, video=video)
    ans = parse_json(reply["text"])
    trs = []
    for tr in (ans.get("transitions") if isinstance(ans, dict) else ans) or []:
        try:
            s = float(tr["start"])
            e = float(tr.get("end", s))
        except (KeyError, TypeError, ValueError):
            continue
        trs.append({"start": min(s, e), "end": max(s, e), "type": tr.get("type", "cut")})
    return {"transitions": sorted(trs, key=lambda x: x["start"]), "fps": info["fps"],
            "duration_s": info["duration_s"], "_cost_usd": reply["cost_usd"],
            "usage": {"input_tokens": reply["input_tokens"],
                      "output_tokens": reply["output_tokens"]}}


def describe(state: dict) -> dict:
    return {"provider": "gemini", "model": state["params"].get("model")}


if __name__ == "__main__":
    serve(load, run, describe)
