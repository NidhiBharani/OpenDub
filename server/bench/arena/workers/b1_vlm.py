"""B1 worker: a VLM says, for each dialogue line, which visible face speaks it (or that the
speaker is off-screen). No hosted active-speaker API exists; this is the research's API pick
(Gemini 3.1 Pro with native video + audio) and its alternates (Claude Opus 5, GPT), which get
frames only.

- ``gemini``: the whole clip (video + audio) goes in one request with the line windows; the model
  returns a ``box_2d`` per line at the line's reference time.
- ``anthropic`` / ``openai``: no audio or video input is used — each line gets ``frames_per_line``
  stills across its window plus the line text when the pack provides it. Expect this to be weak:
  it is lip-reading from stills.

params: provider, model, pricing, frames_per_line (2), frame_side (1024), lines_per_request (12),
max_clip_s (900).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from b_common import box_from_1000, grab_jpeg, make_vlm, parse_json, probe

VIDEO_PROMPT = """This clip has dialogue. For each numbered line below (time window in seconds),
decide who speaks it: if the speaker's face is visible, give that face's bounding box at the
given reference time as box_2d [ymin, xmin, ymax, xmax] normalised to 0-1000; if no visible face
is speaking it (voice-over, off-screen character, narrator), give null.
Lines:
{lines}
Return JSON only: {{"lines": [{{"n": 1, "box_2d": [..] or null}}]}}"""

FRAMES_PROMPT = """Each group of images below shows one dialogue line from a video: {k} frames
spread across the line's time window. For each line decide who speaks it: if the speaker's face is
visible, give its bounding box in the LAST frame of that line's group as box_2d
[ymin, xmin, ymax, xmax] normalised to 0-1000; if no visible face is speaking (voice-over,
off-screen character), give null.
Lines (in image order):
{lines}
Return JSON only: {{"lines": [{{"n": 1, "box_2d": [..] or null}}]}}"""


def load(params: dict, lang: str) -> dict:
    return {"vlm": make_vlm(params), "params": params}


def _line_desc(k: int, ln: dict, t: float) -> str:
    text = f' — "{ln["text"]}"' if ln.get("text") else ""
    return f"{k}. {float(ln['start']):.2f}–{float(ln['end']):.2f} s (reference time {t:.2f} s){text}"


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    lines = item["inputs"].get("lines") or []
    if not lines:
        raise Unsupported("b1_vlm needs inputs.lines (the dialogue windows to attribute)")
    video = item["inputs"]["video"]
    info = probe(video)
    if info["duration_s"] > float(p.get("max_clip_s", 900)):
        raise Unsupported(f"clip longer than max_clip_s ({info['duration_s']:.0f}s)")
    per_req = int(p.get("lines_per_request", 12))
    k_frames = int(p.get("frames_per_line", 2))
    result, cost, n_in, n_out = [], 0.0, 0, 0
    for a in range(0, len(lines), per_req):
        chunk = lines[a:a + per_req]
        tref = [float(ln.get("t", (float(ln["start"]) + float(ln["end"])) / 2)) for ln in chunk]
        desc = "\n".join(_line_desc(k + 1, ln, t) for k, (ln, t) in enumerate(zip(chunk, tref,
                                                                                    strict=True)))
        if p.get("provider") == "gemini":
            reply = state["vlm"].ask(VIDEO_PROMPT.format(lines=desc), video=video)
        else:
            images = []
            for ln, t in zip(chunk, tref, strict=True):
                s, e = float(ln["start"]), float(ln["end"])
                ts = [s + (e - s) * (j + 1) / (k_frames + 1) for j in range(k_frames - 1)] + [t]
                images += [(grab_jpeg(video, x, max_side=int(p.get("frame_side", 1024))),
                            "image/jpeg") for x in ts]
            reply = state["vlm"].ask(FRAMES_PROMPT.format(k=k_frames, lines=desc), images=images)
        ans = parse_json(reply["text"])
        by_n = {int(r.get("n", 0)): r for r in (ans.get("lines") if isinstance(ans, dict)
                                                else ans) or [] if isinstance(r, dict)}
        for k, (ln, _t) in enumerate(zip(chunk, tref, strict=True), 1):
            box = box_from_1000((by_n.get(k) or {}).get("box_2d"))
            result.append({"start": float(ln["start"]), "end": float(ln["end"]), "box": box,
                           "offscreen": box is None})
        cost += reply["cost_usd"]
        n_in += reply["input_tokens"]
        n_out += reply["output_tokens"]
    payload = {"lines": result, "tracks": [], "fps": info["fps"],
               "duration_s": info["duration_s"],
               "usage": {"input_tokens": n_in, "output_tokens": n_out}}
    if cost:
        payload["_cost_usd"] = cost
    return payload


def describe(state: dict) -> dict:
    p = state["params"]
    return {"provider": p.get("provider"), "model": p.get("model")}


if __name__ == "__main__":
    serve(load, run, describe)
