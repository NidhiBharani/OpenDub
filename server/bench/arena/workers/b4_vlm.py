"""B4 worker: a vision-language model votes on the title from a handful of frames, with an
explicit abstain option (API: Gemini / Claude / GPT via ``b_common.Vlm``; local: Qwen3-VL etc.
via ``b_common.LocalVlm``).

params: provider (gemini|anthropic|openai|local), model, pricing (price_in_per_mtok,
price_out_per_mtok), n_frames (8), frame_side (768), effort / thinking_level, plus the local-model
fields of ``LocalVlm``. The stated confidence is recorded but is not a calibrated probability
(research note); the spec scores calibration separately.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from b_common import grab_jpeg, make_vlm, parse_json, probe, sample_times

PROMPT = """These {n} frames are sampled evenly from one video title (timestamps: {times}).
Classify the title's picture as exactly one of:
- "live_action": filmed with a camera (real people/places), even with some VFX
- "2d": drawn / 2D animation (cel, anime, cartoon, vector motion graphics)
- "3d": 3D computer animation (CGI characters and sets, stylised or photoreal)
- "mixed": substantial parts in more than one of the above (e.g. live action with animated
  sequences, live actors composited into CG)
- "abstain": you cannot tell with reasonable confidence
Answer with JSON only: {{"label": "...", "confidence": <0..1>, "reason": "<one sentence>"}}"""

SCHEMA = {"type": "object",
          "properties": {"label": {"type": "string",
                                   "enum": ["live_action", "2d", "3d", "mixed", "abstain"]},
                         "confidence": {"type": "number"}, "reason": {"type": "string"}},
          "required": ["label", "confidence", "reason"], "additionalProperties": False}


def load(params: dict, lang: str) -> dict:
    return {"vlm": make_vlm(params), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    video = item["inputs"]["video"]
    dur = probe(video)["duration_s"]
    times = sample_times(dur, int(p.get("n_frames", 8)))
    images = [(grab_jpeg(video, t, max_side=int(p.get("frame_side", 768))), "image/jpeg")
              for t in times]
    prompt = PROMPT.format(n=len(times), times=", ".join(f"{t:.0f}s" for t in times))
    reply = state["vlm"].ask(prompt, images=images, schema=SCHEMA)
    ans = parse_json(reply["text"])
    label = str(ans.get("label", "abstain")).strip().lower()
    conf = ans.get("confidence")
    payload = {"label": None if label == "abstain" else label, "abstain": label == "abstain",
               "confidence": float(conf) if isinstance(conf, (int, float)) else None,
               "reason": ans.get("reason"), "model": reply["model"],
               "usage": {"input_tokens": reply["input_tokens"],
                         "output_tokens": reply["output_tokens"]}}
    if reply["cost_usd"]:
        payload["_cost_usd"] = reply["cost_usd"]
    return payload


def describe(state: dict) -> dict:
    p = state["params"]
    return {"provider": p.get("provider", "local"), "model": p.get("model"),
            "revision": p.get("revision")}


if __name__ == "__main__":
    serve(load, run, describe)
