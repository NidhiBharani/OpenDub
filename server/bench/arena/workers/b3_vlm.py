"""B3 worker: vision-language models read on-screen text (API: Gemini / Claude / GPT; local:
Qwen3-VL, PaddleOCR-VL, dots.ocr … through transformers).

params:
  provider: gemini | anthropic | openai | local; model; pricing (price_in_per_mtok, …)
  prompt_style: json (default: every text line with a 0–1000 box, JSON) | plain:<prompt>
    (e.g. ``plain:OCR:`` for PaddleOCR-VL — lines of text, no boxes; they score on text_f /
    char_f only)
  box_order: yxyx (Gemini's box_2d convention, default) | xyxy
  frame_side: longest side sent (default 1536)
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from b_common import box_from_1000, item_frames, make_vlm, parse_json

LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean"}

PROMPT = """Read every legible piece of text visible in this video frame (captions, subtitles,
signs, titles, lower thirds, credits, UI text). The video's main language is {language}, but
transcribe text exactly as written in whatever script it appears; do not translate.
Return JSON only: {{"texts": [{{"text": "<one line of text>", "box_2d": [ymin, xmin, ymax, xmax]}}]}}
with one entry per text line and box_2d normalised to 0-1000. Return {{"texts": []}} if there is
no text."""

SCHEMA = {"type": "object", "properties": {"texts": {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"},
                                     "box_2d": {"type": "array", "items": {"type": "number"}}},
    "required": ["text", "box_2d"], "additionalProperties": False}}},
    "required": ["texts"], "additionalProperties": False}


def load(params: dict, lang: str) -> dict:
    return {"vlm": make_vlm(params), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    style = p.get("prompt_style", "json")
    texts, cost, n_in, n_out = [], 0.0, 0, 0
    for fr in item_frames(item, out, max_side=int(p.get("frame_side", 1536))):
        img = [(Path(fr["path"]).read_bytes(),
                "image/png" if fr["path"].lower().endswith(".png") else "image/jpeg")]
        if style.startswith("plain:"):
            reply = state["vlm"].ask(style[len("plain:"):], images=img)
            lines = [ln.strip() for ln in reply["text"].splitlines() if ln.strip()]
            texts += [{"text": ln, "box": None, "t": fr["t"]} for ln in lines]
        else:
            prompt = PROMPT.format(language=LANG_NAMES.get(state["lang"], state["lang"]))
            reply = state["vlm"].ask(prompt, images=img, schema=SCHEMA)
            ans = parse_json(reply["text"])
            for e in (ans.get("texts") if isinstance(ans, dict) else ans) or []:
                if isinstance(e, dict) and str(e.get("text", "")).strip():
                    texts.append({"text": str(e["text"]), "t": fr["t"],
                                  "box": box_from_1000(e.get("box_2d") or e.get("box"),
                                                       p.get("box_order", "yxyx"))})
        cost += reply["cost_usd"]
        n_in += reply["input_tokens"]
        n_out += reply["output_tokens"]
    payload = {"texts": texts, "usage": {"input_tokens": n_in, "output_tokens": n_out}}
    if cost:
        payload["_cost_usd"] = cost
    return payload


def describe(state: dict) -> dict:
    p = state["params"]
    return {"provider": p.get("provider", "local"), "model": p.get("model"),
            "revision": p.get("revision"), "prompt_style": p.get("prompt_style", "json")}


if __name__ == "__main__":
    serve(load, run, describe)
