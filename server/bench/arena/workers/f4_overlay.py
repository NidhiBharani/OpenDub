"""F4 worker: overlay track + editing-template export (no inpainting). Research: "ship first".

For every text instance: an opaque plate in the colour sampled from the ring around the box
covers the source text, and the translation is rendered on it (PIL/Raqm, Noto). The same cues
(box, span, text, colours) are exported as a JSON template a finisher can import.
params: plate (true; false = outlined text over the source, i.e. no erasure), font_dir.
Server env (numpy + Pillow + ffmpeg).
"""
from __future__ import annotations

import json
from pathlib import Path

from _sdk import serve
from f_common import video_payload


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    from f4_common import render_replacement
    from f_frames import read_frames, write_frames

    texts = item["inputs"]["texts"]
    frames, fps = read_frames(item["inputs"]["video"])
    result = render_replacement(frames, fps, texts, state["params"],
                                plate=bool(state["params"].get("plate", True)))
    dst = write_frames(result, fps, out.with_suffix(".mp4"))
    template = out.with_suffix(".template.json")
    template.write_text(json.dumps({"fps": fps, "cues": [
        {k: t[k] for k in ("box", "start", "end", "tgt_text", "tgt_lang", "src_text")}
        for t in texts]}, ensure_ascii=False, indent=1))
    payload = video_payload(out, dst, None, remux=False)
    payload["files"]["template"] = str(template)
    return payload


if __name__ == "__main__":
    serve(load, run)
