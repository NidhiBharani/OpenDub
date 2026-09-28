"""F4 worker: Vozo Visual Translate (frame_translate) — the only shipped end-to-end product.

API (vozo.ai/docs/api_reference, checked 2026-09-28): ``POST https://api.vozo.ai/v1/media/
frame_translate`` with ``Authorization: Bearer <key>``; body ``{video_url, source_language,
target_language, user_prompt?}`` → ``task_id``; poll ``GET …/frame_translate/{task_id}`` until
``status == "done"``; download ``video_url``. Input is **URL only**, so the pack video must be
reachable publicly: set ``url_prefix`` (the pack directory served somewhere public) — the
candidate is disabled until then. Vozo translates by itself: the intended translation is passed
as ``user_prompt`` guidance, so its OCR CER also measures how well it follows it.
Billing is in plan points (dollar rate unverified); ``price_per_item`` records an estimate.
params: url_prefix, pack_root (local path that url_prefix mirrors), price_per_item.
Key: VOZO_API_KEY.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, api_key, serve
from f_common import download, http, poll, video_payload

BASE = "https://api.vozo.ai/v1/media/frame_translate"
LOCALE = {"en": "en-US", "hi": "hi-IN", "ja": "ja-JP"}


def load(params: dict, lang: str):
    return {"key": api_key("VOZO_API_KEY"), "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    if not p.get("url_prefix") or not p.get("pack_root"):
        raise Unsupported("Vozo takes public URLs only: set params.url_prefix and pack_root")
    rel = Path(item["inputs"]["video"]).resolve().relative_to(Path(p["pack_root"]).resolve())
    t0 = item["inputs"]["texts"][0]
    hdr = {"Authorization": f"Bearer {state['key']}"}
    job = http("POST", BASE, headers=hdr, body={
        "video_url": f"{p['url_prefix'].rstrip('/')}/{rel.as_posix()}",
        "source_language": LOCALE.get(t0["src_lang"], t0["src_lang"]),
        "target_language": LOCALE.get(state["lang"], state["lang"]),
        "user_prompt": "Translate the on-screen text exactly as: " + "; ".join(
            f'"{t["src_text"]}" -> "{t["tgt_text"]}"' for t in item["inputs"]["texts"])})
    tid = job.get("task_id") or job["data"]["task_id"]
    done = poll(lambda: http("GET", f"{BASE}/{tid}", headers=hdr),
                done=lambda j: (j.get("status") or j.get("data", {}).get("status")) == "done",
                failed=lambda j: (j.get("status") or j.get("data", {}).get("status")) == "failed",
                interval=15, timeout=4 * 3600)
    url = done.get("video_url") or done["data"]["video_url"]
    dst = download(url, out.with_suffix(".mp4"))
    return video_payload(out, dst, None, remux=False, task_id=tid,
                         _cost_usd=float(p.get("price_per_item", 0.0)))


if __name__ == "__main__":
    serve(load, run)
