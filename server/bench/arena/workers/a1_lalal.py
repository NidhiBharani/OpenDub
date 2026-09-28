"""A1 worker: LALAL.AI API v1 (header ``X-License-Key``; www.lalal.ai/api/v1/openapi.json read
2026-09-28). Upload ``POST /api/v1/upload/`` (raw body, ``Content-Disposition: attachment;
filename=…``) → ``source_id``; ``POST /api/v1/split/stem_separator/`` ``{"source_id",
"presets": {"stem": "voice", "splitter": "auto" | "andromeda" | …, "dereverb_enabled": false}}``;
poll ``POST /api/v1/check/`` until ``tracks[{type: "stem" | "back", url}]`` are ready.
Pricing is by plan minutes (Pro $15/mo incl. 250 min; top-up 750 min / $50 → $0.067/min):
param usd_per_min. params: stem, splitter, bed_by_subtraction.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a1_common import finish, read
from a4_http import audio_seconds, download, poll, request

BASE = "https://www.lalal.ai/api/v1"


def load(params: dict, lang: str):
    return {"key": api_key("LALAL_API_KEY"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p, hdr = state["params"], {"X-License-Key": state["key"]}
    audio = item["inputs"]["audio"]
    up = request("POST", f"{BASE}/upload/", data=Path(audio).read_bytes(), headers={
        **hdr, "Content-Disposition": f"attachment; filename={Path(audio).name}"})
    sid = up.get("source_id") or up.get("id")
    request("POST", f"{BASE}/split/stem_separator/", headers=hdr, json_body={
        "source_id": sid, "presets": {"stem": p.get("stem", "voice"),
                                      "splitter": p.get("splitter", "auto"),
                                      "dereverb_enabled": False}})

    def check():
        r = request("POST", f"{BASE}/check/", headers=hdr, json_body={"source_ids": [sid]})
        res = (r.get("result") or {}).get(sid, r) if isinstance(r.get("result"), dict) else r
        if str(res.get("status", "")).lower() in ("error", "failed"):
            raise RuntimeError(f"LALAL split failed: {res}")
        tracks = res.get("tracks") or []
        return tracks if tracks and all(t.get("url") for t in tracks) else None

    tracks = poll(check, interval=5.0)
    got = {}
    for tr in tracks:
        dst = out.with_name(f"{out.name}.lalal_{tr['type']}.wav")
        download(tr["url"], dst)
        got[tr["type"]] = str(dst)
    dia, sr = read(got["stem"])
    mix, _ = read(audio, sr)
    back = read(got["back"], sr)[0] if "back" in got else None
    res = finish(out, sr, mix[: dia.shape[0]] if mix.shape[0] >= dia.shape[0] else mix, dia,
                 background=back, subtract=bool(p.get("bed_by_subtraction", True)))
    res["_cost_usd"] = audio_seconds(audio) / 60 * float(p.get("usd_per_min", 0.067))
    return res


if __name__ == "__main__":
    serve(load, run)
