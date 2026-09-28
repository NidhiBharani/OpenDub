"""A1/A2 worker: MVSEP hosted separation (mvsep.com/full_api, read 2026-09-28).
``POST https://mvsep.com/api/separation/create`` (form: ``api_token``, ``audiofile``,
``sep_type``, ``add_opt1``, ``output_format`` = 1 wav) → ``hash``; poll ``GET
/api/separation/get?hash=…`` until files are listed; stems are matched by file name.
sep_type: 45 BandIt v2 (add_opt1 0 multi / 1 eng …), 36 BandIt Plus, 56 MVSep DnR v3 ensemble,
24 Demucs4HT DnR; 22 reverb removal / 47 denoise for A2. Credits: 1 per started minute ×
algorithm multiplier; USD price not published → param usd_per_min (default 0).
params: sep_type, add_opt1, stems ({dialogue: "speech"|"dialog", music: "music", effects:
"sfx"|"effects"} name fragments, or {audio: …} for A2), bed_by_subtraction.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import api_key, serve
from a1_common import finish, read
from a4_http import audio_seconds, download, poll, post_form, request

BASE = "https://mvsep.com/api/separation"


def load(params: dict, lang: str):
    return {"key": api_key("MVSEP_API_TOKEN"), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    audio = item["inputs"]["audio"]
    job = post_form(f"{BASE}/create", {"api_token": state["key"], "sep_type": p["sep_type"],
                                       "add_opt1": p.get("add_opt1"), "output_format": 1},
                    {"audiofile": audio}, {})
    h = (job.get("data") or {}).get("hash") or job.get("hash")

    def check():
        r = request("GET", f"{BASE}/get?hash={h}")
        st = r.get("status")
        if st in ("failed", "error"):
            raise RuntimeError(f"MVSEP failed: {r}")
        files = (r.get("data") or {}).get("files") or []
        return files if st == "done" and files else None

    files = poll(check, interval=10.0)
    stems = {}
    for role, frag in (p.get("stems") or {"dialogue": "speech", "music": "music",
                                          "effects": "sfx"}).items():
        hit = [f for f in files if frag.lower() in str(f.get("url", "") + f.get("type", ""))
               .lower()]
        if hit:
            dst = out.with_name(f"{out.name}.mvsep_{role}.wav")
            download(hit[0]["url"], dst)
            stems[role] = str(dst)
    cost = audio_seconds(audio) / 60 * float(p.get("usd_per_min", 0.0))
    if "audio" in stems:
        return {"files": {"audio": stems["audio"]}, "_cost_usd": cost}
    dia, sr = read(stems["dialogue"])
    mix, _ = read(audio, sr)
    res = finish(out, sr, mix, dia,
                 music=read(stems["music"], sr)[0] if "music" in stems else None,
                 effects=read(stems["effects"], sr)[0] if "effects" in stems else None,
                 subtract=bool(p.get("bed_by_subtraction", True)))
    res["_cost_usd"] = cost
    return res


if __name__ == "__main__":
    serve(load, run)
