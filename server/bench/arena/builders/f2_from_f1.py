"""F2 pack from cached F1 self-reenactment outputs (no model runs here).

Takes the F1 ``test`` items with ground truth (``kind: self``) and the arena's cached output of
one F1 candidate for each (default ``latentsync-1.6``: the 512 px crop composited into the frame
is exactly the artefact F2 exists to repair). The lip-synced video becomes the F2 input; the
untouched original is both the identity reference and the ground truth. Run F1 first
(``arena run F1 --lang en --candidates latentsync-1.6``).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .. import db, paths
from ..packs import load_pack
from . import builder
from ._fvideo import merge_pack

SOURCE = "f2_from_f1"

LICENSE = """# F2 from F1 outputs ({lang}, generator: {cand})

Built by `bench/arena/builders/f2_from_f1.py`: inputs are the cached F1 outputs of `{cand}` on
the F1 self-reenactment pack; ground truth is the original clip. Inherits the F1 pack's terms
(see data/eval/F1/{lang}/LICENSE.md): eval-only.
"""


def cached_outputs(capability: str, lang: str, candidate: str) -> dict[str, Path]:
    """item id -> produced video of the newest ok output of ``candidate``."""
    con = db.connect()
    rows = con.execute("SELECT * FROM outputs WHERE capability=? AND lang=? AND candidate=? AND "
                       "status='ok' ORDER BY rowid", (capability, lang, candidate)).fetchall()
    con.close()
    found: dict[str, Path] = {}
    for row in rows:
        payload = db.output_path(row).with_suffix(".json")
        if not payload.exists():
            continue
        video = (json.loads(payload.read_text()).get("files") or {}).get("video")
        if video and Path(video).exists():
            found[row["item"]] = Path(video)
    return found


@builder(SOURCE, capabilities=["F2"], langs=["en"], license="inherits F1 (eval-only)")
def f2_from_f1(lang: str = "en", n: int | None = None, candidate: str | None = None) -> Path:
    """F2 pack: an F1 generator's self-reenactment outputs + the originals as ground truth."""
    cand = candidate or os.environ.get("OPENDUB_F2_GENERATOR", "latentsync-1.6")
    items = [it for it in load_pack("F1", lang) if it.meta.get("kind") == "self"
             and it.refs.get("video")]
    outputs = cached_outputs("F1", lang, cand)
    out = paths.eval_dir() / "F2" / lang
    rows = []
    for it in items:
        video = outputs.get(it.id)
        if video is None:
            continue
        dst = out / SOURCE / cand / f"{Path(it.inputs['video']).stem}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(video, dst)
        rel = str(dst.relative_to(out))
        orig = Path(it.refs["video"])
        # Keep references relative to the F2 pack: copy the original next to it.
        ref = out / SOURCE / "originals" / orig.name
        ref.parent.mkdir(parents=True, exist_ok=True)
        if not ref.exists():
            shutil.copyfile(orig, ref)
        audio = Path(it.inputs["audio"])
        aud = out / SOURCE / "originals" / audio.name
        if not aud.exists():
            shutil.copyfile(audio, aud)
        rows.append({"id": f"{cand}-{it.id}", "group": it.group, "split": "test",
                     "inputs": {"video": rel, "ref_video": str(ref.relative_to(out)),
                                "audio": str(aud.relative_to(out))},
                     "refs": {"video": str(ref.relative_to(out))},
                     "meta": {**it.meta, "f1_candidate": cand, "f1_item": it.id}})
        if n and len(rows) >= n:
            break
    if not rows:
        raise FileNotFoundError(f"no cached ok F1 outputs of {cand!r} for the F1/{lang} "
                                f"self-reenactment items; run F1 with that candidate first")
    return merge_pack("F2", lang, f"{SOURCE}-{cand}", rows,
                      LICENSE.format(lang=lang, cand=cand))
