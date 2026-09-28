"""G1/G3 items from the user's own audit marks (ok / bad) on arena TTS outputs.

Decision 2026-09-27: no human raters; the user audits. Every "bad" mark on a voice output is a
human-flagged defective take, every "ok" a human-accepted one. This builder turns the latest mark
per (candidate, item) of the chosen voice capabilities into judge meta-evaluation items and
*merges* them into the existing G1 and G3 packs (items from other builders are kept):

- G1: inputs {audio, text}, refs {defective: 1 for bad, 0 for ok, text}
- G3: inputs {audio}, refs {ok: 1 for ok, 0 for bad} (ranked by SRCC like a binary MOS)

Paths point at the kept arena outputs (absolute), so the audio is not copied. Group = the
source item (the same line voiced by several candidates stays in one resampling unit).
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import db
from ..packs import load_pack
from . import builder
from ._gcommon import merge_pack

VOICE_CAPS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

LICENSE = """# User audit marks ({cap}, {lang})

Source: the user's ok/bad marks in the arena audit viewer on outputs of {sources}.
{n} marked takes. Audio: the kept arena outputs (paths are absolute; do not publish).
"""


def audit_items(lang: str, capabilities: list[str]) -> tuple[list[dict], list[dict]]:
    con = db.connect()
    g1, g3 = [], []
    try:
        for cap in capabilities:
            marks = db.audit_marks(con, cap, lang)
            if not marks:
                continue
            try:
                items = {it.id: it for it in load_pack(cap, lang, split=None)}
            except FileNotFoundError:
                items = {}
            for (cand_key, item_id), mark in marks.items():
                verdict = mark["verdict"]
                if verdict not in ("ok", "bad"):
                    continue
                row = con.execute("SELECT * FROM outputs WHERE capability=? AND lang=? AND "
                                  "cand_key=? AND item=? AND status='ok' ORDER BY created DESC",
                                  (cap, lang, cand_key, item_id)).fetchone()
                if row is None:
                    continue
                out_json = db.output_path(row).with_suffix(".json")
                if not out_json.exists():
                    continue
                audio = (json.loads(out_json.read_text()).get("files") or {}).get("audio")
                if not audio or not Path(audio).exists():
                    continue
                it = items.get(item_id)
                text = (it.inputs.get("text") or it.refs.get("text")) if it else None
                uid = f"audit-{cap}-{cand_key}-{item_id}"
                meta = {"source": "arena_audit", "capability": cap, "candidate": row["candidate"],
                        "note": mark.get("note", "")}
                bad = int(verdict == "bad")
                if text:
                    g1.append({"id": uid, "group": f"{cap}:{item_id}", "split": "test",
                               "inputs": {"audio": audio, "text": text},
                               "refs": {"text": text, "defective": bad}, "meta": meta})
                g3.append({"id": uid, "group": f"{cap}:{item_id}", "split": "test",
                           "inputs": {"audio": audio}, "refs": {"ok": 1 - bad}, "meta": meta})
    finally:
        con.close()
    return g1, g3


@builder("arena_audit", capabilities=["G1", "G3"], langs=["en", "hi", "ja"],
         license="user's own marks on arena outputs (private)")
def arena_audit(lang: str, n: int | None = None, capabilities: str | list[str] | None = None
                ) -> list[Path]:
    """Merge the user's ok/bad audit marks on voice outputs into the G1 and G3 packs."""
    caps = capabilities.split(",") if isinstance(capabilities, str) else (capabilities or
                                                                          VOICE_CAPS)
    g1, g3 = audit_items(lang, caps)
    if n:
        g1, g3 = g1[:n], g3[:n]
    if not g1 and not g3:
        raise RuntimeError(f"no ok/bad audit marks on {', '.join(caps)} outputs in {lang} yet "
                           "(mark takes in `bench arena serve` first)")
    src = ", ".join(caps)
    out = []
    if g1:
        out.append(merge_pack("G1", lang, g1, "arena_audit",
                              LICENSE.format(cap="G1", lang=lang, sources=src, n=len(g1))))
    out.append(merge_pack("G3", lang, g3, "arena_audit",
                          LICENSE.format(cap="G3", lang=lang, sources=src, n=len(g3))))
    return out
