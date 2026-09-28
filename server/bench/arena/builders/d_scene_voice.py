"""SCENE (SPY×FAMILY ep. 1, official uploads) diarized lines → D-phase voice packs.

Reads the SCENE pack (``data/eval/SCENE/ja/manifest.jsonl``, built by ``spyfamily.py``: aligned
1–3 min clips with ``inputs.audio`` = Japanese mix/dialogue and ``refs.dub_<lang>_audio`` = the
official dub on the same timeline) plus **line transcripts the user produces** (diarization +
transcription + translation are model/user steps; builders never run models). Lines are read
from, in order: ``refs.lines`` / ``meta.lines`` in the manifest row, or a sidecar
``data/eval/SCENE/ja/lines/<item id>.jsonl`` (one JSON object per line)::

    {"start": 12.40, "end": 14.95, "speaker": "loid", "ja": "…",
     "en": "…", "hi": "…",                     # translations (else dub transcript text)
     "dub_en": {"start": 12.3, "end": 15.1, "text": "…"},   # optional, official dub line
     "emotion": "surprised",                   # optional (D2)
     "tagged": {"en": "[laugh] …"},            # optional script with non-verbal tags (D7)
     "nv": [{"tag": "laugh", "start": 12.4, "end": 13.0}]}  # optional source vocalisations

Times are seconds on the SCENE clip's timeline; dub times default to the source line's.
Key aliases are accepted (``text``/``ja_text``, ``en_text``/``text_en``, ``spk``). Per
capability (pack language = dub language en/hi):

- D1/D2/D3: reference = the source line (``ref_mode="line"``, carries the line's emotion) or the
  speaker's longest clean line (``"speaker"``); text = translation; ``refs.human_audio`` = the
  official dub line (the human floor); D3 ``target_s`` = the source line's duration.
- D6: official dub-actor line → original actor's voice (appendix IH pairs).
- D7: lines with non-verbal tags; ``inputs.events`` = their tags and ``inputs.event_audio_<j>`` the
  source vocalisation clips (splice method),
  ``refs.nv_audio`` = the source line.
- D8: Japanese line audio → translation.
Group = speaker. Private evaluation only (official uploads: never commit or redistribute).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths
from . import builder
from ._dvoice import cut, finish, pack_root, read_jsonl

LICENSE = """# SPY×FAMILY ep. 1 lines ({capability}, {lang})

Built by bench/arena/builders/d_scene_voice.py from the SCENE pack (official Muse Asia / Muse
India uploads: Japanese original and the official {lang} dub) and user-produced line
transcripts. Eval-only, private, never committed or redistributed. Group = speaker.
"""
ALIASES = {"ja": ("ja", "text", "ja_text", "source_text"), "speaker": ("speaker", "spk", "who")}


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return default


def _translation(line: dict, lang: str) -> str | None:
    dub = line.get(f"dub_{lang}") or {}
    return _get(line, lang, f"{lang}_text", f"text_{lang}") or (dub.get("text") if
                                                                isinstance(dub, dict) else None)


def load_lines(row: dict, scene_root: Path) -> list[dict]:
    for holder in (row.get("refs") or {}, row.get("meta") or {}, row):
        if isinstance(holder.get("lines"), list):
            return holder["lines"]
    side = scene_root / "lines" / f"{row['id']}.jsonl"
    return read_jsonl(side) if side.exists() else []


def _abs(scene_root: Path, v: str | None) -> Path | None:
    if not v:
        return None
    p = Path(v)
    return p if p.is_absolute() else scene_root / p


@builder("scene_voice", capabilities=["D1", "D2", "D3", "D6", "D7", "D8"], langs=["en", "hi"],
         license="private (official uploads, eval-only)")
def scene_voice(lang: str, n: int | None = None, capability: str = "D1",
                ref_mode: str = "line", min_line_s: float = 0.8, max_line_s: float = 15.0,
                scene: str = "SCENE") -> Path:
    """Cloning/VC/S2ST items from SCENE lines for dub language ``lang`` (en | hi)."""
    cap = capability.upper()
    scene_root = paths.eval_dir() / scene / "ja"
    manifest = scene_root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing: build the SCENE pack first "
                                "(python -m bench arena pack spyfamily --lang ja)")
    root = pack_root(cap, lang)
    items: list[dict[str, Any]] = []
    for row in read_jsonl(manifest):
        lines = load_lines(row, scene_root)
        if not lines:
            continue
        src_audio = _abs(scene_root, (row.get("inputs") or {}).get("audio"))
        refs = row.get("refs") or {}
        dub_audio = _abs(scene_root, _get(refs, f"dub_{lang}_audio", f"dub_audio_{lang}"))
        if src_audio is None:
            continue
        # speaker reference: longest line per speaker within bounds
        longest: dict[str, dict] = {}
        for ln in lines:
            spk = str(_get(ln, *ALIASES["speaker"], default="unknown"))
            d = float(ln["end"]) - float(ln["start"])
            if min_line_s <= d <= max_line_s and d > (float(longest.get(spk, {}).get("end", 0)) -
                                                      float(longest.get(spk, {}).get("start", 0))):
                longest[spk] = ln
        for k, ln in enumerate(lines):
            t0, t1 = float(ln["start"]), float(ln["end"])
            if not (min_line_s <= t1 - t0 <= max_line_s):
                continue
            spk = str(_get(ln, *ALIASES["speaker"], default="unknown"))
            ja = _get(ln, *ALIASES["ja"], default="")
            text = _translation(ln, lang)
            lid = f"{row['id']}-l{k:03d}"
            src_rel = f"audio/src/{lid}.wav"
            cut(src_audio, root / src_rel, t0, t1)
            dub = ln.get(f"dub_{lang}") if isinstance(ln.get(f"dub_{lang}"), dict) else {}
            d0, d1 = float(dub.get("start", t0)), float(dub.get("end", t1))
            dub_rel = None
            if dub_audio is not None and dub_audio.exists():
                dub_rel = f"audio/dub/{lid}.wav"
                cut(dub_audio, root / dub_rel, d0, d1)
            meta = {"src_lang": "ja", "pair": f"ja-{lang}", "speaker": spk,
                    "duration_s": round(t1 - t0, 3), "scene_item": row["id"],
                    "source": "scene_voice"}
            base = {"id": lid, "group": spk, "split": "test"}
            if cap == "D8":
                if text:
                    items.append({**base, "inputs": {"audio": src_rel},
                                  "refs": {"text": text, "source_text": ja}, "meta": meta})
                continue
            if cap == "D6":
                if dub_rel:
                    items.append({**base, "inputs": {"audio": dub_rel, "ref_audio": src_rel,
                                                     **({"text": text} if text else {})},
                                  "refs": {"text": text or ""}, "meta": meta})
                continue
            if not text:
                continue
            ref_rel, ref_text = src_rel, ja
            if ref_mode == "speaker" and spk in longest:
                sl = longest[spk]
                ref_rel = f"audio/spk/{row['id']}-{spk}.wav"
                cut(src_audio, root / ref_rel, float(sl["start"]), float(sl["end"]))
                ref_text = _get(sl, *ALIASES["ja"], default="")
            inputs: dict[str, Any] = {"text": text, "ref_audio": ref_rel, "ref_text": ref_text}
            item_refs: dict[str, Any] = {"text": text}
            if dub_rel:
                item_refs["human_audio"] = dub_rel
            if cap == "D2" and ln.get("emotion"):
                inputs["emotion"] = ln["emotion"]
            if cap == "D3":
                inputs["target_s"] = round((d1 - d0) if dub_rel and dub else (t1 - t0), 3)
            if cap == "D7":
                tagged = (ln.get("tagged") or {}).get(lang) if isinstance(ln.get("tagged"),
                                                                           dict) else None
                if not tagged:
                    continue
                inputs["text"] = tagged
                item_refs["nv_audio"] = src_rel
                tags = []
                for j, ev in enumerate(ln.get("nv") or []):
                    ev_rel = f"audio/nv/{lid}-{j}.wav"
                    cut(src_audio, root / ev_rel, float(ev["start"]), float(ev["end"]))
                    tags.append(str(ev.get("tag", "")))
                    inputs[f"event_audio_{j}"] = ev_rel  # top-level so the pack resolves it
                if tags:
                    inputs["events"] = tags
            items.append({**base, "inputs": inputs, "refs": item_refs, "meta": meta})
            if n and len(items) >= n:
                break
        if n and len(items) >= n:
            break
    return finish(cap, lang, items, LICENSE.format(capability=cap, lang=lang))
