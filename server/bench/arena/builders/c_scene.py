"""SCENE (SPY×FAMILY ep. 1 clips, built by ``builders/spyfamily.py``) → C2 and C1 packs.

The SCENE pack holds audio/video only. C-phase packs need the Japanese *source* words, so they
read a timed Japanese transcript per clip from, in order:

1. ``refs.ja_asr`` / ``refs.src_asr`` in the SCENE manifest (a path to an A4 payload JSON), or
2. ``data/eval/SCENE/ja/refs_asr/<candidate>/<clip id>.ja.json`` — written by the opt-in
   :func:`transcribe_source` (runs an A4 worker; the user runs it, builders never do).

``scene_c2`` (ja-en, ja-hi): one item per dub line (pause rule over the Japanese words), with
the **measured** slot (``slot_s``), two lines of context before and one after. References:

- ja-en: the official English dub (``refs.dub_en_asr``, timed words, same timeline as the clip)
  — the dub words whose midpoint falls inside the line's span (±``ref_pad`` s). Dubs are
  lip-synced to the original lines, so this is a usable per-line human reference; lines with no
  dub words get none (QE-only for that line).
- ja-hi: the Hindi transcript has no word timing (the aligner does not cover Hindi), so per-line
  references come only from timed segments when the transcript has more than one; otherwise the
  items are reference-free (MetricX-QE).

``scene_c1`` (ja): one item per stretch of speech (windows split at ≥ 1.2 s pauses, ≤ 60 s),
inputs.words = the Japanese words; no gold boundaries (the user's audit fills them in).

Groups: the clip (the episode is one title; with a single group the bootstrap would be
degenerate). Private evaluation only, like the SCENE pack itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..packs import pack_dir, write_pack
from . import builder
from ._ctext import pause_lines, windows, words_from_payload

SCENE = "SCENE"
LICENSE = """# {cap} / {lang} — derived from SCENE/ja (SPY×FAMILY episode 1, official uploads)

Derived from the private SCENE pack (see data/eval/SCENE/ja/LICENSE.md): YouTube Terms of
Service apply. **Private evaluation only: never commit, publish or redistribute** these items or
model outputs made from them. Source words: machine transcript ({asr}); references: official
dub transcripts ({refs}). Groups = clips. Builder: server/bench/arena/builders/c_scene.py.
"""


def _scene_rows() -> tuple[Path, list[dict]]:
    root = pack_dir(SCENE, "ja")
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing: build it first "
                                "(`python -m bench arena pack spyfamily --lang ja`)")
    return root, [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]


def _load_json(root: Path, rel: str | None) -> Any:
    if not rel:
        return None
    p = Path(rel) if Path(rel).is_absolute() else root / rel
    return json.loads(p.read_text()) if p.exists() else None


def source_words(root: Path, row: dict, candidate: str | None = None) -> list[dict] | None:
    refs = row.get("refs", {})
    payload = _load_json(root, refs.get("ja_asr") or refs.get("src_asr"))
    if payload is None:
        base = root / "refs_asr"
        cands = [base / candidate] if candidate else sorted(base.glob("*")) if base.exists() \
            else []
        for d in cands:
            f = d / f"{row['id']}.ja.json"
            if f.exists():
                payload = json.loads(f.read_text())
                break
    return words_from_payload(payload) if payload else None


def _ref_for_line(line: dict, dub_words: list[dict], dub_segments: list[dict], lang: str,
                  pad: float) -> str | None:
    lo, hi = line["start"] - pad, line["end"] + pad
    joiner = "" if lang == "ja" else " "
    if dub_words:
        hit = [w["word"] for w in dub_words if lo <= (w["start"] + w["end"]) / 2 <= hi]
        return joiner.join(hit).strip() or None
    if len(dub_segments) > 1:
        hit = [s["text"] for s in dub_segments if lo <= (s["start"] + s["end"]) / 2 <= hi]
        return joiner.join(hit).strip() or None
    return None


@builder("scene_c2", capabilities=["C2"], langs=["ja-en", "ja-hi"],
         license="private evaluation only (YouTube ToS)")
def scene_c2(lang: str, n: int | None = None, candidate: str | None = None,
             ref_pad: float = 0.35, min_pause: float = 0.35, capability: str = "C2") -> Path:
    """SPY×FAMILY dub lines with measured slots and time-aligned official-dub references."""
    src, tgt = lang.split("-", 1)
    root, rows = _scene_rows()
    items, used_asr = [], set()
    for row in rows:
        words = source_words(root, row, candidate)
        if not words:
            continue
        used_asr.add(candidate or "refs_asr")
        dub = _load_json(root, row.get("refs", {}).get(f"dub_{tgt}_asr")) or {}
        dub_words = words_from_payload(dub) if dub else []
        dub_segs = [s for s in dub.get("segments", []) if s.get("start") is not None]
        lines = pause_lines(words, src, min_pause=min_pause)
        texts = [ln["text"] for ln in lines]
        for k, ln in enumerate(lines):
            ref = _ref_for_line(ln, dub_words, dub_segs, tgt, ref_pad)
            items.append({
                "id": f"{row['id']}-L{k:03d}", "group": row["id"], "split": "test",
                "inputs": {"text": ln["text"], "src_lang": src, "tgt_lang": tgt,
                           "slot_s": round(ln["end"] - ln["start"], 3),
                           "context_before": texts[max(0, k - 2):k],
                           "context_after": texts[k + 1:k + 2]},
                "refs": {"text": ref} if ref else {},
                "meta": {"src_lang": src, "tgt_lang": tgt, "source": "spyfamily",
                         "clip": row["id"], "start_s": ln["start"], "end_s": ln["end"],
                         "ref_kind": "dub-time-aligned" if ref else "none"}})
    if not items:
        raise FileNotFoundError("no Japanese transcripts for the SCENE clips: run "
                                "`python -m bench.arena.builders.c_scene --transcribe-source`")
    items = items[:n] if n else items
    return write_pack(capability, lang, items, LICENSE.format(
        cap=capability, lang=lang, asr=", ".join(sorted(used_asr)),
        refs=f"refs.dub_{tgt}_asr, aligned by time"))


@builder("scene_c1", capabilities=["C1"], langs=["ja"],
         license="private evaluation only (YouTube ToS)")
def scene_c1(lang: str = "ja", n: int | None = None, candidate: str | None = None,
             max_window_s: float = 60.0, capability: str = "C1") -> Path:
    """SPY×FAMILY Japanese timed words in ≤ 60 s windows (no gold boundaries)."""
    root, rows = _scene_rows()
    items = []
    for row in rows:
        words = source_words(root, row, candidate)
        if not words:
            continue
        for k, (a, b) in enumerate(windows(words, max_s=max_window_s)):
            ws = words[a:b]
            items.append({"id": f"{row['id']}-W{k:02d}", "group": row["id"], "split": "test",
                          "inputs": {"words": ws, "src_lang": "ja",
                                     "text": "".join(w["word"] for w in ws)},
                          "refs": {},
                          "meta": {"src_lang": "ja", "source": "spyfamily", "clip": row["id"],
                                   "duration_s": round(ws[-1]["end"] - ws[0]["start"], 3)}})
    if not items:
        raise FileNotFoundError("no Japanese transcripts for the SCENE clips: run "
                                "`python -m bench.arena.builders.c_scene --transcribe-source`")
    items = items[:n] if n else items
    return write_pack(capability, lang, items, LICENSE.format(
        cap=capability, lang=lang, asr=candidate or "refs_asr", refs="none (audit)"))


def transcribe_source(candidate: str = "qwen3-asr-1.7b", *, params: dict | None = None,
                      log=print) -> Path:
    """Opt-in (runs a model): transcribe each SCENE clip's Japanese audio with an A4 candidate
    into ``SCENE/ja/refs_asr/<candidate>/<clip>.ja.json``. Never called by a builder."""
    from ..registry import load_candidates, select
    from ..runner import ensure_gpu_room, run_worker_job

    root, rows = _scene_rows()
    cand = select(load_candidates("A4"), [candidate])[0]
    out_dir = root / "refs_asr" / cand.id
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [{"id": r["id"], "inputs": {"audio": str(root / r["inputs"]["audio"])},
              "meta": {"duration_s": r.get("meta", {}).get("duration_s")},
              "out": str(out_dir / f"{r['id']}.ja")} for r in rows if r.get("inputs", {}).get(
                  "audio")]
    ensure_gpu_room(cand.id, cand.requires.vram_gb)
    run_worker_job(worker=cand.worker, env=cand.env,
                   params={**cand.params, "max_new_tokens": 2048, **(params or {})}, lang="ja",
                   items=items, label=f"scene-source-ja-{cand.id}", log=log)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.arena.builders.c_scene")
    p.add_argument("--transcribe-source", action="store_true",
                   help="transcribe the Japanese clip audio with an A4 candidate (runs a model)")
    p.add_argument("--candidate", default="qwen3-asr-1.7b")
    a = p.parse_args(argv)
    if a.transcribe_source:
        print(transcribe_source(a.candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

