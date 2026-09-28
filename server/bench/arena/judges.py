"""Judges turn cached outputs into per-item metric rows; each capability's spec says how it ranks.

Two kinds of judge:

- **Function judges** ``fn(item, output_payload, output_row) -> [(metric, value, weight)]`` run
  in the bench process: text metrics, durations, loudness, anything numpy can do.
- **Model judges** (:class:`ModelJudge`) are models themselves (ASR round-trip, speaker
  similarity, MOS predictors, emotion, lip-sync, MT QE). They run as isolated workers exactly like
  candidates — one process per job, GPU memory freed on exit — and their outputs are cached under
  ``data/arena/outputs/_judges/<judge key>/`` by the candidate output they judged, so adding a
  candidate only judges the new outputs and changing a judge never regenerates candidates.

``weight`` makes aggregation a weighted mean, so corpus-level rates (total edits / total reference
words) stay right and resampleable per group. Judge ids carry a version (``…@3``): bump it when a
judge changes and every output is re-scored on the next ``score`` run.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.pipeline.quality import _distance

from . import db, paths
from .hardware import Requires, check
from .packs import Item
from .registry import Candidate, env_python

Row = tuple[str, float, float]
Judge = Callable[[Item, dict[str, Any], Any], list[Row]]

# Languages written without spaces between words: WER is meaningless, rank by CER.
UNSPACED = {"ja", "zh", "th", "lo", "km", "my"}


# ------------------------------------------------------------------ shared text helpers

def normalize_text(text: str) -> str:
    """NFKC + casefold, punctuation to spaces, marks (Indic matras, virama) kept."""
    clean = "".join(" " if unicodedata.category(ch).startswith("P") else ch
                    for ch in unicodedata.normalize("NFKC", text).casefold())
    return " ".join(clean.split())


_EN_NORMALIZER: Callable[[str], str] | None = None


def english_normalizer() -> Callable[[str], str]:
    """Whisper's EnglishTextNormalizer (spelling variants, numbers, contractions): the standard
    for English WER (Open ASR Leaderboard), so US/UK spelling or "20" vs "twenty" is not an error.
    """
    global _EN_NORMALIZER
    if _EN_NORMALIZER is None:
        from huggingface_hub import hf_hub_download
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        mapping = json.loads(Path(hf_hub_download("openai/whisper-large-v3", "normalizer.json"))
                             .read_text())
        _EN_NORMALIZER = EnglishTextNormalizer(mapping)
    return _EN_NORMALIZER


def normalize_for(lang: str, text: str) -> str:
    if lang == "en":
        # Whisper's normaliser deletes (…) and […] as non-speech annotations, but read-speech
        # references (FLEURS) put spoken words in parentheses: keep their content.
        text = re.sub(r"[()\[\]{}]", " ", text)
        return normalize_text(english_normalizer()(text))
    return normalize_text(text)


def error_rows(lang: str, reference: str, hypothesis: str, prefix: str = "") -> list[Row]:
    """CER (+ WER for spaced languages) and an empty-output flag, weighted by reference length."""
    ref = normalize_for(lang, reference)
    hyp = normalize_for(lang, hypothesis)
    if not ref:
        return []
    rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
    rows: list[Row] = [(f"{prefix}cer", _distance(list(rc), list(hc)) / len(rc), float(len(rc)))]
    if lang not in UNSPACED:
        rw = ref.split()
        rows.append((f"{prefix}wer", _distance(rw, hyp.split()) / len(rw), float(len(rw))))
    # Empty output on a non-empty reference: a dropped line, the costliest failure in a dub.
    rows.append((f"{prefix}empty_output", float(not hc), 1.0))
    return rows


def text_errors(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    return error_rows(item.lang, str(item.refs.get("text", "")), str(out.get("text", "")))


def speed(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    secs = row["seconds"] if row is not None else None
    dur = item.meta.get("duration_s") or out.get("duration_s")
    if not secs or not dur:
        return []
    # weight = seconds, so the weighted mean is total audio / total compute (corpus RTFx).
    return [("rtfx", float(dur) / float(secs), float(secs))]


def cost(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    """API spend per item (weight 1, so the mean is $/item)."""
    c = row["cost_usd"] if row is not None and "cost_usd" in row.keys() else None  # noqa: SIM118 - sqlite3.Row `in` checks values
    return [("cost_usd", float(c), 1.0)] if c is not None else []


def output_file(out: dict[str, Any], key: str = "audio") -> str | None:
    """Path of a file a worker produced (``payload["files"][key]``), if it exists."""
    p = (out.get("files") or {}).get(key)
    return p if p and Path(p).exists() else None


# ------------------------------------------------------------------ specs

@dataclass
class ModelJudge:
    """A model that scores candidate outputs, run as an isolated worker.

    ``inputs(item, output_payload, output_prefix)`` builds the worker item inputs, or returns
    None when this judge does not apply to the item (e.g. no reference voice). ``to_rows(item,
    judge_payload, output_payload)`` turns the worker's JSON into metric rows.
    """

    id: str                            # versioned, e.g. "asr.whisper-large-v3@1"
    worker: str
    inputs: Callable[[Item, dict[str, Any], Path], dict[str, Any] | None]
    to_rows: Callable[[Item, dict[str, Any], dict[str, Any]], list[Row]]
    env: str = "server"
    params: dict[str, Any] = field(default_factory=dict)
    requires: Requires = field(default_factory=Requires)
    languages: list[str] | str = "*"
    lang_param: bool = True            # pass the pack language to the worker

    @property
    def key(self) -> str:
        blob = json.dumps({"id": self.id, "worker": self.worker, "params": self.params},
                          sort_keys=True)
        return f"{self.id.replace('@', '_')}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}"

    def supports(self, lang: str) -> bool:
        return self.languages == "*" or lang in self.languages

    def blockers(self) -> list[str]:
        why = check(self.requires)
        try:
            env_python(self.env)
        except (KeyError, FileNotFoundError):
            why.append(f"env {self.env!r} not set up")
        if not (paths.WORKERS_DIR / f"{self.worker}.py").exists():
            why.append(f"worker {self.worker}.py missing")
        return why


@dataclass
class CorpusMetric:
    """A corpus-level metric (correlation with human ratings, pairwise accuracy) over per-item
    predictions: ranks judges (G*) and QE models (C4) by agreement with humans.

    ``pred`` is the per-item metric a judge wrote; ``target(item)`` the human value (e.g.
    ``item.refs["mos"]``). ``fn`` is "spearman" | "pearson" | "kendall" | "pairwise_acc" or a
    callable (preds, targets) -> float. Bootstrapped over item groups like everything else.
    """

    pred: str
    target: Callable[[Item], float | None]
    fn: str | Callable[[list[float], list[float]], float] = "spearman"


@dataclass
class Spec:
    """How one capability is evaluated and ranked (``bench/arena/specs/<ID>.py``)."""

    id: str
    title: str
    judges: dict[str, Judge]
    primary: dict[str, str]                  # lang -> metric ("*" = default)
    higher_is_better: dict[str, bool]        # metric -> direction
    threshold: dict[str, float] = field(default_factory=dict)  # practical difference
    secondary: list[str] = field(default_factory=list)
    model_judges: list[ModelJudge] | Callable[[str], list[ModelJudge]] = field(
        default_factory=list)                # or a function of the language
    gates: dict[str, tuple[str, float]] = field(default_factory=dict)  # metric -> ("<=", 0.2)
    packs: list[str] = field(default_factory=list)       # builder names that feed this spec
    io: str = ""                             # the item inputs / worker payload contract
    # Per-output metrics computed from other judges' rows, e.g. the ASR-panel mean round-trip
    # CER: name -> fn({metric: (value, weight)}) -> (value, weight) | None.
    derived: dict[str, Callable[[dict[str, tuple[float, float]]], tuple[float, float] | None]] = \
        field(default_factory=dict)
    derived_version: int = 1
    corpus: dict[str, CorpusMetric] = field(default_factory=dict)  # name -> corpus metric

    def primary_for(self, lang: str) -> str:
        return self.primary.get(lang, self.primary["*"])

    def judges_for(self, lang: str) -> list[ModelJudge]:
        mj = self.model_judges(lang) if callable(self.model_judges) else self.model_judges
        return [j for j in mj if j.supports(lang)]

    def judge_ids(self, lang: str) -> set[str]:
        ids = set(self.judges) | {j.id for j in self.judges_for(lang)}
        return ids | ({f"derived@{self.derived_version}"} if self.derived else set())


def mean_of(prefix: str, suffix: str):
    """Derived metric: weighted mean of every ``<prefix>*<suffix>`` metric (e.g. the ASR panel's
    ``rt_*_cer``), so the rank does not hinge on one judge family."""
    def fn(values: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
        hits = [(v, w) for k, (v, w) in values.items()
                if k.startswith(prefix) and k.endswith(suffix)]
        if not hits:
            return None
        return (sum(v for v, _ in hits) / len(hits), sum(w for _, w in hits) / len(hits))
    return fn


def get_spec(capability: str) -> Spec:
    from .specs import SPECS

    if capability not in SPECS:
        raise KeyError(f"no spec for {capability} (add bench/arena/specs/{capability}.py)")
    return SPECS[capability]


# ------------------------------------------------------------------ scoring

def _judge_prefix(judge: ModelJudge, output_key: str) -> Path:
    return paths.outputs_dir() / "_judges" / judge.key / output_key


def score(capability: str, lang: str, candidates: list[Candidate], items: list[Item], *,
          rescore: bool = False, run_models: bool = True, check_gpu: bool = True,
          log: Callable[[str], None] = print) -> int:
    """Score every cached ok output that lacks a row for the current judge versions."""
    from .runner import ensure_gpu_room, output_key, run_worker_job

    spec = get_spec(capability)
    con = db.connect()
    written = 0
    ok_outputs: list[tuple[Candidate, Item, str, Any, dict[str, Any]]] = []
    for cand in candidates:
        for it in items:
            key = output_key(cand, it)
            row = db.get_output(con, key)
            if row is None or row["status"] != "ok":
                continue
            out_path = db.output_path(row).with_suffix(".json")
            if out_path.exists():
                ok_outputs.append((cand, it, key, row, json.loads(out_path.read_text())))

    def put(cand: Candidate, it: Item, key: str, judge_id: str, rows: list[Row]) -> int:
        db.put_scores(con, [{"output_key": key, "judge": judge_id, "metric": m,
                             "ref_hash": it.ref_hash(), "capability": capability, "lang": lang,
                             "cand_key": cand.key, "item": it.id, "grp": it.group, "value": v,
                             "weight": w} for (m, v, w) in rows])
        return len(rows)

    for cand, it, key, row, out in ok_outputs:
        for judge_id, judge in spec.judges.items():
            if rescore or not db.has_score(con, key, judge_id, it.ref_hash()):
                written += put(cand, it, key, judge_id, judge(it, out, row))

    for mj in spec.judges_for(lang):
        todo, job_items = [], []
        for cand, it, key, row, out in ok_outputs:
            if not rescore and db.has_score(con, key, mj.id, it.ref_hash()):
                continue
            prefix = _judge_prefix(mj, key)
            cached = prefix.with_suffix(".json")
            if cached.exists():
                todo.append((cand, it, key, out, cached))
                continue
            inputs = mj.inputs(it, out, db.output_path(row))
            if inputs is None:
                continue
            todo.append((cand, it, key, out, cached))
            job_items.append({"id": key, "inputs": inputs, "meta": it.meta, "out": str(prefix)})
        if job_items:
            blockers = mj.blockers()
            if blockers or not run_models:
                log(f"  - judge {mj.id}: {len(job_items)} outputs unjudged "
                    f"({'; '.join(blockers) or 'model judges disabled'})")
            else:
                if check_gpu:
                    ensure_gpu_room(f"judge {mj.id}", mj.requires.vram_gb)
                run_worker_job(worker=mj.worker, env=mj.env, params=mj.params,
                               lang=lang if mj.lang_param else "", items=job_items,
                               label=f"judge-{capability}-{lang}-{mj.key}", log=log)
        for cand, it, key, out, cached in todo:
            if cached.exists():
                written += put(cand, it, key, mj.id,
                               mj.to_rows(it, json.loads(cached.read_text()), out))
    if spec.derived:
        judge_id = f"derived@{spec.derived_version}"
        base = spec.judge_ids(lang) - {judge_id}
        for cand, it, key, _row, _out in ok_outputs:
            if not rescore and db.has_score(con, key, judge_id, it.ref_hash()):
                continue
            values = {r["metric"]: (r["value"], r["weight"]) for r in con.execute(
                "SELECT metric, value, weight, judge FROM scores WHERE output_key=?", (key,))
                if r["judge"] in base}
            rows = [(name, *res) for name, fn in spec.derived.items()
                    if (res := fn(values)) is not None]
            if rows:
                written += put(cand, it, key, judge_id, rows)
    con.commit()
    con.close()
    return written
