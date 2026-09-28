"""Translate-stage metrics.

Reference-free: time-budget compliance (reusing the app's own `max_chars_for`) and an LLM judge
(adequacy + fluency, 1-5) that reuses the project's configured Ollama endpoint. Ground truth
(when ref_translation.jsonl exists): chrF via sacrebleu.
"""
from __future__ import annotations

import json

import httpx

from app import config
from app.providers.translation._llm import max_chars_for

from .base import Metric, MetricContext, load_jsonl, missing

_JUDGE_SYSTEM = (
    "You are a strict bilingual dubbing-translation judge. For each item you are given the source "
    "line and its English translation. Score two axes from 1 (terrible) to 5 (perfect):\n"
    "- adequacy: does the translation preserve the source meaning?\n"
    "- fluency: is the English natural, colloquial spoken dialogue?\n"
    "Return ONLY a JSON array of objects {\"n\":<int>,\"adequacy\":<1-5>,\"fluency\":<1-5>}, "
    "one per item, no prose."
)


async def score(ctx: MetricContext) -> list[Metric]:
    segs = [s for s in ctx.project.segments if s.translated_text.strip()]
    out: list[Metric] = []
    if not segs:
        return [missing("translate.budget_compliance", "no translated segments")]

    # --- reference-free: time-budget compliance ---
    within = 0
    overshoot = 0.0
    for s in segs:
        budget = max_chars_for(s.duration)
        length = len(s.translated_text)
        if length <= budget:
            within += 1
        else:
            overshoot += (length - budget) / budget
    out.append(
        Metric("translate.budget_compliance", round(within / len(segs), 3), "ratio", "free",
               higher_is_better=True,
               note=f"{within}/{len(segs)} lines within their ~15 char/s spoken-time budget")
    )
    out.append(
        Metric("translate.mean_overshoot", round(overshoot / len(segs), 3), "ratio", "free",
               higher_is_better=False, note="mean fractional overrun of over-budget lines")
    )

    # --- reference-free: LLM judge (reuses the configured Ollama endpoint) ---
    out.extend(await _judge(ctx, segs))

    # --- ground truth: chrF ---
    out.append(await _chrf(ctx, segs))
    return out


async def _judge(ctx: MetricContext, segs) -> list[Metric]:
    opts = config.provider_options("translation.openai_compatible")
    base_url = str(opts.get("base_url") or "").rstrip("/")
    model = opts.get("model")
    if not base_url or not model:
        return [missing("translate.judge_adequacy", "no openai_compatible endpoint configured",
                        unit="1-5")]
    items = [{"n": i, "source": s.source_text, "translation": s.translated_text}
             for i, s in enumerate(segs)]
    headers = {"Authorization": f"Bearer {opts['api_key']}"} if opts.get("api_key") else {}
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model, "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
                    ],
                },
            )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        scores = _parse_scores(content)
    except Exception as e:  # noqa: BLE001 - network / parse: report as uncomputed, never crash
        return [missing("translate.judge_adequacy", f"judge call failed: {str(e)[:80]}", unit="1-5")]

    if not scores:
        return [missing("translate.judge_adequacy", "judge returned no parseable scores", unit="1-5")]
    adeq = [x["adequacy"] for x in scores if "adequacy" in x]
    flu = [x["fluency"] for x in scores if "fluency" in x]
    res = []
    if adeq:
        res.append(Metric("translate.judge_adequacy", round(sum(adeq) / len(adeq), 2), "1-5",
                          "free", higher_is_better=True, note=f"LLM judge ({model}), n={len(adeq)}"))
    if flu:
        res.append(Metric("translate.judge_fluency", round(sum(flu) / len(flu), 2), "1-5",
                          "free", higher_is_better=True, note=f"LLM judge ({model}), n={len(flu)}"))
    return res or [missing("translate.judge_adequacy", "judge scores empty", unit="1-5")]


def _parse_scores(content: str) -> list[dict]:
    content = content.strip()
    if content.startswith("```"):  # strip markdown fences
        content = content.split("```")[1] if "```" in content[3:] else content.strip("`")
        content = content.removeprefix("json").strip()
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(content[start : end + 1])
        return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        return []


async def _chrf(ctx: MetricContext, segs) -> Metric:
    if not ctx.case.has("ref_translation"):
        return missing("translate.chrf", "no ref_translation.jsonl", provenance="gt")
    try:
        import sacrebleu
    except ImportError:
        return missing("translate.chrf", "sacrebleu not installed (bench extra)", provenance="gt")
    refs = [r.get("text", "") for r in load_jsonl(ctx.case.ref_translation)]
    hyps = [s.translated_text for s in sorted(segs, key=lambda s: s.start)]
    if len(refs) != len(hyps):
        return missing("translate.chrf",
                       f"ref/hyp length mismatch ({len(refs)} vs {len(hyps)})", provenance="gt")
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return Metric("translate.chrf", round(chrf, 2), "0-100", "gt", higher_is_better=True,
                  note="sacrebleu chrF vs ref_translation")
