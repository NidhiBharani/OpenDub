"""Serialize bench results to JSON and render human-readable markdown reports / comparisons."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .runner import CaseResult


def result_to_dict(r: CaseResult) -> dict[str, Any]:
    return {
        "case": r.case,
        "project_id": r.project_id,
        "error": r.error,
        "metrics": [m.as_dict() for m in r.metrics],
    }


def save_run(results: list[CaseResult], out_path: Path, meta: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"meta": meta, "results": [result_to_dict(r) for r in results]}, indent=2
    ))


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value}"


def render_markdown(run: dict[str, Any]) -> str:
    lines = ["# OpenDub benchmark report", ""]
    meta = run.get("meta", {})
    if meta:
        lines.append("- " + " · ".join(f"{k}: {v}" for k, v in meta.items()))
        lines.append("")
    for res in run["results"]:
        lines.append(f"## {res['case']}  (`{res['project_id']}`)")
        if res.get("error"):
            lines.append(f"> pipeline error: `{res['error']}`")
        lines.append("")
        lines.append("| metric | value | unit | src | good | note |")
        lines.append("|---|---:|---|---|---|---|")
        for m in res["metrics"]:
            good = {True: "↑", False: "↓", None: ""}[m.get("higher_is_better")]
            lines.append(
                f"| {m['name']} | {_fmt(m['value'])} | {m.get('unit','')} | "
                f"{m.get('provenance','')} | {good} | {m.get('note','')} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Per-metric deltas between two runs, flagging regressions by higher_is_better."""
    def index(run: dict) -> dict[tuple[str, str], dict]:
        out = {}
        for res in run["results"]:
            for m in res["metrics"]:
                out[(res["case"], m["name"])] = m
        return out

    oi, ni = index(old), index(new)
    lines = ["# Benchmark comparison", "", "| case | metric | old | new | Δ | flag |",
             "|---|---|---:|---:|---:|---|"]
    for key in sorted(ni):
        nm = ni[key]
        om = oi.get(key)
        ov, nv = (om or {}).get("value"), nm.get("value")
        if not isinstance(ov, (int, float)) or not isinstance(nv, (int, float)):
            continue
        delta = nv - ov
        flag = ""
        hib = nm.get("higher_is_better")
        if hib is not None and abs(delta) > 1e-9:
            improved = (delta > 0) == hib
            flag = "✅ better" if improved else "🔴 regressed"
        lines.append(
            f"| {key[0]} | {key[1]} | {ov} | {nv} | {delta:+.3g} | {flag} |"
        )
    return "\n".join(lines)


def regression_gate(old: dict[str, Any], new: dict[str, Any],
                    max_regression: float = 0.0,
                    required: list[str] | None = None) -> list[str]:
    """Return gate failures for case errors, lost measurements, and directional regressions.

    ``max_regression`` is an absolute allowance in each metric's unit. Use ``required``
    for metrics that must exist even when the baseline had no value (e.g. optional ASR).
    """
    if max_regression < 0:
        raise ValueError('max_regression must be nonnegative')
    required = required or []
    baseline = {r['case']: r for r in old.get('results', [])}
    current = {r['case']: r for r in new.get('results', [])}
    failures: list[str] = []
    for case, before in baseline.items():
        after = current.get(case)
        if after is None:
            failures.append(f'{case}: case missing from new run')
            continue
        if after.get('error'):
            failures.append(f"{case}: pipeline error: {after['error']}")
        old_metrics = {m['name']: m for m in before.get('metrics', [])}
        new_metrics = {m['name']: m for m in after.get('metrics', [])}
        for name, metric in new_metrics.items():
            if name.endswith('.error'):
                failures.append(f"{case}/{name}: {metric.get('note', 'metric module failed')}")
        for name, previous in old_metrics.items():
            old_value = previous.get('value')
            if not isinstance(old_value, (int, float)) or not math.isfinite(old_value):
                continue
            latest = new_metrics.get(name)
            value = latest.get('value') if latest else None
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f'{case}/{name}: measurable baseline became unavailable')
                continue
            direction = latest.get('higher_is_better')
            if direction is None:
                direction = previous.get('higher_is_better')
            if ((direction is True and value < old_value - max_regression) or
                    (direction is False and value > old_value + max_regression)):
                failures.append(f'{case}/{name}: {old_value} -> {value} regressed')
    for case, after in current.items():
        if case not in baseline and after.get('error'):
            failures.append(f"{case}: pipeline error: {after['error']}")
        metrics = {m['name']: m for m in after.get('metrics', [])}
        if not metrics:
            failures.append(f'{case}: no metrics in new run')
        if case not in baseline:
            for name, metric in metrics.items():
                if name.endswith('.error'):
                    failures.append(f"{case}/{name}: {metric.get('note', 'metric module failed')}")
        for name in required:
            value = (metrics.get(name) or {}).get('value')
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append(f'{case}/{name}: required metric unavailable')
    return failures
