"""Serialize bench results to JSON and render human-readable markdown reports / comparisons."""
from __future__ import annotations

import json
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
