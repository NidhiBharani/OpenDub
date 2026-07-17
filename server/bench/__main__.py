"""OpenDub benchmark CLI.

    python -m bench run [case ...]          # run pipeline + score, write results/<ts>.json
    python -m bench report <run.json>       # render a run as a markdown table
    python -m bench compare <old.json> <new.json>

Run from the server/ directory (so `app` and `bench` are importable) with the venv python:
    .venv/bin/python -m bench run moshi
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import cases as cases_mod
from . import report as report_mod
from .runner import run_case

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


async def _cmd_run(args: argparse.Namespace) -> int:
    selected = args.cases or None
    cases = cases_mod.load_cases(selected)
    if not cases:
        print("no matching cases")
        return 1
    results = []
    for case in cases:
        print(f"▶ running {case.name} …", flush=True)
        res = await run_case(case, timeout=args.timeout)
        if res.error:
            print(f"  pipeline error: {res.error}")
        results.append(res)

    stamp = args.now or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{stamp}-{_git_hash()}.json"
    report_mod.save_run(results, out, meta={"timestamp": stamp, "git": _git_hash(),
                                            "cases": [c.name for c in cases]})
    print(f"\nwrote {out}\n")
    print(report_mod.render_markdown(json.loads(out.read_text())))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run = json.loads(Path(args.run).read_text())
    print(report_mod.render_markdown(run))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    old = json.loads(Path(args.old).read_text())
    new = json.loads(Path(args.new).read_text())
    print(report_mod.render_compare(old, new))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="bench", description="OpenDub pipeline benchmarking")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline over cases and score them")
    r.add_argument("cases", nargs="*", help="case names (default: all in cases.yaml)")
    r.add_argument("--timeout", type=float, default=1800.0, help="per-case timeout (s)")
    r.add_argument("--now", help=argparse.SUPPRESS)  # deterministic timestamp for tests
    r.set_defaults(func=_cmd_run, is_async=True)

    rep = sub.add_parser("report", help="render a run json as markdown")
    rep.add_argument("run")
    rep.set_defaults(func=_cmd_report, is_async=False)

    cmp = sub.add_parser("compare", help="compare two run jsons")
    cmp.add_argument("old")
    cmp.add_argument("new")
    cmp.set_defaults(func=_cmd_compare, is_async=False)

    args = p.parse_args()
    if getattr(args, "is_async", False):
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
