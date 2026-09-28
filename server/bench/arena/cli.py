"""``python -m bench arena …`` subcommands. The user orchestrates runs; nothing runs implicitly.

    arena hw                                         what this machine can run
    arena plan [A4 D1 … | all] --lang en hi ja       runnable candidates/judges here, and why not
    arena env list | setup <env> [--dry-run] | check <env>
                                                     isolated worker environments
    arena pack <builder> --lang en hi ja [--n 300]   build eval packs (builders/)
    arena run A4 [D1 …|all] --lang en hi ja [--candidates …] [--limit N] [--budget-usd 5]
                                                     generate missing outputs, judge, report
    arena score A4 --lang ja [--rescore] [--no-model-judges]
                                                     re-score cached outputs (no generation)
    arena report A4 --lang ja                        leaderboard + audit page on disk
    arena merge /path/to/other/data                  fold another machine's arena in (Spark)
    arena smoke [A4 …|all] [--setup-envs] | --app     load/unload smoke test of runnable models
    arena hf-access                                  gated HF repos vs your token (→ docs)
    arena docs                                       write docs/arena-models.md from the registry
    arena ladder prj_… [--asr-check] [--metrics]    stage-by-stage audit of one real dub run
    arena serve [--host 192.168.x.y] [--port 8765]   audit viewer with ok/bad/note marks
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from . import db, envs, hardware, judges, paths, runner
from .builders import BUILDERS
from .packs import load_pack, pack_dir
from .registry import Candidate, capabilities, load_candidates, select
from .report import audit_data, audit_html, leaderboard


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _caps(names: list[str]) -> list[str]:
    return capabilities() if names in (["all"], []) else names


def _write_report(cap: str, lang: str, cands: list[Candidate], items) -> str:
    md, rows = leaderboard(cap, lang, cands, items)
    out = paths.arena_dir() / "reports" / cap
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{lang}.md").write_text(md)
    (out / f"{lang}.json").write_text(json.dumps(rows, indent=1, default=str))
    (out / f"{lang}.html").write_text(audit_html(audit_data(cap, lang, cands, items), md))
    return md


def cmd_hw(a: argparse.Namespace) -> int:
    print(json.dumps(hardware.detect().as_dict(), indent=1))
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    hw = hardware.detect()
    print(f"# arena plan on {hw.host} ({hw.profile}: {hw.arch}, "
          f"{hw.vram_gb:g} GB GPU, {hw.ram_gb:g} GB RAM, keys: {', '.join(hw.api_keys) or '-'})\n")
    total = runnable = 0
    for cap in _caps(a.capabilities):
        cands = load_candidates(cap)
        try:
            spec = judges.get_spec(cap)
        except KeyError:
            spec = None
        print(f"## {cap}{' — ' + spec.title if spec else ' (no spec yet)'}")
        for lang in a.lang:
            has_pack = (pack_dir(cap, lang) / "manifest.jsonl").exists()
            print(f"  [{lang}] pack: {'ready' if has_pack else 'missing'}")
            for c in cands:
                if not c.supports(lang):
                    continue
                total += 1
                why = c.blockers(hw)
                runnable += not why
                mark = "✓" if not why else "✗"
                print(f"    {mark} {c.id:34} {c.kind:6} {'; '.join(why)}")
            if spec:
                for j in spec.judges_for(lang):
                    why = j.blockers()
                    print(f"    {'✓' if not why else '✗'} judge {j.id:28} {'; '.join(why)}")
        print()
    print(f"{runnable}/{total} candidate×language runs are possible on this machine")
    return 0


def cmd_env(a: argparse.Namespace) -> int:
    if a.env_cmd == "list":
        for e in envs.all_envs():
            print(f"{'✓' if envs.is_ready(e) else '·'} {e.name:24} {e.description[:90]}")
        return 0
    env = envs.load_env(a.name)
    if a.env_cmd == "setup":
        return envs.setup(env, dry_run=a.dry_run, arch=a.arch)
    if not envs.is_ready(env):
        print(f"{env.name}: not set up")
        return 1
    return _check(env)


def _check(env: envs.Env) -> int:
    import subprocess
    import sys

    python = sys.executable if env.python == "auto" else str(Path(env.python).expanduser())
    if not env.check:
        print(f"{env.name}: ready (no check defined)")
        return 0
    rc = subprocess.run([python, "-c", env.check], check=False).returncode
    print(f"{env.name}: check {'passed' if rc == 0 else 'FAILED'}")
    return rc


def cmd_pack(a: argparse.Namespace) -> int:
    b = BUILDERS[a.builder]
    for lang in a.lang:
        out = b.fn(lang, n=a.n)
        for root in out if isinstance(out, list) else [out]:
            print(f"{a.builder} {lang}: {root}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    for cap in _caps(a.capabilities):
        cands = select(load_candidates(cap), a.candidates)
        for lang in a.lang:
            try:
                items = load_pack(cap, lang, split=a.split, limit=a.limit)
            except FileNotFoundError as e:
                print(f"▷ {cap} {lang}: {e}")
                continue
            print(f"▶ {cap} {lang}: {len(items)} items × {len(cands)} candidates")
            runner.generate(cap, lang, cands, items, force=a.force, check_gpu=not a.no_gpu_check,
                            check_hardware=not a.ignore_hardware, budget_usd=a.budget_usd)
            n = judges.score(cap, lang, cands, items, run_models=not a.no_model_judges,
                             check_gpu=not a.no_gpu_check)
            print(f"  scored {n} metric rows")
            print(_write_report(cap, lang, cands, items))
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    for cap in _caps(a.capabilities):
        cands = select(load_candidates(cap), a.candidates)
        for lang in a.lang:
            items = load_pack(cap, lang, split=a.split, limit=a.limit)
            n = judges.score(cap, lang, cands, items, rescore=a.rescore,
                             run_models=not a.no_model_judges, check_gpu=not a.no_gpu_check)
            print(f"{cap} {lang}: {n} metric rows written")
            print(_write_report(cap, lang, cands, items))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    for cap in _caps(a.capabilities):
        cands = select(load_candidates(cap), a.candidates)
        for lang in a.lang:
            items = load_pack(cap, lang, split=a.split, limit=a.limit)
            print(_write_report(cap, lang, cands, items))
    print(f"reports in {paths.arena_dir() / 'reports'}")
    return 0


def cmd_merge(a: argparse.Namespace) -> int:
    counts = db.merge(Path(a.other))
    print("merged: " + ", ".join(f"{k} +{v}" for k, v in counts.items()))
    return 0


def cmd_smoke(a: argparse.Namespace) -> int:
    from . import smoke

    if a.app:
        smoke.smoke_app()
    else:
        recs = smoke.smoke_all(_caps(a.capabilities), a.candidates, setup_envs=a.setup_envs,
                               include_api=a.include_api, min_free_disk_gb=a.min_free_disk_gb,
                               rerun=a.rerun, evict_weights=a.evict_weights)
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        print("smoke: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if not a.no_docs:
        from .docs import write

        print(f"docs: {write()}")
    return 0


def cmd_hf_access(a: argparse.Namespace) -> int:
    from .hfaccess import scan

    scan()
    if not a.no_docs:
        from .docs import write

        print(f"docs: {write()}")
    return 0


def cmd_docs(a: argparse.Namespace) -> int:
    from .docs import write

    print(write(Path(a.out) if a.out else None))
    return 0


def cmd_ladder(a: argparse.Namespace) -> int:
    from . import ladder

    for pid in a.projects:
        out = ladder.write(pid, asr_check=a.asr_check, judge=a.metrics)
        print(f"{pid}: {out}  (served at /ladder/{pid})")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .serve import serve

    serve(a.host or _lan_ip(), a.port)
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    ap = sub.add_parser("arena", help="rank candidate models per capability and language")
    asub = ap.add_subparsers(dest="arena_cmd", required=True)

    asub.add_parser("hw", help="print detected hardware").set_defaults(func=cmd_hw,
                                                                       is_async=False)

    pl = asub.add_parser("plan", help="what can run on this machine")
    pl.add_argument("capabilities", nargs="*", default=["all"])
    pl.add_argument("--lang", nargs="+", default=["en", "hi", "ja"])
    pl.set_defaults(func=cmd_plan, is_async=False)

    ev = asub.add_parser("env", help="isolated worker environments")
    esub = ev.add_subparsers(dest="env_cmd", required=True)
    esub.add_parser("list").set_defaults(func=cmd_env, is_async=False)
    es = esub.add_parser("setup")
    es.add_argument("name")
    es.add_argument("--dry-run", action="store_true", help="print the commands only")
    es.add_argument("--arch", help="recipe to use (default: this machine's)")
    es.set_defaults(func=cmd_env, is_async=False)
    ec = esub.add_parser("check")
    ec.add_argument("name")
    ec.set_defaults(func=cmd_env, is_async=False)

    p = asub.add_parser("pack", help="build eval packs")
    p.add_argument("builder", choices=sorted(BUILDERS))
    p.add_argument("--lang", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300)
    p.set_defaults(func=cmd_pack, is_async=False)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("capabilities", nargs="+", help="capability ids, or 'all'")
        q.add_argument("--lang", nargs="+", required=True)
        q.add_argument("--candidates", nargs="*")
        q.add_argument("--limit", type=int)
        q.add_argument("--split", default="test")

    r = asub.add_parser("run", help="generate missing outputs, judge them, write reports")
    common(r)
    r.add_argument("--force", action="store_true", help="regenerate even cached outputs")
    r.add_argument("--no-gpu-check", action="store_true")
    r.add_argument("--ignore-hardware", action="store_true",
                   help="run candidates even if `requires` says this machine is too small")
    r.add_argument("--no-model-judges", action="store_true")
    r.add_argument("--budget-usd", type=float, help="per-job spend cap for API candidates")
    r.set_defaults(func=cmd_run, is_async=False)

    s = asub.add_parser("score", help="score cached outputs without generating")
    common(s)
    s.add_argument("--rescore", action="store_true")
    s.add_argument("--no-model-judges", action="store_true")
    s.add_argument("--no-gpu-check", action="store_true")
    s.set_defaults(func=cmd_score, is_async=False)

    rep = asub.add_parser("report", help="write leaderboard + audit page")
    common(rep)
    rep.set_defaults(func=cmd_report, is_async=False)

    mg = asub.add_parser("merge", help="merge another machine's data/arena into this one")
    mg.add_argument("other", help="the other machine's data dir (containing arena/)")
    mg.set_defaults(func=cmd_merge, is_async=False)

    sm = asub.add_parser("smoke", help="load/unload smoke test of runnable models")
    sm.add_argument("capabilities", nargs="*", default=["all"])
    sm.add_argument("--candidates", nargs="*")
    sm.add_argument("--app", action="store_true",
                    help="test the server's in-process models via the model manager instead")
    sm.add_argument("--setup-envs", action="store_true",
                    help="create missing worker envs from their recipes first")
    sm.add_argument("--include-api", action="store_true", help="also call hosted APIs (costs)")
    sm.add_argument("--min-free-disk-gb", type=float, default=15.0)
    sm.add_argument("--rerun", action="store_true", help="re-test candidates already ok here")
    sm.add_argument("--evict-weights", action="store_true",
                    help="delete weights each test downloaded (keeps previously cached ones)")
    sm.add_argument("--no-docs", action="store_true", help="don't regenerate the docs after")
    sm.set_defaults(func=cmd_smoke, is_async=False)

    hfa = asub.add_parser("hf-access", help="check gated Hugging Face repos against your token")
    hfa.add_argument("--no-docs", action="store_true")
    hfa.set_defaults(func=cmd_hf_access, is_async=False)

    dc = asub.add_parser("docs", help="write docs/arena-models.md from the registry")
    dc.add_argument("--out")
    dc.set_defaults(func=cmd_docs, is_async=False)

    lad = asub.add_parser("ladder", help="stage-by-stage audit page for dub projects")
    lad.add_argument("projects", nargs="+")
    lad.add_argument("--asr-check", action="store_true",
                     help="re-transcribe every take and fitted clip (round-trip CER)")
    lad.add_argument("--metrics", action="store_true",
                     help="add whole-pipeline bench metrics per stage (runs the LLM judge)")
    lad.set_defaults(func=cmd_ladder, is_async=False)

    sv = asub.add_parser("serve", help="serve the audit viewer on the LAN")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(func=cmd_serve, is_async=False)
