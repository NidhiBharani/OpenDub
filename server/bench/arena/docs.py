"""Generate ``docs/arena-models.md`` from the registry: every benchmarked model, by capability, with
kind (local / API / method), size, licence, languages, hardware needs, where it runs, where to get
it, and the load/unload smoke-test results recorded on each machine.

    python -m bench arena docs [--out docs/arena-models.md]
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from . import hardware, paths
from .hardware import Gpu, Hardware, check
from .registry import Candidate, capabilities, load_candidates

# Reference machines the registry is written for (hardware.check ignores API keys here: a key is
# something you add, not hardware you lack).
MACHINES = {
    "Thalassa": Hardware(host="thalassa", arch="x86_64", ram_gb=24, disk_free_gb=90,
                         gpus=[Gpu("RTX 5060 Ti", 15.9, "12.0")],
                         tools=["docker", "ffmpeg", "ffprobe", "uv", "git", "yt-dlp"]),
    "DGX Spark": Hardware(host="spark", arch="aarch64", ram_gb=128, disk_free_gb=3500,
                          gpus=[Gpu("NVIDIA GB10", 112, "12.1", unified=True)],
                          tools=["docker", "ffmpeg", "ffprobe", "uv", "git", "nvcc"]),
}

PHASES = {"A": "Source analysis · audio", "B": "Source analysis · picture", "C": "Text",
          "D": "Voice", "E": "Audio post-production", "F": "Picture out", "G": "Judges"}
L1 = ("en", "hi", "ja")


def _title(cap: str) -> str:
    try:
        from .judges import get_spec

        return get_spec(cap).title
    except KeyError:
        return ""


def _langs(c: Candidate) -> str:
    if c.languages == "*":
        return "all (✓ en hi ja)"
    langs = list(c.languages)
    have = [lang for lang in L1 if any(lang == x or x.endswith(f"-{lang}") or
                                       x.startswith(f"{lang}-") for x in langs)]
    extra = len([x for x in langs if x not in L1])
    return (" ".join(have) or "—") + (f" +{extra}" if extra else "")


def _needs(c: Candidate) -> str:
    r = c.requires
    bits = []
    if r.vram_gb:
        bits.append(f"{r.vram_gb:g} GB VRAM")
    elif r.gpu:
        bits.append("GPU")
    if r.ram_gb:
        bits.append(f"{r.ram_gb:g} GB RAM")
    if r.arch:
        bits.append("/".join(r.arch))
    if r.min_compute_capability:
        bits.append(f"SM ≥ {r.min_compute_capability}")
    if r.api_keys:
        bits.append("key " + ", ".join(k.replace("_API_KEY", "").replace("_KEY", "")
                                       for k in r.api_keys))
    if r.tools:
        bits.append("+" + ",".join(r.tools))
    return ", ".join(bits) or "CPU"


def _fits(c: Candidate, hw: Hardware) -> str:
    if not c.enabled:
        return "—"
    why = [w for w in check(c.requires, hw) if not w.startswith("missing API key")]
    return "✓" if not why else "✗"


def _get(c: Candidate) -> str:
    url = c.url
    model = c.params.get("model") if isinstance(c.params.get("model"), str) else None
    if not url and model and "/" in model and not model.startswith(("/", "~", "hf://")):
        url = f"https://huggingface.co/{model}"
    if not url:
        return "—"
    label = url.split("://", 1)[-1].removeprefix("www.").removeprefix("huggingface.co/")
    label = label if len(label) <= 48 else label[:45] + "…"
    return f"[{label}]({url})"


def _lic(c: Candidate) -> str:
    return (c.license or "?") + (" · NC" if c.ship_ok is False else "")


def _smoke_cell(rec: dict[str, Any] | None) -> str:
    if not rec:
        return "not run"
    s = rec["status"]
    if s == "ok":
        peak = rec.get("peak_vram_mib")
        return "✓ " + (f"{peak / 1024:.1f} GB" if peak else "loaded") + \
            (f", {rec['load_s']:.0f}s load" if rec.get("load_s") else "")
    return {"skipped": "skipped", "env_missing": "env not set up"}.get(s, f"✗ {s}")


def _esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _gated_section(rows: list[dict[str, Any]]) -> list[str]:
    gated = [r for r in rows if r.get("gated")]
    if not gated:
        return [("No gated repos recorded. Run `python -m bench arena hf-access` (needs "
                "`hf auth login`) to check."), ""]

    def fits(r: dict[str, Any], machine: str) -> bool:
        return any(c.get(machine) for c in r["candidates"])

    gated.sort(key=lambda r: (r["access"] != "needs-agree", not fits(r, "Thalassa"), r["repo"]))
    need = sum(r["access"] == "needs-agree" for r in gated)
    out = [(f"{len(gated)} gated Hugging Face repos are used by enabled candidates; "
           f"**{need} still need you to click “Agree”** on the repo page while logged in as "
           "the token's account (a one-time licence acceptance; there is no API for it). "
           "Log in once with `server/.venv/bin/hf auth login`; every worker env reads the "
           "saved token. Re-check with `python -m bench arena hf-access`, then regenerate "
           "these docs."), "",
           "| ☐ | Repo | Your access | Needed by | Thalassa | Spark | Gate |",
           "|---|---|---|---|---|---|---|"]
    for r in gated:
        users = sorted({f"{c['capability']} {c['candidate']}" for c in r["candidates"]})
        shown = ", ".join(users[:4]) + (f" +{len(users) - 4}" if len(users) > 4 else "")
        access = {"ok": "✓ granted", "needs-agree": "**click Agree**"}.get(r["access"],
                                                                         r["access"])
        box = "☑" if r["access"] == "ok" else "☐"
        url = f"https://huggingface.co/{'datasets/' if r.get('type') == 'dataset' else ''}" \
              f"{r['repo']}"
        out.append(f"| {box} | [{r['repo']}]({url}) | {access} | {_esc(shown)} | "
                   f"{'✓' if fits(r, 'Thalassa') else '—'} | "
                   f"{'✓' if fits(r, 'DGX Spark') else '—'} | {r['gated']} |")
    missing = [r for r in rows if r.get("type") is None]
    if missing:
        out += ["", (f"{len(missing)} referenced ids are not Hugging Face repos (GitHub / "
                "torch.hub / ModelScope ids or paths inside a cloned repo) and need no token.")]
    return out + [""]


def render(host_smoke: dict[tuple[str, str], dict[str, Any]] | None = None,
           app_smoke: list[dict[str, Any]] | None = None,
           smoke_rows: list[dict[str, Any]] | None = None) -> str:
    host_smoke = host_smoke or {}
    here = hardware.detect()
    from .hfaccess import load as load_access

    access_rows = load_access()
    gated_by_key: dict[str, list[dict[str, Any]]] = {}
    for r in access_rows:
        if r.get("gated"):
            for c in r["candidates"]:
                gated_by_key.setdefault(c["cand_key"], []).append(r)
    caps = [c for c in capabilities() if c[0] in PHASES]
    by_cap, broken = {}, []
    for c in caps:
        try:
            by_cap[c] = load_candidates(c)
        except Exception as exc:  # noqa: BLE001 - one bad YAML must not hide the rest
            broken.append(f"{c}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    caps = [c for c in caps if c in by_cap]
    total = sum(len(v) for v in by_cap.values())
    n_local = sum(c.kind == "local" for v in by_cap.values() for c in v)
    n_api = sum(c.kind == "api" for v in by_cap.values() for c in v)
    n_method = sum(c.kind == "method" for v in by_cap.values() for c in v)
    n_off = sum(not c.enabled for v in by_cap.values() for c in v)

    out = [
        "# Models covered by the benchmark",
        "",
        (f"Generated {dt.datetime.now(dt.UTC).date().isoformat()} by `python -m bench arena docs` from "
        "`server/bench/candidates/*.yaml` — edit the YAML, not this file."),
        "",
        (f"**{total} candidates** across {len(caps)} capabilities: {n_local} local models, "
        f"{n_api} hosted APIs, {n_method} methods (compositions of other models); "
        f"{n_off} registered but disabled (no released weights, gated, deferred, or a product "
        "without an API — the notes say which)."),
        "",
        "How to read the columns:",
        "",
        ("- **Kind** — `local` runs on your GPU/CPU in its own environment; `api` is a hosted "
        "service (needs a key, costs money per item); `method` composes other candidates."),
        ("- **Needs** — what the model requires to run (`requires` in the YAML): peak VRAM, "
        "CPU architecture, API key, tools."),
        ("- **Thalassa / Spark** — whether the hardware can run it: Thalassa = RTX 5060 Ti 16 GB, "
        "x86_64, 24 GB RAM; DGX Spark = GB10, aarch64, ~112 GB unified memory. API keys are "
        "not counted as hardware."),
        ("- **Licence** — `NC` = non-commercial weights (allowed as a default for now, by "
        "decision of 2026-09-27)."),
        (f"- **Smoke ({here.host})** — load/unload smoke test on this machine "
        "(`python -m bench arena smoke`): ✓ = loaded, produced an output, and gave all GPU "
        "memory back when it exited."),
        "- **Get it** — model weights page, repository, or API documentation.",
        ("- 🔒 — uses a **gated Hugging Face repo**: needs `hf auth login` plus a one-time "
        "“Agree” on the repo page (see [Hugging Face access](#hugging-face-access)); "
        "🔓 = gated but your token already has access."),
        "",
        *([f"> **Not included — invalid candidate file(s):** {'; '.join(broken)}", ""]
          if broken else []),
        "## Summary",
        "",
        ("| Capability | Candidates | Local | API | Method | Disabled | Fit Thalassa | Fit Spark "
        f"| Smoke ✓ ({here.host}) |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cap in caps:
        cs = by_cap[cap]
        ok = sum(1 for c in cs if (host_smoke.get((cap, c.key)) or {}).get("status") == "ok")
        out.append(
            f"| [{cap}](#{cap.lower()}) {_esc(_title(cap))} | {len(cs)} | "
            f"{sum(c.kind == 'local' for c in cs)} | {sum(c.kind == 'api' for c in cs)} | "
            f"{sum(c.kind == 'method' for c in cs)} | {sum(not c.enabled for c in cs)} | "
            f"{sum(_fits(c, MACHINES['Thalassa']) == '✓' for c in cs)} | "
            f"{sum(_fits(c, MACHINES['DGX Spark']) == '✓' for c in cs)} | {ok} |")

    out += ["", "## Hugging Face access", "", *_gated_section(access_rows)]

    for letter, phase in PHASES.items():
        phase_caps = [c for c in caps if c.startswith(letter)]
        if not phase_caps:
            continue
        out += ["", f"## {letter} — {phase}"]
        for cap in phase_caps:
            out += ["", f"### {cap}", "", f"**{_esc(_title(cap))}**", "",
                    ("| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | "
                    f"Smoke ({here.host}) | Get it | Env |"),
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in sorted(by_cap[cap], key=lambda c: (not c.enabled, c.kind != "local",
                                                        not c.baseline, c.id)):
                name = f"**{c.id}**" + (" (baseline)" if c.baseline else "")
                if not c.enabled:
                    name = f"~~{c.id}~~ (disabled)"
                gates = gated_by_key.get(c.key, [])
                if gates:
                    name += " 🔒" if any(g["access"] != "ok" for g in gates) else " 🔓"
                size = f"{c.params_b:g} B" if c.params_b else "—"
                out.append(
                    f"| {name} | {c.kind} | {size} | {_esc(_lic(c))} | {_esc(_langs(c))} | "
                    f"{_esc(_needs(c))} | {_fits(c, MACHINES['Thalassa'])} | "
                    f"{_fits(c, MACHINES['DGX Spark'])} | "
                    f"{_esc(_smoke_cell(host_smoke.get((cap, c.key))))} | {_get(c)} | "
                    f"`{c.env}` |")
            notes = [f"- **{c.id}**: {_esc(c.notes)}" for c in by_cap[cap] if c.notes]
            if notes:
                out += ["", "<details><summary>Notes</summary>", "", *notes, "", "</details>"]

    out += ["", "## Load / unload smoke tests", ""]
    rows = smoke_rows or []
    if not rows and not app_smoke:
        out.append("No smoke tests recorded yet. Run `python -m bench arena smoke all "
                   "--setup-envs` on each machine, then `python -m bench arena docs`.")
    if rows:
        out += [("What each status means: `ok` loaded, produced an output and returned GPU "
                "memory to the baseline after the worker exited; `leak` left GPU memory or a "
                "process (model server, Ollama model) behind; `load_failed` / `run_failed` see "
                "error; `env_missing` the environment was not set up; `skipped` not runnable on "
                "this machine (reason given)."), "",
                ("| Host | Capability | Candidate | Status | Load s | Peak VRAM | Left after exit "
                "| Downloaded | Error / reason |"),
                "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            peak = f"{r['peak_vram_mib'] / 1024:.1f} GB" if r.get("peak_vram_mib") else "—"
            resid = f"{r['residual_vram_mib']:.0f} MiB" if r.get("residual_vram_mib") \
                is not None else "—"
            dl = f"{r['downloaded_gb']:.1f} GB" if r.get("downloaded_gb") else "—"
            err = _esc((r.get("error") or r.get("leaked") or "")[:160])
            out.append(f"| {r['host']} | {r['capability']} | {r['candidate']} | {r['status']} | "
                       f"{r['load_s'] if r.get('load_s') is not None else '—'} | {peak} | "
                       f"{resid} | {dl} | {err} |")
    if app_smoke:
        out += ["", "### App model manager (server process)", "",
                ("Models the OpenDub server loads in-process, acquired through "
                "`app/providers/_runtime.py` and then unloaded. *Left after unload* is GPU "
                "memory above the level before loading (the CUDA context itself stays once "
                "created)."), "",
                "| Host | Model | Status | Load s | GPU while loaded | Left after unload | Note |",
                "|---|---|---|---|---|---|---|"]
        for r in app_smoke:
            out.append(f"| {r['host']} | {r['candidate']} | {r['status']} | "
                       f"{r.get('load_s') or '—'} | "
                       f"{(r.get('peak_vram_mib') or 0) / 1024:.1f} GB | "
                       f"{r.get('residual_vram_mib') or 0:.0f} MiB | "
                       f"{_esc((r.get('error') or '')[:120])} |")
    return "\n".join(out) + "\n"


def write(out: Path | None = None) -> Path:
    from .smoke import results

    rows = results()
    host = hardware.detect().host
    host_smoke = {(r["capability"], r["cand_key"]): r for r in rows if r["host"] == host}
    app = [r for r in rows if r["capability"] == "APP"]
    worker_rows = [r for r in rows if r["capability"] != "APP"]
    out = out or paths.REPO_DIR / "docs" / "arena-models.md"
    out.write_text(render(host_smoke, app, worker_rows))
    return out
