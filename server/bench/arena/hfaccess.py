"""Which Hugging Face repos does the registry use, which are gated, and can this token get them?

    python -m bench arena hf-access        → data/arena/hf_access.json (read by `arena docs`)

Gated repos need a one-time "Agree" on the model page by the account that owns the token (there
is no API for it). Dependencies that a model pulls at run time but that no candidate names
(pyannote's segmentation / embedding models) are listed in ``RUNTIME_DEPS``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import paths
from .hardware import check
from .registry import capabilities, load_candidates

# Gated repos a pipeline downloads by itself at run time.
RUNTIME_DEPS = {
    "pyannote/speaker-diarization-3.1": ["pyannote/segmentation-3.0",
                                         "pyannote/wespeaker-voxceleb-resnet34-LM"],
    "pyannote/speaker-diarization-community-1": ["pyannote/segmentation-3.0",
                                                 "pyannote/embedding"],
}

_REPO = re.compile(r"^(?:hf://)?([A-Za-z0-9][\w.-]*/[\w.-]+)")
_URL = re.compile(r"https://huggingface\.co/(?:datasets/)?([\w.-]+/[\w.-]+)")


def access_path() -> Path:
    return paths.arena_dir() / "hf_access.json"


def _repos_in(v: Any, out: set[str]) -> None:
    if isinstance(v, str):
        s = v.strip()
        m = _URL.match(s)
        if m:
            out.add(m.group(1))
        elif not s.startswith(("/", "~", "http", ".")):
            m = _REPO.match(s)
            if m:
                out.add("/".join(m.group(1).split("/")[:2]))
    elif isinstance(v, dict):
        for x in v.values():
            _repos_in(x, out)
    elif isinstance(v, list):
        for x in v:
            _repos_in(x, out)


def scan(log=print) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    from .docs import MACHINES

    uses: dict[str, list[dict[str, Any]]] = {}
    for cap in capabilities():
        for c in load_candidates(cap):
            if not c.enabled or c.kind == "api":
                continue
            fit = {name: not [w for w in check(c.requires, hw)
                              if not w.startswith("missing API key")]
                   for name, hw in MACHINES.items()}
            repos: set[str] = set()
            _repos_in(c.params, repos)
            _repos_in(c.url, repos)
            for r in list(repos):
                repos |= set(RUNTIME_DEPS.get(r, []))
            for r in repos:
                uses.setdefault(r, []).append({"capability": cap, "candidate": c.id,
                                               "cand_key": c.key, **fit})
    api = HfApi()
    rows = []
    for repo, cands in sorted(uses.items()):
        rec: dict[str, Any] = {"repo": repo, "type": None, "gated": False, "access": "not-on-hf",
                               "candidates": cands}
        for rtype in ("model", "dataset"):
            try:
                info = api.repo_info(repo, repo_type=rtype)
            except RepositoryNotFoundError:
                continue
            except HfHubHTTPError as e:
                rec["access"] = f"http-{getattr(e.response, 'status_code', '?')}"
                break
            rec.update(type=rtype, gated=getattr(info, "gated", False) or False)
            try:
                api.auth_check(repo, repo_type=rtype)
                rec["access"] = "ok"
            except GatedRepoError:
                rec["access"] = "needs-agree"
            break
        rows.append(rec)
    access_path().write_text(json.dumps(rows, indent=1))
    gated = [r for r in rows if r["gated"]]
    log(f"{len(rows)} repos referenced; {sum(r['type'] is not None for r in rows)} on HF; "
        f"{len(gated)} gated ({sum(r['access'] == 'needs-agree' for r in gated)} still need "
        f"'Agree'); written to {access_path()}")
    return rows


def load() -> list[dict[str, Any]]:
    p = access_path()
    return json.loads(p.read_text()) if p.exists() else []
