"""Worker environments: one isolated interpreter per model family, with setup recipes per arch.

``bench/envs/<name>.yaml``::

    name: qwen3asr
    python: ~/.opendub/envs/qwen3asr/bin/python     # or "auto" = the bench's own interpreter
    description: Qwen3-ASR via qwen-asr (pins transformers 4.57)
    setup:                                          # shell commands, run in order
      x86_64:
        - uv venv {prefix} --python 3.12
        - uv pip install --python {python} torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
        - uv pip install --python {python} qwen-asr==0.0.6
      aarch64:                                      # DGX Spark (SM 12.1, CUDA 13)
        - uv venv {prefix} --python 3.12
        - uv pip install --python {python} torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
        - uv pip install --python {python} qwen-asr==0.0.6
    check: "import qwen_asr, torch; assert torch.cuda.is_available()"   # python -c snippet

Placeholders: ``{prefix}`` (the env dir, parent of bin/), ``{python}``, ``{workers}`` (the workers
directory), ``{repo}``, ``{home}``. Nothing here runs unless the user calls
``python -m bench arena env setup <name>``.
"""
from __future__ import annotations

import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths

ENVS_DIR = paths.SERVER_DIR / "bench" / "envs"


@dataclass
class Env:
    name: str
    python: str = "auto"
    description: str = ""
    setup: dict[str, list[str]] = field(default_factory=dict)
    check: str = ""
    notes: str = ""

    @property
    def prefix(self) -> Path | None:
        if self.python == "auto":
            return None
        return Path(self.python).expanduser().parent.parent

    def commands(self, arch: str | None = None) -> list[str]:
        arch = arch or platform.machine()
        steps = self.setup.get(arch) or self.setup.get("any") or []
        subs = {"prefix": str(self.prefix or ""), "python": str(Path(self.python).expanduser())
                if self.python != "auto" else sys.executable,
                "workers": str(paths.WORKERS_DIR), "repo": str(paths.REPO_DIR),
                "home": str(Path.home())}
        return [s.format(**subs) for s in steps]


def load_env(name: str, root: Path | None = None) -> Env:
    root = root or ENVS_DIR
    path = root / f"{name}.yaml"
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        raw.setdefault("name", name)
        return Env(**raw)
    legacy = paths.CANDIDATES_DIR / "_envs.yaml"
    envs = (yaml.safe_load(legacy.read_text()) or {}) if legacy.exists() else {}
    if name in envs:
        return Env(name=name, **envs[name])
    if name == "server":
        return Env(name="server", python="auto")
    raise KeyError(f"worker env {name!r} is not defined ({path})")


def all_envs(root: Path | None = None) -> list[Env]:
    root = root or ENVS_DIR
    names = {p.stem for p in root.glob("*.yaml")} if root.exists() else set()
    legacy = paths.CANDIDATES_DIR / "_envs.yaml"
    if legacy.exists():
        names |= set((yaml.safe_load(legacy.read_text()) or {}).keys())
    return [load_env(n, root) for n in sorted(names)]


def is_ready(env: Env) -> bool:
    return env.python == "auto" or Path(env.python).expanduser().exists()


def setup(env: Env, *, dry_run: bool = False, arch: str | None = None) -> int:
    cmds = env.commands(arch)
    if not cmds:
        print(f"{env.name}: no setup recipe for {arch or platform.machine()}")
        return 1
    for cmd in cmds:
        print(f"$ {cmd}")
        if not dry_run:
            rc = subprocess.run(cmd, shell=True, check=False).returncode
            if rc:
                print(f"{env.name}: step failed with exit {rc}")
                return rc
    if env.check and not dry_run:
        python = str(Path(env.python).expanduser()) if env.python != "auto" else sys.executable
        rc = subprocess.run([python, "-c", env.check], check=False).returncode
        print(f"{env.name}: check {'passed' if rc == 0 else 'FAILED'}")
        return rc
    return 0


def quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
