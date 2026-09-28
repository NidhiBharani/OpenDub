"""What this machine can run: GPUs, memory, architecture, tools and API keys.

Candidates declare ``requires`` (see :class:`Requires`); the arena runs a candidate only where the
detected hardware satisfies it, so one registry works on Thalassa (RTX 5060 Ti 16 GB, x86_64),
a DGX Spark (GB10, 128 GB unified memory, aarch64) and API-only machines alike.

``OPENDUB_ARENA_HW=/path/hw.json`` replaces detection with a fixed description (tests, dry runs
for another box): the JSON has the same fields as :class:`Hardware`.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import paths

# GPUs whose memory is the system's unified memory: nvidia-smi reports no dedicated total.
UNIFIED_MEMORY_GPUS = ("GB10", "GH200", "GB200", "Grace", "Jetson", "Thor")


@dataclass
class Gpu:
    name: str
    vram_gb: float           # dedicated memory, or the usable share of unified memory
    compute_capability: str  # "12.0" (RTX 50xx), "12.1" (GB10), "9.0" (H100)
    unified: bool = False


@dataclass
class Hardware:
    host: str
    arch: str                              # platform.machine(): x86_64 | aarch64
    ram_gb: float
    disk_free_gb: float
    gpus: list[Gpu] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)  # names of env vars that are set
    tools: list[str] = field(default_factory=list)     # docker, nvcc, ffmpeg, yt-dlp …
    driver_cuda: str | None = None                     # CUDA version the driver supports

    @property
    def vram_gb(self) -> float:
        """Largest single-GPU memory; a job runs on one GPU."""
        return max((g.vram_gb for g in self.gpus), default=0.0)

    @property
    def compute_capability(self) -> str | None:
        best = max(self.gpus, key=lambda g: g.vram_gb, default=None)
        return best.compute_capability if best else None

    @property
    def profile(self) -> str:
        """Short label for reports: which of the known machine classes this is."""
        if not self.gpus:
            return "cpu"
        g = max(self.gpus, key=lambda g: g.vram_gb)
        if g.unified and self.arch == "aarch64":
            return f"spark-{round(g.vram_gb)}g" if "GB10" in g.name else f"unified-{round(g.vram_gb)}g"
        return f"{self.arch}-{round(g.vram_gb)}g"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["profile"] = self.profile
        return d


# API keys the arena knows how to use; a candidate names the ones it needs in requires.api_keys.
KNOWN_API_KEYS = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "ELEVENLABS_API_KEY", "DEEPL_API_KEY", "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_REGION", "AWS_ACCESS_KEY_ID", "PYANNOTEAI_API_KEY", "AUDIOSHAKE_API_KEY",
    "LALAL_API_KEY", "REPLICATE_API_TOKEN", "SYNC_API_KEY", "HEYGEN_API_KEY", "CARTESIA_API_KEY",
    "MINIMAX_API_KEY", "DASHSCOPE_API_KEY", "ASSEMBLYAI_API_KEY", "SARVAM_API_KEY",
    "HUME_API_KEY", "FISH_AUDIO_API_KEY", "RESEMBLE_API_KEY", "HF_TOKEN", "MISTRAL_API_KEY",
    "DEEPGRAM_API_KEY", "SPEECHMATICS_API_KEY", "OPENROUTER_API_KEY",
)
KNOWN_TOOLS = ("docker", "nvcc", "ffmpeg", "ffprobe", "yt-dlp", "uv", "git", "espeak-ng",
               "mfa", "rubberband", "c2patool", "ollama")


def _nvidia_gpus() -> tuple[list[Gpu], str | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return [], None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem, cc = parts[0], parts[1], parts[2]
        unified = any(tag in name for tag in UNIFIED_MEMORY_GPUS)
        try:
            vram = float(mem) / 1024
        except ValueError:  # "[N/A]" on unified-memory parts
            vram = 0.0
        gpus.append(Gpu(name=name, vram_gb=round(vram, 1), compute_capability=cc,
                        unified=unified or vram == 0.0))
    cuda = None
    try:
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15,
                              check=False).stdout
        marker = "CUDA Version:"
        if marker in head:
            cuda = head.split(marker, 1)[1].split()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return gpus, cuda


def _hf_token_saved() -> bool:
    """`hf auth login` stores the token in a file instead of the environment; count it as
    HF_TOKEN (huggingface_hub reads it in every worker env)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = Path(os.environ.get("HF_TOKEN_PATH", home / "token"))
    try:
        return path.is_file() and bool(path.read_text().strip())
    except OSError:
        return False


def _has_key(name: str) -> bool:
    return bool(os.environ.get(name)) or (name == "HF_TOKEN" and _hf_token_saved())


def _ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return 0.0


@lru_cache(maxsize=1)
def detect() -> Hardware:
    override = os.environ.get("OPENDUB_ARENA_HW")
    if override:
        raw = json.loads(Path(override).read_text())
        raw.pop("profile", None)
        raw["gpus"] = [Gpu(**g) for g in raw.get("gpus", [])]
        return Hardware(**raw)
    gpus, cuda = _nvidia_gpus()
    ram = _ram_gb()
    for g in gpus:
        if g.unified:
            # Unified memory is shared with the CPU; leave headroom for the OS and the arena.
            g.vram_gb = round(max(0.0, ram - 16.0) if ram else g.vram_gb, 1)
    target = paths.data_dir() if paths.data_dir().exists() else Path.home()
    disk = shutil.disk_usage(target).free / 2**30
    return Hardware(
        host=socket.gethostname(), arch=platform.machine(), ram_gb=ram,
        disk_free_gb=round(disk, 1), gpus=gpus,
        api_keys=[k for k in KNOWN_API_KEYS if _has_key(k)],
        tools=[t for t in KNOWN_TOOLS if shutil.which(t)], driver_cuda=cuda)


@dataclass
class Requires:
    """What a candidate (or judge) needs to run. Every field is optional; empty = runs anywhere."""

    gpu: bool = False                  # needs an NVIDIA GPU at all
    vram_gb: float = 0.0               # peak GPU memory for one worker
    ram_gb: float = 0.0                # host RAM (CPU offload, big tokenisers, video decode)
    disk_gb: float = 0.0               # weights + caches to download
    arch: list[str] = field(default_factory=list)       # e.g. ["x86_64"]; empty = any
    min_compute_capability: str | None = None           # e.g. "8.0" for bf16 flash kernels
    api_keys: list[str] = field(default_factory=list)   # env vars that must be set
    tools: list[str] = field(default_factory=list)      # executables that must be on PATH

    @classmethod
    def parse(cls, raw: dict[str, Any] | None, vram_gb: float = 0.0) -> Requires:
        raw = dict(raw or {})
        if vram_gb and not raw.get("vram_gb"):
            raw["vram_gb"] = vram_gb
        if raw.get("vram_gb"):
            raw.setdefault("gpu", True)
        for key in ("arch", "api_keys", "tools"):
            if isinstance(raw.get(key), str):
                raw[key] = [raw[key]]
        return cls(**raw)


def _cc(v: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


def check(req: Requires, hw: Hardware | None = None) -> list[str]:
    """Reasons this hardware cannot run ``req``; an empty list means it can."""
    hw = hw or detect()
    why: list[str] = []
    if req.arch and hw.arch not in req.arch:
        why.append(f"arch {hw.arch} not in {req.arch}")
    if (req.gpu or req.vram_gb) and not hw.gpus:
        why.append("needs an NVIDIA GPU")
    elif req.vram_gb and hw.vram_gb < req.vram_gb:
        why.append(f"needs {req.vram_gb:g} GB VRAM, have {hw.vram_gb:g}")
    if req.min_compute_capability and hw.gpus and \
            _cc(hw.compute_capability) < _cc(req.min_compute_capability):
        why.append(f"needs compute capability ≥ {req.min_compute_capability}, "
                   f"have {hw.compute_capability}")
    if req.ram_gb and hw.ram_gb < req.ram_gb:
        why.append(f"needs {req.ram_gb:g} GB RAM, have {hw.ram_gb:g}")
    if req.disk_gb and hw.disk_free_gb < req.disk_gb:
        why.append(f"needs {req.disk_gb:g} GB free disk, have {hw.disk_free_gb:g}")
    missing_keys = [k for k in req.api_keys if k not in hw.api_keys and not _has_key(k)]
    if missing_keys:
        why.append(f"missing API key(s): {', '.join(missing_keys)}")
    missing_tools = [t for t in req.tools if t not in hw.tools]
    if missing_tools:
        why.append(f"missing tool(s): {', '.join(missing_tools)}")
    return why
