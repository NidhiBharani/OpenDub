"""Wav2Lip lip-sync provider.

Wav2Lip has its own (often old/conflicting) dependency stack, so it is always driven as a
subprocess against a separately cloned repo + venv, configured via `repo_dir` / `python_bin` /
`checkpoint` — this module never imports wav2lip/torch/etc. in-process.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from ..base import ConfigField, LipSyncProvider, ProgressFn, ProviderMeta, register

_PERCENT_RE = re.compile(r"(\d{1,3})%\|")


async def _run_streaming(
    cmd: list[str], cwd: Path, progress: ProgressFn, tail_len: int = 15
) -> tuple[int, list[str]]:
    """Run `cmd` in `cwd`, feeding decoded output lines to `progress`.

    Wav2Lip prints tqdm progress bars ("NN%|...|...") for frame reading / mel-chunk inference to
    stderr; tqdm redraws with carriage returns rather than newlines, so we split on either. Stdout
    and stderr are merged so failures are fully captured. Returns (returncode, last `tail_len`
    non-empty lines) for error reporting.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    buf = b""
    tail: list[str] = []
    last_frac = 0.0
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf or b"\r" in buf:
            n_idx = buf.find(b"\n")
            r_idx = buf.find(b"\r")
            idx = min(i for i in (n_idx, r_idx) if i != -1)
            raw, buf = buf[:idx], buf[idx + 1 :]
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            tail.append(line)
            if len(tail) > tail_len:
                tail.pop(0)
            m = _PERCENT_RE.search(line)
            if m:
                last_frac = max(0.0, min(0.99, int(m.group(1)) / 100.0))
            progress(last_frac, line)
    remainder = buf.decode("utf-8", errors="replace").strip()
    if remainder:
        tail.append(remainder)
        if len(tail) > tail_len:
            tail.pop(0)
    returncode = await proc.wait()
    return returncode, tail


@register
class Wav2LipProvider(LipSyncProvider):
    meta = ProviderMeta(
        id="lipsync.wav2lip",
        kind="lipsync",
        name="Wav2Lip",
        description=(
            "Rudrabha/Wav2Lip mouth-sync, driven as a subprocess against a separately cloned repo "
            "+ checkpoint (typically its own venv, given Wav2Lip's old pinned deps)."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="repo_dir",
                label="Repo directory",
                type="string",
                help="Path to a cloned Wav2Lip repo (https://github.com/Rudrabha/Wav2Lip).",
            ),
            ConfigField(
                key="checkpoint",
                label="Checkpoint path",
                type="string",
                help="Path to wav2lip_gan.pth.",
            ),
            ConfigField(
                key="python_bin",
                label="Python interpreter",
                type="string",
                default="python3",
                help="Interpreter of the Wav2Lip venv (with its own torch/librosa/etc.).",
            ),
            ConfigField(
                key="resize_factor",
                label="Resize factor",
                type="number",
                default=1,
                help="Downsample the input video by this factor before inference.",
            ),
            ConfigField(
                key="nosmooth",
                label="Disable box smoothing",
                type="boolean",
                default=False,
                help="Useful when Wav2Lip's face-detection smoothing causes visible jitter.",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        repo_dir = self.opt("repo_dir")
        checkpoint = self.opt("checkpoint")
        if not repo_dir:
            return False, "repo_dir is not configured"
        if not checkpoint:
            return False, "checkpoint is not configured"
        if not (Path(repo_dir) / "inference.py").exists():
            return False, f"inference.py not found in {repo_dir}"
        if not Path(checkpoint).exists():
            return False, f"checkpoint not found at {checkpoint}"
        return True, "ready"

    async def sync(
        self, video: Path, audio: Path, out_video: Path, progress: ProgressFn
    ) -> None:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"Wav2Lip is not available: {reason}")

        repo_dir = Path(self.opt("repo_dir"))
        checkpoint = Path(self.opt("checkpoint"))
        python_bin = self.opt("python_bin", "python3")
        resize_factor = self.opt("resize_factor", 1)
        nosmooth = bool(self.opt("nosmooth", False))

        out_video.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin,
            "inference.py",
            "--checkpoint_path", str(checkpoint),
            "--face", str(video),
            "--audio", str(audio),
            "--outfile", str(out_video),
            "--resize_factor", str(resize_factor),
        ]
        if nosmooth:
            cmd.append("--nosmooth")

        progress(0.0, "starting Wav2Lip inference")
        returncode, tail = await _run_streaming(cmd, repo_dir, progress)
        if returncode != 0:
            lines = "\n".join(tail[-15:]) or "(no output captured)"
            raise RuntimeError(
                f"Wav2Lip inference failed (exit code {returncode}). Last output:\n{lines}"
            )
        if not out_video.exists():
            raise RuntimeError(
                f"Wav2Lip exited successfully but did not produce an output file at {out_video}"
            )
        progress(1.0, "done")
