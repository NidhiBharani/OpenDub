"""LatentSync lip-sync provider (ByteDance LatentSync).

Like Wav2Lip, LatentSync has its own heavy diffusion-model dependency stack, so it is always
driven as a subprocess against a separately cloned repo + checkpoints/venv — this module never
imports latentsync/torch/diffusers/etc. in-process.
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

    LatentSync's inference script prints tqdm-style diffusion-step progress ("NN%|...|...") to
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
    try:
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
    except asyncio.CancelledError:
        # Job cancellation: kill the child so it stops burning GPU/CPU and writing outputs.
        if proc.returncode is None:
            proc.kill()
        raise
    remainder = buf.decode("utf-8", errors="replace").strip()
    if remainder:
        tail.append(remainder)
        if len(tail) > tail_len:
            tail.pop(0)
    returncode = await proc.wait()
    return returncode, tail


@register
class LatentSyncProvider(LipSyncProvider):
    meta = ProviderMeta(
        id="lipsync.latentsync",
        kind="lipsync",
        name="LatentSync",
        description=(
            "ByteDance LatentSync diffusion-based lip sync, driven as a subprocess against a "
            "separately cloned repo + checkpoints/venv."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="repo_dir",
                label="Repo directory",
                type="string",
                help="Path to a cloned LatentSync repo.",
            ),
            ConfigField(
                key="script",
                label="Inference script",
                type="string",
                default="scripts/inference.py",
                help="Path (relative to repo_dir) of the inference script.",
            ),
            ConfigField(
                key="unet_config",
                label="UNet config",
                type="string",
                default="configs/unet/stage2.yaml",
                help="Path to the UNet config yaml (relative to repo_dir, unless absolute).",
            ),
            ConfigField(
                key="checkpoint",
                label="Checkpoint path",
                type="string",
                default="checkpoints/latentsync_unet.pt",
                help="Path to the UNet checkpoint (relative to repo_dir unless absolute).",
            ),
            ConfigField(
                key="python_bin",
                label="Python interpreter",
                type="string",
                default="python3",
                help="Interpreter of the LatentSync venv (with its own torch/diffusers/etc.).",
            ),
            ConfigField(
                key="inference_steps",
                label="Inference steps",
                type="number",
                default=20,
                help="Diffusion denoising steps; higher = slower but often better quality.",
            ),
            ConfigField(
                key="guidance_scale",
                label="Guidance scale",
                type="number",
                default=1.5,
                help="Classifier-free guidance scale.",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        repo_dir = self.opt("repo_dir")
        if not repo_dir:
            return False, "repo_dir is not configured"
        script = self.opt("script", "scripts/inference.py")
        if not (Path(repo_dir) / script).exists():
            return False, f"{script} not found in {repo_dir}"
        return True, "ready"

    async def sync(
        self, video: Path, audio: Path, out_video: Path, progress: ProgressFn
    ) -> None:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"LatentSync is not available: {reason}")

        repo_dir = Path(self.opt("repo_dir"))
        script = self.opt("script", "scripts/inference.py")
        unet_config = self.opt("unet_config", "configs/unet/stage2.yaml")
        checkpoint = self.opt("checkpoint", "checkpoints/latentsync_unet.pt")
        python_bin = self.opt("python_bin", "python3")
        # LatentSync's inference script declares --inference_steps as argparse type=int: coerce
        # float/string values (the web form stores numbers as floats) before building argv.
        inference_steps = int(float(self.opt("inference_steps", 20) or 20))
        guidance_scale = self.opt_float("guidance_scale", 1.5)

        out_video.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin,
            script,
            "--unet_config_path", str(unet_config),
            "--inference_ckpt_path", str(checkpoint),
            "--video_path", str(video),
            "--audio_path", str(audio),
            "--video_out_path", str(out_video),
            "--inference_steps", str(inference_steps),
            "--guidance_scale", str(guidance_scale),
        ]

        progress(0.0, "starting LatentSync inference")
        returncode, tail = await _run_streaming(cmd, repo_dir, progress)
        if returncode != 0:
            lines = "\n".join(tail[-15:]) or "(no output captured)"
            raise RuntimeError(
                f"LatentSync inference failed (exit code {returncode}). Last output:\n{lines}"
            )
        if not out_video.exists():
            raise RuntimeError(
                f"LatentSync exited successfully but did not produce an output file at {out_video}"
            )
        progress(1.0, "done")
