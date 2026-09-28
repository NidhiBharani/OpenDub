"""Demucs (htdemucs) two-stems source separation.

Demucs/torch have their own heavy, sometimes conflicting dependency stack, so this provider never
imports them in-process: `available()` only probes for the packages via `importlib.util.find_spec`,
and `separate()` runs `python -m demucs.separate` as a subprocess (using the *current* interpreter,
`sys.executable`, so demucs must be installed into the server's own environment — see the
`separation` extra in pyproject.toml).
"""
from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path

from ..base import ConfigField, ProgressFn, ProviderMeta, SeparationProvider, register

_PERCENT_RE = re.compile(r"(\d{1,3})%\|")


async def _run_streaming(
    cmd: list[str], progress: ProgressFn, tail_len: int = 20
) -> tuple[int, list[str]]:
    """Run `cmd`, feeding decoded output lines to `progress` as they arrive.

    demucs (via tqdm) redraws its progress bar with carriage returns rather than newlines, so we
    split on either `\\r` or `\\n`. Stdout and stderr are merged so failures are fully captured
    regardless of which stream a tool writes to. Returns (returncode, last `tail_len` non-empty
    lines) for error reporting.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
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
class DemucsSeparationProvider(SeparationProvider):
    meta = ProviderMeta(
        id="separation.demucs",
        kind="separation",
        name="Demucs (htdemucs)",
        description=(
            "Meta's Demucs music source separation model, run in two-stems mode to split dialogue "
            "(vocals) from background music/effects."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="model",
                label="Model",
                type="select",
                default="htdemucs",
                options=["htdemucs", "htdemucs_ft", "mdx_extra"],
                help="Pretrained demucs model name (passed as -n).",
            ),
            ConfigField(
                key="device",
                label="Device",
                type="select",
                default="auto",
                options=["auto", "cuda", "cpu"],
                help="Compute device; 'auto' lets demucs pick.",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return self._can_import("demucs", "torch")

    async def separate(
        self, audio: Path, vocals_out: Path, background_out: Path, progress: ProgressFn
    ) -> None:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(f"Demucs is not available: {reason}")

        model = self.opt("model", "htdemucs")
        device = self.opt("device", "auto")

        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        background_out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="opendub-demucs-") as tmpdir:
            cmd = [
                sys.executable, "-m", "demucs.separate",
                "--two-stems", "vocals",
                "-n", model,
                "--shifts", str(max(1, min(10, int(self.opt_float("shifts", 2))))),
                "--overlap", str(max(0.0, min(0.9, self.opt_float("overlap", 0.5)))),
                "-o", tmpdir,
            ]
            if device != "auto":
                cmd += ["-d", device]
            cmd += [str(audio)]

            progress(0.0, f"running demucs ({model}) ...")
            returncode, tail = await _run_streaming(cmd, progress)
            if returncode != 0:
                lines = "\n".join(tail) or "(no output captured)"
                raise RuntimeError(
                    f"demucs separation failed (exit code {returncode}). Last output:\n{lines}"
                )

            # demucs writes to <out>/<model>/<track_name>/{vocals,no_vocals}.wav where track_name
            # is the input file's basename without extension.
            track_name = audio.stem
            src_dir = Path(tmpdir) / model / track_name
            vocals_src = src_dir / "vocals.wav"
            no_vocals_src = src_dir / "no_vocals.wav"
            if not vocals_src.exists() or not no_vocals_src.exists():
                found = sorted(p.name for p in src_dir.iterdir()) if src_dir.exists() else []
                raise RuntimeError(
                    f"demucs finished but expected output was not found in {src_dir} "
                    f"(found: {found or 'directory missing'})"
                )

            from ...media.ffmpeg import to_std_wav

            progress(0.95, "converting outputs to standard wav")
            await to_std_wav(vocals_src, vocals_out)
            await to_std_wav(no_vocals_src, background_out)

        progress(1.0, "done")
