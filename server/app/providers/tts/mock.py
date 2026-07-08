"""tts.mock — always-available placeholder TTS.

Uses espeak-ng for basic (robotic but intelligible) English speech when it is installed on the
host; otherwise falls back to a speech-shaped sine-tone placeholder sized to roughly match the
expected line duration, via `app.media.ffmpeg.tone_wav`. This keeps the full pipeline runnable
with zero TTS model / API dependencies installed.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from ...media import ffmpeg
from ...models import TTSRequest
from ..base import ProgressFn, ProviderMeta, TTSProvider, register


@register
class MockTTSProvider(TTSProvider):
    meta = ProviderMeta(
        id="tts.mock",
        kind="tts",
        name="Beep voice (mock)",
        description=(
            "Always-available placeholder. Uses espeak-ng for intelligible (robotic) English "
            "speech if it is installed on the host; otherwise synthesizes a speech-shaped tone "
            "sized to roughly match the expected duration, so the full pipeline can run without "
            "any TTS model or API configured."
        ),
        runtime="local",
        fields=[],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        if shutil.which("espeak-ng"):
            return True, "ready (espeak-ng)"
        return True, "ready (tone placeholder — install espeak-ng for intelligible speech)"

    async def synthesize(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        progress(0.0, "starting mock synthesis")
        if shutil.which("espeak-ng"):
            await self._synth_espeak(req, out_wav, progress)
        else:
            await self._synth_tone(req, out_wav, progress)
        progress(1.0, "done")

    async def _synth_espeak(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        progress(0.2, "running espeak-ng")
        text = req.text.strip() or " "
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            txt_path = tmp_dir / "line.txt"
            raw_wav = tmp_dir / "raw.wav"
            txt_path.write_text(text, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                "espeak-ng", "-v", "en", "-s", "165", "-f", str(txt_path), "-w", str(raw_wav),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"espeak-ng failed (exit {proc.returncode}): "
                    f"{stderr.decode(errors='replace')[:400]}"
                )
            progress(0.7, "standardizing audio")
            await ffmpeg.to_std_wav(raw_wav, out_wav)

    async def _synth_tone(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        estimated = max(0.6, 0.055 * len(req.text))
        floor = max(req.target_duration, 0.6)
        duration = min(estimated, floor)
        progress(0.5, f"generating {duration:.1f}s placeholder tone")
        await ffmpeg.tone_wav(out_wav, duration)
