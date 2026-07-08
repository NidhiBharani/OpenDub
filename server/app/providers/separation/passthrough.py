"""No-op source separation: treat the whole input as vocals and produce a silent background track.

Useful when the source audio is already clean dialogue (no music/SFX to strip) or as a safe,
zero-dependency default so the pipeline is runnable out of the box.
"""
from __future__ import annotations

from pathlib import Path

from ..base import ProgressFn, ProviderMeta, SeparationProvider, register


@register
class PassthroughSeparationProvider(SeparationProvider):
    meta = ProviderMeta(
        id="separation.passthrough",
        kind="separation",
        name="No separation",
        description=(
            "Skips source separation entirely: vocals = the original audio unchanged, "
            "background = silence. Always available, no dependencies."
        ),
        runtime="local",
        fields=[],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return True, "ready"

    async def separate(
        self, audio: Path, vocals_out: Path, background_out: Path, progress: ProgressFn
    ) -> None:
        from ...media.ffmpeg import silent_wav, to_std_wav, wav_duration

        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        background_out.parent.mkdir(parents=True, exist_ok=True)

        progress(0.0, "copying original audio as vocals")
        await to_std_wav(audio, vocals_out)

        progress(0.5, "generating silent background track")
        duration = await wav_duration(audio)
        await silent_wav(background_out, duration=duration)

        progress(1.0, "done")
