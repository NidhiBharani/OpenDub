"""Speaker diarization via pyannote.audio's pretrained speaker-diarization-3.1 pipeline.

Optional dependency (`pip install pyannote.audio`, plus a HuggingFace token that has accepted the
pyannote model's user agreement); imported lazily so the server boots without it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...models import ASRSegment
from ..base import ConfigField, DiarizationProvider, ProgressFn, ProviderMeta, register


@register
class PyannoteDiarization(DiarizationProvider):
    meta = ProviderMeta(
        id="diarization.pyannote",
        kind="diarization",
        name="Pyannote 3.1",
        description=(
            "Local speaker diarization via pyannote.audio's speaker-diarization-3.1 pipeline."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="hf_token",
                label="HuggingFace token",
                type="secret",
                help=(
                    "Needs access to pyannote/speaker-diarization-3.1 "
                    "(accept its user agreement)."
                ),
            ),
            ConfigField(
                key="device", label="Device", type="select", default="auto",
                options=["auto", "cuda", "cpu"],
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        ok, reason = self._can_import("pyannote.audio")
        if not ok:
            return ok, reason
        if not self.opt("hf_token"):
            return False, "hf_token not configured"
        return True, "ready"

    async def diarize(
        self, audio: Path, segments: list[ASRSegment], progress: ProgressFn
    ) -> list[str]:
        if not segments:
            return []
        token = self.opt("hf_token")
        if not token:
            raise RuntimeError("Pyannote diarization: hf_token not configured")
        device_opt = self.opt("device", "auto")

        def _run() -> list[tuple[float, float, str]]:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=token
            )
            try:
                import torch

                if device_opt in ("cuda", "cpu"):
                    pipeline.to(torch.device(device_opt))
                elif device_opt in ("", "auto") and torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
            except Exception:
                pass  # fall back to pyannote's own default device selection

            diarization = pipeline(str(audio))
            return [
                (turn.start, turn.end, str(speaker))
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

        progress(0.05, "loading diarization model")
        turns = await asyncio.to_thread(_run)
        progress(0.7, "mapping speakers to segments")

        labels: list[str] = []
        prev_label = "S0"
        for seg in segments:
            best_label: str | None = None
            best_overlap = 0.0
            for t_start, t_end, spk in turns:
                overlap = min(seg.end, t_end) - max(seg.start, t_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = spk
            label = best_label if best_label is not None else prev_label
            labels.append(label)
            prev_label = label

        progress(1.0, "done")
        return labels
