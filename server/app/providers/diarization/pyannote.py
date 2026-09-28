"""Speaker diarization via pyannote.audio's pretrained speaker-diarization-3.1 pipeline.

Optional dependency (`pip install pyannote.audio`, plus a HuggingFace token that has accepted the
pyannote model's user agreement); imported lazily so the server boots without it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...models import ASRSegment
from .._runtime import manager
from ..base import ConfigField, DiarizationProvider, ProgressFn, ProviderMeta, register
from ._overlap import assign_labels_by_overlap

PIPELINE_ID = "pyannote/speaker-diarization-3.1"


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
        device_opt = str(self.opt("device", "auto") or "auto")

        def _load():
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(PIPELINE_ID, use_auth_token=token)
            try:
                import torch

                if device_opt in ("cuda", "cpu"):
                    pipeline.to(torch.device(device_opt))
                elif device_opt == "auto" and torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
            except (ImportError, RuntimeError):
                pass  # fall back to pyannote's own default device selection
            return pipeline

        def _run() -> list[tuple[float, float, str]]:
            # The token only grants access; it is not part of the key (keys are listed by the API).
            with manager.acquire(("diarization.pyannote", PIPELINE_ID, device_opt), _load,
                                 est_vram_gb=0.0 if device_opt == "cpu" else 1.0) as pipeline:
                diarization = pipeline(str(audio))
            return [
                (turn.start, turn.end, str(speaker))
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

        progress(0.05, "loading diarization model")
        turns = await asyncio.to_thread(_run)
        progress(0.7, "mapping speakers to segments")
        labels = assign_labels_by_overlap(segments, turns)
        progress(1.0, "done")
        return labels
