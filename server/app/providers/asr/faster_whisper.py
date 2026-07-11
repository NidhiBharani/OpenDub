"""Local ASR via faster-whisper (CTranslate2 reimplementation of OpenAI Whisper).

Optional dependency (`pip install faster-whisper`); imported lazily so the server boots without it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...models import ASRSegment
from ..base import ASRProvider, ConfigField, ProgressFn, ProviderMeta, register


@register
class FasterWhisperASR(ASRProvider):
    meta = ProviderMeta(
        id="asr.faster_whisper",
        kind="asr",
        name="Faster-Whisper",
        description=(
            "Local transcription via faster-whisper (CTranslate2 Whisper). Runs on GPU or CPU."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="model",
                label="Model",
                type="select",
                default="large-v3-turbo",
                options=["large-v3", "large-v3-turbo", "medium", "small", "base"],
                help="Whisper model size/variant to load.",
            ),
            ConfigField(
                key="device",
                label="Device",
                type="select",
                default="auto",
                options=["auto", "cuda", "cpu"],
            ),
            ConfigField(
                key="compute_type",
                label="Compute type",
                type="select",
                default="auto",
                options=["auto", "float16", "int8"],
                help="Quantization/precision used by CTranslate2.",
            ),
            ConfigField(
                key="vad",
                label="Voice activity filter",
                type="boolean",
                default=True,
                help="Skip non-speech regions with Silero VAD before decoding.",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return self._can_import("faster_whisper")

    async def transcribe(
        self, audio: Path, language: str, progress: ProgressFn
    ) -> list[ASRSegment]:
        from ...media.ffmpeg import wav_duration

        model_name = self.opt("model", "large-v3-turbo")
        device = self.opt("device", "auto")
        compute_type = self.opt("compute_type", "auto")
        vad = self.opt_bool("vad", True)
        lang = None if language in ("", "auto") else language

        duration = await wav_duration(audio)

        def _run() -> list[ASRSegment]:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            seg_iter, _info = model.transcribe(
                str(audio),
                language=lang,
                vad_filter=vad,
                beam_size=5,
                condition_on_previous_text=False,
            )
            out: list[ASRSegment] = []
            for seg in seg_iter:
                out.append(
                    ASRSegment(start=seg.start, end=seg.end, text=(seg.text or "").strip())
                )
                frac = min(1.0, seg.end / duration) if duration > 0 else 0.0
                progress(frac, f"transcribing {seg.end:.1f}s / {duration:.1f}s")
            return out

        progress(0.0, f"loading {model_name} ({device}/{compute_type})")
        result = await asyncio.to_thread(_run)
        progress(1.0, "done")
        return result
