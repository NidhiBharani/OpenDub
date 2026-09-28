"""Local ASR via faster-whisper (CTranslate2 reimplementation of OpenAI Whisper).

Optional dependency (`pip install faster-whisper`); imported lazily so the server boots without it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...models import ASRSegment, Word
from .._runtime import manager
from ..base import ASRProvider, ConfigField, ProgressFn, ProviderMeta, register

# Rough fp16 VRAM per checkpoint (GB), for the model manager's GPU budget. int8 is about half.
_VRAM_GB = {"large-v3": 3.5, "large-v2": 3.5, "large-v3-turbo": 1.8, "medium": 1.6,
            "small": 0.7, "base": 0.3, "tiny": 0.2}


def _preload_cuda_libs() -> None:
    """Load the pip-installed CUDA libraries CTranslate2 links against (cuBLAS 12, cuDNN 9).

    CTranslate2 wheels are built for CUDA 12 and dlopen `libcublas.so.12` by bare name, which the
    loader can't find inside site-packages (and a CUDA 13 torch ships only `.so.13`). Preloading
    them globally makes the later dlopen resolve; missing files are skipped (CPU still works).
    """
    import ctypes
    import sysconfig

    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for rel in (
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cudnn/lib/libcudnn.so.9",
    ):
        lib = nvidia / rel
        if lib.exists():
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def whisper_vram_gb(model_name: str, device: str, compute_type: str) -> float | None:
    """Estimated VRAM for a faster-whisper model; 0 on CPU, None when unknown."""
    if device == "cpu":
        return 0.0
    base = _VRAM_GB.get(model_name.rsplit("/", 1)[-1].removeprefix("faster-whisper-"))
    if base is None:
        return None
    return base / 2 if "int8" in compute_type else base


def acquire_whisper(model_name: str, device: str = "auto", compute_type: str = "auto",
                    **kwargs: object):
    """Shared, managed faster-whisper model (context manager). Blocking: call from a thread."""

    def load():
        from faster_whisper import WhisperModel

        _preload_cuda_libs()
        return WhisperModel(model_name, device=device, compute_type=compute_type, **kwargs)

    return manager.acquire(
        ("asr.faster_whisper", model_name, device, compute_type),
        load,
        est_vram_gb=whisper_vram_gb(model_name, device, compute_type),
    )


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

        model_name = str(self.opt("model", "large-v3-turbo"))
        device = str(self.opt("device", "auto"))
        compute_type = str(self.opt("compute_type", "auto"))
        vad = self.opt_bool("vad", True)
        lang = None if language in ("", "auto") else language

        duration = await wav_duration(audio)

        def _run() -> list[ASRSegment]:
            with acquire_whisper(model_name, device, compute_type) as model:
                return _decode(model)

        def _decode(model) -> list[ASRSegment]:
            # The segment iterator is lazy: it must be drained while the model is acquired.
            seg_iter, _info = model.transcribe(
                str(audio),
                language=lang,
                vad_filter=vad,
                beam_size=5,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
            out: list[ASRSegment] = []
            for seg in seg_iter:
                # Segment bounds can swallow long silences/music around the speech (the VAD merges
                # regions), which would give the dub a wildly wrong slot - tighten to the words.
                start, end = seg.start, seg.end
                if seg.words:
                    start, end = seg.words[0].start, seg.words[-1].end
                words = [
                    Word(start=w.start, end=w.end, text=w.word, confidence=w.probability)
                    for w in (seg.words or [])
                ]
                out.append(
                    ASRSegment(start=start, end=end, text=(seg.text or "").strip(), words=words)
                )
                frac = min(1.0, seg.end / duration) if duration > 0 else 0.0
                progress(frac, f"transcribing {seg.end:.1f}s / {duration:.1f}s")
            return out

        progress(0.0, f"loading {model_name} ({device}/{compute_type})")
        result = await asyncio.to_thread(_run)
        progress(1.0, "done")
        return result
