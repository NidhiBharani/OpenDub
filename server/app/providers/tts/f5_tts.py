"""tts.f5_tts — F5-TTS zero-shot voice cloning (local).

Mirrors tts.xtts's reference-selection strategy: prefers the segment's own source audio (≥1.5s)
as the style/emotion conditioning clip, falling back to the speaker identity reference. The
reference transcript passed to F5-TTS is left empty on purpose — F5-TTS then runs its own internal
ASR pass over the reference clip to align the cloning conditioning, so no ground-truth transcript
of the reference audio is required (at the cost of a little extra latency per call).
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from ...media import ffmpeg
from ...models import TTSRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TTSProvider, register

DEFAULT_MODEL = "F5TTS_v1_Base"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_model(model_name: str, device: str) -> Any:
    """Load (or fetch cached) F5-TTS model. Runs inside a worker thread."""
    key = (model_name, device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from f5_tts.api import F5TTS  # heavy, lazy import

        model = F5TTS(model=model_name, device=device)
        _MODEL_CACHE[key] = model
        return model


def _run_infer(model: Any, ref_file: str, gen_text: str, out_path: Path) -> None:
    # ref_text intentionally blank: F5-TTS transcribes the reference clip itself.
    model.infer(ref_file=ref_file, ref_text="", gen_text=gen_text, file_wave=str(out_path))


@register
class F5TTSProvider(TTSProvider):
    meta = ProviderMeta(
        id="tts.f5_tts",
        kind="tts",
        name="F5-TTS",
        description=(
            "Local zero-shot voice cloning (F5-TTS). Conditions on the segment's own source audio "
            "when available (≥1.5s) to carry that line's emotion into the dub, falling back to "
            "the speaker identity reference otherwise. The reference transcript is left blank, so "
            "F5-TTS runs its own ASR over the reference clip to align cloning (adds a little "
            "latency, needs no ground-truth transcript). First call per model/device loads the "
            "model; subsequent calls reuse the cached instance."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="device", label="Device", type="select", default="auto",
                options=["auto", "cuda", "cpu"],
                help="auto picks cuda if a GPU is available, else cpu.",
            ),
            ConfigField(
                key="segment_style", label="Use segment audio as style reference", type="boolean",
                default=True,
                help="Prefer this line's own source audio (≥1.5s) as the emotion/prosody "
                     "conditioning clip; falls back to the speaker reference otherwise.",
            ),
            ConfigField(key="model", label="Model", type="string", default=DEFAULT_MODEL),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return self._can_import("f5_tts")

    async def synthesize(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        progress(0.0, "selecting conditioning audio")
        reference = await self._pick_reference(req)
        device = self._resolve_device()
        model_name = str(self.opt("model", DEFAULT_MODEL) or DEFAULT_MODEL)

        progress(0.05, f"loading {model_name} ({device})")
        model = await asyncio.to_thread(_load_model, model_name, device)

        progress(0.4, "synthesizing (reference ASR + inference)")
        text = req.text.strip() or " "
        tmp_out = out_wav.with_name(f"{out_wav.stem}.f5_raw.wav")
        await asyncio.to_thread(_run_infer, model, reference, text, tmp_out)

        progress(0.9, "standardizing audio")
        await ffmpeg.to_std_wav(tmp_out, out_wav)
        tmp_out.unlink(missing_ok=True)
        progress(1.0, "done")

    async def _pick_reference(self, req: TTSRequest) -> str:
        use_segment = bool(self.opt("segment_style", True))
        if use_segment and req.segment_reference:
            try:
                duration = await ffmpeg.wav_duration(Path(req.segment_reference))
            except Exception:
                duration = 0.0
            if duration >= 1.5:
                return req.segment_reference
        if req.speaker_reference:
            return req.speaker_reference
        if req.segment_reference:
            return req.segment_reference
        raise RuntimeError(
            "tts.f5_tts requires a speaker_reference or segment_reference audio clip to clone a voice"
        )

    def _resolve_device(self) -> str:
        device = str(self.opt("device", "auto") or "auto").lower()
        if device in ("cuda", "cpu"):
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
