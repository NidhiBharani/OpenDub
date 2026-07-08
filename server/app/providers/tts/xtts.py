"""tts.xtts — Coqui XTTS-v2 zero-shot voice cloning (local).

Prefers the segment's own source audio as the conditioning clip (it carries that line's emotional
delivery and prosody) when it exists and is at least 1.5s long; otherwise falls back to the
speaker identity reference. Loading the model takes several seconds, so the loaded instance is
cached process-wide, keyed by (model_name, device), guarded by a lock so concurrent segments don't
each pay the load cost.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

from ...media import ffmpeg
from ...models import TTSRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TTSProvider, register

DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _load_model(model_name: str, device: str) -> Any:
    """Load (or fetch cached) XTTS model. Runs inside a worker thread."""
    key = (model_name, device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS  # heavy, lazy import

        model = TTS(model_name)
        model.to(device)
        _MODEL_CACHE[key] = model
        return model


def _run_tts(
    model: Any, text: str, speaker_wav: str, language: str, speed: float, out_path: Path
) -> None:
    model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=str(out_path),
        speed=speed,
    )


@register
class XTTSProvider(TTSProvider):
    meta = ProviderMeta(
        id="tts.xtts",
        kind="tts",
        name="Coqui XTTS-v2",
        description=(
            "Local zero-shot voice cloning (Coqui XTTS-v2). Conditions on the segment's own "
            "source audio when available (≥1.5s) to carry that line's emotion and prosody into "
            "the dub, falling back to the speaker's identity reference otherwise. First call per "
            "model/device loads the model (~10s); subsequent calls reuse the cached instance."
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
            ConfigField(key="model_name", label="Model", type="string", default=DEFAULT_MODEL),
            ConfigField(key="speed", label="Speed", type="number", default=1.0),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        return self._can_import("TTS")

    async def synthesize(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        progress(0.0, "selecting conditioning audio")
        reference = await self._pick_reference(req)
        device = self._resolve_device()
        model_name = str(self.opt("model_name", DEFAULT_MODEL) or DEFAULT_MODEL)

        progress(0.05, f"loading {model_name} ({device})")
        model = await asyncio.to_thread(_load_model, model_name, device)

        progress(0.4, "synthesizing")
        text = req.text.strip() or " "
        language = (req.language or "en")[:2].lower()
        speed = float(self.opt("speed", 1.0) or 1.0)
        tmp_out = out_wav.with_name(f"{out_wav.stem}.xtts_raw.wav")
        await asyncio.to_thread(_run_tts, model, text, reference, language, speed, tmp_out)

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
            "tts.xtts requires a speaker_reference or segment_reference audio clip to clone a voice"
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
