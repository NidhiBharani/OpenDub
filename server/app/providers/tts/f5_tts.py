"""tts.f5_tts — F5-TTS zero-shot voice cloning (local).

Reference selection: prefers the segment's own source audio (≥1.5s) as the style/emotion
conditioning clip, falling back to the speaker identity reference. The
reference transcript passed to F5-TTS is left empty on purpose — F5-TTS then runs its own internal
ASR pass over the reference clip to align the cloning conditioning, so no ground-truth transcript
of the reference audio is required (at the cost of a little extra latency per call).
"""
from __future__ import annotations

import asyncio
import os
import threading
import uuid
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


def _run_infer(
    model: Any, ref_file: str, ref_text: str, gen_text: str, out_path: Path
) -> None:
    # f5_tts's seed_everything() writes random.randint(0, sys.maxsize) into PYTHONHASHSEED —
    # almost always outside the valid [0, 2**32-1] range — which makes every python subprocess
    # spawned afterwards (demucs, latentsync, wav2lip) die at interpreter startup. Snapshot the
    # variable and restore it after inference so the process environment stays clean.
    hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        # With ref_text="" F5-TTS transcribes the reference clip itself; we pass the known
        # transcript when we have one (see synthesize) because the internal ASR fails on hard
        # audio (sung/stylized lines), collapsing the duration estimate to near-zero output.
        model.infer(
            ref_file=ref_file, ref_text=ref_text, gen_text=gen_text, file_wave=str(out_path)
        )
    finally:
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed


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
        reference, ref_text = await self._pick_reference(req)
        device = self._resolve_device()
        model_name = str(self.opt("model", DEFAULT_MODEL) or DEFAULT_MODEL)

        progress(0.05, f"loading {model_name} ({device})")
        model = await asyncio.to_thread(_load_model, model_name, device)

        progress(0.4, "synthesizing")
        text = req.text.strip() or " "
        # Unique per call: an abandoned (uncancellable) synthesis thread from a cancelled job
        # must never share a tmp path with a later job's write.
        tmp_out = out_wav.with_name(f"{out_wav.stem}.f5_raw.{uuid.uuid4().hex[:8]}.wav")
        await asyncio.to_thread(_run_infer, model, reference, ref_text, text, tmp_out)

        # Degenerate-output guard: when conditioning goes wrong (e.g. F5's internal ASR returns
        # nothing for a sung reference), it emits a near-empty clip. Retry once on the speaker
        # identity reference before accepting the result.
        duration = await ffmpeg.wav_duration(tmp_out)
        if duration < 0.25 and len(text) > 2 and req.speaker_reference and (
            reference != req.speaker_reference
        ):
            progress(0.6, "output degenerate; retrying with speaker reference")
            tmp_out.unlink(missing_ok=True)
            await asyncio.to_thread(
                _run_infer, model, req.speaker_reference, "", text, tmp_out
            )

        progress(0.9, "standardizing audio")
        await ffmpeg.to_std_wav(tmp_out, out_wav)
        tmp_out.unlink(missing_ok=True)
        progress(1.0, "done")

    async def _pick_reference(self, req: TTSRequest) -> tuple[str, str]:
        """Choose the conditioning clip and (if known) its transcript.

        Returns (reference_path, ref_text). ref_text is only non-empty for the segment
        reference, whose transcript is the segment's own source_text; passing it spares
        F5-TTS an internal ASR pass that fails on hard (e.g. sung) audio."""
        use_segment = self.opt_bool("segment_style", True)
        if use_segment and req.segment_reference:
            try:
                duration = await ffmpeg.wav_duration(Path(req.segment_reference))
            except Exception:
                duration = 0.0
            if duration >= 1.5:
                return req.segment_reference, req.segment_reference_text.strip()
        if req.speaker_reference:
            return req.speaker_reference, ""
        if req.segment_reference:
            return req.segment_reference, req.segment_reference_text.strip()
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
