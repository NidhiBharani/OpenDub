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

_MODEL_CACHE: dict[tuple[str, str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _ensure_audio_loader() -> None:
    """Make `torchaudio.load` work when torchcodec can't load its FFmpeg libraries.

    torchaudio >= 2.9 decodes through torchcodec, which dlopens the system FFmpeg; that fails in
    common setups (no shared FFmpeg, or a conda python whose bundled libstdc++/glib shadow the
    system ones). F5-TTS only loads reference WAVs, so fall back to soundfile in that case.
    """
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    # torchcodec imports fine and only dlopens FFmpeg on first decode, so probe with a real load.
    with tempfile.NamedTemporaryFile(suffix=".wav") as probe:
        sf.write(probe.name, np.zeros(1600, dtype="float32"), 16000)
        try:
            torchaudio.load(probe.name)
        except Exception:  # noqa: BLE001 - RuntimeError/OSError/ImportError depending on version
            broken = True
        else:
            broken = False
    if not broken:
        return

    def _sf_load(path: Any, *_args: Any, **_kwargs: Any) -> tuple[Any, int]:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _sf_load


def _resolve_file(path: str) -> str:
    """Local path as-is; `hf://org/repo/file` is downloaded into the Hugging Face cache."""
    if path.startswith("hf://"):
        from cached_path import cached_path

        return str(cached_path(path))
    return path


def _load_model(model_name: str, device: str, ckpt_file: str = "", vocab_file: str = "") -> Any:
    """Load (or fetch cached) F5-TTS model. Runs inside a worker thread."""
    key = (model_name, device, ckpt_file, vocab_file)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from f5_tts.api import F5TTS  # heavy, lazy import

        _ensure_audio_loader()
        model = F5TTS(
            model=model_name,
            ckpt_file=_resolve_file(ckpt_file) if ckpt_file else "",
            vocab_file=_resolve_file(vocab_file) if vocab_file else "",
            device=device,
        )
        _MODEL_CACHE[key] = model
        return model


def _run_infer(
    model: Any, ref_file: str, ref_text: str, gen_text: str, out_path: Path, speed: float = 1.0
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
            ref_file=ref_file, ref_text=ref_text, gen_text=gen_text, file_wave=str(out_path),
            speed=speed,
        )
    finally:
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed


_VERIFY_MODEL: list[Any] = []


def _norm_words(text: str) -> list[str]:
    import unicodedata

    kept = "".join(
        ch if unicodedata.category(ch)[0] in ("L", "M", "N") else " " for ch in text.lower()
    )
    return kept.split()


def _verify_take(wav: Path, target_text: str, language: str) -> tuple[float, float]:
    """Read a take back with Whisper. Returns (score 0..1, seconds of leaked lead-in to trim).

    Cross-lingual cloning on a single-language checkpoint often re-speaks the tail of the
    reference clip before the actual line, or garbles it. The score is the character similarity
    between what Whisper hears (after skipping up to 3 leaked leading words) and the script.
    """
    from difflib import SequenceMatcher

    from faster_whisper import WhisperModel

    from ..asr.faster_whisper import _preload_cuda_libs

    with _MODEL_LOCK:
        if not _VERIFY_MODEL:
            _preload_cuda_libs()
            _VERIFY_MODEL.append(WhisperModel("large-v3-turbo", device="auto", compute_type="auto"))
    segs, _info = _VERIFY_MODEL[0].transcribe(
        str(wav), language=language, word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False,
    )
    words = [w for seg in segs for w in (seg.words or [])]
    target = " ".join(_norm_words(target_text))
    if not words or not target:
        return 0.0, 0.0
    best_score, best_trim = 0.0, 0.0
    for skip in range(min(4, len(words))):
        heard = " ".join(" ".join(_norm_words(w.word)) for w in words[skip:])
        score = SequenceMatcher(None, heard, target).ratio()
        if score > best_score + 0.02:  # only skip words when it clearly helps
            best_score = score
            best_trim = max(0.0, words[skip].start - 0.05) if skip else 0.0
    return best_score, best_trim


_TRANSLIT_CACHE: dict[tuple[str, str, str], str] = {}

# LLMs follow "Hindi (Devanagari script)" far more reliably than a bare "hi".
_SCRIPT_NAMES = {
    "hi": "Hindi (Devanagari script)",
    "mr": "Marathi (Devanagari script)",
    "bn": "Bengali script",
    "ta": "Tamil script",
    "te": "Telugu script",
    "ru": "Russian (Cyrillic script)",
    "ja": "Japanese katakana",
    "ko": "Korean hangul",
    "ar": "Arabic script",
    "el": "Greek script",
    "th": "Thai script",
}


async def _transliterate(text: str, language: str, base_url: str, model: str) -> str:
    """Spell `text` phonetically in `language`'s script via an OpenAI-compatible LLM.

    A single-language checkpoint (e.g. Hindi) has no Latin letters in its vocab, so an English
    reference transcript can't be aligned to the reference audio: the clone babbles or leaks
    reference words into the output. Returns `text` unchanged if the LLM can't be reached.
    """
    key = (text, language, model)
    if key in _TRANSLIT_CACHE:
        return _TRANSLIT_CACHE[key]
    import httpx

    script = _SCRIPT_NAMES.get(language, f"the native script of language '{language}'")
    prompt = (
        f"Write the following sentence phonetically in {script}, the way a native reader would "
        "spell these exact foreign words. Do NOT translate; every word must be in that script. "
        f"Reply with the transliteration only.\n\n{text}"
    )
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            out = str(resp.json()["choices"][0]["message"]["content"]).strip()
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return text
    out = out.splitlines()[0].strip() if out else text
    out = out.replace("\u200d", "").replace("\u200c", "")  # joiners aren't in F5 vocabs
    _TRANSLIT_CACHE[key] = out
    return out


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
            ConfigField(
                key="model", label="Model", type="string", default=DEFAULT_MODEL,
                help="F5-TTS architecture config (F5TTS_v1_Base, F5TTS_Base, F5TTS_Small, ...). The "
                     "stock checkpoints speak English and Chinese only.",
            ),
            ConfigField(
                key="ckpt_file", label="Custom checkpoint", type="string", default="",
                placeholder="hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors",
                help="Optional fine-tuned checkpoint for another language (local path or hf://). "
                     "Must match the Model config above (the Hindi one is F5TTS_Small).",
            ),
            ConfigField(
                key="vocab_file", label="Custom vocab", type="string", default="",
                placeholder="hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt",
                help="vocab.txt that the custom checkpoint was trained with.",
            ),
            ConfigField(
                key="speed", label="Speaking speed", type="number", default=1.0,
                help="F5-TTS speed factor. Lower it (about 0.7) if a custom checkpoint rushes or "
                     "drops words.",
            ),
            ConfigField(
                key="verify", label="Verify takes with Whisper", type="boolean", default=False,
                help="Custom checkpoints only: read each take back with Faster-Whisper in the "
                     "target language, trim leaked reference words from its start, and retry at "
                     "other speeds when it doesn't match the script. Slower, far more reliable.",
            ),
            ConfigField(
                key="translit_base_url", label="Reference transliteration LLM", type="string",
                default="", placeholder="http://localhost:11434/v1",
                help="Custom checkpoints only: OpenAI-compatible server used to spell the reference "
                     "clip's transcript in the target language's script, which single-language "
                     "checkpoints need to clone across languages. Empty = off.",
            ),
            ConfigField(
                key="translit_model", label="Transliteration model", type="string",
                default="", placeholder="gemma3:12b",
            ),
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
        ckpt_file = str(self.opt("ckpt_file", "") or "").strip()
        vocab_file = str(self.opt("vocab_file", "") or "").strip()
        model = await asyncio.to_thread(_load_model, model_name, device, ckpt_file, vocab_file)
        speed = self.opt_float("speed", 1.0) or 1.0
        translit_url = str(self.opt("translit_base_url", "") or "").strip()
        translit_model = str(self.opt("translit_model", "") or "").strip()
        if ckpt_file and ref_text and translit_url and translit_model:
            ref_text = await _transliterate(ref_text, req.language, translit_url, translit_model)

        progress(0.4, "synthesizing")
        text = req.text.strip() or " "
        # Unique per call: an abandoned (uncancellable) synthesis thread from a cancelled job
        # must never share a tmp path with a later job's write.
        tmp_out = out_wav.with_name(f"{out_wav.stem}.f5_raw.{uuid.uuid4().hex[:8]}.wav")
        if ckpt_file and self.opt_bool("verify", False):
            await self._synthesize_verified(
                model, reference, ref_text, text, tmp_out, speed, req.language, progress
            )
        else:
            await asyncio.to_thread(
                _run_infer, model, reference, ref_text, text, tmp_out, speed
            )

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
                _run_infer, model, req.speaker_reference, "", text, tmp_out, speed
            )

        progress(0.9, "standardizing audio")
        await ffmpeg.to_std_wav(tmp_out, out_wav)
        tmp_out.unlink(missing_ok=True)
        progress(1.0, "done")

    async def _synthesize_verified(
        self, model: Any, reference: str, ref_text: str, text: str, tmp_out: Path,
        speed: float, language: str, progress: ProgressFn,
    ) -> None:
        """Best-of-N synthesis: keep the take Whisper reads back closest to the script."""
        best: tuple[float, float, Path] | None = None
        attempts = [speed, speed * 1.2, speed * 0.85, speed, speed * 1.2]
        for n, attempt_speed in enumerate(attempts):
            cand = tmp_out.with_name(f"{tmp_out.stem}.try{n}.wav")
            await asyncio.to_thread(
                _run_infer, model, reference, ref_text, text, cand, attempt_speed
            )
            score, trim = await asyncio.to_thread(_verify_take, cand, text, language)
            progress(0.4 + 0.1 * (n + 1), f"take check {n + 1}: {score:.0%} match")
            if best is None or score > best[0]:
                if best is not None:
                    best[2].unlink(missing_ok=True)
                best = (score, trim, cand)
            else:
                cand.unlink(missing_ok=True)
            if score >= 0.85:
                break
        if best is None:  # unreachable: attempts is never empty
            raise RuntimeError("tts.f5_tts produced no candidate take")
        _score, trim, cand = best
        if trim > 0:
            await ffmpeg.trim_start(cand, tmp_out, trim)
            cand.unlink(missing_ok=True)
        else:
            cand.replace(tmp_out)

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
