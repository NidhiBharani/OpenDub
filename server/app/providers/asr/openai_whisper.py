"""Cloud ASR via the OpenAI (or OpenAI-compatible) `/audio/transcriptions` endpoint."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import httpx

from ...models import ASRSegment
from ..base import ASRProvider, ConfigField, ProgressFn, ProviderMeta, register

_MAX_BYTES = 24 * 1024 * 1024  # OpenAI's hard limit is 25 MB; split before we hit it
_CHUNK_SECONDS = 600.0  # 10 minutes
# Chunks are re-encoded to Whisper's native 16 kHz mono s16 (32 kB/s): a 600 s chunk is ~19.2 MB,
# safely under the limit — at the pipeline's 48 kHz stereo it would be ~115 MB and always rejected.
_CHUNK_SAMPLE_RATE = 16000
_CHUNK_CHANNELS = 1


@register
class OpenAIWhisperASR(ASRProvider):
    meta = ProviderMeta(
        id="asr.openai_whisper",
        kind="asr",
        name="OpenAI Whisper API",
        description=(
            "Cloud transcription via OpenAI's /audio/transcriptions endpoint "
            "(or a compatible one)."
        ),
        runtime="cloud",
        fields=[
            ConfigField(key="api_key", label="API key", type="secret", help="OpenAI API key."),
            ConfigField(
                key="base_url",
                label="Base URL",
                type="string",
                default="https://api.openai.com/v1",
            ),
            ConfigField(key="model", label="Model", type="string", default="whisper-1"),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        if not self.opt("api_key"):
            return False, "api_key not configured"
        return True, "ready"

    async def transcribe(
        self, audio: Path, language: str, progress: ProgressFn
    ) -> list[ASRSegment]:
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("OpenAI Whisper: api_key not configured")
        base_url = self.opt("base_url", "https://api.openai.com/v1").rstrip("/")
        model = self.opt("model", "whisper-1")
        lang = None if language in ("", "auto") else language

        size = audio.stat().st_size
        if size <= _MAX_BYTES:
            progress(0.05, "uploading audio")
            segments = await self._call(base_url, api_key, model, audio, lang)
            progress(1.0, "done")
            return segments

        from ...media.ffmpeg import slice_audio, wav_duration

        duration = await wav_duration(audio)
        n_chunks = max(1, math.ceil(duration / _CHUNK_SECONDS))
        out: list[ASRSegment] = []
        with tempfile.TemporaryDirectory(prefix="opendub_asr_") as td:
            tmp_dir = Path(td)
            for i in range(n_chunks):
                chunk_start = i * _CHUNK_SECONDS
                chunk_end = min(duration, chunk_start + _CHUNK_SECONDS)
                if chunk_end <= chunk_start:
                    continue
                chunk_path = tmp_dir / f"chunk_{i:03d}.wav"
                await slice_audio(
                    audio,
                    chunk_path,
                    chunk_start,
                    chunk_end,
                    sample_rate=_CHUNK_SAMPLE_RATE,
                    channels=_CHUNK_CHANNELS,
                )
                progress(i / n_chunks, f"transcribing chunk {i + 1}/{n_chunks}")
                chunk_segments = await self._call(base_url, api_key, model, chunk_path, lang)
                for seg in chunk_segments:
                    out.append(
                        ASRSegment(
                            start=seg.start + chunk_start, end=seg.end + chunk_start, text=seg.text
                        )
                    )
        progress(1.0, "done")
        return out

    async def _call(
        self, base_url: str, api_key: str, model: str, path: Path, language: str | None
    ) -> list[ASRSegment]:
        data: dict[str, str] = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=600.0) as client:
            with path.open("rb") as f:
                files = {"file": (path.name, f, "audio/wav")}
                resp = await client.post(
                    f"{base_url}/audio/transcriptions", headers=headers, data=data, files=files
                )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI Whisper API error {resp.status_code}: {resp.text[:300]}"
                )
            payload = resp.json()

        raw_segments = payload.get("segments") or []
        out = [
            ASRSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
            )
            for seg in raw_segments
        ]
        if not out:
            text = str(payload.get("text") or "").strip()
            if text:
                from ...media.ffmpeg import wav_duration

                dur = await wav_duration(path)
                out.append(ASRSegment(start=0.0, end=dur, text=text))
        return out
