"""tts.elevenlabs — cloud TTS with instant voice cloning (ElevenLabs).

Clones each speaker's identity once from `speaker_reference` (cached in-process, keyed by a sha1
hash of the reference audio bytes, so repeated takes for the same speaker reuse the same cloned
voice_id instead of re-cloning). Eleven v3 supports bracketed audio tags in its text input;
other ElevenLabs models continue to interpret the translated text naturally.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx

from ...media import ffmpeg
from ...models import TTSRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TTSProvider, register

API_BASE = "https://api.elevenlabs.io"
DEFAULT_MODEL = "eleven_multilingual_v2"

# process-wide cache: sha1(reference bytes) -> ElevenLabs voice_id
_VOICE_CACHE: dict[str, str] = {}


def _v3_text(text: str, emotion: str) -> str:
    """Map known delivery hints to documented Eleven v3 tags, never arbitrary user text."""
    import re

    words = set(re.findall(r"[a-z]+", emotion.lower()))
    for labels, tag in (
        ({"whisper", "whispering", "whispers"}, "whispers"),
        ({"shout", "shouting", "shouts"}, "shouts"),
        ({"angry", "anger"}, "angry"),
        ({"happy", "happiness", "joyful"}, "happy"),
        ({"sad", "sadness"}, "sad"),
        ({"curious"}, "curious"),
        ({"excited", "excitement"}, "excited"),
    ):
        if words & labels:
            return f"[{tag}] {text}"
    return text


def _friendly_error(resp: httpx.Response, action: str) -> RuntimeError:
    detail = ""
    try:
        body = resp.json()
        d = body.get("detail")
        if isinstance(d, dict):
            detail = str(d.get("message") or d.get("status") or d)
        elif d:
            detail = str(d)
        else:
            detail = str(body)
    except ValueError:  # not JSON
        detail = resp.text[:300]
    if resp.status_code == 401:
        return RuntimeError(f"ElevenLabs: invalid or unauthorized API key ({action}). {detail}")
    if resp.status_code == 402:
        return RuntimeError(f"ElevenLabs: quota exceeded / payment required ({action}). {detail}")
    if resp.status_code == 429:
        return RuntimeError(f"ElevenLabs: rate limited ({action}), try again shortly. {detail}")
    return RuntimeError(f"ElevenLabs {action} failed ({resp.status_code}): {detail}")


@register
class ElevenLabsProvider(TTSProvider):
    meta = ProviderMeta(
        id="tts.elevenlabs",
        kind="tts",
        name="ElevenLabs",
        description=(
            "Cloud instant voice cloning. Clones the speaker's identity from speaker_reference "
            "(cached per speaker for reuse across takes). Eleven v3 uses supported audio tags "
            "for emotion hints; other models interpret the spoken text naturally."
        ),
        runtime="cloud",
        fields=[
            ConfigField(
                key="api_key", label="API key", type="secret",
                help="ElevenLabs API key, from https://elevenlabs.io/app/settings/api-keys",
            ),
            ConfigField(key="model", label="Model", type="string", default=DEFAULT_MODEL),
            ConfigField(key="stability", label="Stability", type="number", default=0.4),
            ConfigField(key="similarity", label="Similarity boost", type="number", default=0.8),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        api_key = self.opt("api_key")
        if not api_key:
            return False, "no API key configured"
        if not deep:
            return True, "API key configured"
        try:
            resp = httpx.get(f"{API_BASE}/v1/user", headers={"xi-api-key": api_key}, timeout=15)
        except httpx.HTTPError as exc:
            return False, f"network error reaching ElevenLabs: {exc}"
        if resp.status_code == 200:
            return True, "ready"
        if resp.status_code == 401:
            return False, "invalid API key"
        return False, f"ElevenLabs error {resp.status_code}: {resp.text[:200]}"

    async def synthesize(self, req: TTSRequest, out_wav: Path, progress: ProgressFn) -> None:
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("tts.elevenlabs requires an API key")
        if not req.speaker_reference:
            raise RuntimeError(
                "tts.elevenlabs requires a speaker_reference audio clip to clone the voice"
            )
        ref_path = Path(req.speaker_reference)
        if not ref_path.exists():
            raise RuntimeError(f"speaker reference not found: {ref_path}")

        progress(0.0, "preparing voice clone")
        ref_bytes = await asyncio.to_thread(ref_path.read_bytes)
        digest = hashlib.sha1(ref_bytes).hexdigest()

        async with httpx.AsyncClient(timeout=120, headers={"xi-api-key": api_key}) as client:
            voice_id = _VOICE_CACHE.get(digest)
            if voice_id is None:
                progress(0.1, "creating voice clone")
                resp = await client.post(
                    f"{API_BASE}/v1/voices/add",
                    data={"name": f"opendub-{digest[:8]}"},
                    files={"files": (ref_path.name, ref_bytes, "audio/wav")},
                )
                if resp.status_code >= 400:
                    raise _friendly_error(resp, "voice clone")
                voice_id = resp.json()["voice_id"]
                _VOICE_CACHE[digest] = voice_id

            progress(0.4, "synthesizing speech")
            model_id = str(self.opt("model", DEFAULT_MODEL) or DEFAULT_MODEL)
            text = req.text
            if model_id == "eleven_v3" and self.opt_bool("emotion_tags", True):
                text = _v3_text(text, req.emotion)
            settings = {"stability": self.opt_float("stability", 0.4)}
            if model_id != "eleven_v3":
                settings["similarity_boost"] = self.opt_float("similarity", 0.8)
            resp = await client.post(
                f"{API_BASE}/v1/text-to-speech/{voice_id}",
                headers={"Accept": "audio/mpeg"},
                json={
                    "text": text,
                    "model_id": model_id,
                    "voice_settings": settings,
                },
            )
            if resp.status_code >= 400:
                raise _friendly_error(resp, "text-to-speech")
            audio_bytes = resp.content

        progress(0.8, "standardizing audio")
        tmp_mp3 = out_wav.with_name(f"{out_wav.stem}.el_raw.mp3")
        await asyncio.to_thread(tmp_mp3.write_bytes, audio_bytes)
        await ffmpeg.to_std_wav(tmp_mp3, out_wav)
        tmp_mp3.unlink(missing_ok=True)
        progress(1.0, "done")
