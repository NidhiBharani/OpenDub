"""Cloud lip-sync via Replicate.

Replicate hosts many video-to-video lip-sync models and — unlike most lip-sync APIs, which only
accept public URLs — its files endpoint takes a direct upload, so OpenDub's local render and dub
audio can be sent as-is. Flow: upload both files, create a prediction against the configured model
(latest version), poll it, download the result video.

Caveat for the anime-first use case: every production lip-sync model is trained on **human** faces
and will fail face detection on 2D anime frames (the same limitation as the local LatentSync/
Wav2Lip providers). This provider is for live-action / photoreal sources; leave lipsync set to
`lipsync.none` for anime.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from .._http import download_file, poll_until
from ..base import ConfigField, LipSyncProvider, ProgressFn, ProviderMeta, register


@register
class ReplicateLipSyncProvider(LipSyncProvider):
    meta = ProviderMeta(
        id="lipsync.replicate",
        kind="lipsync",
        name="Replicate (cloud)",
        description=(
            "Cloud lip-sync via a Replicate-hosted model (uploads local files directly). "
            "Human-face models only — not for 2D anime. Needs a Replicate API token."
        ),
        runtime="cloud",
        fields=[
            ConfigField(key="api_key", label="API token", type="secret",
                        help="Replicate API token (r8_...)."),
            ConfigField(
                key="model", label="Model", type="string", default="bytedance/latentsync",
                help="Replicate model as 'owner/name'; its latest version is used. Must accept "
                     "video + audio inputs (e.g. bytedance/latentsync, cjwbw/video-retalking).",
            ),
            ConfigField(
                key="video_field", label="Video input field", type="string", default="video",
                help="Name of the model's video input parameter.",
            ),
            ConfigField(
                key="audio_field", label="Audio input field", type="string", default="audio",
                help="Name of the model's audio input parameter.",
            ),
            ConfigField(key="base_url", label="Base URL", type="string",
                        default="https://api.replicate.com/v1"),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        if not self.opt("api_key"):
            return False, "api_key not configured"
        return True, "ready"

    async def sync(
        self, video: Path, audio: Path, out_video: Path, progress: ProgressFn
    ) -> None:
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("lipsync.replicate: api_key not configured")
        base = str(self.opt("base_url", "https://api.replicate.com/v1")).rstrip("/")
        model = str(self.opt("model", "bytedance/latentsync"))
        vfield = self.opt("video_field", "video")
        afield = self.opt("audio_field", "audio")
        headers = {"Authorization": f"Bearer {api_key}"}
        out_video.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), headers=headers) as client:
            progress(0.05, "uploading video")
            video_url = await self._upload(client, base, video, "video/mp4")
            progress(0.15, "uploading audio")
            audio_url = await self._upload(client, base, audio, "audio/wav")

            progress(0.25, f"creating prediction ({model})")
            create = await client.post(
                f"{base}/models/{model}/predictions",
                json={"input": {vfield: video_url, afield: audio_url}},
            )
            _raise_for(create, "predictions")
            pred = create.json()
            pred_id = pred["id"]

            async def fetch() -> dict:
                r = await client.get(f"{base}/predictions/{pred_id}")
                _raise_for(r, "predictions/get")
                return r.json()

            def on_poll(p: dict) -> None:
                progress(0.35, f"lip-syncing ({p.get('status', '...')})")

            done = await poll_until(
                fetch,
                is_done=lambda p: p.get("status") == "succeeded",
                is_failed=lambda p: p.get("status") in ("failed", "canceled"),
                on_poll=on_poll,
            )

            output = done.get("output")
            url = output[-1] if isinstance(output, list) else output
            if not isinstance(url, str):
                raise RuntimeError(  # noqa: TRY004 - a bad upstream response, not a caller type error
                    f"Replicate returned no output video URL: {str(output)[:200]}")
            progress(0.9, "downloading result")
            await download_file(client, url, out_video)
        progress(1.0, "done")

    async def _upload(
        self, client: httpx.AsyncClient, base: str, path: Path, content_type: str
    ) -> str:
        """Upload a local file to Replicate's files API; return its served URL."""
        with path.open("rb") as f:
            resp = await client.post(
                f"{base}/files", files={"content": (path.name, f, content_type)}
            )
        _raise_for(resp, "files")
        data = resp.json()
        url = (data.get("urls") or {}).get("get") or data.get("url")
        if not url:
            raise RuntimeError(f"Replicate file upload returned no URL: {str(data)[:200]}")
        return url


def _raise_for(resp: httpx.Response, what: str) -> None:
    if resp.status_code >= 400:
        raise RuntimeError(f"Replicate {what} error {resp.status_code}: {resp.text[:300]}")
