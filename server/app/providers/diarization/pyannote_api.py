"""Cloud diarization via the pyannoteAI API (api.pyannote.ai).

The hosted counterpart of the local `diarization.pyannote` provider — same model lineage, no local
GPU or HF token, premium accuracy. Useful for anime with many distinct character voices. Flow per
the pyannoteAI docs: create a temporary media object and upload the audio to it, POST a diarize
job referencing that media, poll the job, then map the returned speaker turns onto the ASR
segments (shared overlap logic with the local provider).
"""
from __future__ import annotations

from pathlib import Path

import httpx

from ...models import ASRSegment
from .._http import poll_until
from ..base import ConfigField, DiarizationProvider, ProgressFn, ProviderMeta, register
from ._overlap import assign_labels_by_overlap


@register
class PyannoteAPIDiarization(DiarizationProvider):
    meta = ProviderMeta(
        id="diarization.pyannote_api",
        kind="diarization",
        name="pyannoteAI (cloud)",
        description=(
            "Cloud speaker diarization via api.pyannote.ai — the hosted, premium counterpart of "
            "the local pyannote provider. No local GPU or HF token; needs a pyannoteAI API key."
        ),
        runtime="cloud",
        fields=[
            ConfigField(key="api_key", label="API key", type="secret",
                        help="pyannoteAI API key (Bearer token)."),
            ConfigField(key="base_url", label="Base URL", type="string",
                        default="https://api.pyannote.ai/v1"),
            ConfigField(
                key="num_speakers", label="Number of speakers", type="number", default=0,
                help="Fix the speaker count if known (0 = let the model decide).",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        if not self.opt("api_key"):
            return False, "api_key not configured"
        return True, "ready"

    async def diarize(
        self, audio: Path, segments: list[ASRSegment], progress: ProgressFn
    ) -> list[str]:
        if not segments:
            return []
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("diarization.pyannote_api: api_key not configured")
        base = str(self.opt("base_url", "https://api.pyannote.ai/v1")).rstrip("/")
        num_speakers = int(self.opt_float("num_speakers", 0) or 0)
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), headers=headers) as client:
            # 1. reserve a temporary media object and PUT the audio to its presigned URL.
            media_url = f"media://opendub/{audio.stem}-{abs(hash(str(audio))) % 10**8}.wav"
            progress(0.05, "reserving upload slot")
            inp = await client.post(f"{base}/media/input", json={"url": media_url})
            _raise_for(inp, "media/input")
            presigned = inp.json()["url"]

            progress(0.1, "uploading audio")
            with audio.open("rb") as f:
                put = await client.put(presigned, content=f.read(),
                                       headers={"Content-Type": "audio/wav"})
            if put.status_code >= 400:
                raise RuntimeError(f"pyannoteAI upload error {put.status_code}: {put.text[:200]}")

            # 2. start the diarize job.
            progress(0.2, "queuing diarization job")
            body: dict = {"url": media_url}
            if num_speakers > 0:
                body["numSpeakers"] = num_speakers
            job = await client.post(f"{base}/diarize", json=body)
            _raise_for(job, "diarize")
            job_id = job.json()["jobId"]

            # 3. poll the job.
            async def fetch() -> dict:
                r = await client.get(f"{base}/jobs/{job_id}")
                _raise_for(r, "jobs")
                return r.json()

            def on_poll(p: dict) -> None:
                progress(0.3, f"diarizing ({p.get('status', '...')})")

            done = await poll_until(
                fetch,
                is_done=lambda p: p.get("status") == "succeeded",
                is_failed=lambda p: p.get("status") in ("failed", "canceled"),
                on_poll=on_poll,
            )

        # 4. map turns -> per-segment labels.
        turns = [
            (float(t["start"]), float(t["end"]), str(t["speaker"]))
            for t in done.get("output", {}).get("diarization", [])
        ]
        progress(0.9, "mapping speakers to segments")
        labels = assign_labels_by_overlap(segments, turns) if turns else ["S0"] * len(segments)
        progress(1.0, "done")
        return labels


def _raise_for(resp: httpx.Response, what: str) -> None:
    if resp.status_code >= 400:
        raise RuntimeError(f"pyannoteAI {what} error {resp.status_code}: {resp.text[:300]}")
