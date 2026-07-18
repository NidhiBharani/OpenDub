"""Cloud source separation via the LALAL.AI API.

Good default for anime, which is music- and SFX-heavy: LALAL.AI's neural splitter pulls clean
dialogue (vocals) out from the score/effects bed so the dub sits on the original background. Flow
per the LALAL.AI API docs: upload the audio, request a vocals/instrumental split, poll until the
job finishes, then download the two stems. Requires an API-key licence.

Endpoints are configurable (``base_url``) in case the API surface moves; the default targets the
documented https://www.lalal.ai/api/ routes.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from .._http import download_file, poll_until
from ..base import ConfigField, ProgressFn, ProviderMeta, SeparationProvider, register


@register
class LalalaiSeparationProvider(SeparationProvider):
    meta = ProviderMeta(
        id="separation.lalalai",
        kind="separation",
        name="LALAL.AI (cloud)",
        description=(
            "Cloud vocal/instrumental separation via LALAL.AI. Pulls dialogue out of the music + "
            "SFX bed — a strong default for anime. Needs a LALAL.AI API licence key."
        ),
        runtime="cloud",
        fields=[
            ConfigField(key="api_key", label="API key", type="secret",
                        help="LALAL.AI licence key (sent as 'Authorization: license <key>')."),
            ConfigField(
                key="splitter", label="Splitter", type="select", default="phoenix",
                options=["phoenix", "orion", "perseus"],
                help="Neural model. 'phoenix'/'perseus' are the newest, highest-quality.",
            ),
            ConfigField(
                key="stem", label="Stem", type="select", default="vocals",
                options=["vocals", "voice"],
                help="What counts as the foreground. 'vocals' (sung+spoken) suits anime dialogue.",
            ),
            ConfigField(key="base_url", label="Base URL", type="string",
                        default="https://www.lalal.ai/api"),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        if not self.opt("api_key"):
            return False, "api_key not configured"
        return True, "ready"

    async def separate(
        self, audio: Path, vocals_out: Path, background_out: Path, progress: ProgressFn
    ) -> None:
        from ...media.ffmpeg import to_std_wav

        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("separation.lalalai: api_key not configured")
        base = str(self.opt("base_url", "https://www.lalal.ai/api")).rstrip("/")
        splitter = self.opt("splitter", "phoenix")
        stem = self.opt("stem", "vocals")
        headers = {"Authorization": f"license {api_key}"}

        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        background_out.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), headers=headers) as client:
            progress(0.05, "uploading audio to LALAL.AI")
            with audio.open("rb") as f:
                up = await client.post(
                    f"{base}/upload/",
                    headers={**headers, "Content-Disposition": f'attachment; filename="{audio.name}"'},
                    content=f.read(),
                )
            _raise_for(up, "upload")
            file_id = up.json()["id"]

            progress(0.15, "queuing split job")
            params = [{"id": file_id, "stem": stem, "splitter": splitter}]
            split = await client.post(f"{base}/split/", data={"params": json.dumps(params)})
            _raise_for(split, "split")

            def is_done(p: dict) -> bool:
                task = p.get("result", {}).get(file_id, {}).get("task", {})
                return task.get("state") == "success"

            def is_failed(p: dict) -> bool:
                task = p.get("result", {}).get(file_id, {}).get("task", {})
                return task.get("state") == "error"

            def on_poll(p: dict) -> None:
                task = p.get("result", {}).get(file_id, {}).get("task", {})
                progress(0.2 + 0.5 * float(task.get("progress", 0) or 0) / 100.0,
                         f"separating ({task.get('progress', 0)}%)")

            async def check() -> dict:
                r = await client.post(f"{base}/check/", data={"id": file_id})
                _raise_for(r, "check")
                return r.json()

            done = await poll_until(check, is_done=is_done, is_failed=is_failed, on_poll=on_poll)
            split_info = done["result"][file_id]["split"]
            stem_url = split_info["stem_track"]      # foreground (dialogue/vocals)
            back_url = split_info["back_track"]       # background (music + SFX)

            progress(0.75, "downloading separated stems")
            tmp_voc = vocals_out.with_suffix(".lalal.tmp")
            tmp_bg = background_out.with_suffix(".lalal.tmp")
            try:
                await download_file(client, stem_url, tmp_voc)
                await download_file(client, back_url, tmp_bg)
                progress(0.9, "standardizing audio")
                await to_std_wav(tmp_voc, vocals_out)
                await to_std_wav(tmp_bg, background_out)
            finally:
                tmp_voc.unlink(missing_ok=True)
                tmp_bg.unlink(missing_ok=True)
        progress(1.0, "done")


def _raise_for(resp: httpx.Response, what: str) -> None:
    if resp.status_code >= 400:
        raise RuntimeError(f"LALAL.AI {what} error {resp.status_code}: {resp.text[:300]}")
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if isinstance(body, dict) and body.get("status") == "error":
        raise RuntimeError(f"LALAL.AI {what} error: {body.get('error', body)}")
