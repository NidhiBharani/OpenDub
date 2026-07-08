"""DeepL translation provider (cloud, API key required).

Unlike the LLM-based providers in this package, DeepL is a dedicated machine-translation API: it has
no concept of "dubbing", speaker emotion, or per-line time budgets - it simply translates text as
accurately as possible. It therefore **ignores the segment's spoken-duration budget entirely**;
translations may run longer or shorter than the original line's timing, which downstream mix/atempo
fitting has to absorb. Use one of the LLM providers when duration-aware phrasing matters more than
raw translation fidelity.
"""
from __future__ import annotations

import httpx

from ...models import TranslationRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TranslationProvider, register
from ._llm import post_with_retries

MAX_BATCH_SIZE = 50


@register
class DeepLTranslationProvider(TranslationProvider):
    meta = ProviderMeta(
        id="translation.deepl",
        kind="translation",
        name="DeepL",
        description=(
            "Cloud machine translation via the DeepL API. High translation fidelity, but not "
            "dubbing-aware: it has no notion of emotion, speaker, or spoken-duration budgets, so "
            "translated lines may not fit their original timing. Requires an API key."
        ),
        runtime="cloud",
        fields=[
            ConfigField(
                key="api_key",
                label="API key",
                type="secret",
                default="",
                help=(
                    "DeepL API key. Free-tier keys end with ':fx' and are routed to the free API "
                    "endpoint automatically."
                ),
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        api_key = self.opt("api_key")
        if not api_key:
            return False, "api_key is required"
        if not deep:
            return True, "configured"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{_base_url(api_key)}/v2/usage",
                    headers=_headers(api_key),
                )
            if resp.status_code >= 400:
                return False, f"DeepL API returned HTTP {resp.status_code}: {resp.text[:200]}"
            return True, "ready"
        except Exception as e:
            return False, f"could not reach DeepL API: {e}"

    async def translate(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        progress: ProgressFn,
    ) -> list[str]:
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("translation.deepl: api_key is not configured")
        if not requests:
            return []

        base_url = _base_url(api_key)
        headers = _headers(api_key)
        target = _deepl_target_lang(target_lang)
        source = _deepl_source_lang(source_lang)

        out: list[str] = []
        batches = [requests[i : i + MAX_BATCH_SIZE] for i in range(0, len(requests), MAX_BATCH_SIZE)]
        total = len(batches)
        async with httpx.AsyncClient() as client:
            for bi, batch in enumerate(batches):
                form: list[tuple[str, str]] = [("target_lang", target)]
                if source:
                    form.append(("source_lang", source))
                form.extend(("text", req.text) for req in batch)

                resp = await post_with_retries(
                    client, f"{base_url}/v2/translate", data=form, headers=headers
                )
                payload = resp.json()
                try:
                    translations = payload["translations"]
                except (KeyError, TypeError) as e:
                    raise RuntimeError(
                        f"unexpected DeepL response shape: {str(payload)[:300]}"
                    ) from e
                if len(translations) != len(batch):
                    raise RuntimeError(
                        f"DeepL returned {len(translations)} translations for a batch of "
                        f"{len(batch)} lines"
                    )
                out.extend(t.get("text", "") for t in translations)
                progress((bi + 1) / total, f"translated {len(out)}/{len(requests)} lines")

        return out


def _base_url(api_key: str) -> str:
    return "https://api-free.deepl.com" if api_key.endswith(":fx") else "https://api.deepl.com"


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"DeepL-Auth-Key {api_key}"}


def _deepl_target_lang(target_lang: str) -> str:
    lang = (target_lang or "").strip()
    if lang.lower() == "en":
        return "EN-US"
    return lang.upper()


def _deepl_source_lang(source_lang: str) -> str | None:
    lang = (source_lang or "").strip()
    if not lang or lang.lower() == "auto":
        return None
    return lang.upper()
