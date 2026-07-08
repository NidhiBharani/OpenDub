"""OpenAI GPT translation provider (cloud, API key required).

Talks directly to the OpenAI chat-completions API via httpx (no ``openai`` SDK dependency).
"""
from __future__ import annotations

import httpx

from ...models import TranslationRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TranslationProvider, register
from ._llm import post_with_retries, translate_dubbing


@register
class OpenAITranslationProvider(TranslationProvider):
    meta = ProviderMeta(
        id="translation.openai",
        kind="translation",
        name="OpenAI GPT",
        description="Cloud translation via the OpenAI chat-completions API. Requires an API key.",
        runtime="cloud",
        fields=[
            ConfigField(
                key="api_key",
                label="API key",
                type="secret",
                default="",
                help="OpenAI API key (sk-...).",
            ),
            ConfigField(
                key="model",
                label="Model",
                type="string",
                default="gpt-4o-mini",
                placeholder="gpt-4o-mini",
                help="OpenAI model id used for translation.",
            ),
            ConfigField(
                key="base_url",
                label="Base URL",
                type="string",
                default="https://api.openai.com/v1",
                placeholder="https://api.openai.com/v1",
                help="Override to use an Azure/OpenAI-proxy endpoint.",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        api_key = self.opt("api_key")
        if not api_key:
            return False, "api_key is required"
        if not deep:
            return True, "configured"
        base_url = str(self.opt("base_url", "https://api.openai.com/v1")).rstrip("/")
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{base_url}/models",
                    headers=_headers(api_key),
                )
            if resp.status_code >= 400:
                return False, f"OpenAI API returned HTTP {resp.status_code}: {resp.text[:200]}"
            return True, "ready"
        except Exception as e:
            return False, f"could not reach {base_url}: {e}"

    async def translate(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        progress: ProgressFn,
    ) -> list[str]:
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("translation.openai: api_key is not configured")
        model = self.opt("model", "gpt-4o-mini")
        base_url = str(self.opt("base_url", "https://api.openai.com/v1")).rstrip("/")
        headers = _headers(api_key)

        async with httpx.AsyncClient() as client:

            async def complete(messages: list[dict[str, str]]) -> str:
                resp = await post_with_retries(
                    client,
                    f"{base_url}/chat/completions",
                    json={"model": model, "messages": messages, "temperature": 0.3},
                    headers=headers,
                )
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(
                        f"unexpected response shape from {base_url}: {str(data)[:300]}"
                    ) from e

            return await translate_dubbing(complete, requests, source_lang, target_lang, progress)


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
