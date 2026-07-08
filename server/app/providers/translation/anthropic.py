"""Anthropic Claude translation provider (cloud, API key required).

Talks directly to the Anthropic Messages API via httpx (no ``anthropic`` SDK dependency).
"""
from __future__ import annotations

import httpx

from ...models import TranslationRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TranslationProvider, register
from ._llm import post_with_retries, translate_dubbing

ANTHROPIC_VERSION = "2023-06-01"
API_BASE = "https://api.anthropic.com/v1"


@register
class AnthropicTranslationProvider(TranslationProvider):
    meta = ProviderMeta(
        id="translation.anthropic",
        kind="translation",
        name="Anthropic Claude",
        description="Cloud translation via the Anthropic Messages API. Requires an API key.",
        runtime="cloud",
        fields=[
            ConfigField(
                key="api_key",
                label="API key",
                type="secret",
                default="",
                help="Anthropic API key (sk-ant-...).",
            ),
            ConfigField(
                key="model",
                label="Model",
                type="string",
                default="claude-sonnet-4-5",
                placeholder="claude-sonnet-4-5",
                help="Anthropic model id used for translation.",
            ),
            ConfigField(
                key="max_tokens",
                label="Max tokens",
                type="number",
                default=4096,
                help="Max output tokens per request (bounds how large a batch's response can be).",
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
                    f"{API_BASE}/models",
                    headers=_headers(api_key),
                )
            if resp.status_code >= 400:
                return False, f"Anthropic API returned HTTP {resp.status_code}: {resp.text[:200]}"
            return True, "ready"
        except Exception as e:
            return False, f"could not reach Anthropic API: {e}"

    async def translate(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        progress: ProgressFn,
    ) -> list[str]:
        api_key = self.opt("api_key")
        if not api_key:
            raise RuntimeError("translation.anthropic: api_key is not configured")
        model = self.opt("model", "claude-sonnet-4-5")
        max_tokens = int(self.opt("max_tokens", 4096) or 4096)
        headers = _headers(api_key)

        async with httpx.AsyncClient() as client:

            async def complete(messages: list[dict[str, str]]) -> str:
                system_parts = [m["content"] for m in messages if m["role"] == "system"]
                convo = [m for m in messages if m["role"] != "system"]
                body: dict = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": convo,
                }
                if system_parts:
                    body["system"] = "\n\n".join(system_parts)
                resp = await post_with_retries(
                    client, f"{API_BASE}/messages", json=body, headers=headers
                )
                data = resp.json()
                try:
                    return "".join(
                        block.get("text", "")
                        for block in data.get("content", [])
                        if block.get("type") == "text"
                    )
                except (AttributeError, TypeError) as e:
                    raise RuntimeError(
                        f"unexpected response shape from Anthropic API: {str(data)[:300]}"
                    ) from e

            return await translate_dubbing(complete, requests, source_lang, target_lang, progress)


def _headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
