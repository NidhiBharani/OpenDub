"""OpenAI-compatible chat-completions translation provider.

Works with any server implementing the ``/v1/chat/completions`` API: Ollama, vLLM, LM Studio,
text-generation-webui, etc. Runs fully local/self-hosted (a base_url pointing at a cloud-hosted
OpenAI-compatible endpoint is also possible, but the common case - and the default - is local).
"""
from __future__ import annotations

import httpx

from ...models import TranslationRequest
from ..base import ConfigField, ProgressFn, ProviderMeta, TranslationProvider, register
from ._llm import post_with_retries, translate_dubbing


@register
class OpenAICompatibleTranslationProvider(TranslationProvider):
    meta = ProviderMeta(
        id="translation.openai_compatible",
        kind="translation",
        name="OpenAI-compatible / Ollama",
        description=(
            "Talks to any OpenAI-compatible chat-completions server: Ollama, vLLM, LM Studio, "
            "text-generation-webui, etc. Fully self-hosted/local."
        ),
        runtime="local",
        fields=[
            ConfigField(
                key="base_url",
                label="Base URL",
                type="string",
                default="http://localhost:11434/v1",
                placeholder="http://localhost:11434/v1",
                help=(
                    "OpenAI-compatible API base URL. Ollama: http://localhost:11434/v1 - "
                    "vLLM/LM Studio: whatever host:port they serve on, with /v1 appended."
                ),
            ),
            ConfigField(
                key="model",
                label="Model",
                type="string",
                default="qwen2.5:14b",
                placeholder="qwen2.5:14b",
                help="Model name as known to the server (e.g. an Ollama tag).",
            ),
            ConfigField(
                key="api_key",
                label="API key",
                type="secret",
                default="",
                help="Only needed if your server requires auth (most local servers don't).",
            ),
            ConfigField(
                key="temperature",
                label="Temperature",
                type="number",
                default=0.3,
                help="Sampling temperature for translation.",
            ),
        ],
    )

    def available(self, deep: bool = False) -> tuple[bool, str]:
        base_url = self.opt("base_url")
        model = self.opt("model")
        if not base_url or not model:
            return False, "base_url and model are required"
        if not deep:
            return True, "configured"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{str(base_url).rstrip('/')}/models",
                    headers=_auth_headers(self.opt("api_key")),
                )
            if resp.status_code >= 400:
                return False, f"server returned HTTP {resp.status_code}"
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
        base_url = str(self.opt("base_url", "http://localhost:11434/v1")).rstrip("/")
        model = self.opt("model", "qwen2.5:14b")
        temperature = float(self.opt("temperature", 0.3) or 0.3)
        headers = _auth_headers(self.opt("api_key"))

        async with httpx.AsyncClient() as client:

            async def complete(messages: list[dict[str, str]]) -> str:
                resp = await post_with_retries(
                    client,
                    f"{base_url}/chat/completions",
                    json={"model": model, "messages": messages, "temperature": temperature},
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


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
