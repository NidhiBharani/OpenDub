"""Mock translation provider - deterministic, offline, always available.

Useful for exercising the full pipeline (and for tests) without any network access or optional
dependency. Simply prefixes each source line with "(en) " and truncates it, on a word boundary, to
fit the segment's time budget.
"""
from __future__ import annotations

from ...models import TranslationRequest
from ..base import ProgressFn, ProviderMeta, TranslationProvider, register
from ._llm import max_chars_for


@register
class MockTranslationProvider(TranslationProvider):
    meta = ProviderMeta(
        id="translation.mock",
        kind="translation",
        name="Mock translator",
        description=(
            "Deterministic offline stub: prefixes each line with '(en) ' and truncates it to the "
            "segment's time budget on a word boundary. No network, no dependencies - always "
            "available for testing the pipeline end to end."
        ),
        runtime="local",
        fields=[],
    )

    async def translate(
        self,
        requests: list[TranslationRequest],
        source_lang: str,
        target_lang: str,
        progress: ProgressFn,
    ) -> list[str]:
        out: list[str] = []
        total = len(requests) or 1
        for i, req in enumerate(requests):
            out.append(_mock_translate(req))
            progress((i + 1) / total, f"translated {i + 1}/{len(requests)}")
        return out


def _mock_translate(req: TranslationRequest) -> str:
    text = f"(en) {req.text}".strip()
    budget = max_chars_for(req.duration)
    if len(text) <= budget:
        return text
    return _truncate_on_word_boundary(text, budget)


def _truncate_on_word_boundary(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    cut = text[:budget]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip()
