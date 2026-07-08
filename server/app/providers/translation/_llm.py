"""Shared helpers for LLM-based, dubbing-aware translation providers.

Every LLM translation provider (anthropic.py, openai.py, openai_compatible.py) builds an async
``complete(messages) -> str`` closure around its own HTTP call (model, auth headers, endpoint) and
hands it to :func:`translate_dubbing`, which owns everything provider-agnostic:

- the dubbing system prompt (natural spoken dialogue, time-budget aware, emotion-preserving)
- batching up to :data:`MAX_BATCH_SIZE` lines per request
- building the numbered-JSON user message (with ``max_chars`` computed from each line's duration)
- defensively parsing the model's JSON response (stripping code fences, tolerating stray prose)
- the retry-once / fall-back-to-one-line-per-request policy on malformed output
- per-batch progress reporting

``messages`` is the familiar OpenAI-style list of ``{"role": "system"|"user"|"assistant", "content":
str}`` dicts. Providers whose wire format differs (e.g. Anthropic's separate ``system`` field) adapt
it inside their own ``complete()``.

This module also exposes :func:`post_with_retries`, a small httpx retry helper used by every HTTP-based
provider in this package (including the non-LLM DeepL provider) for consistent 429/5xx backoff.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

import httpx

from ..base import ProgressFn
from ...models import TranslationRequest

# ---- batching / prompt constants ---------------------------------------------------------------

MAX_BATCH_SIZE = 25
CHARS_PER_SECOND = 15.0  # English speaking rate used for the "max ~N characters" time budget
MIN_MAX_CHARS = 12

CompleteFn = Callable[[list[dict[str, str]]], Awaitable[str]]


def max_chars_for(duration: float) -> int:
    """Approximate character budget for a spoken line of the given slot duration (seconds)."""
    return max(MIN_MAX_CHARS, round(max(0.0, duration) * CHARS_PER_SECOND))


def build_system_prompt(source_lang: str, target_lang: str) -> str:
    src = ""
    if source_lang and source_lang.strip().lower() not in ("", "auto"):
        src = f" from {source_lang}"
    return (
        f"You are an expert dubbing translator adapting dialogue{src} into {target_lang} for a "
        "VOICE DUB. The output will be performed by a voice actor or a text-to-speech system, "
        "never read as subtitles.\n\n"
        "Follow these rules for every line:\n"
        "1. Write natural, colloquial SPOKEN dialogue in the target language - how a person would "
        "actually say it out loud - not a stiff, literal, or written-style translation.\n"
        "2. Each line must fit its time budget: you are given `max_chars`, an approximate character "
        "budget (derived from the line's spoken duration, ~15 characters per second of English "
        "speech) that the translation should roughly fit so the dub can be spoken in the time "
        "available. Prefer concise, natural phrasing over padding; it is fine to land a little "
        "under budget, but do not substantially exceed it.\n"
        "3. Preserve each line's emotion, register, tone, and the speaker's intent - match the "
        "energy of a shout, a whisper, sarcasm, hesitation, affection, and so on. Never flatten or "
        "neutralize delivery.\n"
        "4. Keep character names and honorifics natural and consistent for a target-language "
        "audience; never invent, drop, or mistranslate a name.\n"
        "5. Use `context_before` / `context_after` only to resolve ambiguity (pronouns, references, "
        "continuing thoughts) - never translate those context lines themselves.\n"
        "6. Output ONLY the translated line content, inside the required JSON - never add "
        "translator notes, bracketed explanations, parentheticals about tone, or any other "
        "meta-commentary.\n"
    )


def build_user_content(batch: list[tuple[int, TranslationRequest]], target_lang: str) -> str:
    lines = [
        {
            "n": n,
            "speaker": req.speaker_name or "unknown",
            "emotion": req.emotion or "neutral",
            "seconds": round(req.duration, 2),
            "max_chars": max_chars_for(req.duration),
            "text": req.text,
            "context_before": list(req.context_before),
            "context_after": list(req.context_after),
        }
        for n, req in batch
    ]
    payload = json.dumps(lines, ensure_ascii=False)
    return (
        f"Translate the following {len(lines)} line(s) of dialogue into {target_lang} for dubbing.\n"
        "Each object has: n (line id), speaker, emotion, seconds (slot duration), max_chars "
        "(approximate character budget for the translation so it fits the slot when spoken aloud), "
        "text (the source line), context_before/context_after (neighboring lines for disambiguation "
        "only - do not translate them).\n\n"
        f"{payload}\n\n"
        'Respond with STRICT JSON ONLY: an array of objects {"n": <int>, "translation": <string>}, '
        "one per input line, covering every n exactly once. No markdown code fences, no prose "
        "before or after the array."
    )


def _messages(
    batch: list[tuple[int, TranslationRequest]],
    target_lang: str,
    system_prompt: str,
    error_note: str,
) -> list[dict[str, str]]:
    user_content = build_user_content(batch, target_lang)
    if error_note:
        user_content += (
            "\n\nYour previous response could not be used: "
            f"{error_note}\nRespond again with ONLY a valid JSON array in the exact format "
            "requested, covering every line's n exactly once."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---- defensive JSON parsing ----------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_json_array(text: str) -> str:
    text = _strip_fences(text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def parse_translations(raw: str, expected_ns: set[int]) -> dict[int, str]:
    """Parse a strict-JSON-array response into ``{n: translation}``.

    Raises ``ValueError`` (never propagated as a hard failure by itself - callers decide whether to
    retry) on any JSON, shape, or count/id mismatch.
    """
    candidate = _extract_json_array(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e}): {raw[:200]!r}") from e
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}: {raw[:200]!r}")

    out: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict) or "n" not in item or "translation" not in item:
            raise ValueError(f"malformed item (expected {{n, translation}}): {item!r}")
        try:
            n = int(item["n"])
        except (TypeError, ValueError):
            raise ValueError(f"non-integer 'n': {item.get('n')!r}") from None
        translation = item["translation"]
        out[n] = translation if isinstance(translation, str) else str(translation)

    missing = expected_ns - out.keys()
    extra = out.keys() - expected_ns
    if missing or extra:
        raise ValueError(
            f"n mismatch: missing={sorted(missing)} extra={sorted(extra)} "
            f"(expected {len(expected_ns)} lines, got {len(out)})"
        )
    return out


# ---- batching / retry / fallback orchestration ---------------------------------------------------


async def _translate_batch_with_fallback(
    complete: CompleteFn,
    batch: list[tuple[int, TranslationRequest]],
    target_lang: str,
    system_prompt: str,
) -> dict[int, str]:
    expected = {n for n, _ in batch}
    error_note = ""
    last_error = ""
    raw = ""
    for _attempt in range(2):  # first try + one retry with an error note
        raw = await complete(_messages(batch, target_lang, system_prompt, error_note))
        try:
            return parse_translations(raw, expected)
        except ValueError as e:
            error_note = str(e)
            last_error = error_note

    # Still bad after one retry: fall back to one line per request for this batch.
    out: dict[int, str] = {}
    for n, req in batch:
        raw = await complete(_messages([(n, req)], target_lang, system_prompt, ""))
        try:
            out.update(parse_translations(raw, {n}))
        except ValueError as e:
            raise RuntimeError(
                f"translation provider returned malformed output for line {n} "
                f"(previous batch error: {last_error[:150]}): {e} - response excerpt: {raw[:300]!r}"
            ) from e
    return out


async def translate_dubbing(
    complete: CompleteFn,
    requests: list[TranslationRequest],
    source_lang: str,
    target_lang: str,
    progress: ProgressFn,
) -> list[str]:
    """Translate every request via an LLM ``complete()`` function, batching up to
    :data:`MAX_BATCH_SIZE` lines per call, with the retry/fallback policy described in the module
    docstring. Returns one translation per request, in the same order as ``requests``.
    """
    if not requests:
        return []

    system_prompt = build_system_prompt(source_lang, target_lang)
    indexed = list(enumerate(requests))
    batches = [indexed[i : i + MAX_BATCH_SIZE] for i in range(0, len(indexed), MAX_BATCH_SIZE)]
    results: dict[int, str] = {}
    total = len(batches)
    done_lines = 0
    for bi, batch in enumerate(batches):
        translations = await _translate_batch_with_fallback(complete, batch, target_lang, system_prompt)
        results.update(translations)
        done_lines += len(batch)
        progress((bi + 1) / total, f"translated {done_lines}/{len(requests)} lines")

    return [results[i] for i in range(len(requests))]


# ---- shared HTTP retry helper (used by all LLM providers + DeepL) --------------------------------


async def post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    retries: int = 2,
    timeout: float = 120.0,
    **kwargs: Any,
) -> httpx.Response:
    """POST with retry on 429/5xx and network errors, exponential backoff (1s, 2s, ...).

    Non-retryable 4xx responses raise immediately. On final failure, raises ``RuntimeError`` with an
    excerpt of the provider's error message.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.post(url, timeout=timeout, **kwargs)
        except httpx.HTTPError as e:
            last_err = e
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            elif resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            else:
                return resp
        if attempt < retries:
            await asyncio.sleep(float(2**attempt))
    raise RuntimeError(str(last_err) if last_err else f"request to {url} failed")
