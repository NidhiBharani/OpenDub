"""Shared rubric for the LLM reviewer workers (g6_*.py); not a worker itself. Stdlib only.

A reviewer backend implements ``ask(state, parts, prompt) -> (text, usage)`` where ``parts`` is a
list of ``("audio", path) | ("video", path) | ("text", str)`` in order. This module turns an item
into queries (:func:`plan`) and the answers into a payload (:func:`combine`), so every backend
(Gemini, GPT audio, Qwen-Omni, Audio Flamingo, a text LLM over transcripts) runs the same protocol.

Protocol rules from the 2026 LALM-judge audits (compute-tiers G6/G4):
- never show other judges' outputs or specialist labels in the prompt (label copying);
- no A/B preference prompts; single-clip rubric questions. Where two clips must be compared
  (reference voice vs take, source vs dub emotion) the question is same/different, asked in both
  orders and averaged (position bias);
- every fault needs evidence (a quote / what was heard) and a timestamp; a fault claim with no
  evidence is discounted (``p`` halved) as a cheap hallucination check.

Tasks (``params.task``):
- ``review`` (G6): defects wrong_speaker, off_sync, clipping, untranslated. ``protocol:
  holistic`` = one prompt for all four; ``decomposed`` = one focused sub-judge per defect, each
  seeing only the inputs it needs (AudioJudge-style lexical / identity / signal / sync split).
  Payload metrics ``p_<defect>`` in 0..1. Audio-only backends cannot see the picture: off_sync is
  reported as 0 (a miss), with ``no_video: true``.
- ``speaker`` (G2): same-speaker rating 1..5 of take vs reference -> ``sim_llm`` in 0..1.
- ``mos`` (G3): naturalness 1..5 -> ``mos_llm``.
- ``emotion`` (G4): "same feeling" 1..5 of dub vs source -> ``emo_sim_llm`` in 0..1.
- ``sync`` (G5): lip-sync fault probability -> ``desync`` = p, ``offset_ms`` null.
"""
from __future__ import annotations

import json
import re

DEFECTS = ("wrong_speaker", "off_sync", "clipping", "untranslated")
LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "ru": "Russian", "ta": "Tamil", "te": "Telugu",
              "bn": "Bengali", "ar": "Arabic"}

SYSTEM = ("You are a meticulous dubbing QC reviewer. Judge only what you can hear and see in the "
          "clips. Answer with a single JSON object and nothing else.")

_FAULT_SCHEMA = ('{"p": <probability 0..1 that the fault is present>, "evidence": "<what you '
                 'heard or saw, empty if none>", "timestamp": "<m:ss in the take, or null>"}')
_RATING_SCHEMA = '{"rating": <integer 1..5>, "evidence": "<one short sentence>"}'


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code or "the target language")


class Query(dict):
    """{key, parts, prompt, swap?}; a dict so it serialises into the payload for audit."""


def _line(inputs: dict) -> str:
    line = f'The take is supposed to say: "{inputs["text"]}".' if inputs.get("text") else ""
    if inputs.get("src_text"):
        line += f' (It dubs the source line: "{inputs["src_text"]}".)'
    return line


def plan(task: str, protocol: str, inputs: dict, lang: str, caps: dict) -> list[Query]:
    """The queries one item needs. ``caps``: {"video": bool}."""
    L = lang_name(lang)
    take, ref, video = inputs.get("audio"), inputs.get("ref_audio"), inputs.get("video")
    if isinstance(ref, list):
        ref = ref[0] if ref else None
    if task == "review":
        if protocol == "holistic":
            parts: list = [("text", "Clip 1 — the dubbed take:"), ("audio", take)]
            if ref:
                parts += [("text", "Clip 2 — a reference recording of the intended voice:"),
                          ("audio", ref)]
            if video and caps.get("video"):
                parts += [("text", "Clip 3 — the take in its video:"), ("video", video)]
            keys = [d for d in DEFECTS if d != "off_sync" or (video and caps.get("video"))]
            if not ref:
                keys = [k for k in keys if k != "wrong_speaker"]
            if not inputs.get("text"):
                keys = [k for k in keys if k != "untranslated"]
            ask = {"wrong_speaker": "the take is voiced by a different person than the "
                                    "reference voice (Clip 2)",
                   "off_sync": "the speech is audibly out of sync with the lips in the video",
                   "clipping": "the take has audible clipping / hard digital distortion",
                   "untranslated": f"the take is not spoken in {L} (e.g. left in the source "
                                   "language) or does not say the intended line"}
            prompt = (f"Review this line of a {L} dub. {_line(inputs)}\nFor each fault below, "
                      "estimate the probability it is present:\n"
                      + "\n".join(f"- {k}: {ask[k]}" for k in keys)
                      + "\nReturn JSON: {" + ", ".join(f'"{k}": {_FAULT_SCHEMA}' for k in keys)
                      + "}")
            return [Query(key="holistic", parts=parts, prompt=prompt, keys=keys)]
        qs = [Query(key="untranslated", parts=[("audio", take)], prompt=(
                  f"Listen to this take from a {L} dub. {_line(inputs)} Is the take NOT spoken "
                  f"in {L} (for example left in the original language), or does it clearly not "
                  f"say the intended line? Return JSON: {_FAULT_SCHEMA}"))] \
            if inputs.get("text") else []
        qs += [Query(key="clipping", parts=[("audio", take)], prompt=(
                  "Listen to this recording. Does it contain audible clipping or hard digital "
                  "distortion (flattened peaks, crackle on loud syllables)? Ignore accent, "
                  f"content and language. Return JSON: {_FAULT_SCHEMA}"))]
        if ref:
            for swap in (False, True):
                a, b = (ref, take) if not swap else (take, ref)
                qs.append(Query(key="wrong_speaker", swap=swap, parts=[
                    ("text", "Recording A:"), ("audio", a), ("text", "Recording B:"),
                    ("audio", b)], prompt=(
                    "Recordings A and B may be in different languages. Ignoring language, "
                    "words, emotion and recording quality: are they spoken by DIFFERENT people? "
                    f"Return JSON: {_FAULT_SCHEMA}")))
        if video and caps.get("video"):
            qs.append(Query(key="off_sync", parts=[("video", video)], prompt=(
                "Watch the speaker's mouth in this video. Is the speech audibly out of sync "
                "with the lip movements (audio early or late)? If no face is visible, answer "
                f"p = 0. Return JSON: {_FAULT_SCHEMA}")))
        return qs
    if task == "speaker":
        return [Query(key="speaker", swap=swap, parts=[
            ("text", "Recording A:"), ("audio", a), ("text", "Recording B:"), ("audio", b)],
            prompt=("Recordings A and B may be in different languages. Ignoring language and "
                    "words, how likely are they the same speaker? 1 = certainly different, "
                    f"5 = certainly the same. Return JSON: {_RATING_SCHEMA}"))
            for swap, (a, b) in ((False, (ref, take)), (True, (take, ref)))]
    if task == "mos":
        return [Query(key="mos", parts=[("audio", take)], prompt=(
            f"Rate the naturalness of this {L} speech recording on the MOS scale: 1 = bad "
            "(clearly synthetic or broken), 5 = excellent (indistinguishable from a natural "
            f"recording). Return JSON: {_RATING_SCHEMA}"))]
    if task == "emotion":
        return [Query(key="emotion", swap=swap, parts=[
            ("text", "Recording A:"), ("audio", a), ("text", "Recording B:"), ("audio", b)],
            prompt=("A and B may be in different languages and say different words. Ignoring "
                    "language, words and voice: how similar is the emotion and delivery "
                    "(feeling, energy)? 1 = very different, 5 = the same feeling. "
                    f"Return JSON: {_RATING_SCHEMA}"))
            for swap, (a, b) in ((False, (ref, take)), (True, (take, ref)))]
    if task == "sync":
        return [Query(key="off_sync", parts=[("video", video)], prompt=(
            "Watch the speaker's mouth in this video. Is the speech out of sync with the lip "
            f"movements? Return JSON: {_FAULT_SCHEMA}"))]
    raise ValueError(f"unknown reviewer task {task!r}")


def parse_json(text: str) -> dict:
    """First JSON object in a model answer (tolerates code fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in answer: {text[:200]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError(f"unterminated JSON in answer: {text[:200]!r}")


def _p(obj) -> float:
    if isinstance(obj, (int, float)):
        p, evidence = float(obj), "?"
    else:
        p = float((obj or {}).get("p", 0.0))
        evidence = str((obj or {}).get("evidence") or "").strip()
    p = min(1.0, max(0.0, p))
    return p * 0.5 if p >= 0.5 and not evidence else p  # unsupported fault claim: discount


def _rating(obj) -> float:
    r = float((obj or {}).get("rating", 3))
    return min(5.0, max(1.0, r))


def combine(task: str, queries: list[Query], answers: list[dict], caps: dict,
            inputs: dict) -> dict:
    """Merge parsed answers (same order as ``queries``) into the worker payload."""
    metrics: dict[str, float] = {}
    if task == "review":
        vals: dict[str, list[float]] = {}
        for q, a in zip(queries, answers):
            if q["key"] == "holistic":
                for k in q["keys"]:
                    vals.setdefault(k, []).append(_p(a.get(k)))
            else:
                vals.setdefault(q["key"], []).append(_p(a))
        for d in DEFECTS:
            metrics[f"p_{d}"] = sum(vals[d]) / len(vals[d]) if d in vals else 0.0
        no_video = bool(inputs.get("video")) and not caps.get("video")
        return {"metrics": metrics, "flags": [d for d in DEFECTS if metrics[f"p_{d}"] >= 0.5],
                "no_video": no_video}
    if task in ("speaker", "emotion", "mos"):
        ratings = [_rating(a) for a in answers]
        r = sum(ratings) / len(ratings)
        name = {"speaker": "sim_llm", "emotion": "emo_sim_llm", "mos": "mos_llm"}[task]
        value = r if task == "mos" else (r - 1.0) / 4.0
        return {"metrics": {name: value}, "score": value, "ratings": ratings}
    if task == "sync":
        p = _p(answers[0]) if answers else 0.0
        return {"metrics": {"p_off_sync": p}, "offset_ms": None, "desync": p, "score": -p}
    raise ValueError(task)


def concat_for_single_audio(parts: list, tmp_dir, sr: int = 16000) -> tuple[list, str]:
    """For backends that accept one audio clip per request: join the clips with 1 s of silence
    and describe the layout in text. Returns (new parts, layout note)."""
    import numpy as np
    from judge_audio import load_mono, write_wav

    clips, labels, t = [], [], 0.0
    pending_label = ""
    for kind, val in parts:
        if kind == "text":
            pending_label = val
        elif kind == "audio":
            wav = load_mono(val, sr)
            labels.append(f"{pending_label or 'clip'} from {t:.1f}s to {t + len(wav) / sr:.1f}s")
            clips += [wav, np.zeros(sr, dtype=np.float32)]
            t += len(wav) / sr + 1.0
            pending_label = ""
    path = write_wav(f"{tmp_dir}/joined.wav", np.concatenate(clips[:-1]), sr)
    note = "The audio contains several clips separated by 1 s of silence: " + "; ".join(labels)
    return [("audio", path)], note


def cost_usd(usage: dict, prices: dict) -> float | None:
    """``prices``: USD per 1M tokens, keys matching ``usage`` (e.g. text_in, audio_in, out)."""
    if not prices:
        return None
    return sum(float(usage.get(k, 0)) * float(v) / 1e6 for k, v in prices.items())


MIME = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mp3", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg", ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".mkv": "video/x-matroska"}


def mime_of(path: str) -> str:
    from pathlib import Path

    return MIME.get(Path(path).suffix.lower(), "application/octet-stream")


def compact_audio(path: str, tmp_dir, sr: int = 16000) -> str:
    """Mono 16 kHz WAV copy (small enough to inline; every backend hears the same signal)."""
    import hashlib

    from judge_audio import load_mono, write_wav

    name = hashlib.sha1(str(path).encode()).hexdigest()[:12]
    return write_wav(f"{tmp_dir}/{name}.wav", load_mono(path, sr), sr)


def with_retries(fn, *, tries: int = 5, base: float = 2.0, retryable=lambda exc: True):
    """Call ``fn()`` with exponential backoff on retryable errors (rate limits, 5xx)."""
    import random
    import time

    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            if attempt == tries - 1 or not retryable(exc):
                raise
            time.sleep(base * 2 ** attempt + random.random())
    raise RuntimeError("unreachable")


def run_item(ask, state, item: dict, lang: str, *, task: str, protocol: str, caps: dict,
             prices: dict | None = None) -> dict:
    """Plan, ask every query, parse (one re-ask on unparseable output), combine, add cost."""
    inputs = item["inputs"]
    queries = plan(task, protocol, inputs, lang, caps)
    answers, raw, usage = [], [], {}
    for q in queries:
        parsed = None
        for _ in range(2):
            text, use = ask(state, q["parts"], q["prompt"])
            for k, v in (use or {}).items():
                usage[k] = usage.get(k, 0) + (v or 0)
            raw.append(text[:2000])
            try:
                parsed = parse_json(text)
                break
            except (ValueError, json.JSONDecodeError):
                continue
        answers.append(parsed or {})
    payload = combine(task, queries, answers, caps, inputs)
    payload.update(task=task, protocol=protocol, answers=raw, usage=usage,
                   unparsed=sum(1 for a in answers if not a))
    c = cost_usd(usage, prices or {})
    if c is not None:
        payload["_cost_usd"] = c
    return payload
