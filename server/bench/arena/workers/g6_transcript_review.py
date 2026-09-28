"""G6 method worker: transcript-plus-signal review by a text LLM (Claude), no audio model judging.

The cheap, deterministic third opinion from the G6 research ("transcript plus non-verbal-token
review"): faster-whisper transcribes the take with language identification ON (here the language
is the thing being checked, so it is deliberately not forced), a DSP pass measures clipping, and
Claude — which takes text and images only, no audio or video — reviews the structured facts
against the intended line. It cannot judge voice identity or lip sync, so it reports 0 for
those (misses by design); its value is the lexical/untranslated and clipping axes, where it is
immune to audio-judge shortcut failures.

params: model (claude-opus-5), asr (faster-whisper params: model, compute_type, device),
max_tokens, prices (USD per 1M tokens: ``in``, ``out``). Pricing checked 2026-09-28 (claude-api
skill model table): claude-opus-5 $5 / $25 per 1M input / output tokens. No server-side model
fallback is configured: a refusal is recorded as an error rather than silently judged by a
different model (a judge's identity must stay fixed).

Key: ANTHROPIC_API_KEY. Env: g6_api (anthropic SDK + faster-whisper).
"""
from __future__ import annotations

from pathlib import Path

import g6_rubric as R
from _sdk import CUDA12_LIBS, api_key, preload_pip_cuda_libs, serve


def load(params: dict, lang: str):
    import anthropic

    preload_pip_cuda_libs(*CUDA12_LIBS)
    from faster_whisper import WhisperModel

    asr = {"model": "large-v3", "compute_type": "float16", "device": "cuda",
           **params.get("asr", {})}
    whisper = WhisperModel(asr["model"], device=asr["device"], compute_type=asr["compute_type"])
    client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"))
    return {"client": client, "whisper": whisper, "asr": asr, "params": params, "lang": lang,
            "model": params.get("model", "claude-opus-5")}


def clipping_stats(path: str) -> dict:
    """Share of samples at the digital ceiling and the longest run of consecutive ones."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    x = np.abs(data).max(axis=1)
    peak = float(x.max()) if x.size else 0.0
    ceiling = max(peak, 1e-9) * 0.999
    hot = x >= ceiling if peak >= 0.5 else np.zeros_like(x, dtype=bool)
    longest = run = 0
    for h in hot:
        run = run + 1 if h else 0
        longest = max(longest, run)
    return {"peak_dbfs": round(20 * float(np.log10(max(peak, 1e-9))), 2),
            "clipped_sample_share": round(float(hot.mean()) if x.size else 0.0, 6),
            "longest_clipped_run_samples": int(longest), "sample_rate": int(rate)}


def ask_claude(state: dict, prompt: str) -> tuple[str, dict]:
    import anthropic

    def call():
        return state["client"].messages.create(
            model=state["model"], max_tokens=int(state["params"].get("max_tokens", 4000)),
            system=R.SYSTEM, messages=[{"role": "user", "content": prompt}])

    resp = R.with_retries(call, retryable=lambda e: isinstance(
        e, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)))
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}


def run(state: dict, item: dict, out: Path) -> dict:
    inputs = item["inputs"]
    take = inputs["audio"]
    segs, info = state["whisper"].transcribe(take, language=None, beam_size=5,
                                             condition_on_previous_text=False)
    transcript = " ".join(s.text.strip() for s in segs).strip()
    clip = clipping_stats(take)
    L = R.lang_name(state["lang"])
    keys = list(R.DEFECTS)
    facts = (f"Automatic transcript of the take (Whisper {state['asr']['model']}, language "
             f"auto-detected as '{info.language}' with probability "
             f"{info.language_probability:.2f}): \"{transcript}\"\n"
             f"Signal measurements: {clip}")
    prompt = (f"You review one line of a {L} dub without hearing it; you get an automatic "
              f"transcript and signal measurements instead. {R._line(inputs)}\n{facts}\n"
              "Estimate the probability of each fault. You cannot hear the voice or see the "
              "video, so give p = 0 for wrong_speaker and off_sync.\n"
              f"- untranslated: the take is not spoken in {L} or does not say the intended line "
              "(small ASR slips are not faults)\n"
              "- clipping: audible clipping / hard digital distortion\n"
              "Return JSON: {" + ", ".join(f'"{k}": {R._FAULT_SCHEMA}' for k in keys) + "}")
    text, usage = ask_claude(state, prompt)
    try:
        answer = R.parse_json(text)
    except ValueError:
        answer = {}
    q = R.Query(key="holistic", parts=[], prompt=prompt, keys=keys)
    payload = R.combine("review", [q], [answer], {"video": False}, inputs)
    payload.update(task="review", protocol="transcript", answers=[text[:2000]], usage=usage,
                   transcript=transcript, detected_language=info.language, clipping=clip)
    c = R.cost_usd(usage, state["params"].get("prices") or {})
    if c is not None:
        payload["_cost_usd"] = c
    return payload


def describe(state: dict) -> dict:
    import anthropic
    import faster_whisper

    return {"model": state["model"], "asr": state["asr"]["model"],
            "anthropic": anthropic.__version__, "faster_whisper": faster_whisper.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
