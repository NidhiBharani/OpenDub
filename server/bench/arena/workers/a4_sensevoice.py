"""A4 worker: SenseVoice-Small (FunAudioLLM) via FunASR.

params: model (FunAudioLLM/SenseVoiceSmall), hub (hf|ms), use_itn. Language forced
(``language=lang``; card names zh/yue/en/ja/ko). The rich-transcription tags (<|ja|><|NEUTRAL|>
<|Speech|>…) are stripped with ``rich_transcription_postprocess``. No word timestamps.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve

LANGS = {"zh", "yue", "en", "ja", "ko"}


def load(params: dict, lang: str):
    from funasr import AutoModel

    model = AutoModel(model=params.get("model", "FunAudioLLM/SenseVoiceSmall"),
                      hub=params.get("hub", "hf"), device="cuda:0", disable_update=True)
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    if state["lang"] not in LANGS:
        raise Unsupported(f"SenseVoice has no language code for {state['lang']!r}")
    res = state["model"].generate(input=item["inputs"]["audio"], language=state["lang"],
                                  use_itn=bool(state["params"].get("use_itn", True)))
    raw = res[0]["text"] if res else ""
    return {"text": rich_transcription_postprocess(raw).strip(), "segments": [], "raw": raw,
            "language": state["lang"]}


def describe(state: dict) -> dict:
    import funasr

    return {"model": state["params"].get("model"), "funasr": funasr.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
