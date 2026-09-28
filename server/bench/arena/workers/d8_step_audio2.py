"""D8 worker (evaluate-only): Step-Audio 2 mini (stepfun-ai/Step-Audio-2-mini, Apache-2.0) speech
LLM prompted for speech-to-speech translation, following the repo's examples.py ``s2st_test``
(commit 76e272b, 2026-09-28): system prompt asking to translate and speak, human audio turn,
assistant turn ``"<tts_start>"`` with ``eot: False``; audio tokens < 6561 go to ``Token2wav``.

Per the examples, S2ST output supports en and zh (S2TT also ja): only the en pack runs.
The source clip is passed as Token2wav's ``prompt_wav`` so the output keeps the source voice.
params: model_path (default ~/.opendub/src/Step-Audio2/Step-Audio-2-mini), max_tokens (2048),
temperature (0.7), voice_prompt (source | default).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve

PROMPTS = {"en": "Please listen carefully to this audio and then translate its content into "
                 "English speech.",
           "zh": "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。"}


def load(params: dict, lang: str):
    dv.require_lang(lang, PROMPTS, "Step-Audio 2 S2ST")
    repo = dv.add_repo_to_path("Step-Audio2")
    from stepaudio2 import StepAudio2
    from token2wav import Token2wav

    path = params.get("model_path") or str(repo / "Step-Audio-2-mini")
    return {"model": StepAudio2(path), "t2w": Token2wav(f"{path}/token2wav"), "repo": repo,
            "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src = (item.get("inputs") or {}).get("audio")
    if not src:
        raise Unsupported("S2ST item has no inputs.audio")
    dv.seed_all(dv.item_seed(item, p))
    messages = [{"role": "system", "content": PROMPTS[state["lang"]]},
                {"role": "human", "content": [{"type": "audio", "audio": src}]},
                {"role": "assistant", "content": "<tts_start>", "eot": False}]
    _tokens, text, audio = state["model"](messages, max_tokens=int(p.get("max_tokens", 2048)),
                                          temperature=float(p.get("temperature", 0.7)),
                                          do_sample=True)
    audio = [x for x in audio if x < 6561]
    prompt = src if p.get("voice_prompt", "source") == "source" else \
        str(state["repo"] / "assets" / "default_female.wav")
    data = state["t2w"](audio, prompt_wav=prompt)
    wav = dv.wav_path(out)
    wav.write_bytes(data)
    return {**dv.audio_payload(wav), "text": text}


if __name__ == "__main__":
    serve(load, run)
