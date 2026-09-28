"""A8 worker: emotion2vec+ (SJTU / Alibaba; FunASR). ``AutoModel(model=…, hub="hf")``;
``generate(path, granularity="utterance", extract_embedding=False)`` → ``[{key, labels,
scores}]`` over 9 classes (angry, disgusted, fearful, happy, neutral, other, sad, surprised,
unknown; labels may be bilingual "生气/angry"). 16 kHz. Checked against the HF card / GitHub
README 2026-09-28. Licence: code MIT, weights "model-license" (read before shipping).
params: model (emotion2vec/emotion2vec_plus_large | _base | _seed), hub.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a8_llm import duration


def load(params: dict, lang: str):
    from funasr import AutoModel

    model = AutoModel(model=params.get("model", "emotion2vec/emotion2vec_plus_large"),
                      hub=params.get("hub", "hf"), device="cuda:0", disable_update=True)
    return {"model": model, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    audio = item["inputs"]["audio"]
    res = state["model"].generate(audio, granularity="utterance", extract_embedding=False)
    row = res[0] if res else {}
    dur = duration(audio)
    labels = [str(lab).split("/")[-1] for lab in row.get("labels", [])]
    scores = [float(s) for s in row.get("scores", [])]
    ranked = sorted(zip(labels, scores), key=lambda t: -t[1])
    return {"segments": [{"start": 0.0, "end": dur, "label": lab, "score": s}
                         for lab, s in ranked],
            "emotion": ranked[0][0] if ranked else None}


if __name__ == "__main__":
    serve(load, run)
