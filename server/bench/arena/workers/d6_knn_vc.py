"""D6 worker: kNN-VC (bshall/knn-vc, MIT; WavLM-Large features + prematched HiFi-GAN) through
torch.hub, as in the repo README: ``torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True,
trust_repo=True, pretrained=True, device=…)``, ``get_features(src)``,
``get_matching_set([refs])``, ``match(query, matching_set, topk=4)`` → 16 kHz tensor.

Non-parametric: it cannot invent content, so it is the regression floor for source leak vs SIM.
Quality grows with minutes of target audio; a single short reference is its worst case.
params: topk (4), prematched (true), hub_ref ("bshall/knn-vc:<commit>" to pin).
"""
from __future__ import annotations

from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    import torch

    device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(params.get("hub_ref", "bshall/knn-vc"), "knn_vc",
                           prematched=bool(params.get("prematched", True)), trust_repo=True,
                           pretrained=True, device=device)
    return {"model": model, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    src = (item.get("inputs") or {}).get("audio")
    if not src:
        raise Unsupported("VC item has no inputs.audio")
    ref, _ = dv.ref_of(item)
    refs = [ref, *((item.get("inputs") or {}).get("extra_refs") or [])]
    m = state["model"]
    # kNN-VC expects 16 kHz mono wavs: resample every input once.
    tmp = []
    for k, path in enumerate([src, *refs]):
        conv = Path(out).with_name(f"{Path(out).name}.in{k}.16k.wav")
        dv.ffmpeg_to_wav(path, conv, 16000)
        tmp.append(conv)
    try:
        query = m.get_features(str(tmp[0]))
        matching = m.get_matching_set([str(t) for t in tmp[1:]])
        wav_t = m.match(query, matching, topk=int(state["params"].get("topk", 4)))
    finally:
        for t in tmp:
            t.unlink(missing_ok=True)
    wav = dv.write_wav(dv.wav_path(out), wav_t, 16000)
    return dv.audio_payload(wav)


def describe(state: dict) -> dict:
    return {"model": "bshall/knn-vc", "hub_ref": state["params"].get("hub_ref", "bshall/knn-vc")}


if __name__ == "__main__":
    serve(load, run, describe)
