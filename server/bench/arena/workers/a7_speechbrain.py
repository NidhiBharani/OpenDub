"""A7 worker: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, Apache-2.0 code,
VoxCeleb-derived weights) — the encoder OpenDub's quality stage loads today.
``EncoderClassifier.from_hparams(source=…, run_opts={"device": "cuda"})``;
``encode_batch(load_audio(path)[None])`` (16 kHz). params: model.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a7_common import score_item


def load(params: dict, lang: str):
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source=params.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
        savedir=str(Path.home() / ".opendub" / "models" / "speechbrain_ecapa"),
        run_opts={"device": "cuda"})
    return {"enc": enc, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    enc = state["enc"]

    def embed(path: str):
        return enc.encode_batch(enc.load_audio(path).unsqueeze(0)).squeeze().cpu().numpy()

    return score_item(embed, item, out, state["params"])


if __name__ == "__main__":
    serve(load, run)
