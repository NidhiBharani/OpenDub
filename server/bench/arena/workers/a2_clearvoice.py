"""A2 worker: ClearerVoice-Studio (Alibaba, Apache-2.0; ``clearvoice`` 0.1.2).
``ClearVoice(task='speech_enhancement', model_names=[name])``; ``cv(input_path=…,
online_write=False)`` then ``cv.write(out, output_path=…)``. Models: MossFormer2_SE_48K,
FRCRN_SE_16K, MossFormerGAN_SE_16K (README checked 2026-09-28). params: model.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

RATES = {"MossFormer2_SE_48K": 48000, "FRCRN_SE_16K": 16000, "MossFormerGAN_SE_16K": 16000}


def load(params: dict, lang: str):
    from clearvoice import ClearVoice

    name = params.get("model", "MossFormer2_SE_48K")
    return {"cv": ClearVoice(task="speech_enhancement", model_names=[name]), "name": name}


def run(state: dict, item: dict, out: Path) -> dict:
    res = state["cv"](input_path=item["inputs"]["audio"], online_write=False)
    dst = out.with_suffix(".wav")
    state["cv"].write(res, output_path=str(dst))
    return {"files": {"audio": str(dst)}, "sample_rate": RATES.get(state["name"])}


if __name__ == "__main__":
    serve(load, run)
