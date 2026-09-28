"""E2 worker: ClearerVoice-Studio speech super-resolution (Alibaba Speech Lab), ``MossFormer2_SR_48K``.

The research ledger names "HiFi-SR via ClearerVoice-Studio"; the released toolkit exposes 48 kHz
super-resolution as ``MossFormer2_SR_48K`` (``pip install clearvoice``, Apache-2.0 toolkit; the
checkpoint terms are not stated separately — see notes). Weights download on first use.

params: model (MossFormer2_SR_48K).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from clearvoice import ClearVoice

    name = params.get("model", "MossFormer2_SR_48K")
    return {"cv": ClearVoice(task="speech_super_resolution", model_names=[name]), "model": name}


def run(state: dict, item: dict, out: Path) -> dict:
    path = out.with_suffix(".wav")
    wav = state["cv"](input_path=item["inputs"]["audio"], online_write=False)
    state["cv"].write(wav, output_path=str(path))
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": state["model"]}


if __name__ == "__main__":
    serve(load, run, describe)
