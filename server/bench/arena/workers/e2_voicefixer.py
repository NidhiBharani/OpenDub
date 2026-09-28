"""E2 worker: VoiceFixer (Liu et al., MIT, ``pip install voicefixer``), analysis-resynthesis
speech restoration with a 44.1 kHz neural vocoder. Old and vocoder-based (highest hallucination
risk in the ledger); benchmarked for completeness.

params: mode (0 = default restoration; 1 adds a high-frequency pre-emphasis; 2 = train mode).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import torch
    from voicefixer import VoiceFixer

    return {"vf": VoiceFixer(), "cuda": torch.cuda.is_available(), "p": params}


def run(state: dict, item: dict, out: Path) -> dict:
    path = out.with_suffix(".wav")
    state["vf"].restore(input=item["inputs"]["audio"], output=str(path), cuda=state["cuda"],
                        mode=int(state["p"].get("mode", 0)))
    return {"files": {"audio": str(path)}}


def describe(state: dict) -> dict:
    return {"model": "voicefixer", "cuda": state["cuda"]}


if __name__ == "__main__":
    serve(load, run, describe)
