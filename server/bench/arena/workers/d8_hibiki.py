"""D8 worker (evaluate-only): Kyutai Hibiki / Hibiki-Zero simultaneous S2ST (weights CC-BY-4.0).

Hibiki translates fr→en; Hibiki-Zero fr/es/pt/de(/it)→en. Neither accepts ja or hi sources, so the
candidates are registered disabled for the en/hi/ja scope; the worker raises Unsupported for them.
The documented path is the ``moshi`` package CLI::

    python -m moshi.run_inference --hf-repo kyutai/hibiki-1b-pytorch-bf16 in.wav out.wav

run once per item (the model reloads each time). Hibiki-Zero ships its own package
(github.com/kyutai-labs/hibiki-zero); set ``module`` accordingly.
params: hf_repo, module ("moshi.run_inference"), extra_args, sources (accepted source langs).
Output 24 kHz; the text stream is not captured by this CLI path (``text`` left empty, so the
D8 ASR-chrF judge re-transcribes the audio).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    dv.require_lang(lang, ("en",), "Hibiki")
    return {"params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src = (item.get("inputs") or {}).get("audio")
    if not src or dv.src_lang(item) not in set(p.get("sources", ["fr"])):
        raise Unsupported(f"Hibiki does not translate from {dv.src_lang(item)!r}")
    wav = dv.wav_path(out)
    subprocess.run([sys.executable, "-m", p.get("module", "moshi.run_inference"),
                    "--hf-repo", p.get("hf_repo", "kyutai/hibiki-1b-pytorch-bf16"),
                    *p.get("extra_args", []), src, str(wav)], check=True)
    return {**dv.audio_payload(wav), "text": ""}


if __name__ == "__main__":
    serve(load, run)
