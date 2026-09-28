"""A1/A2 worker: Tencent AuK / AuK-Flash (1.5B, MIT, released 2026-09-09) — instruction-driven
generative separation / enhancement. It regenerates audio (24 kHz) instead of masking, so for
A1 the bed is always mix − dialogue (after resampling to the mix rate) and any timing drift of
the regenerated dialogue shows up as bleed in the bed.

Repo README (github.com/Tencent-Hunyuan/AuK, checked 2026-09-28): ``AukInfer(config,
checkpoint).generate(messages, gen_seconds=…)`` → (audio, sr); needs Qwen/Qwen2.5-Omni-3B too;
~25 GiB bf16 (17 GiB with ``cpu_offload``). params: config, checkpoint, instruction, role
(dialogue | audio), cpu_offload.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a1_common import finish, read

A1_INSTR = "Extract all speech and dialogue, remove music and sound effects."
A2_INSTR = ("Preserve all speakers, remove noise and reverberation, and output clean speech of "
            "the same length.")


def load(params: dict, lang: str):
    from auk.infer.infer_auk import AukInfer

    kw = {"cpu_offload": True} if params.get("cpu_offload") else {}
    home = Path.home() / ".opendub" / "src" / "AuK" / "ckpts" / "AuK-Flash"
    eng = AukInfer(params.get("config", str(home / "config.yaml")),
                   params.get("checkpoint", str(home / "auk_flash.safetensors")), **kw)
    return {"eng": eng, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import soundfile as sf

    p = state["params"]
    role = p.get("role", "dialogue")
    audio = item["inputs"]["audio"]
    info = sf.info(audio)
    instr = p.get("instruction") or (A1_INSTR if role == "dialogue" else A2_INSTR)
    msgs = [{"role": "user", "content": [{"type": "text", "text": instr},
                                         {"type": "audio", "audio": audio}]}]
    y, sr = state["eng"].generate(msgs, gen_seconds=float(info.duration))
    y = np.asarray(getattr(y, "cpu", lambda: y)(), dtype=np.float32).reshape(-1)
    if role == "audio":
        dst = out.with_suffix(".wav")
        sf.write(str(dst), y, sr)
        return {"files": {"audio": str(dst)}, "sample_rate": sr}
    import librosa

    mix, msr = read(audio)
    dia = librosa.resample(y, orig_sr=sr, target_sr=msr)
    return finish(out, msr, mix, np.repeat(dia[None], mix.shape[0], axis=0), subtract=True)


if __name__ == "__main__":
    serve(load, run)
