"""A6/A7 worker: NVIDIA NeMo speaker models.

task=diarize: Sortformer (``SortformerEncLabelModel.from_pretrained(id).diarize(audio=[path],
batch_size=1)`` → per-file segments "begin end speaker_k"; max 4 speakers; 16 kHz mono). The
streaming v2/v2.1 checkpoints get the card's streaming settings (chunk_len 340, right context
40, fifo 40, spkcache update 300, spkcache 188) unless ``streaming: false``.
task=embed (A7): TitaNet-L ``EncDecSpeakerLabelModel.from_pretrained(id).get_embedding(path)``.
Model cards checked 2026-09-28. params: model, task, streaming.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

STREAMING = {"chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40,
             "spkcache_update_period": 300, "spkcache_len": 188}


def load(params: dict, lang: str):
    task = params.get("task", "diarize")
    if task == "embed":
        import nemo.collections.asr as nemo_asr

        m = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(params["model"])
        return {"task": task, "model": m.cuda().eval(), "params": params}
    from nemo.collections.asr.models import SortformerEncLabelModel

    m = SortformerEncLabelModel.from_pretrained(params["model"]).cuda().eval()
    if params.get("streaming", "streaming" in params["model"]):
        for k, v in STREAMING.items():
            setattr(m.sortformer_modules, k, v)
    return {"task": task, "model": m, "params": params}


def _parse(seg) -> dict | None:
    if isinstance(seg, str):
        parts = seg.split()
        if len(parts) >= 3:
            return {"start": float(parts[0]), "end": float(parts[1]), "speaker": parts[2]}
        return None
    if isinstance(seg, (list, tuple)) and len(seg) >= 3:
        return {"start": float(seg[0]), "end": float(seg[1]), "speaker": str(seg[2])}
    return None


def run(state: dict, item: dict, out: Path) -> dict:
    from a1_common import read

    audio = item["inputs"]["audio"]
    x, sr = read(audio, 16000, mono=True)
    if sr != 16000 or not audio.endswith(".wav"):
        import soundfile as sf

        audio = str(out.with_suffix(".16k.wav"))
        sf.write(audio, x[0], 16000)
    if state["task"] == "embed":
        from a7_common import score_item

        def emb(path: str):
            import numpy as np

            return np.asarray(state["model"].get_embedding(path).cpu(), np.float32).reshape(-1)

        return score_item(emb, item, out, state["params"])
    res = state["model"].diarize(audio=[audio], batch_size=1)
    segs = res[0] if res and isinstance(res[0], list) else res
    return {"turns": [t for t in (_parse(s) for s in segs) if t]}


if __name__ == "__main__":
    serve(load, run)
