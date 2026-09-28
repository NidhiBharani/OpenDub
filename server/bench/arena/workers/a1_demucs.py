"""A1 worker: Demucs v4 (htdemucs / htdemucs_ft / mdx_extra), the current OpenDub separator.

params: model, shifts, overlap, bed_by_subtraction (default true, OpenDub's rule).
dialogue = the "vocals" source; bed = mix − dialogue (or the sum of the other sources).
Runs in the server env (demucs is the ``separation`` extra); models download from dl.fbaipublicfiles
on first use. 44.1 kHz stereo native.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a1_common import finish, read


def load(params: dict, lang: str):
    import torch
    from demucs.pretrained import get_model

    model = get_model(params.get("model", "htdemucs"))
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return {"model": model, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np
    import torch
    from demucs.apply import apply_model

    m, p = state["model"], state["params"]
    mix, sr = read(item["inputs"]["audio"], m.samplerate)
    if mix.shape[0] == 1:
        mix = np.repeat(mix, 2, axis=0)
    x = torch.from_numpy(mix[:2]).to(next(m.parameters()).device)
    ref = x.mean(0)
    xn = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        est = apply_model(m, xn[None], shifts=int(p.get("shifts", 2)),
                          overlap=float(p.get("overlap", 0.5)), progress=False)[0]
    est = (est * (ref.std() + 1e-8) + ref.mean()).cpu().numpy()
    src = list(m.sources)
    voc = est[src.index("vocals")]
    others = sum(est[k] for k, s in enumerate(src) if s != "vocals")
    return finish(out, sr, mix[:2], voc, background=others,
                  subtract=bool(p.get("bed_by_subtraction", True)))


def describe(state: dict) -> dict:
    import demucs

    return {"model": state["params"].get("model", "htdemucs"), "demucs": demucs.__version__}


if __name__ == "__main__":
    serve(load, run, describe)
