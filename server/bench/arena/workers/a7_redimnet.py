"""A7 worker: ReDimNet / ReDimNet2 via torch.hub (MIT).

ReDimNet2 (PalabraAI/redimnet2, Interspeech 2026): ``torch.hub.load("PalabraAI/redimnet2",
"redimnet2", model_name="b6", train_type="lm", pretrained=True)``; ReDimNet v1
(IDRnD/ReDimNet): ``torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b6",
train_type="ptn", dataset="vox2")``. Input [N, T] 16 kHz waveform → [N, 192] (repo READMEs,
2026-09-28). params: repo, entry, model_name, train_type, dataset, ref (git tag/commit for
``repo:ref``).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a7_common import score_item


def load(params: dict, lang: str):
    import torch

    repo = params.get("repo", "PalabraAI/redimnet2")
    if params.get("ref"):
        repo = f"{repo}:{params['ref']}"
    kw = {k: params[k] for k in ("model_name", "train_type", "dataset") if params.get(k)}
    if "redimnet2" in repo.lower():
        kw.setdefault("pretrained", True)
    model = torch.hub.load(repo, params.get("entry", "redimnet2"), trust_repo=True, **kw)
    return {"model": model.cuda().eval(), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import torch

    def embed(path: str):
        y, _ = librosa.load(path, sr=16000, mono=True)
        with torch.no_grad():
            return state["model"](torch.from_numpy(y)[None].cuda()).squeeze(0).cpu().numpy()

    return score_item(embed, item, out, state["params"])


if __name__ == "__main__":
    serve(load, run)
