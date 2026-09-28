"""A7 worker: WeSpeaker model zoo (Apache-2.0 code; VoxCeleb-derived weights CC BY 4.0).
``wespeaker.load_model_local(dir)`` on a snapshot of an HF repo (Wespeaker/wespeaker-voxceleb-
resnet293-LM, -resnet221-LM, -campplus-LM, -ecapa-tdnn512-LM, -redimnet2-B6-LM, SimAM-ResNet100
…; each holds ``avg_model.pt`` + ``config.yaml``) or ``wespeaker.load_model('english')``;
``set_device('cuda:0')``; ``extract_embedding(path)`` (docs/python_package.md, 2026-09-28).
params: model (HF repo id or a load_model name), revision.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a7_common import score_item


def load(params: dict, lang: str):
    import wespeaker

    mid = params.get("model", "english")
    if "/" in mid:
        from huggingface_hub import snapshot_download

        model = wespeaker.load_model_local(snapshot_download(mid, revision=params.get(
            "revision")))
    else:
        model = wespeaker.load_model(mid)
    model.set_device("cuda:0")
    return {"model": model, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    return score_item(lambda p: state["model"].extract_embedding(p).cpu().numpy(), item, out,
                      state["params"])


if __name__ == "__main__":
    serve(load, run)
