"""E2 worker: AP-BWE (Lu et al., TASLP 2024), parallel amplitude/phase GAN bandwidth extension.

Code: https://github.com/yxlu-0102/AP-BWE (MIT), cloned by the env recipe to
``~/.opendub/src/ap-bwe``; 48 kHz checkpoints (MIT, weights_LICENSE.txt) from the authors' Google
Drive folder into ``~/.opendub/models/ap-bwe/<variant>/`` (each variant dir holds ``config.json``
and ``g_*``). The worker discovers variants by reading each ``config.json``
(``lr_sampling_rate`` → ``hr_sampling_rate`` 48000) and picks the one matching the item's input
rate; other rates are ``Unsupported``. Inference mirrors ``inference/inference_48k.py``: upsample
the low-rate input to 48 kHz, STFT → model(amp, pha) → iSTFT.

params: repo, ckpt_root, device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    import torch

    repo = Path(params.get("repo", "~/.opendub/src/ap-bwe")).expanduser()
    sys.path.insert(0, str(repo))
    from datasets.dataset import amp_pha_istft, amp_pha_stft
    from env import AttrDict
    from models.model import APNet_BWE_Model

    device = torch.device(params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    variants = {}
    for cfg in sorted(Path(params.get("ckpt_root", "~/.opendub/models/ap-bwe")).expanduser()
                      .glob("*/config.json")):
        h = AttrDict(json.loads(cfg.read_text()))
        ckpts = sorted(cfg.parent.glob("g_*"))
        if int(h.hr_sampling_rate) == 48000 and ckpts:
            variants[int(h.lr_sampling_rate)] = (h, ckpts[-1])
    if not variants:
        raise RuntimeError("no 48 kHz AP-BWE checkpoints found (see bench/envs/e2_ap_bwe.yaml)")
    return {"torch": torch, "device": device, "variants": variants, "models": {},
            "Model": APNet_BWE_Model, "stft": amp_pha_stft, "istft": amp_pha_istft}


def _model(state: dict, sr: int):
    if sr not in state["models"]:
        h, ckpt = state["variants"][sr]
        m = state["Model"](h).to(state["device"])
        sd = state["torch"].load(str(ckpt), map_location=state["device"])
        m.load_state_dict(sd["generator"])
        state["models"][sr] = (h, m.eval())
    return state["models"][sr]


def run(state: dict, item: dict, out: Path) -> dict:
    import soundfile as sf
    import torchaudio.functional as aF

    torch = state["torch"]
    x, sr = sf.read(item["inputs"]["audio"], dtype="float32", always_2d=True)
    if sr not in state["variants"]:
        raise Unsupported(f"no AP-BWE checkpoint for {sr} Hz input "
                          f"(have {sorted(state['variants'])})")
    h, model = _model(state, sr)
    wav = torch.from_numpy(x.mean(axis=1)).unsqueeze(0).to(state["device"])
    lr_up = aF.resample(wav, orig_freq=sr, new_freq=h.hr_sampling_rate)
    with torch.no_grad():
        amp, pha, _ = state["stft"](lr_up, h.n_fft, h.hop_size, h.win_size)
        amp_g, pha_g, _ = model(amp, pha)
        y = state["istft"](amp_g, pha_g, h.n_fft, h.hop_size, h.win_size)
    y = y.squeeze().cpu().numpy()[: lr_up.shape[-1]]
    path = out.with_suffix(".wav")
    sf.write(path, y, int(h.hr_sampling_rate), subtype="PCM_16")
    return {"files": {"audio": str(path)}, "params": {"lr_sampling_rate": sr}}


def describe(state: dict) -> dict:
    return {"variants": {str(k): str(v[1]) for k, v in state["variants"].items()},
            "device": str(state["device"])}


if __name__ == "__main__":
    serve(load, run, describe)
