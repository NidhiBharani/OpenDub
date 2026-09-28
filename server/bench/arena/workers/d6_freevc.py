"""D6 worker: FreeVC (OlaWod/FreeVC, MIT) from its repo (``~/.opendub/src/FreeVC``), following
the repo's convert.py: WavLM-Large content (``utils.get_cmodel``), ``SynthesizerTrn`` + checkpoint,
speaker encoder for the non-"-s" variants, ``net_g.infer(c, g=…)`` or ``infer(c, mel=…)``.

params:
  variant   freevc (16 kHz, speaker encoder) | freevc-s (mel-conditioned) | freevc-24 (24 kHz out)
  hpfile / ptfile  override configs/<variant>.json and checkpoints/<variant>.pth
The env recipe fetches checkpoints/, wavlm/WavLM-Large.pt and speaker_encoder/ckpt into the repo
(links in the FreeVC README; not verified on this machine).
"""
from __future__ import annotations

import os
from pathlib import Path

import d_voice as dv
from _sdk import Unsupported, serve


def load(params: dict, lang: str):
    import torch

    repo = dv.add_repo_to_path("FreeVC")
    os.chdir(repo)
    import utils
    from models import SynthesizerTrn

    variant = params.get("variant", "freevc")
    hps = utils.get_hparams_from_file(params.get("hpfile", f"configs/{variant}.json"))
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length, **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(params.get("ptfile", f"checkpoints/{variant}.pth"), net_g, None, True)
    cmodel = utils.get_cmodel(0)
    smodel = None
    if hps.model.use_spk:
        from speaker_encoder.voice_encoder import SpeakerEncoder

        smodel = SpeakerEncoder("speaker_encoder/ckpt/pretrained_bak_5805000.pt")
    out_sr = 24000 if variant == "freevc-24" else hps.data.sampling_rate
    torch.set_grad_enabled(False)
    return {"net_g": net_g, "cmodel": cmodel, "smodel": smodel, "hps": hps, "utils": utils,
            "out_sr": out_sr, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    import librosa
    import torch
    from mel_processing import mel_spectrogram_torch

    hps, utils = state["hps"], state["utils"]
    src = (item.get("inputs") or {}).get("audio")
    if not src:
        raise Unsupported("VC item has no inputs.audio")
    ref, _ = dv.ref_of(item)
    sr = hps.data.sampling_rate
    wav_tgt, _ = librosa.load(ref, sr=sr)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    if state["smodel"] is not None:
        g = torch.from_numpy(state["smodel"].embed_utterance(wav_tgt)).unsqueeze(0).cuda()
        cond = {"g": g}
    else:
        t = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
        cond = {"mel": mel_spectrogram_torch(t, hps.data.filter_length, hps.data.n_mel_channels,
                                             sr, hps.data.hop_length, hps.data.win_length,
                                             hps.data.mel_fmin, hps.data.mel_fmax)}
    wav_src, _ = librosa.load(src, sr=sr)
    c = utils.get_content(state["cmodel"], torch.from_numpy(wav_src).unsqueeze(0).cuda())
    audio = state["net_g"].infer(c, **cond)[0][0].data.cpu().float().numpy()
    wav = dv.write_wav(dv.wav_path(out), audio, state["out_sr"])
    return dv.audio_payload(wav)


def describe(state: dict) -> dict:
    return {"variant": state["params"].get("variant", "freevc")}


if __name__ == "__main__":
    serve(load, run, describe)
