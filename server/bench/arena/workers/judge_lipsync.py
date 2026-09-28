"""Judge worker: audio-visual (lip) sync of a talking-face video.

item.inputs: {video, audio?}. When ``audio`` is given and the video has no audio stream, the two
are muxed first (``-c:v copy``), so the judge always hears the audio being evaluated.

Payload: ``{"metrics": {...}, "offset_ms": <signed or null>, "desync": <magnitude>, "score": …}``.
**Sign convention (arena-wide): positive ``offset_ms`` = the audio lags the picture.** Each backend
converts its native convention (``params.sign`` overrides the default). ``desync`` is a
higher-is-worse magnitude every backend provides, so G5 can compare offset estimators with
confidence-only scorers; ``score`` = ``-desync`` (higher = better synced).

One worker, three backends with conflicting deps, one env each (``judge_lipsync_<model>``):

- ``synchformer`` (MIT; v-iashin/Synchformer, repo at ``repo`` default
  ~/.opendub/src/Synchformer): 21 offset classes on a 0.2 s grid over ±2 s; ``exp_name``
  23-12-23T18-33-57 = LRS3-trained (talking heads; default), 24-01-04T16-39-21 = AudioSet.
  The video is re-encoded to 25 fps / 256 px short side / 16 kHz (as the repo's example.py does),
  scored on sliding windows, and the class posteriors are averaged. Native convention: a positive
  class means the audio starts earlier than the picture (audio leads), so ``sign`` = -1.
  Metrics: ``sync_offset_ms_synchformer`` (posterior mean, arena sign),
  ``sync_offset_argmax_ms_synchformer``, ``sync_conf_synchformer`` (max posterior),
  ``sync_p0_synchformer`` (posterior of the 0 s class).
- ``syncnet`` (joonson/syncnet_python, MIT code, VGG research weights; repo default
  ~/.opendub/src/syncnet_python): face detection/tracking via run_pipeline.py, then
  SyncNetInstance.evaluate in process. Metrics ``lse_c`` (confidence), ``lse_d`` (min mean
  distance), ``sync_offset_ms_syncnet`` (frames × 40 ms at 25 fps; ``sign`` default +1,
  unverified), ``sync_conf_syncnet`` (= LSE-C). The best-confidence face track is reported.
  Never use it to rank SyncNet-supervised generators (Wav2Lip family).
- ``peavs`` (Apache-2.0; amazon-science/avgen-eval-toolkit, repo default
  ~/.opendub/src/avgen-eval-toolkit): the human-calibrated 1..5 sync score. Its scripts work on a
  folder, so each item is scored in its own temporary folder. Metrics ``peavs``,
  ``sync_conf_peavs``; no signed offset.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _sdk import Unsupported, serve
from judge_audio import device_of

SYNCHFORMER_LRS3 = "23-12-23T18-33-57"


def _ffprobe_has_audio(path: str) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "json", path], capture_output=True, text=True,
                         check=False).stdout
    try:
        return bool(json.loads(out or "{}").get("streams"))
    except json.JSONDecodeError:
        return False


def _prepared_video(inputs: dict, tmp: Path) -> str:
    video = inputs.get("video")
    if not video:
        raise ValueError("lip-sync judge needs inputs.video")
    audio = inputs.get("audio")
    if audio and audio != video and not _ffprobe_has_audio(video):
        muxed = tmp / "muxed.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", video, "-i", audio,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                        "-shortest", str(muxed)], check=True)
        return str(muxed)
    if not _ffprobe_has_audio(video):
        raise Unsupported("video has no audio track and no inputs.audio was given")
    return video


# ------------------------------------------------------------------ Synchformer

def _synchformer(params: dict, device: str):
    import torch
    from omegaconf import OmegaConf

    repo = Path(os.path.expanduser(params.get("repo", "~/.opendub/src/Synchformer")))
    if not (repo / "example.py").exists():
        raise RuntimeError(f"Synchformer repo not found at {repo}")
    os.chdir(repo)  # the repo resolves ./logs/sync_models/... relative to the cwd
    sys.path.insert(0, str(repo))
    from dataset.dataset_utils import get_video_and_audio
    from dataset.transforms import make_class_grid
    from scripts.train_utils import get_model, get_transforms, prepare_inputs
    from utils.utils import check_if_file_exists_else_download

    exp = params.get("exp_name", SYNCHFORMER_LRS3)
    cfg_path = f"./logs/sync_models/{exp}/cfg-{exp}.yaml"
    ckpt_path = f"./logs/sync_models/{exp}/{exp}.pt"
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)
    cfg = OmegaConf.load(cfg_path)
    # same patch as example.py: feature-extractor weights live inside the sync checkpoint
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model.")
    _, model = get_model(cfg, torch.device(device))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    max_off = float(cfg.data.max_off_sec)
    n_cls = int(cfg.model.params.transformer.params.off_head_cfg.params.out_features)
    grid = make_class_grid(-max_off, max_off, n_cls)
    transform = get_transforms(cfg, ["test"])["test"]
    window = float(params.get("window_s") or cfg.data.get("crop_len_sec") or 5.0)
    hop = float(params.get("hop_s", 2.0))
    sign = float(params.get("sign", -1.0))
    half = bool(cfg.get("training", {}).get("use_half_precision", True))

    def score(video: str, tmp: Path) -> dict:
        enc = tmp / "sf_25fps_256_16k.mp4"
        vf = ("fps=25,scale=iw*256/'min(iw,ih)':ih*256/'min(iw,ih)',"
              "crop='trunc(iw/2)'*2:'trunc(ih/2)'*2")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", video, "-vf", vf,
                        "-ar", "16000", str(enc)], check=True)
        rgb, audio, meta = get_video_and_audio(str(enc), get_meta=True)
        dur = min(rgb.shape[0] / 25.0, audio.shape[-1] / 16000.0)
        if dur < window:
            raise Unsupported(f"clip {dur:.2f}s shorter than the {window:g}s model window")
        starts, s = [], 0.0
        while s + window <= dur + 1e-6:
            starts.append(round(s, 2))
            s += hop
        probs = []
        for v0 in starts:
            item = {"video": rgb, "audio": audio, "meta": meta, "path": str(enc),
                    "split": "test", "targets": {"v_start_i_sec": v0, "offset_sec": 0.0}}
            batch = torch.utils.data.default_collate([transform(item)])
            aud, vid, _ = prepare_inputs(batch, torch.device(device))
            with torch.inference_mode(), torch.autocast("cuda", enabled=half and "cuda" in device):
                _, logits = model(vid, aud)
            probs.append(torch.softmax(logits.float(), dim=-1)[0].cpu())
        p = torch.stack(probs).mean(0)
        g = torch.as_tensor(grid, dtype=torch.float32)
        mean_s, argmax_s = float((p * g).sum()), float(g[int(p.argmax())])
        zero = int(torch.argmin(g.abs()))
        offset_ms = sign * 1000.0 * mean_s
        return {"metrics": {"sync_offset_ms_synchformer": offset_ms,
                            "sync_offset_argmax_ms_synchformer": sign * 1000.0 * argmax_s,
                            "sync_conf_synchformer": float(p.max()),
                            "sync_p0_synchformer": float(p[zero])},
                "offset_ms": offset_ms, "desync": abs(offset_ms), "windows": len(starts)}
    return score, {"exp_name": exp, "grid_s": [float(x) for x in grid]}


# ------------------------------------------------------------------ SyncNet (LSE-C / LSE-D)

def _syncnet(params: dict, device: str):
    import types

    repo = Path(os.path.expanduser(params.get("repo", "~/.opendub/src/syncnet_python")))
    model_file = repo / "data" / "syncnet_v2.model"
    if not model_file.exists():
        raise RuntimeError(f"SyncNet weights missing at {model_file} (run download_model.sh)")
    sys.path.insert(0, str(repo))
    from SyncNetInstance import SyncNetInstance

    net = SyncNetInstance()
    net.loadParameters(str(model_file))
    sign = float(params.get("sign", 1.0))
    vshift = int(params.get("vshift", 15))

    def score(video: str, tmp: Path) -> dict:
        data_dir, ref = tmp / "syncnet", "item"
        subprocess.run([sys.executable, "run_pipeline.py", "--videofile", video, "--reference",
                        ref, "--data_dir", str(data_dir)], cwd=repo, check=True,
                       capture_output=True)
        crops = sorted(glob.glob(str(data_dir / "pycrop" / ref / "0*.avi")))
        if not crops:
            raise Unsupported("no face track found")
        opt = types.SimpleNamespace(batch_size=20, vshift=vshift, data_dir=str(data_dir),
                                    videofile=video, reference=ref,
                                    avi_dir=str(data_dir / "pyavi"),
                                    tmp_dir=str(data_dir / "pytmp"),
                                    work_dir=str(data_dir / "pywork"),
                                    crop_dir=str(data_dir / "pycrop"))
        best = None
        for crop in crops:
            offset, conf, dists = net.evaluate(opt, videofile=crop)
            lse_d = float(dists.mean(axis=0).min()) if getattr(dists, "ndim", 0) == 2 \
                else float("nan")
            rec = (float(conf), float(offset), lse_d)
            if best is None or rec[0] > best[0]:
                best = rec
        conf, offset, lse_d = best
        offset_ms = sign * 40.0 * offset
        return {"metrics": {"lse_c": conf, "lse_d": lse_d, "sync_offset_ms_syncnet": offset_ms,
                            "sync_conf_syncnet": conf},
                "offset_ms": offset_ms, "desync": abs(offset_ms), "tracks": len(crops)}
    return score, {"weights": "syncnet_v2.model"}


# ------------------------------------------------------------------ PEAVS

def _peavs(params: dict, device: str):
    repo = Path(os.path.expanduser(params.get("repo", "~/.opendub/src/avgen-eval-toolkit")))
    peavs = repo / "PEAVS"
    if not (peavs / "metric.py").exists():
        raise RuntimeError(f"avgen-eval-toolkit (PEAVS) not found at {repo}")
    gpu = device if device.startswith("cuda:") else "cuda:0"

    def score(video: str, tmp: Path) -> dict:
        folder = tmp / "peavs_in"
        folder.mkdir(exist_ok=True)
        shutil.copy(video, folder / "item.mp4")
        for stale in (peavs / "av_features_extraction" / "output_feats", peavs / "results"):
            shutil.rmtree(stale, ignore_errors=True)
        subprocess.run([sys.executable, "main.py", f"device={gpu}", f"video_paths={folder}"],
                       cwd=peavs / "av_features_extraction", check=True, capture_output=True)
        subprocess.run([sys.executable, "metric.py"], cwd=peavs, check=True, capture_output=True)
        with (peavs / "results" / "results.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError("PEAVS wrote no result row")
        vals = [float(v) for k, v in rows[0].items() if k != "filename" and _is_float(v)]
        s = vals[-1]
        return {"metrics": {"peavs": s, "sync_conf_peavs": s}, "offset_ms": None,
                "desync": 5.0 - s}
    return score, {"repo": str(repo)}


def _is_float(v: str) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


_LOADERS = {"synchformer": _synchformer, "syncnet": _syncnet, "peavs": _peavs}


def load(params: dict, lang: str):
    model = params.get("model", "synchformer")
    if model not in _LOADERS:
        raise ValueError(f"unknown lip-sync model {model!r}; one of {', '.join(_LOADERS)}")
    device = device_of(params)
    if device == "cuda":
        device = "cuda:0"
    fn, ident = _LOADERS[model](params, device)
    return {"model": model, "fn": fn, "device": device, "ident": ident}


def run(state: dict, item: dict, out: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="opendub-sync-") as td:
        tmp = Path(td)
        payload = state["fn"](_prepared_video(item["inputs"], tmp), tmp)
    payload["score"] = -payload["desync"]
    payload["model"] = state["model"]
    return payload


def describe(state: dict) -> dict:
    return {"model": state["model"], "device": state["device"], **state["ident"]}


if __name__ == "__main__":
    serve(load, run, describe)
