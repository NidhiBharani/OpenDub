"""B1 worker: face front-end (detect → shot-cut-aware IoU tracks → smoothed crops) + an
active-speaker head, following the TalkNet-ASD demo pipeline (``demoTalkNet.py``) with the
research's fixes: tracks are hard-cut at every shot boundary and never bridge a cut.

params:
  asd: talknet | lrasd | loconet | none
       (none = "largest face speaks": the face-only baseline — what lip-syncing whichever face is
       on screen amounts to)
  repo: checkout of the ASD repo (defaults under ~/.opendub/src: TalkNet-ASD, LR-ASD, TalkNCE)
  weights: checkpoint path (defaults: TalkNet pretrain_TalkSet.model, LR-ASD
       weight/finetuning_TalkSet.model, TalkNCE talknce_ava_pretrained.model)
  detector: s3fd (TalkNet's bundled S3FD, default) | yunet (OpenCV FaceDetectorYN, MIT; needs
       yunet_model path) | scrfd (insightface det_10g via ``insightface``; weights non-commercial)
  det_scale (0.25), det_conf (0.9), min_track (10 frames), max_gap (10 frames), iou_thres (0.5),
  min_face (1% of frame height), crop_scale (0.40), durations ([1, 2, 3, 4, 5, 6] s, averaged as
  in the demo), fps (25: the models are trained on 25 fps crops), analysis_height (720),
  scene_threshold (27: PySceneDetect ContentDetector).

Output: ``tracks`` (normalised boxes at 25 fps, per-frame speaking probability = softmax of the
model's 2-way AV head) and ``lines`` when ``inputs.lines`` is given (best track per line by mean
probability ≥ 0.5, else off-screen) — the judge can also derive lines from tracks itself.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

from _sdk import serve
from b_common import probe

HOME = Path.home() / ".opendub"
DEFAULT_REPO = {"talknet": HOME / "src" / "TalkNet-ASD", "lrasd": HOME / "src" / "LR-ASD",
                "loconet": HOME / "src" / "TalkNCE"}
DEFAULT_WEIGHTS = {"talknet": "pretrain_TalkSet.model", "lrasd": "weight/finetuning_TalkSet.model",
                   "loconet": "talknce_ava_pretrained.model"}


@contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ------------------------------------------------------------------ face detectors

class _S3FD:
    """TalkNet's S3FD, imported as a standalone package so it never clashes with the ASD repo's
    own ``model`` package; its weight path is relative to the TalkNet checkout."""

    def __init__(self, talknet: Path, scale: float, conf: float):
        sys.path.insert(0, str(talknet / "model" / "faceDetector"))
        with _cwd(talknet):
            import s3fd  # type: ignore[import-not-found]

            self.det = s3fd.S3FD(device="cuda")
        sys.path.pop(0)
        self.scale, self.conf = scale, conf

    def __call__(self, rgb):
        return [list(map(float, b)) for b in self.det.detect_faces(rgb, conf_th=self.conf,
                                                                   scales=[self.scale])]


class _YuNet:
    def __init__(self, model: str, conf: float):
        import cv2

        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(model, "", (320, 320), score_threshold=conf)

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        return [[f[0], f[1], f[0] + f[2], f[1] + f[3], f[14]] for f in (faces if faces is not None
                                                                          else [])]


class _Scrfd:
    def __init__(self, conf: float):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        self.app.prepare(ctx_id=0, det_thresh=conf)

    def __call__(self, rgb):
        return [[*map(float, f.bbox), float(f.det_score)] for f in self.app.get(rgb[:, :, ::-1])]


# ------------------------------------------------------------------ media

def _frames(video: str, fps: float, height: int):
    """Yield RGB frames resampled to ``fps`` and scaled to ``height`` (ffmpeg rawvideo pipe)."""
    import numpy as np

    info = probe(video)
    h = min(height, int(info["height"]) or height)
    w = round(info["width"] * h / max(1.0, info["height"]) / 2) * 2
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps},scale={w}:{h}", "-f",
         "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _audio16k(video: str):
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16)


def _shot_starts(video: str, fps: float, threshold: float) -> list[int]:
    """Frame indices (at ``fps``) where a new shot starts, from PySceneDetect ContentDetector."""
    from scenedetect import ContentDetector, detect

    scenes = detect(video, ContentDetector(threshold=threshold))
    return sorted({round(s.get_seconds() * fps) for s, _e in scenes[1:]})


# ------------------------------------------------------------------ tracking (TalkNet demo)

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def track_faces(dets: list[list[list[float]]], cuts: list[int], p: dict) -> list[dict]:
    """IoU chaining within each shot, as in TalkNet's ``track_shot``, then linear interpolation
    over gaps and a median filter on centre/size. Returns [{frames: [...], x, y, s}]."""
    import numpy as np
    from scipy import signal
    from scipy.interpolate import interp1d

    iou_thres, max_gap = float(p.get("iou_thres", 0.5)), int(p.get("max_gap", 10))
    min_track = int(p.get("min_track", 10))
    bounds = [0, *[c for c in cuts if 0 < c < len(dets)], len(dets)]
    tracks = []
    for a, b in pairwise(bounds):
        faces = [[{"frame": f, "bbox": d[:4]} for d in dets[f]] for f in range(a, b)]
        while True:
            track = []
            for frame_faces in faces:
                for face in list(frame_faces):
                    if not track:
                        track.append(face)
                        frame_faces.remove(face)
                    elif face["frame"] - track[-1]["frame"] <= max_gap:
                        if _iou(face["bbox"], track[-1]["bbox"]) > iou_thres:
                            track.append(face)
                            frame_faces.remove(face)
                            continue
                    else:
                        break
            if not track:
                break
            if len(track) > min_track:
                fn = np.array([f["frame"] for f in track])
                bb = np.array([f["bbox"] for f in track])
                full = np.arange(fn[0], fn[-1] + 1)
                bbi = np.stack([interp1d(fn, bb[:, k])(full) for k in range(4)], axis=1)
                s = np.maximum(bbi[:, 3] - bbi[:, 1], bbi[:, 2] - bbi[:, 0]) / 2
                k = 13 if len(full) >= 13 else (len(full) // 2) * 2 - 1 or 1
                tracks.append({"frames": full.tolist(),
                               "bbox": bbi.tolist(),
                               "s": signal.medfilt(s, kernel_size=k).tolist(),
                               "x": signal.medfilt((bbi[:, 0] + bbi[:, 2]) / 2, k).tolist(),
                               "y": signal.medfilt((bbi[:, 1] + bbi[:, 3]) / 2, k).tolist()})
    return tracks


def crop_tracks(video: str, tracks: list[dict], p: dict) -> list:
    """112×112 grayscale mouth-centred crops per track (TalkNet ``crop_video`` + the 224→112
    centre crop of ``evaluate_network``)."""
    import cv2
    import numpy as np

    cs = float(p.get("crop_scale", 0.40))
    by_frame: dict[int, list[tuple[int, int]]] = {}
    for ti, tr in enumerate(tracks):
        for k, f in enumerate(tr["frames"]):
            by_frame.setdefault(f, []).append((ti, k))
    crops = [[] for _ in tracks]
    for f, rgb in enumerate(_frames(video, float(p.get("fps", 25)),
                                    int(p.get("analysis_height", 720)))):
        for ti, k in by_frame.get(f, []):
            tr = tracks[ti]
            bs = tr["s"][k]
            bsi = int(bs * (1 + 2 * cs))
            frame = np.pad(rgb, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant",
                           constant_values=110)
            my, mx = tr["y"][k] + bsi, tr["x"][k] + bsi
            face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)),
                         int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY), (224, 224))
            crops[ti].append(face[56:168, 56:168])
    return [np.array(c) for c in crops]


# ------------------------------------------------------------------ ASD heads

def _prob(loss_av, out):
    import torch

    x = out.squeeze(1) if out.dim() > 2 else out
    return torch.softmax(loss_av.FC(x).float(), dim=-1)[:, 1].cpu().numpy()


class TalkNetLike:
    """TalkNet (cross-attention) and LR-ASD (no cross-attention) share the demo interface."""

    def __init__(self, kind: str, repo: Path, weights: Path):
        sys.path.insert(0, str(repo))
        if kind == "talknet":
            from talkNet import talkNet  # type: ignore[import-not-found]

            self.s = talkNet()
        else:
            from ASD import ASD  # type: ignore[import-not-found]

            self.s = ASD()
        self.s.loadParameters(str(weights))
        self.s.eval()
        self.cross = kind == "talknet"

    def scores(self, mfcc, faces, durations: list[int]):
        import numpy as np
        import torch

        length = min(len(mfcc) / 100, len(faces) / 25)
        mfcc, faces = mfcc[:round(length * 100)], faces[:round(length * 25)]
        runs = []
        with torch.no_grad():
            for d in durations:
                out_scores = []
                for i in range(int(np.ceil(length / d))):
                    a = torch.FloatTensor(mfcc[i * d * 100:(i + 1) * d * 100]).unsqueeze(0).cuda()
                    v = torch.FloatTensor(faces[i * d * 25:(i + 1) * d * 25]).unsqueeze(0).cuda()
                    if a.shape[1] < 4 or v.shape[1] < 1:
                        continue
                    ea = self.s.model.forward_audio_frontend(a)
                    ev = self.s.model.forward_visual_frontend(v)
                    if self.cross:
                        ea, ev = self.s.model.forward_cross_attention(ea, ev)
                    out = self.s.model.forward_audio_visual_backend(ea, ev)
                    out_scores.extend(_prob(self.s.lossAV, out).tolist())
                runs.append(out_scores[:len(faces)])
        n = min(len(r) for r in runs) if runs else 0
        return np.mean([r[:n] for r in runs], axis=0) if n else np.zeros(0)


class LoCoNetHead:
    """LoCoNet (+TalkNCE) inference on one target track with up to NUM_SPEAKERS-1 context tracks,
    reproducing ``val_loader`` + ``evaluate_network`` of kaistmm/TalkNCE: VGGish log-mel audio at
    4 frames per video frame, 112×112 grayscale faces, context slots padded by repeating the
    target (as the loader does when a frame has no other speaker)."""

    def __init__(self, repo: Path, weights: Path):
        import types

        import torch
        import yaml

        sys.path.insert(0, str(repo))
        # loss_multi imports utils.distributed (not in the public repo): identity stubs suffice
        # for single-process inference.
        if "utils.distributed" not in sys.modules:
            dist = types.ModuleType("utils.distributed")
            dist.all_reduce = lambda xs, average=False: xs
            dist.all_gather = lambda xs: xs
            pkg = types.ModuleType("utils")
            pkg.distributed = dist
            sys.modules.setdefault("utils", pkg)
            sys.modules["utils.distributed"] = dist
        from loss_multi import lossAV  # type: ignore[import-not-found]
        from model.loconet_encoder import locoencoder  # type: ignore[import-not-found]
        from torchvggish import vggish_input  # type: ignore[import-not-found]

        class Cfg(dict):
            __getattr__ = dict.__getitem__

        def wrap(d):
            return Cfg({k: wrap(v) if isinstance(v, dict) else v for k, v in d.items()})

        self.cfg = wrap(yaml.safe_load((repo / "configs" / "test.yaml").read_text()))
        with _cwd(repo):
            self.enc = locoencoder(self.cfg).cuda().eval()
        self.loss = lossAV(self.cfg).cuda().eval()
        state = torch.load(str(weights), map_location="cpu")
        state = {k.replace("model.module.", ""): v for k, v in state.items()}
        mine = {f"model.{k}": v for k, v in self.enc.state_dict().items()}
        mine.update({f"lossAV.{k}": v for k, v in self.loss.state_dict().items()})
        missing = [k for k in mine if k not in state]
        if len(missing) > len(mine) // 10:
            raise RuntimeError(f"checkpoint does not match LoCoNet ({len(missing)} missing keys, "
                               f"e.g. {missing[:3]})")
        self.enc.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")},
                                 strict=False)
        self.loss.load_state_dict({k[7:]: v for k, v in state.items()
                                   if k.startswith("lossAV.")}, strict=False)
        self.vggish_input = vggish_input
        self.S = int(self.cfg.MODEL.NUM_SPEAKERS)
        self.clip = int(self.cfg.MODEL.CLIP_LENGTH)

    def scores(self, audio16k, target, contexts, fps: float = 25.0):
        """target: (T, 112, 112); contexts: list of (T, 112, 112) aligned to the target frames
        (zeros where the context face is absent)."""
        import numpy as np
        import torch

        T = len(target)
        out = np.zeros(T)
        slots = [target, *contexts[:self.S - 1]]
        while len(slots) < self.S:
            slots.append(slots[1] if len(slots) > 1 else target)
        for a in range(0, T, self.clip):
            b = min(T, a + self.clip)
            n = b - a
            seg = audio16k[int(a / fps * 16000):int(b / fps * 16000)]
            mel = self.vggish_input.waveform_to_examples(seg, 16000, n, fps, return_tensor=False)
            audio = torch.FloatTensor(mel)[None, None].cuda()
            vis = torch.FloatTensor(np.stack([s[a:b] for s in slots]))[None].cuda()
            with torch.no_grad():
                bb, s, t = vis.shape[:3]
                v = vis.view(bb * s, *vis.shape[2:])
                ea = self.enc.forward_audio_frontend(audio).repeat(s, 1, 1)
                ev = self.enc.forward_visual_frontend(v)
                ea, ev = self.enc.forward_cross_attention(ea, ev)
                av = self.enc.forward_audio_visual_backend(ea, ev, bb, s)
                av = av.view(bb, s, t, -1)[:, 0].reshape(bb * t, -1)
                out[a:b] = _prob(self.loss, av)[:n]
        return out


# ------------------------------------------------------------------ worker

def load(params: dict, lang: str) -> dict:
    kind = params.get("asd", "talknet")
    det_kind = params.get("detector", "s3fd")
    conf = float(params.get("det_conf", 0.9))
    talknet_repo = Path(params.get("talknet_repo", DEFAULT_REPO["talknet"])).expanduser()
    if det_kind == "s3fd":
        det = _S3FD(talknet_repo, float(params.get("det_scale", 0.25)), conf)
    elif det_kind == "yunet":
        det = _YuNet(str(Path(params["yunet_model"]).expanduser()), conf)
    else:
        det = _Scrfd(conf)
    head = None
    if kind != "none":
        repo = Path(params.get("repo", DEFAULT_REPO[kind])).expanduser()
        weights = Path(params.get("weights", DEFAULT_WEIGHTS[kind])).expanduser()
        weights = weights if weights.is_absolute() else repo / weights
        head = LoCoNetHead(repo, weights) if kind == "loconet" else TalkNetLike(kind, repo,
                                                                                 weights)
    return {"kind": kind, "det": det, "head": head, "params": params}


def _largest_face_scores(tracks: list[dict], n_frames: int):
    """Face-only baseline: in every frame the largest face 'speaks' (p = 1), others p = 0."""
    import numpy as np

    best = {}
    for ti, tr in enumerate(tracks):
        for k, f in enumerate(tr["frames"]):
            if f not in best or tr["s"][k] > best[f][1]:
                best[f] = (ti, tr["s"][k])
    return [np.array([1.0 if best.get(f, (None,))[0] == ti else 0.0 for f in tr["frames"]])
            for ti, tr in enumerate(tracks)]


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np

    p = state["params"]
    video = item["inputs"]["video"]
    fps = float(p.get("fps", 25))
    info = probe(video)
    dets, shape = [], None
    for rgb in _frames(video, fps, int(p.get("analysis_height", 720))):
        shape = rgb.shape
        min_face = float(p.get("min_face", 0.01)) * rgb.shape[0]
        dets.append([d for d in state["det"](rgb) if d[3] - d[1] >= min_face])
    if shape is None:
        raise RuntimeError("no frames decoded")
    H, W = shape[0], shape[1]
    cuts = _shot_starts(video, fps, float(p.get("scene_threshold", 27.0)))
    tracks = track_faces(dets, cuts, p)

    if state["kind"] == "none":
        probs = _largest_face_scores(tracks, len(dets))
    else:
        crops = crop_tracks(video, tracks, p)
        audio = _audio16k(video)
        probs = []
        if state["kind"] == "loconet":
            for ti, tr in enumerate(tracks):
                f0, f1 = tr["frames"][0], tr["frames"][-1]
                ctx = []
                for tj, other in enumerate(tracks):
                    if tj == ti or other["frames"][-1] < f0 or other["frames"][0] > f1:
                        continue
                    arr = np.zeros_like(crops[ti])
                    for k, f in enumerate(other["frames"]):
                        if f0 <= f <= f1 and k < len(crops[tj]):
                            arr[f - f0] = crops[tj][k]
                    ctx.append(arr)
                probs.append(state["head"].scores(audio, crops[ti], ctx, fps))
        else:
            import python_speech_features

            durations = [int(d) for d in p.get("durations", [1, 2, 3, 4, 5, 6])]
            for ti, tr in enumerate(tracks):
                seg = audio[int(tr["frames"][0] / fps * 16000):
                            int((tr["frames"][-1] + 1) / fps * 16000)]
                mfcc = python_speech_features.mfcc(seg, 16000, numcep=13, winlen=0.025,
                                                   winstep=0.010)
                sc = state["head"].scores(mfcc, crops[ti], durations)
                full = np.zeros(len(tr["frames"]))
                full[:len(sc)] = sc
                probs.append(full)

    out_tracks = []
    for ti, tr in enumerate(tracks):
        boxes = [[round(f / fps, 3), b[0] / W, b[1] / H, b[2] / W, b[3] / H]
                 for f, b in zip(tr["frames"], tr["bbox"], strict=True)]
        out_tracks.append({"id": f"t{ti}", "boxes": [[round(v, 4) for v in b] for b in boxes],
                           "speaking": [round(float(x), 4) for x in probs[ti]],
                           "face_height_px": round(float(np.median(tr["s"])) * 2, 1)})
    payload = {"tracks": out_tracks, "threshold": 0.5, "fps": fps,
               "duration_s": info["duration_s"], "cuts": [round(c / fps, 3) for c in cuts]}
    lines = item["inputs"].get("lines") or []
    if lines:
        payload["lines"] = _assign_lines(out_tracks, lines)
    return payload


def _assign_lines(tracks: list[dict], lines: list[dict]) -> list[dict]:
    res = []
    for ln in lines:
        s, e = float(ln["start"]), float(ln["end"])
        best, best_p = None, 0.5
        for tr in tracks:
            vals = [p for b, p in zip(tr["boxes"], tr["speaking"], strict=False) if s <= b[0] <= e]
            if vals and sum(vals) / len(vals) >= best_p:
                best, best_p = tr["id"], sum(vals) / len(vals)
        res.append({"start": s, "end": e, "track": best, "offscreen": best is None,
                    "score": round(best_p, 4) if best else None})
    return res


def describe(state: dict) -> dict:
    p = state["params"]
    return {"asd": state["kind"], "detector": p.get("detector", "s3fd"),
            "weights": str(p.get("weights", DEFAULT_WEIGHTS.get(state["kind"], "")))}


if __name__ == "__main__":
    serve(load, run, describe)
