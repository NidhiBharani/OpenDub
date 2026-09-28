"""D7 judge worker: non-verbal event tagging with CED (mispeech/ced-base, Apache-2.0, AudioSet
527 classes, 16 kHz) — the "tagged by BEATs/CED" judge of the D7 spec (a judge lineage no D7
candidate trains against).

Item inputs: {audio, ref_audio?, expected?: [labels]}. Both clips are tagged on sliding windows
(``win_s`` 1.0, ``hop_s`` 0.25); a window whose best non-verbal class group exceeds ``threshold``
opens an event, adjacent windows of the same group merge. The reference events come from
``expected`` (tags in the script) when given, else from tagging ``ref_audio``.

Returns {"metrics": {nv_f1, nv_precision, nv_recall, nv_f1_timed, nv_ref_events, nv_out_events},
"weights": {...}} — presence F1 counts events per group (timing-free: a dub line is not aligned
with its source), timed F1 matches onsets within ``tol_s`` (0.2) for time-aligned outputs
(splice, VC). Groups: laugh, cry, sigh, breath, scream, cough, groan.
params: model, revision, threshold (0.3), win_s, hop_s, tol_s.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve

GROUPS = {"laugh": ("laugh", "giggle", "snicker", "chuckle", "chortle"),
          "cry": ("crying", "sobbing", "whimper", "wail", "baby cry"),
          "sigh": ("sigh",),
          "breath": ("breathing", "gasp", "pant", "wheeze"),
          "scream": ("scream", "yell", "shout"),
          "cough": ("cough", "throat clearing", "sneeze", "sniff"),
          "groan": ("groan", "grunt", "moan")}
TEXT_TAGS = {"laugh": ("laugh", "laughter", "giggle", "chuckle", "笑"), "cry": ("cry", "sob"),
             "sigh": ("sigh",), "breath": ("breath", "inhale", "exhale", "gasp"),
             "scream": ("scream", "shout", "yell"), "cough": ("cough", "clears throat", "sniff",
                                                              "snort"),
             "groan": ("groan", "grunt", "moan", "uhm")}


def group_of_tag(tag: str) -> str | None:
    t = tag.lower().strip("[]() ")
    for g, keys in TEXT_TAGS.items():
        if any(k in t for k in keys):
            return g
    return None


def load(params: dict, lang: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    name = params.get("model", "mispeech/ced-base")
    fe = AutoFeatureExtractor.from_pretrained(name, revision=params.get("revision"),
                                              trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(
        name, revision=params.get("revision"), trust_remote_code=True).to("cuda").eval()
    labels = model.config.id2label
    idx = {g: [int(i) for i, lab in labels.items() if any(k in lab.lower() for k in keys)]
           for g, keys in GROUPS.items()}
    torch.set_grad_enabled(False)
    return {"fe": fe, "model": model, "idx": idx, "params": params}


def events(state: dict, path: str) -> list[tuple[float, float, str]]:
    import librosa
    import numpy as np
    import torch

    p = state["params"]
    y, _ = librosa.load(path, sr=16000, mono=True)
    win, hop = int(16000 * float(p.get("win_s", 1.0))), int(16000 * float(p.get("hop_s", 0.25)))
    thr = float(p.get("threshold", 0.3))
    starts = list(range(0, max(1, len(y) - win + 1), hop)) or [0]
    out: list[tuple[float, float, str]] = []
    for s in starts:
        chunk = y[s:s + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        feats = state["fe"](chunk, sampling_rate=16000, return_tensors="pt").to("cuda")
        probs = torch.sigmoid(state["model"](**feats).logits)[0].float().cpu().numpy()
        best = max(((max(probs[i] for i in ids), g) for g, ids in state["idx"].items() if ids),
                   default=(0.0, ""))
        if best[0] < thr:
            continue
        t0, t1 = s / 16000, (s + win) / 16000
        if out and out[-1][2] == best[1] and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], t1, best[1])
        else:
            out.append((t0, t1, best[1]))
    return out


def f1_rows(ref: list[tuple[float, float, str]], hyp: list[tuple[float, float, str]],
            tol: float = 0.2) -> dict:
    """Presence F1 (per-group event counts) and timed F1 (onsets within ``tol`` seconds)."""
    from collections import Counter

    rc, hc = Counter(e[2] for e in ref), Counter(e[2] for e in hyp)
    tp = sum(min(rc[g], hc[g]) for g in rc)
    prec = tp / sum(hc.values()) if hc else (1.0 if not rc else 0.0)
    rec = tp / sum(rc.values()) if rc else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    used, ttp = set(), 0
    for r in ref:
        for j, h in enumerate(hyp):
            if j not in used and h[2] == r[2] and abs(h[0] - r[0]) <= tol:
                used.add(j)
                ttp += 1
                break
    tp_p = ttp / len(hyp) if hyp else (1.0 if not ref else 0.0)
    tr = ttp / len(ref) if ref else 1.0
    return {"nv_f1": f1, "nv_precision": prec, "nv_recall": rec,
            "nv_f1_timed": 2 * tp_p * tr / (tp_p + tr) if tp_p + tr else 0.0,
            "nv_ref_events": float(len(ref)), "nv_out_events": float(len(hyp))}


def run(state: dict, item: dict, out: Path) -> dict:
    inp = item["inputs"]
    hyp = events(state, inp["audio"])
    if inp.get("expected"):
        ref = [(0.0, 0.0, g) for g in (group_of_tag(t) for t in inp["expected"]) if g]
    elif inp.get("ref_audio"):
        ref = events(state, inp["ref_audio"])
    else:
        ref = []
    metrics = f1_rows(ref, hyp, float(state["params"].get("tol_s", 0.2)))
    if inp.get("expected"):
        metrics["nv_f1_timed"] = None  # no reference timing for script tags
    w = float(max(1, len(ref)))
    return {"metrics": metrics, "weights": {"nv_f1": w, "nv_precision": w, "nv_recall": w,
                                            "nv_f1_timed": w},
            "events": [list(e) for e in hyp]}


if __name__ == "__main__":
    serve(load, run)
