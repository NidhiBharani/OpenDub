"""A6 — speaker diarization with overlap.

DER with **no collar and overlap scored** (primary), plus the 0.25 s-collar DER, its
miss / false-alarm / confusion parts, JER, speaker-count error and the dubbing-specific
**line-level speaker-attribution error** (each reference line gets the hypothesis speaker with
the most overlap, mapped through the optimal speaker mapping; wrong or none = error).

Implemented in numpy on a 10 ms grid with the optimal one-to-one speaker mapping (Hungarian,
``scipy.optimize.linear_sum_assignment``), which matches NIST md-eval / pyannote.metrics DER with
``collar=0, skip_overlap=False`` up to grid rounding. DER rows are weighted by reference speech
time, so the weighted mean over items is the corpus DER (sum of errors / sum of reference).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..judges import Row, Spec, cost, speed
from ..packs import Item

RES_S = 0.01


def read_rttm(path: str | Path) -> list[dict[str, Any]]:
    """``SPEAKER <file> <chan> <start> <dur> <NA> <NA> <speaker> …`` lines → turns."""
    turns = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            turns.append({"start": start, "end": start + dur, "speaker": parts[7]})
    return turns


def ref_turns(item: Item) -> list[dict[str, Any]] | None:
    if item.refs.get("turns") is not None:
        return list(item.refs["turns"])
    rttm = item.refs.get("rttm")
    if rttm and Path(str(rttm)).exists():
        return read_rttm(rttm)
    return None


def _matrix(turns: list[dict[str, Any]], n: int):
    import numpy as np

    spk = sorted({str(t["speaker"]) for t in turns})
    m = np.zeros((n, len(spk)), dtype=bool)
    idx = {s: k for k, s in enumerate(spk)}
    for t in turns:
        a = max(0, round(float(t["start"]) / RES_S))
        b = min(n, round(float(t["end"]) / RES_S))
        if b > a:
            m[a:b, idx[str(t["speaker"])]] = True
    return m, spk


def _mapping(ref, hyp) -> dict[int, int]:
    """Optimal one-to-one ref→hyp speaker map maximising overlapped frames."""
    import numpy as np

    if ref.shape[1] == 0 or hyp.shape[1] == 0:
        return {}
    overlap = ref.T.astype(np.int64) @ hyp.astype(np.int64)
    try:
        from scipy.optimize import linear_sum_assignment

        r, h = linear_sum_assignment(-overlap)
        pairs = list(zip(r.tolist(), h.tolist()))
    except ImportError:  # greedy fallback
        pairs, used_r, used_h = [], set(), set()
        for flat in np.argsort(-overlap, axis=None):
            i, j = divmod(int(flat), overlap.shape[1])
            if i not in used_r and j not in used_h:
                pairs.append((i, j))
                used_r.add(i)
                used_h.add(j)
    return {i: j for i, j in pairs if overlap[i, j] > 0}


def der_parts(ref, hyp, mapping: dict[int, int], mask=None) -> tuple[float, float, float, float]:
    """(miss, false alarm, confusion, total reference) in frames over the scored ``mask``."""
    import numpy as np

    n_ref = ref.sum(axis=1)
    n_hyp = hyp.sum(axis=1)
    correct = np.zeros(len(ref), dtype=np.int64)
    for i, j in mapping.items():
        correct += (ref[:, i] & hyp[:, j])
    if mask is not None:
        n_ref, n_hyp, correct = n_ref[mask], n_hyp[mask], correct[mask]
    miss = np.maximum(0, n_ref - n_hyp).sum()
    fa = np.maximum(0, n_hyp - n_ref).sum()
    conf = (np.minimum(n_ref, n_hyp) - correct).sum()
    return float(miss), float(fa), float(conf), float(n_ref.sum())


def collar_mask(turns: list[dict[str, Any]], n: int, collar: float):
    import numpy as np

    mask = np.ones(n, dtype=bool)
    c = round(collar / RES_S)
    for t in turns:
        for edge in (float(t["start"]), float(t["end"])):
            k = round(edge / RES_S)
            mask[max(0, k - c):min(n, k + c)] = False
    return mask


def diarization_rows(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    import numpy as np

    ref_t = ref_turns(item)
    if not ref_t:
        return []
    hyp_t = [t for t in (out.get("turns") or []) if float(t["end"]) > float(t["start"])]
    end = max([float(t["end"]) for t in ref_t + hyp_t]
              + [float(item.meta.get("duration_s") or 0)])
    n = round(end / RES_S) + 1
    ref, ref_spk = _matrix(ref_t, n)
    hyp, hyp_spk = _matrix(hyp_t, n)
    mapping = _mapping(ref, hyp)
    rows: list[Row] = []
    miss, fa, conf, total = der_parts(ref, hyp, mapping)
    if total:
        rows += [("der", (miss + fa + conf) / total, total * RES_S),
                 ("der_miss", miss / total, total * RES_S),
                 ("der_fa", fa / total, total * RES_S),
                 ("der_confusion", conf / total, total * RES_S)]
    miss, fa, conf, total_c = der_parts(ref, hyp, mapping, collar_mask(ref_t, n, 0.25))
    if total_c:
        rows.append(("der_collar", (miss + fa + conf) / total_c, total_c * RES_S))
    # JER (DIHARD III): mean over reference speakers of 1 − IoU with the mapped hyp speaker.
    jers = []
    for i in range(ref.shape[1]):
        j = mapping.get(i)
        if j is None:
            jers.append(1.0)
            continue
        inter = float((ref[:, i] & hyp[:, j]).sum())
        union = float((ref[:, i] | hyp[:, j]).sum())
        jers.append(1.0 - inter / union if union else 0.0)
    if jers:
        rows.append(("jer", float(np.mean(jers)), 1.0))
    rows.append(("spk_count_err", float(abs(len(hyp_spk) - len(ref_spk))), 1.0))
    # Line-level attribution: which hyp speaker would a dub line be voiced by?
    lines = item.refs.get("lines") or ref_t
    inv = {j: i for i, j in mapping.items()}
    errors = 0
    for ln in lines:
        a = max(0, round(float(ln["start"]) / RES_S))
        b = min(n, round(float(ln["end"]) / RES_S))
        if b <= a:
            continue
        cover = hyp[a:b].sum(axis=0) if hyp.shape[1] else np.zeros(0)
        if not len(cover) or cover.max() == 0:
            errors += 1
            continue
        j = int(np.argmax(cover))
        i = inv.get(j)
        if i is None or ref_spk[i] != str(ln["speaker"]):
            errors += 1
    if lines:
        rows.append(("line_attr_err", errors / len(lines), float(len(lines))))
    return rows


SPEC = Spec(
    id="A6",
    title="Diarization with overlap",
    judges={"diarization@1": diarization_rows, "speed@1": speed, "cost@1": cost},
    primary={"*": "der"},
    higher_is_better={"der": False, "der_collar": False, "der_miss": False, "der_fa": False,
                      "der_confusion": False, "jer": False, "spk_count_err": False,
                      "line_attr_err": False, "rtfx": True, "cost_usd": False},
    threshold={"der": 0.01, "line_attr_err": 0.01},
    secondary=["line_attr_err", "der_collar", "jer", "der_miss", "der_fa", "der_confusion",
               "spk_count_err", "rtfx", "cost_usd"],
    packs=["a6-voxconverse", "a6-ami", "a6-rttm-ih"],
    io="""item.inputs: {audio} (whole recording); item.refs: {rttm: path} or {turns: [...]},
optional {lines: [{start, end, speaker}]} for line attribution (default: reference turns).
payload: {turns: [{start, end, speaker}]} (overlapping turns allowed; speaker labels arbitrary).
Joint ASR+diarization workers may also return text/segments.""",
)
