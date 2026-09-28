"""E1 worker: timing fit by uniform stretch or by a pause-structure DP time map, with a choice of
time-scale modification (TSM) backend.

Modes (``params.mode``):

- ``uniform``: trim outer silence, stretch the take's speech span to the source speech span
  (``inputs.src_speech``; else the slot minus 30 ms margins) and place it at the source onset.
- ``pause_dp``: prosodic-alignment style. Detect the take's pauses (energy VAD), then choose by
  dynamic programming which take pauses to anchor on which source pauses (``inputs.src_pauses`` /
  ``src_speech``, known in the pipeline from the source line's A4 word timings). Each anchored
  group of take speech is mapped onto the matching source speech group; its rate is held within
  ``rate_dev`` of the line's global rate (dubbers keep rate steady and move pauses), pauses absorb
  the rest, and the whole line is compressed further only if it would overflow the slot.
  Cost of a group pair = w_rate·log²(rate/global rate) + pen_take·(take pauses left unanchored)
  + pen_src·(source pause seconds left unmatched). The method is published (Amazon prosodic
  alignment line; SPaDA-style DP) without a reference implementation: this is ours.

Backends (``params.backend``): ``atempo`` (ffmpeg WSOLA), ``rubberband`` (Rubber Band CLI, GPL —
shelled out, never linked; ``engine: finer`` = R3, ``faster`` = R2), ``signalsmith``
(Signalsmith Stretch via the MIT ``python-stretch`` binding; env ``e1_signalsmith``).

Output: native rate, mono, exactly the slot length unless the take cannot fit even at
``max_rate`` (then longer; the judge scores it as an overflow). Payload carries the time map.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    import numpy as np
    import soundfile as sf

    backend = params.get("backend", "atempo")
    state = {"np": np, "sf": sf, "p": params, "backend": backend}
    if backend == "signalsmith":
        import python_stretch

        state["ps"] = python_stretch
    elif backend in ("atempo", "rubberband"):
        exe = "ffmpeg" if backend == "atempo" else "rubberband"
        if not shutil.which(exe):
            raise RuntimeError(f"{exe} not on PATH")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return state


# ------------------------------------------------------------------ analysis

def vad(np, x, sr: int, min_pause: float = 0.12) -> list[list[float]]:
    """Speech runs (energy, 20 ms frames / 10 ms hop, adaptive threshold)."""
    hop, win = int(0.01 * sr), int(0.02 * sr)
    if len(x) < win:
        return []
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 12, -65)
    act = db > thr
    runs, i = [], 0
    while i < n:
        if act[i]:
            j = i
            while j + 1 < n and act[j + 1]:
                j += 1
            runs.append([(i * hop + (win - hop) / 2) / sr, (j * hop + (win + hop) / 2) / sr])
            i = j + 1
        else:
            i += 1
    merged: list[list[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_pause:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [r for r in merged if r[1] - r[0] >= 0.05]


def source_segments(inputs: dict, target: float) -> list[list[float]] | None:
    if inputs.get("src_speech"):
        return [list(map(float, s)) for s in inputs["src_speech"]]
    pauses = inputs.get("src_pauses")
    if pauses:
        edges = [0.0] + [v for p in sorted(pauses) for v in p] + [target]
        return [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2)
                if edges[i + 1] > edges[i]]
    return None


def speech_ratio(take: list[list[float]], src: list[list[float]]) -> float:
    """Global rate: take speech seconds per source speech second."""
    return sum(e - s for s, e in take) / max(sum(e - s for s, e in src), 1e-3)


def align(take: list[list[float]], src: list[list[float]], w_rate: float, pen_take: float,
          pen_src: float) -> list[tuple[int, int, int, int]]:
    """DP over matched pause anchors → [(take_a, take_b, src_c, src_d)] inclusive run ranges."""
    n, m = len(take), len(src)
    r0 = speech_ratio(take, src)
    ct = [0.0]
    for s, e in take:
        ct.append(ct[-1] + e - s)
    cs = [0.0]
    for s, e in src:
        cs.append(cs[-1] + e - s)
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: dict[tuple[int, int], tuple[int, int]] = {}
    dp[0][0] = 0.0
    for j in range(1, n + 1):
        for k in range(1, m + 1):
            best, arg = inf, None
            for j0 in range(j):
                for k0 in range(k):
                    if dp[j0][k0] == inf:
                        continue
                    t_len = ct[j] - ct[j0]      # speech seconds in the group (no pauses)
                    s_len = cs[k] - cs[k0]
                    if t_len <= 0 or s_len <= 0:
                        continue
                    c = w_rate * math.log((t_len / s_len) / r0) ** 2
                    c += pen_take * (j - 1 - j0)
                    c += pen_src * sum(src[q + 1][0] - src[q][1] for q in range(k0, k - 1))
                    if dp[j0][k0] + c < best:
                        best, arg = dp[j0][k0] + c, (j0, k0)
            if arg is not None:
                dp[j][k], back[(j, k)] = best, arg
    groups, j, k = [], n, m
    while (j, k) != (0, 0):
        j0, k0 = back[(j, k)]
        groups.append((j0, j - 1, k0, k - 1))
        j, k = j0, k0
    return groups[::-1]


def plan(take: list[list[float]], src: list[list[float]], target: float, p: dict):
    """[(take_start, take_end, out_start, out_dur)] for each group."""
    if p.get("mode", "pause_dp") == "uniform" or len(take) < 2 or len(src) < 2:
        groups = [(0, len(take) - 1, 0, len(src) - 1)]
    else:
        groups = align(take, src, float(p.get("w_rate", 4.0)), float(p.get("pen_take", 0.05)),
                       float(p.get("pen_src", 0.3)))
    r0 = speech_ratio(take, src)
    lo, hi = float(p.get("min_rate", 0.7)), float(p.get("max_rate", 1.4))
    dev = float(p.get("rate_dev", 0.10))
    gap = float(p.get("min_gap", 0.08))
    segs = []
    for a, b, c, d in groups:
        t0, t1 = take[a][0], take[b][1]
        want = src[d][1] - src[c][0]
        rate = (t1 - t0) / max(want, 1e-3)
        if len(groups) > 1:
            rate = min(max(rate, r0 * (1 - dev)), r0 * (1 + dev))
        segs.append([t0, t1, src[c][0], min(max(rate, lo), hi)])
    gaps = gap * (len(segs) - 1)
    for _ in range(8):  # compress uniformly (up to max_rate) until the line fits the slot
        speech = sum((s[1] - s[0]) / s[3] for s in segs)
        avail = target - 0.02 - gaps
        if speech <= avail or all(s[3] >= hi for s in segs):
            break
        f = speech / avail if avail > 0 else hi
        for s in segs:
            s[3] = min(hi, s[3] * max(f, 1.001))
    out, prev_end = [], 0.0
    durs = [(s[1] - s[0]) / s[3] for s in segs]
    for i, (s, dur) in enumerate(zip(segs, durs, strict=True)):
        rest = sum(durs[i:]) + gap * (len(segs) - 1 - i)
        start = max(prev_end + (gap if i else 0.0), min(s[2], target - 0.02 - rest))
        start = max(start, prev_end + (gap if i else 0.0), 0.0)
        out.append((s[0], s[1], start, dur))
        prev_end = start + dur
    return out


# ------------------------------------------------------------------ TSM backends

def stretch(state: dict, seg, sr: int, out_len: int):
    np, sf, backend = state["np"], state["sf"], state["backend"]
    if out_len <= 0 or not len(seg):
        return np.zeros(max(0, out_len))
    tempo = len(seg) / out_len
    if abs(tempo - 1.0) < 1e-3:
        y = seg
    elif backend == "signalsmith":
        y = None
        for tf in (tempo, 1.0 / tempo):  # binding's timeFactor direction is checked, not assumed
            s = state["ps"].Signalsmith.Stretch()
            s.preset(1, sr)
            s.timeFactor = tf
            cand = np.asarray(s.process(seg[None, :].astype(np.float32)))[0].astype(np.float64)
            if abs(len(cand) - out_len) <= 0.05 * out_len:
                y = cand
                break
        if y is None:
            raise RuntimeError("python-stretch output length does not match the time factor")
    else:
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
            sf.write(src, seg, sr, subtype="FLOAT")
            if backend == "atempo":
                stages, f = [], tempo
                while f > 2.0:
                    stages.append(2.0)
                    f /= 2.0
                while f < 0.5:
                    stages.append(0.5)
                    f /= 0.5
                stages.append(f)
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                       "-filter:a", ",".join(f"atempo={v:.6f}" for v in stages), "-c:a",
                       "pcm_f32le", str(dst)]
            else:
                engine = ["-3"] if state["p"].get("engine", "finer") == "finer" else []
                cmd = ["rubberband", "-q", *engine, "--formant", "-D", f"{out_len / sr:.6f}",
                       str(src), str(dst)]
            subprocess.run(cmd, check=True, capture_output=True)
            y, _ = sf.read(dst, dtype="float64", always_2d=True)
            y = y.mean(axis=1)
    if len(y) >= out_len:
        y = y[:out_len]
    else:
        y = np.pad(y, (0, out_len - len(y)))
    fade = min(len(y) // 4, int(0.004 * sr))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y


def run(state: dict, item: dict, out: Path) -> dict:
    np, sf, p = state["np"], state["sf"], state["p"]
    inp = item["inputs"]
    target = float(inp["target_s"])
    x, sr = sf.read(inp["audio"], dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    dur_in = len(x) / sr
    take = vad(np, x, sr)
    if not take:
        raise RuntimeError("no speech detected in the take")
    take = [[max(0.0, s - 0.03), min(dur_in, e + 0.03)] for s, e in take]
    src = source_segments(inp, target) or [[0.03, max(0.06, target - 0.03)]]
    if p.get("mode", "pause_dp") == "uniform":
        take, src = [[take[0][0], take[-1][1]]], [[src[0][0], src[-1][1]]]
    groups = plan(take, src, target, p)
    end = max(target, max(g[2] + g[3] for g in groups))
    y = np.zeros(round(end * sr))
    time_map = [[0.0, 0.0]]
    for t0, t1, o0, dur in groups:
        seg = x[int(t0 * sr):int(t1 * sr)]
        k0, n_out = round(o0 * sr), round(dur * sr)
        z = stretch(state, seg, sr, n_out)
        y[k0:k0 + len(z)] += z[: len(y) - k0]
        time_map += [[round(t0, 4), round(o0, 4)], [round(t1, 4), round(o0 + dur, 4)]]
    time_map.append([round(dur_in, 4), round(len(y) / sr, 4)])
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.999:
        y *= 0.999 / peak
    wav = out.with_suffix(".wav")
    sf.write(wav, y, sr, subtype="PCM_16")
    rates = [round((g[1] - g[0]) / g[3], 4) for g in groups]
    return {"files": {"audio": str(wav)}, "time_map": time_map,
            "params": {"mode": p.get("mode", "pause_dp"), "backend": state["backend"],
                       "groups": len(groups), "rates": rates,
                       "overflow": len(y) / sr > target + 0.02}}


def describe(state: dict) -> dict:
    info = {"backend": state["backend"], "mode": state["p"].get("mode", "pause_dp")}
    if state["backend"] == "rubberband":
        v = subprocess.run(["rubberband", "--version"], capture_output=True, text=True,
                           check=False)
        info["rubberband"] = (v.stdout or v.stderr).strip()
    return info


if __name__ == "__main__":
    serve(load, run, describe)
