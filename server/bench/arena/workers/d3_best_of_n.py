"""Method worker (D1/D3 …): best-of-N sampling over another D candidate, re-ranked by a verifier.

The method candidate runs in the INNER candidate's env: this module imports the inner worker
(``params.inner.worker``), calls its ``load`` once and its ``run`` N times per item with
``item["seed"] = base + k`` (every D worker honours ``item["seed"]``), transcribes each take with
the ASR verifier subprocess (``d3_asr_verifier.py`` in ``verifier.env``, never the reporting
judge model) and keeps the best take.

params:
  inner     {worker: d1_f5_tts, params: {...}}   — copy of the inner candidate's worker/params
  n         takes per item (4)
  verifier  {env: server, model: large-v3-turbo, compute_type: float16}
  accept_cer  a take is acceptable when verifier CER ≤ this (0.10)
  window    D3 duration window: |dur − target_s| / target_s ≤ window (0.10) when target_s is set
  rank      "cer" | "cer+duration" (default: cer+duration when target_s is set)
  early_stop stop sampling at the first acceptable take (false: always N, so accept-rate@N is
            measured on N takes)
  keep_takes keep every take's wav next to the output for the audit viewer (true)

Payload: the chosen take's files.audio plus ``takes`` [{k, seed, duration_s, cer, in_window,
accepted}], ``accept_rate`` and ``n`` — the D3 spec reads them for accept-rate@N.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import d_voice as dv
from _sdk import serve


def load(params: dict, lang: str):
    inner = params["inner"]
    mod = dv.import_worker(inner["worker"])
    state = mod.load(inner.get("params", {}), lang)
    vcfg = dict(params.get("verifier") or {})
    verifier = dv.Verifier(vcfg.pop("env", "server"), vcfg)
    return {"mod": mod, "inner": state, "verifier": verifier, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    n = int(p.get("n", 4))
    text = dv.text_of(item)
    tgt = dv.target_s(item)
    window = float(p.get("window", 0.10))
    accept_cer = float(p.get("accept_cer", 0.10))
    rank = p.get("rank") or ("cer+duration" if tgt else "cer")
    base = dv.item_seed(item, p.get("inner", {}).get("params", {}))
    takes, best = [], None
    for k in range(n):
        tout = Path(out).with_name(f"{Path(out).name}-take{k}")
        try:
            payload = state["mod"].run(state["inner"], {**item, "seed": base + k}, tout)
        except Exception as exc:  # noqa: BLE001 - one failed take is a rejected take
            takes.append({"k": k, "seed": base + k, "error": f"{type(exc).__name__}: {exc}",
                          "accepted": False})
            continue
        audio = payload["files"]["audio"]
        dur = dv.duration_s(audio)
        c = dv.cer(text, state["verifier"].transcribe(audio, state["lang"]))
        in_win = None if not tgt else abs(dur - tgt) / tgt <= window
        ok = c <= accept_cer and in_win is not False
        score = c + (abs(dur - tgt) / tgt if tgt and rank == "cer+duration" else 0.0)
        takes.append({"k": k, "seed": base + k, "duration_s": round(dur, 4), "cer": round(c, 4),
                      "in_window": in_win, "accepted": ok, "audio": audio})
        # prefer accepted takes, then the lowest score
        key = (not ok, score)
        if best is None or key < best[0]:
            best = (key, audio)
        if ok and p.get("early_stop"):
            break
    if best is None:
        raise RuntimeError(f"all {n} takes failed: {takes[-1].get('error')}")
    wav = dv.wav_path(out)
    shutil.copyfile(best[1], wav)
    if not p.get("keep_takes", True):
        for t in takes:
            if t.get("audio"):
                Path(t["audio"]).unlink(missing_ok=True)
    accepted = sum(bool(t["accepted"]) for t in takes)
    return dv.audio_payload(wav, takes=takes, n=len(takes),
                            accept_rate=round(accepted / max(1, len(takes)), 4),
                            chosen=next(t["k"] for t in takes if t.get("audio") == best[1]))


def describe(state: dict) -> dict:
    inner = state["params"]["inner"]
    d = {"inner_worker": inner["worker"], "n": state["params"].get("n", 4)}
    if hasattr(state["mod"], "describe"):
        d["inner"] = state["mod"].describe(state["inner"])
    return d


if __name__ == "__main__":
    serve(load, run, describe)
