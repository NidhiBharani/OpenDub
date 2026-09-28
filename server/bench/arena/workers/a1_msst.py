"""A1/A2 worker: ZFTurbo Music-Source-Separation-Training (``msst`` PyPI 0.1.0, MIT) — BandIt v2
(DnR v3 Zenodo checkpoints, CC BY-SA 4.0), BandIt Plus (DnR v2), Mel-Band RoFormer vocals /
dereverb / denoise checkpoints, SCNet …

params: model_type (bandit_v2 | bandit | mel_band_roformer | …), config, checkpoint (URLs or
local paths; URLs are fetched once into ~/.opendub/models/msst/ when the job runs), stems
({dialogue: <instrument>, music: …, effects: …} for A1, or {audio: <instrument>} for A2
enhancement), bed_by_subtraction (A1, default true).

Python API per the msst README (checked 2026-09-28): ``msst.inference(model_type=…,
config_path=…, checkpoint_path=…, input_folder=…, output_folder=…, device_ids=0)``; outputs are
written as ``<output>/<file_name>/<instrument>.wav`` (default ``--filename_template``) — the
worker globs for ``*<instrument>*`` to stay robust to template changes. The model is reloaded
for every item (the public API has no persistent session), so RTF includes load time.
"""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path

from _sdk import serve
from a1_common import finish, read

CACHE = Path.home() / ".opendub" / "models" / "msst"


def fetch(ref: str) -> str:
    if not ref.startswith(("http://", "https://")):
        return str(Path(ref).expanduser())
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / ref.rstrip("/").split("/")[-1].split("?")[0]
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(ref, tmp)
        tmp.rename(dest)
    return str(dest)


def load(params: dict, lang: str):
    import msst

    return {"msst": msst, "config": fetch(params["config"]),
            "ckpt": fetch(params["checkpoint"]), "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    audio = item["inputs"]["audio"]
    with tempfile.TemporaryDirectory(prefix="msst-") as tmp:
        t = Path(tmp)
        (t / "in").mkdir()
        src = t / "in" / f"mix{Path(audio).suffix}"
        shutil.copy(audio, src)
        state["msst"].inference(model_type=p["model_type"], config_path=state["config"],
                                checkpoint_path=state["ckpt"], input_folder=str(t / "in"),
                                output_folder=str(t / "out"), device_ids=0)
        produced = sorted((t / "out").rglob("*.wav")) + sorted((t / "out").rglob("*.flac"))
        stems = {}
        for role, instr in (p.get("stems") or {"dialogue": "speech"}).items():
            hit = [f for f in produced if f.stem == instr or f.stem.endswith(f"_{instr}")
                   or f.stem.endswith(instr)]
            if not hit:
                raise RuntimeError(f"no {instr!r} stem among {[f.name for f in produced]}")
            stems[role], sr = read(str(hit[0]))
        if "audio" in stems:  # A2: enhanced speech
            import soundfile as sf

            dst = out.with_suffix(".wav")
            sf.write(str(dst), stems["audio"].T, sr)
            return {"files": {"audio": str(dst)}, "sample_rate": sr}
        mix, _ = read(audio, sr)
        if mix.shape[0] != stems["dialogue"].shape[0]:
            import numpy as np

            mix = np.repeat(mix.mean(0, keepdims=True), stems["dialogue"].shape[0], axis=0)
        return finish(out, sr, mix, stems["dialogue"], music=stems.get("music"),
                      effects=stems.get("effects"), background=stems.get("background"),
                      subtract=bool(p.get("bed_by_subtraction", True)))


def describe(state: dict) -> dict:
    from importlib.metadata import version

    return {"model_type": state["params"].get("model_type"),
            "checkpoint": state["params"].get("checkpoint"), "msst": version("msst")}


if __name__ == "__main__":
    serve(load, run, describe)
