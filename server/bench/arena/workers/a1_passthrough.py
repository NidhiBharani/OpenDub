"""A1 worker: no separation (OpenDub's ``separation.passthrough`` default) and the classical
supplied-stem / 5.1-centre path.

params: mode = passthrough (dialogue = bed = mix: the floor every separator must beat) |
centre (multichannel input: dialogue = the centre channel, index ``centre_channel`` (default 2,
SMPTE L R C LFE Ls Rs); bed = the other channels, or mix − dialogue with bed_by_subtraction).
Stereo/mono input in centre mode is Unsupported: there is no centre to extract.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import Unsupported, serve
from a1_common import finish, read


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    mix, sr = read(item["inputs"]["audio"])
    if p.get("mode", "passthrough") == "passthrough":
        return finish(out, sr, mix, mix, background=mix, subtract=False)
    c = int(p.get("centre_channel", 2))
    if mix.shape[0] <= c:
        raise Unsupported(f"{mix.shape[0]}-channel input has no centre channel")
    import numpy as np

    dialogue = np.repeat(mix[c:c + 1], mix.shape[0], axis=0) / mix.shape[0]
    rest = mix.copy()
    rest[c] = 0.0
    return finish(out, sr, mix, dialogue, background=rest,
                  subtract=bool(p.get("bed_by_subtraction", False)))


if __name__ == "__main__":
    serve(load, run)
