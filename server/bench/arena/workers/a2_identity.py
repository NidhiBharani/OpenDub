"""A2 worker: no enhancement (OpenDub's A2 default is off: the raw clip is the reference).
Copies the input so every enhancer is compared against leaving the clip alone."""
from __future__ import annotations

import shutil
from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    return {}


def run(state: dict, item: dict, out: Path) -> dict:
    src = Path(item["inputs"]["audio"])
    dst = out.with_suffix(src.suffix)
    shutil.copy(src, dst)
    return {"files": {"audio": str(dst)}}


if __name__ == "__main__":
    serve(load, run)
