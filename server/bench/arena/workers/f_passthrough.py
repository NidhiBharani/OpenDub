"""F2/F3/F4 baseline worker: deliver the input picture unchanged (what OpenDub does today).

F2: the raw lip-synced video (no restoration). F3: decline to lip-sync animation, without a
mouth track (``f3_decline`` adds one). F4: the source-language text stays on screen.
params: none. Runs in the server env; stdlib + ffmpeg only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from _sdk import serve
from f_common import item_io, video_payload


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    video, audio = item_io(item)
    if audio is None:
        dst = out.with_suffix(".mp4")
        shutil.copyfile(video, dst)
        return video_payload(out, dst, None, remux=False)
    return video_payload(out, video, audio)


if __name__ == "__main__":
    serve(load, run)
