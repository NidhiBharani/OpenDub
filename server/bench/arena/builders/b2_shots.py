"""Shot-boundary packs for B2 from the public SBD datasets. Shot boundaries do not depend on the
spoken language, so every builder here writes the same items for whichever pack language is
asked for (build it once per language you rank).

- ``clipshots`` — ClipShots test (github.com/Tangshitao/ClipShots, MIT repo): annotations
  ``annotations/test.json`` = ``{"<vid>.mp4": {"transitions": [[s, e], ...], "frame_num": N}}``
  (fetched by script); videos (YouTube/Weibo, Google Drive folder
  1AAhTbNroSFsygHBXa88emCU7f50MxI8t or Baidu) must be downloaded by hand → ``media_dir``.
- ``autoshot_shot`` — the SHOT short-video set from AutoShot (github.com/wentaozhu/AutoShot, MIT):
  data on Google Drive folder 1xZN6tvefXXmpZlIZ6GoSUUxpDQQOSNfJ / Baidu → ``media_dir`` with the
  videos and ``kuaishou_v2.txt`` (one video per line: name followed by shot start/end frame
  pairs; parsed leniently — the exact layout is unverified).
- ``bbc_planet_earth`` — BBC Planet Earth (Baraldi et al. 2015) or RAI (``dataset="rai"``) from
  the AImageLab page (Google Drive) → ``media_dir`` (videos) + ``ann_dir`` (BBC:
  ``annotations/shots/<idx>*`` 0-based "start end" frame pairs per line; RAI:
  ``labels/<idx>_gt.txt``). Layout follows TransNetV2's ``consolidate_datasets.py``.

Frame intervals: a transition ``[s, e]`` with ``e - s ≤ 1`` is a hard cut at frame ``e`` (first
frame of the new shot); longer spans are gradual. Times = frame / fps of the video file.
"""
from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import download, find_media, link_or_copy, need_dir, probe, sample, source_dir

CLIPSHOTS_ANN = "https://raw.githubusercontent.com/Tangshitao/ClipShots/master/annotations/{}.json"


def _transitions_from_pairs(pairs: list[tuple[int, int]], fps: float) -> list[dict]:
    out = []
    for s, e in sorted(pairs):
        if e - s <= 1:
            t = round(e / fps, 4)
            out.append({"start": t, "end": t, "type": "cut"})
        else:
            out.append({"start": round(s / fps, 4), "end": round(e / fps, 4), "type": "gradual"})
    return out


def _transitions_from_shots(shots: list[tuple[int, int]], fps: float) -> list[dict]:
    pairs = []
    shots = sorted(shots)
    for (_s0, e0), (s1, _e1) in pairwise(shots):
        pairs.append((e0, s1) if s1 - e0 > 1 else (s1 - 1, s1))
    return _transitions_from_pairs(pairs, fps)


def _item(prefix: str, name: str, video: Path, lang: str, capability: str,
          transitions_fn, extra_meta: dict | None = None) -> dict:
    rel = f"videos/{video.name}"
    link_or_copy(video, pack_dir(capability, lang) / rel)
    info = probe(video)
    return {"id": f"{prefix}-{Path(name).stem}", "group": Path(name).stem, "split": "test",
            "inputs": {"video": rel},
            "refs": {"transitions": transitions_fn(info["fps"])},
            "meta": {"fps": info["fps"], "duration_s": round(info["duration_s"], 3),
                     "source": prefix, **(extra_meta or {})}}


@builder("clipshots", capabilities=["B2"], langs=["en", "hi", "ja"], license="MIT (annotations)")
def clipshots(lang: str, n: int | None = 100, *, media_dir: str | None = None,
              split: str = "test", seed: int = 0, capability: str = "B2") -> Path:
    """B2 pack from ClipShots (annotations by script, videos from ``media_dir``)."""
    media = need_dir(media_dir, "clipshots", "ClipShots videos from the Google Drive/Baidu links "
                     "in github.com/Tangshitao/ClipShots")
    ann = json.loads(download(CLIPSHOTS_ANN.format(split),
                              source_dir("clipshots") / f"{split}.json").read_text())
    rows = []
    for name in sample(sorted(ann), None, seed):
        video = find_media(media, Path(name).stem) or (media / name if (media / name).exists()
                                                       else None)
        if video is None:
            continue
        pairs = [(int(s), int(e)) for s, e in ann[name].get("transitions", [])]
        rows.append(_item("clipshots", name, video, lang, capability,
                          lambda fps, p=pairs: _transitions_from_pairs(p, fps),
                          {"frame_num": ann[name].get("frame_num")}))
        if n and len(rows) >= n:
            break
    if not rows:
        raise FileNotFoundError(f"clipshots: none of the {split} videos found in {media}")
    return write_pack(capability, lang, rows, f"""# ClipShots ({split}) → B2

Source: https://github.com/Tangshitao/ClipShots (annotations, MIT repo); videos from the
dataset's Drive/Baidu links (YouTube/Weibo originals), eval-only, not redistributed.
Groups: video. Cuts ±2 frames; gradual transitions by overlap (bench/arena/specs/b2.py).
""")


def _parse_shot_lines(text: str) -> dict[str, list[tuple[int, int]]]:
    """``name s e s e …`` per line (whitespace/comma separated) → {name: [(s, e), ...]}."""
    out: dict[str, list[tuple[int, int]]] = {}
    for line in text.splitlines():
        toks = [t for t in re.split(r"[\s,]+", line.strip()) if t]
        if len(toks) < 3:
            continue
        nums = [int(t) for t in toks[1:] if t.lstrip("-").isdigit()]
        out[toks[0]] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    return out


@builder("autoshot_shot", capabilities=["B2"], langs=["en", "hi", "ja"], license="MIT")
def autoshot_shot(lang: str, n: int | None = 200, *, media_dir: str | None = None,
                  labels: str = "kuaishou_v2.txt", seed: int = 0, capability: str = "B2") -> Path:
    """B2 pack from AutoShot's SHOT short-video test set (user-downloaded)."""
    media = need_dir(media_dir, "autoshot_shot", "SHOT videos + kuaishou_v2.txt from the AutoShot "
                     "Google Drive folder")
    lab = media / labels if (media / labels).exists() else next(media.rglob(labels), None)
    if lab is None:
        raise FileNotFoundError(f"autoshot_shot: {labels} not found under {media}")
    shots = _parse_shot_lines(lab.read_text())
    rows = []
    for name in sample(sorted(shots), None, seed):
        video = find_media(media, Path(name).stem) or next(media.rglob(f"{Path(name).stem}.*"),
                                                           None)
        if video is None or video.suffix in (".txt", ".json"):
            continue
        rows.append(_item("shot", name, video, lang, capability,
                          lambda fps, s=shots[name]: _transitions_from_shots(s, fps)))
        if n and len(rows) >= n:
            break
    return write_pack(capability, lang, rows, """# AutoShot SHOT → B2

Source: https://github.com/wentaozhu/AutoShot (MIT); SHOT videos (Kuaishou short videos) from
the authors' Drive/Baidu release, eval-only, not redistributed. Groups: video.
""")


@builder("bbc_planet_earth", capabilities=["B2"], langs=["en", "hi", "ja"],
         license="research use (AImageLab)")
def bbc_planet_earth(lang: str, n: int | None = None, *, media_dir: str | None = None,
                     ann_dir: str | None = None, dataset: str = "bbc",
                     capability: str = "B2") -> Path:
    """B2 pack from BBC Planet Earth (11 episodes) or RAI (``dataset="rai"``, 10 videos)."""
    media = need_dir(media_dir, dataset, "videos from the AImageLab shot-detection page")
    ann = need_dir(ann_dir or media_dir, dataset, "annotation directory from the same page")
    rows = []
    if dataset == "bbc":
        files = sorted((ann / "annotations" / "shots").glob("*")) or sorted(ann.rglob("*shots*"))
        offset = 1  # BBC shot frames are 0-based (TransNetV2 adds 1)
    else:
        files = sorted(ann.rglob("*_gt.txt"))
        offset = 0
    for f in files:
        m = re.match(r"(\d+)", f.name)
        if not m or not f.is_file():
            continue
        idx = m.group(1)
        video = next((v for v in sorted(media.iterdir()) if v.suffix.lower() in
                      (".mp4", ".mkv", ".avi", ".mov", ".webm") and re.search(rf"(^|\D){int(idx)}"
                                                                             r"(\D|$)", v.stem)),
                     None)
        if video is None:
            continue
        shots = []
        for line in f.read_text().splitlines():
            nums = [int(x) for x in re.split(r"[\s,]+", line.strip()) if x.lstrip("-").isdigit()]
            if len(nums) >= 2:
                shots.append((nums[0] + offset, nums[1] + offset))
        rows.append(_item(dataset, f"{dataset}{idx}", video, lang, capability,
                          lambda fps, s=shots: _transitions_from_shots(s, fps)))
    rows = rows[:n] if n else rows
    return write_pack(capability, lang, rows, f"""# {dataset.upper()} shot dataset → B2

Source: AImageLab (Baraldi et al. 2015), https://aimagelab.ing.unimore.it/ — BBC Planet Earth /
RAI shot annotations; videos and labels downloaded by the user, eval-only. Groups: episode.
""")
