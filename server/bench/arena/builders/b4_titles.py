"""Title-level content-type packs for B4 (live_action / 2d / 3d / mixed): one item per title.

Seed titles (all freely downloadable; URLs checked 2026-09-28):
- Blender open movies (CC BY 3.0/4.0, download.blender.org): Big Buck Bunny, Sintel, Elephants
  Dream, Caminandes: Gran Dillama (3D CG); Tears of Steel (live action + CG → mixed).
- archive.org public-domain 2D animation: Kumo to Chūrippu (1943), Momotarō: Umi no Shinpei
  (1945), Fleischer Superman (The Mechanical Monsters, Electric Earthquake), Popeye (Patriotic
  Popeye), Betty Boop (Minnie the Moocher), The Big Bad Wolf.
- archive.org public-domain live action: His Girl Friday (1940), The General (1926), Night of
  the Living Dead (1968), Detour (1945).

That is 16 titles; the plan asks for ≥ 100, so pass ``titles_file`` (YAML/JSON list of
``{id, label, url | archive_id | path, lang?, split?}``) to extend (``replace=True`` to replace)
the seed list — e.g. more archive.org cartoons / films, deepghs data, or in-house titles.

Content type does not depend on the spoken language, so by default every pack language gets every
title (``match_lang=True`` keeps only titles whose ``lang`` matches, "*" = no dialogue).
Splits: ~30% of each pure class (at least one when the class has ≥ 3 titles) is ``dev`` — the
SigLIP probe's training titles — the rest ``test``. Groups: title.
"""
from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path

from ..packs import pack_dir, write_pack
from . import builder
from ._b_common import download, link_or_copy, probe, source_dir

BL = "https://download.blender.org"
SEED_TITLES = [
    {"id": "big-buck-bunny", "label": "3d", "lang": "*",
     "url": f"{BL}/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip"},
    {"id": "sintel", "label": "3d", "lang": "en",
     "url": f"{BL}/durian/movies/Sintel.2010.720p.mkv.zip"},
    {"id": "elephants-dream", "label": "3d", "lang": "en",
     "url": f"{BL}/ED/elephantsdream-720-h264-st-aac.mov"},
    {"id": "caminandes-gran-dillama", "label": "3d", "lang": "*",
     "url": f"{BL}/demo/movies/caminandes_gran_dillama.mp4.zip"},
    {"id": "tears-of-steel", "label": "mixed", "lang": "en",
     "url": f"{BL}/demo/movies/ToS/tears_of_steel_720p.mov"},
    {"id": "kumo-to-churippu-1943", "label": "2d", "lang": "ja",
     "archive_id": "kumo-to-churippu-1943"},
    {"id": "momotaro-umi-no-shinpei-1945", "label": "2d", "lang": "ja",
     "archive_id": "momotaro-umi-no-shinpei-1945"},
    {"id": "superman-mechanical-monsters", "label": "2d", "lang": "en",
     "archive_id": "superman_the_mechanical_monsters"},
    {"id": "superman-electric-earthquake", "label": "2d", "lang": "en",
     "archive_id": "superman_electric_earthquake"},
    {"id": "popeye-patriotic-popeye", "label": "2d", "lang": "en",
     "archive_id": "popeye_patriotic_popeye"},
    {"id": "betty-boop-minnie-the-moocher", "label": "2d", "lang": "en",
     "archive_id": "bb_minnie_the_moocher"},
    {"id": "the-big-bad-wolf", "label": "2d", "lang": "en", "archive_id": "the_big_bad_wolf"},
    {"id": "his-girl-friday", "label": "live_action", "lang": "en",
     "archive_id": "his_girl_friday"},
    {"id": "the-general-1926", "label": "live_action", "lang": "*",
     "archive_id": "TheGeneral1926"},
    {"id": "night-of-the-living-dead", "label": "live_action", "lang": "en",
     "archive_id": "night_of_the_living_dead_dvd"},
    {"id": "detour-1945", "label": "live_action", "lang": "en", "archive_id": "Detour"},
]
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".ogv", ".mpeg", ".mpg")


def archive_video_url(identifier: str) -> str:
    """The smallest reasonable video file of an archive.org item (MPEG4 preferred)."""
    meta_url = f"https://archive.org/metadata/{identifier}"
    with urllib.request.urlopen(meta_url, timeout=60) as r:
        meta = json.load(r)
    files = [f for f in meta.get("files", []) if str(f.get("name", "")).lower().endswith(VIDEO_EXT)]
    if not files:
        raise FileNotFoundError(f"archive.org item {identifier!r} has no video file")

    def rank(f):
        mp4 = str(f["name"]).lower().endswith(".mp4")
        size = int(f.get("size") or 0)
        return (not mp4, size < 5_000_000, size)
    return f"https://archive.org/download/{identifier}/{min(files, key=rank)['name']}"


def fetch_title(t: dict, src: Path) -> Path:
    if t.get("path"):
        return Path(t["path"]).expanduser()
    url = t.get("url") or archive_video_url(t["archive_id"])
    dst = download(url, src / t["id"] / Path(url.split("?")[0]).name)
    if dst.suffix == ".zip":
        with zipfile.ZipFile(dst) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(VIDEO_EXT)]
            if not names:
                raise FileNotFoundError(f"{dst} contains no video")
            out = dst.parent / Path(names[0]).name
            if not out.exists():
                out.write_bytes(zf.read(names[0]))
            return out
    return dst


def _load_titles(titles_file: str | None, replace: bool) -> list[dict]:
    titles = [] if replace else [dict(t) for t in SEED_TITLES]
    if titles_file:
        p = Path(titles_file).expanduser()
        if p.suffix in (".yaml", ".yml"):
            import yaml

            extra = yaml.safe_load(p.read_text()) or []
        else:
            extra = json.loads(p.read_text())
        known = {t["id"] for t in titles}
        titles += [t for t in extra if t["id"] not in known]
    return titles


def assign_splits(titles: list[dict], dev_frac: float = 0.3, seed: int = 0) -> None:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for t in titles:
        by_label.setdefault(t["label"], []).append(t)
    for label, ts in by_label.items():
        fixed = [t for t in ts if t.get("split")]
        free = sorted((t for t in ts if not t.get("split")), key=lambda t: t["id"])
        rng.shuffle(free)
        k = 0 if label == "mixed" else round(dev_frac * len(ts))
        if label != "mixed" and len(ts) >= 3:
            k = max(1, k)
        k = max(0, k - sum(t["split"] == "dev" for t in fixed))
        for i, t in enumerate(free):
            t["split"] = "dev" if i < k else "test"


@builder("b4_titles", capabilities=["B4"], langs=["en", "hi", "ja"],
         license="CC BY (Blender) / public domain (archive.org) / per titles_file")
def b4_titles(lang: str, n: int | None = None, *, titles_file: str | None = None,
              replace: bool = False, match_lang: bool = False, dev_frac: float = 0.3,
              seed: int = 0, capability: str = "B4") -> Path:
    """B4 pack: one item per title with its content-type label."""
    titles = _load_titles(titles_file, replace)
    if match_lang:
        titles = [t for t in titles if t.get("lang", "*") in (lang, "*")]
    assign_splits(titles, dev_frac, seed)
    titles = titles[:n] if n else titles
    src = source_dir("b4_titles")
    rows = []
    for t in titles:
        video = fetch_title(t, src)
        rel = f"titles/{t['id']}{video.suffix}"
        link_or_copy(video, pack_dir(capability, lang) / rel)
        info = probe(video)
        rows.append({"id": f"title-{t['id']}", "group": t.get("group", t["id"]),
                     "split": t["split"], "inputs": {"video": rel},
                     "refs": {"label": t["label"]},
                     "meta": {"title": t["id"], "duration_s": round(info["duration_s"], 3),
                              "orig_lang": t.get("lang", "*"),
                              "source": t.get("url") or t.get("archive_id") or "local"}})
    return write_pack(capability, lang, rows, """# Content-type titles → B4

Blender open movies: CC BY (Blender Foundation, https://studio.blender.org/films/), downloaded
from download.blender.org. archive.org titles: public domain per their item pages. Extra titles
from titles_file: see that file for sources and terms. Labels: live_action / 2d / 3d / mixed,
assigned by title (Tears of Steel = mixed: live action with CG). Groups: title; ~30% of each pure
class is the dev split used to fit trained heads.
""")
