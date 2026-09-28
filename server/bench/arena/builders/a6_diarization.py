"""A6 packs: VoxConverse test, AMI test (Mix-Headset), and in-house RTTM sets.

- ``a6-voxconverse`` (CC BY 4.0; copyright of the videos stays with their owners): audio zip from
  robots.ox.ac.uk, RTTMs v0.3 from github.com/joonson/voxconverse (``test/<file>.rttm``,
  pinned commit). Multi-speaker YouTube debates/news, mostly English → the ``en`` pack.
- ``a6-ami`` (CC BY 4.0): test meetings of the BUT AMI-diarization-setup (``lists/
  test.meetings.txt``, ``only_words/rttms/test``), Mix-Headset audio from the Edinburgh mirror.
- ``a6-rttm-ih``: user recordings in ``data/eval/_sources/a6_ih/<lang>/`` with same-stem
  ``.rttm`` (and optional ``.lines.json`` [{start, end, speaker}] dub lines). There is no free
  conversational ja diarization set, and DISPLACE (hi) needs a signed request — add those here.

Groups are recordings (the independent unit for diarization). ``n`` limits recordings.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .. import paths
from . import builder
from ._aphase import audio_files, duration, fetch, need, source_dir, write_merged

VOX_COMMIT = "24bf60be297701cd7e4ef18550c6d390c1b87365"
VOX_ZIP = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
VOX_RTTM = f"https://raw.githubusercontent.com/joonson/voxconverse/{VOX_COMMIT}/test/{{}}.rttm"
AMI_COMMIT = "2509d8933721023fab4def2618aabd5c28eb82e9"
AMI_SETUP = f"https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/{AMI_COMMIT}"
AMI_AUDIO = ("https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/"
             "{m}.Mix-Headset.wav")


def _item(out: Path, audio: Path, rttm: Path, rid: str, source: str,
          lines: Path | None = None) -> dict:
    dst_a, dst_r = out / "audio" / audio.name, out / "rttm" / f"{rid}.rttm"
    for s, d in ((audio, dst_a), (rttm, dst_r)):
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            shutil.copy(s, d)
    refs: dict = {"rttm": f"rttm/{rid}.rttm"}
    if lines is not None and lines.exists():
        refs["lines"] = json.loads(lines.read_text())
    return {"id": f"{source}-{rid}", "group": rid, "split": "test",
            "inputs": {"audio": f"audio/{audio.name}"}, "refs": refs,
            "meta": {"duration_s": round(duration(dst_a), 3)}}


@builder("a6-voxconverse", capabilities=["A6"], langs=["en"], license="CC BY 4.0")
def a6_voxconverse(lang: str = "en", n: int | None = None) -> Path:
    """A6 pack: VoxConverse test (232 recordings; RTTM v0.3)."""
    src = source_dir("voxconverse")
    zpath = fetch(VOX_ZIP, src / "voxconverse_test_wav.zip")
    wav_dir = src / "test_wav"
    if not wav_dir.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(wav_dir)
    out = paths.eval_dir() / "A6" / lang
    rows = []
    for audio in audio_files(wav_dir, ("*.wav",)):
        rttm = fetch(VOX_RTTM.format(audio.stem), src / "rttm" / f"{audio.stem}.rttm")
        rows.append(_item(out, audio, rttm, audio.stem, "voxconverse"))
        if n and len(rows) >= n:
            break
    lic = ("# VoxConverse test\n\nSource: https://github.com/joonson/voxconverse (RTTM v0.3, "
           f"commit {VOX_COMMIT[:10]}), audio {VOX_ZIP}. CC BY 4.0; video copyright remains with "
           "the owners. Groups: recording.\n")
    return write_merged("A6", lang, rows, lic, "voxconverse")


@builder("a6-ami", capabilities=["A6"], langs=["en"], license="CC BY 4.0")
def a6_ami(lang: str = "en", n: int | None = None) -> Path:
    """A6 pack: AMI test meetings, Mix-Headset, only_words RTTMs (BUT setup)."""
    src = source_dir("ami")
    meetings = fetch(f"{AMI_SETUP}/lists/test.meetings.txt", src / "test.meetings.txt")
    out = paths.eval_dir() / "A6" / lang
    rows = []
    for m in [x.strip() for x in meetings.read_text().split() if x.strip()]:
        audio = fetch(AMI_AUDIO.format(m=m), src / "audio" / f"{m}.Mix-Headset.wav")
        rttm = fetch(f"{AMI_SETUP}/only_words/rttms/test/{m}.rttm", src / "rttm" / f"{m}.rttm")
        rows.append(_item(out, audio, rttm, m, "ami"))
        if n and len(rows) >= n:
            break
    lic = ("# AMI test (Mix-Headset)\n\nAudio: AMI Meeting Corpus (CC BY 4.0), Edinburgh mirror. "
           f"RTTMs: BUTSpeechFIT/AMI-diarization-setup only_words (commit {AMI_COMMIT[:10]}). "
           "Groups: meeting.\n")
    return write_merged("A6", lang, rows, lic, "ami")


@builder("a6-rttm-ih", capabilities=["A6"], langs=["en", "hi", "ja", "ko", "ta"],
         license="in-house (user media)")
def a6_rttm_ih(lang: str, n: int | None = None, source: str | None = None) -> Path:
    """A6 pack from user recordings with RTTM references (+ optional dub-line files)."""
    root = need(Path(source) if source else source_dir("a6_ih") / lang, f"A6 in-house ({lang})",
                "Put audio files there with same-stem .rttm (and optional .lines.json).")
    out = paths.eval_dir() / "A6" / lang
    rows = []
    for audio in audio_files(root):
        rttm = audio.with_suffix(".rttm")
        if rttm.exists():
            rows.append(_item(out, audio, rttm, audio.stem, "in-house",
                              audio.with_name(audio.stem + ".lines.json")))
        if n and len(rows) >= n:
            break
    return write_merged("A6", lang, rows, f"# A6 in-house ({lang})\n\nUser media from {root}; "
                        "private, evaluation only. Groups: recording.\n", "in-house")
