"""E2 packs: bandwidth extension.

- ``e2-vctk-sim``: VCTK 0.92 (CC BY 4.0) 48 kHz mic1 speech of the 8 held-out test speakers used
  by AP-BWE / NU-Wave-style BWE papers (p360 p361 p362 p363 p364 p374 p376 s5, from AP-BWE's
  ``VCTK-Corpus-0.92/test.txt``), low-passed by polyphase resampling to 8/16/24 kHz. The 48 kHz
  original is the reference. ``local_dir=`` takes any other 48 kHz corpus laid out as
  ``<speaker>/<utt>.wav|flac`` with an optional ``transcripts.tsv`` (``utt<TAB>text``), which is
  how hi/ja simulated packs can be added.
- ``e2-arena-takes``: real TTS outputs (D1 by default) at their native rate, reference-free.
"""
from __future__ import annotations

import math
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .. import paths
from . import builder
from .e_common import arena_outputs, merge_pack, read_mono

VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
VCTK_TEST_SPEAKERS = ["p360", "p361", "p362", "p363", "p364", "p374", "p376", "s5"]

SIM_LICENSE = """# E2 simulated band-limited speech ({source})

Source: {source_line}
Derived: 48 kHz originals (references) resampled to {rates} Hz with scipy.signal.resample_poly
(the protocol FLowHigh recommends); inputs are stored at the low rate. Eval use.
Groups: utterance (all input rates of one utterance share a group).
"""
VCTK_LINE = ("CSTR VCTK Corpus 0.92, https://datashare.ed.ac.uk/handle/10283/3443, CC BY 4.0 "
             "(Yamagishi, Veaux, MacDonald 2019). Test speakers " + " ".join(VCTK_TEST_SPEAKERS)
             + " (held out by AP-BWE/FLowHigh-style VCTK training).")

ARENA_LICENSE = """# E2 real TTS takes from arena outputs ({capability}, {lang})

Outputs of arena candidates for {capability}; licences follow the generating model. Eval-only.
"""


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    tmp.replace(dst)


def vctk_corpus() -> list[dict[str, Any]]:
    """[{utt, speaker, path, text}] for the VCTK test speakers (mic1), extracted on demand."""
    src = paths.eval_dir() / "_sources" / "vctk"
    zpath = src / "VCTK-Corpus-0.92.zip"
    out = src / "test"
    if not out.exists() or not any(out.rglob("*.flac")):
        if not zpath.exists():
            print(f"downloading VCTK 0.92 (~11 GB) from {VCTK_URL}")
            _download(VCTK_URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                parts = Path(name).parts
                if len(parts) >= 2 and parts[-2] in VCTK_TEST_SPEAKERS and (
                        name.endswith(("_mic1.flac", ".txt"))):
                    dst = out / parts[-2] / parts[-1]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as fsrc, dst.open("wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
    utts = []
    for flac in sorted(out.rglob("*_mic1.flac")):
        spk = flac.parent.name
        stem = flac.name.replace("_mic1.flac", "")
        txt = flac.parent / f"{stem}.txt"
        utts.append({"utt": stem, "speaker": spk, "path": flac,
                     "text": txt.read_text().strip() if txt.exists() else None})
    return utts


def local_corpus(local_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(local_dir).expanduser()
    texts: dict[str, str] = {}
    tsv = root / "transcripts.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                texts[Path(k).stem] = v.strip()
    files = sorted([*root.rglob("*.wav"), *root.rglob("*.flac")])
    return [{"utt": f.stem, "speaker": f.parent.name, "path": f, "text": texts.get(f.stem)}
            for f in files]


@builder("e2-vctk-sim", capabilities=["E2"], langs=["en", "hi", "ja"], license="CC BY 4.0")
def e2_vctk_sim(lang: str = "en", n: int | None = 200, seed: int = 0,
                rates: tuple[int, ...] = (16000, 24000), local_dir: str | None = None) -> Path:
    """E2 simulated pack: 48 kHz speech low-passed to 16/24 kHz (VCTK test speakers for en)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    if local_dir:
        corpus, source, line = local_corpus(local_dir), f"local-{Path(local_dir).name}", \
            f"local corpus {local_dir} (licence: see that corpus)"
    elif lang == "en":
        corpus, source, line = vctk_corpus(), "vctk", VCTK_LINE
    else:
        raise ValueError(f"no public 48 kHz multi-speaker corpus wired for {lang}; pass "
                         f"local_dir=<48 kHz corpus> (e.g. Rasa for hi, JSUT/JVNV for ja)")
    rng = random.Random(seed)
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for u in corpus:
        by_spk.setdefault(u["speaker"], []).append(u)
    per = max(1, math.ceil((n or len(corpus)) / max(1, len(by_spk))))
    root = paths.eval_dir() / "E2" / lang
    rows: list[dict[str, Any]] = []
    for spk, utts in sorted(by_spk.items()):
        rng.shuffle(utts)
        chosen, pool = utts[:per], utts[per:] or utts
        for i, u in enumerate(chosen):
            x, sr = read_mono(u["path"])
            if sr != 48000:
                continue
            clean = root / "clean" / spk / f"{u['utt']}.wav"
            clean.parent.mkdir(parents=True, exist_ok=True)
            sf.write(clean, x, 48000, subtype="PCM_16")
            ref = next((p for p in pool if p["utt"] != u["utt"]), None) if len(utts) > 1 else None
            ref_path = None
            if ref is not None:
                rx, _ = read_mono(ref["path"], 48000)
                ref_path = root / "spk_ref" / spk / f"{ref['utt']}.wav"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(ref_path, rx, 48000, subtype="PCM_16")
                pool = pool[1:] + pool[:1]
            for rate in rates:
                g = math.gcd(48000, rate)
                low = resample_poly(x, rate // g, 48000 // g)
                inp = root / f"in{rate // 1000}k" / spk / f"{u['utt']}.wav"
                inp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(inp, low, rate, subtype="PCM_16")
                refs: dict[str, Any] = {"clean": str(clean.relative_to(root))}
                if u["text"]:
                    refs["text"] = u["text"]
                if ref_path is not None:
                    refs["ref_audio"] = str(ref_path.relative_to(root))
                rows.append({"id": f"{source}-{u['utt']}-{rate // 1000}k", "group": u["utt"],
                             "split": "test", "inputs": {"audio": str(inp.relative_to(root))},
                             "refs": refs,
                             "meta": {"duration_s": round(len(x) / 48000, 3), "input_sr": rate,
                                      "cutoff_hz": rate / 2, "speaker": spk}})
            if n and len({r["group"] for r in rows}) >= n:
                break
    lic = SIM_LICENSE.format(source=source, source_line=line, rates=", ".join(map(str, rates)))
    return merge_pack("E2", lang, rows, lic, f"sim-{source}")


@builder("e2-arena-takes", capabilities=["E2"], langs=["en", "hi", "ja"],
         license="per generating model")
def e2_arena_takes(lang: str, n: int | None = None, capability: str = "D1",
                   candidates: list[str] | None = None) -> Path:
    """E2 pack from real TTS outputs (default D1) at their native rate; reference-free."""
    root = paths.eval_dir() / "E2" / lang
    rows: list[dict[str, Any]] = []
    for o in arena_outputs(capability, lang, candidates):
        it = o["item"]
        x, sr = read_mono(o["audio"])
        if sr >= 44100:
            continue  # already full band: nothing to extend
        dst = root / "arena" / capability / o["candidate"] / f"{it.id}{Path(o['audio']).suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(o["audio"], dst)
        refs: dict[str, Any] = {"text": it.inputs.get("text") or it.refs.get("text", "")}
        ref = it.inputs.get("ref_audio") or it.refs.get("ref_audio")
        if ref and Path(str(ref)).exists():
            refs["ref_audio"] = str(ref)
        rows.append({"id": f"e2a-{capability}-{o['candidate']}-{it.id}", "group": it.group,
                     "split": it.split, "inputs": {"audio": str(dst.relative_to(root))},
                     "refs": refs,
                     "meta": {"duration_s": round(len(x) / sr, 3), "input_sr": sr,
                              "cutoff_hz": sr / 2, "from": capability,
                              "candidate": o["candidate"]}})
        if n and len(rows) >= n:
            break
    return merge_pack("E2", lang, rows, ARENA_LICENSE.format(capability=capability, lang=lang),
                      f"arena-{capability}")
