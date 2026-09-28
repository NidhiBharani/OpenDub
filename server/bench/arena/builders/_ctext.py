"""Shared helpers for the C-phase (text) pack builders. Not a builder (underscore: not
auto-imported by ``builders/__init__``)."""
from __future__ import annotations

import io
import os
import random
import re
import tarfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .. import paths

LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese"}


def split_direction(lang: str) -> tuple[str, str]:
    if "-" not in lang:
        raise ValueError(f"text packs are keyed by direction (e.g. 'ja-en'), got {lang!r}")
    src, tgt = lang.split("-", 1)
    return src, tgt


def sources_dir(name: str) -> Path:
    d = paths.eval_dir() / "_sources" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_hf_dataset(repo: str, config: str | None, split: str, *, gated: bool = False):
    """``datasets.load_dataset`` with a clear message for gated repos (needs HF_TOKEN)."""
    from datasets import load_dataset

    token = hf_token()
    if gated and not token:
        raise RuntimeError(f"{repo} is gated on Hugging Face: accept its terms on the dataset "
                           "page and export HF_TOKEN")
    cache = str(sources_dir("hf_datasets"))
    return load_dataset(repo, config, split=split, token=token, cache_dir=cache)


def download(url: str, dest: Path, *, timeout: float = 600.0) -> Path:
    """Fetch ``url`` to ``dest`` once (skipped when present)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "opendub-arena"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def github_tarball(repo: str, refs: Iterable[str], dest_dir: Path) -> Path:
    """Download ``repo`` (owner/name) at the first ref that exists; returns the .tar.gz."""
    last: Exception | None = None
    for ref in refs:
        dest = dest_dir / f"{repo.replace('/', '__')}-{ref.replace('/', '_')}.tar.gz"
        if dest.exists():
            return dest
        kind = "heads" if ref in ("main", "master") else "tags"
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/{kind}/{ref}"
        try:
            return download(url, dest)
        except OSError as exc:  # 404 for a tag spelling that does not exist
            last = exc
    raise RuntimeError(f"could not download {repo} at any of {list(refs)}: {last}")


def tar_members(path: Path, want: Callable[[str], bool]) -> dict[str, bytes]:
    """{member name: bytes} for members whose name satisfies ``want`` (single pass)."""
    out: dict[str, bytes] = {}
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile() and want(m.name):
                f = tf.extractfile(m)
                if f is not None:
                    out[m.name] = f.read()
    return out


def sample(rows: list[Any], n: int | None, seed: int) -> list[Any]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def sample_groups(rows: list[dict], n: int | None, seed: int,
                  key: Callable[[dict], str]) -> list[dict]:
    """Sample whole groups (documents) until ``n`` rows: keeps context and the bootstrap unit."""
    if not n or n >= len(rows):
        return rows
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    out: list[dict] = []
    for g in order:
        if len(out) >= n:
            break
        out.extend(groups[g])
    return out          # whole groups: may overshoot n by part of one document


def with_context(lines: list[str], k: int, before: int = 2, after: int = 1) -> dict[str, list]:
    return {"context_before": lines[max(0, k - before):k], "context_after": lines[k + 1:k + 1 +
                                                                                   after]}


def mt_row(item_id: str, group: str, src: str, tgt: str, text: str, ref: str | None,
           source: str, **extra: Any) -> dict[str, Any]:
    """One C2 manifest row."""
    ctx = extra.pop("context", {}) or {}
    meta = {"src_lang": src, "tgt_lang": tgt, "source": source, **extra.pop("meta", {})}
    row = {"id": item_id, "group": group, "split": extra.pop("split", "test"),
           "inputs": {"text": text, "src_lang": src, "tgt_lang": tgt, **ctx, **extra},
           "refs": {"text": ref} if ref else {}, "meta": meta}
    return row


def license_md(title: str, source: str, terms: str, notes: str = "") -> str:
    return f"# {title}\n\nSource: {source}\nLicense: {terms}\n\n{notes}".rstrip() + "\n"


# ------------------------------------------------------------------ SRT

_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _secs(ts: str) -> float:
    m = _TS.search(ts)
    if not m:
        raise ValueError(f"bad SRT timestamp {ts!r}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def parse_srt(text: str) -> list[dict[str, Any]]:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue
        if "-->" not in lines[0]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        a, b = lines[0].split("-->")
        cues.append({"start": _secs(a), "end": _secs(b),
                     "text": "\n".join(ln.strip() for ln in lines[1:])})
    return cues


def read_srt(path: Path) -> list[dict[str, Any]]:
    return parse_srt(path.read_text(encoding="utf-8-sig"))


def words_from_payload(payload: dict | list) -> list[dict[str, Any]]:
    """Timed words from an A4/A5 payload ({segments: [{words}]}, {words}) or a plain list."""
    if isinstance(payload, list):
        words = payload
    elif payload.get("words"):
        words = payload["words"]
    else:
        words = [w for seg in payload.get("segments", []) for w in seg.get("words", [])]
    out = []
    for w in words:
        text = str(w.get("word", w.get("text", ""))).strip()
        if text and w.get("end") is not None and w.get("start") is not None:
            out.append({"start": float(w["start"]), "end": float(w["end"]), "word": text})
    return out


def windows(words: list[dict], max_s: float = 60.0, gap_s: float = 1.2) -> list[tuple[int, int]]:
    """Split a long word list into windows at pauses ≥ ``gap_s`` (or hard at ``max_s``)."""
    out, start = [], 0
    for k in range(1, len(words) + 1):
        end_here = k == len(words)
        if not end_here:
            gap = words[k]["start"] - words[k - 1]["end"]
            span = words[k]["end"] - words[start]["start"]
            if (gap >= gap_s and words[k - 1]["end"] - words[start]["start"] >= 5.0) \
                    or span > max_s:
                out.append((start, k))
                start = k
        else:
            out.append((start, k))
    return [w for w in out if w[1] > w[0]]


def pause_lines(words: list[dict], lang: str, min_pause: float = 0.35,
                max_s: float = 12.0) -> list[dict[str, Any]]:
    """Dub lines from timed words (the C1 baseline rule, stdlib-only copy for builders)."""
    ends = tuple("。．.!?！？…")
    joiner = "" if lang in ("ja", "zh") else " "
    lines, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            if gap >= min_pause or (cur[-1]["word"].rstrip("」』)\"'").endswith(ends)
                                    and gap >= 0.12) or w["end"] - cur[0]["start"] > max_s:
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [{"start": ln[0]["start"], "end": ln[-1]["end"],
             "text": joiner.join(x["word"] for x in ln).strip(), "words": ln} for ln in lines]


def read_text_bytes(data: bytes) -> str:
    return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace").read()
