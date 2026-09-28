"""Eval packs: frozen inputs + references for one capability in one language.

``data/eval/<capability>/<lang>/manifest.jsonl``, one item per line::

    {"id": "fleurs-1834", "group": "1834", "split": "test",
     "inputs": {"audio": "audio/1834.wav"}, "refs": {"text": "…"}, "meta": {"duration_s": 6.1}}

Relative paths resolve against the pack directory. ``group`` is the independent resampling unit
for the bootstrap (speaker, file, video or title; never a frame). ``LICENSE.md`` next to the
manifest records where the data came from and under what terms.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths


@dataclass
class Item:
    id: str
    lang: str
    inputs: dict[str, Any]
    refs: dict[str, Any] = field(default_factory=dict)
    group: str = ""
    split: str = "test"
    meta: dict[str, Any] = field(default_factory=dict)

    def input_hash(self) -> str:
        """Hash of the inputs *content* (files by bytes), so moving a pack keeps the cache valid."""
        h = hashlib.sha256()
        for key in sorted(self.inputs):
            val = self.inputs[key]
            h.update(key.encode())
            p = Path(val) if isinstance(val, str) else None
            if p is not None and p.is_file():
                h.update(_file_digest(p).encode())
            else:
                h.update(json.dumps(val, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    def ref_hash(self) -> str:
        blob = json.dumps(self.refs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _DIGESTS[key] = h.hexdigest()
    return _DIGESTS[key]


def pack_dir(capability: str, lang: str) -> Path:
    return paths.eval_dir() / capability / lang


def load_pack(capability: str, lang: str, *, split: str | None = "test",
              limit: int | None = None) -> list[Item]:
    root = pack_dir(capability, lang)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"no eval pack at {manifest} (build it with `bench arena pack`)")
    items: list[Item] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if split and raw.get("split", "test") != split:
            continue
        inputs = {k: _resolve(root, v) for k, v in raw.get("inputs", {}).items()}
        refs = {k: _resolve(root, v) for k, v in raw.get("refs", {}).items()}
        items.append(Item(id=raw["id"], lang=lang, inputs=inputs, refs=refs,
                          group=str(raw.get("group") or raw["id"]),
                          split=raw.get("split", "test"), meta=raw.get("meta", {})))
        if limit and len(items) >= limit:
            break
    return items


def _resolve(root: Path, val: Any) -> Any:
    """Relative file references become absolute; everything else passes through."""
    if not isinstance(val, str) or val.startswith("/") or len(val) > 240 or "\n" in val:
        return val
    try:
        return str((root / val).resolve()) if (root / val).is_file() else val
    except OSError:  # e.g. a reference sentence that merely looks like a name
        return val


def write_pack(capability: str, lang: str, rows: list[dict[str, Any]], license_md: str) -> Path:
    root = pack_dir(capability, lang)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "LICENSE.md").write_text(license_md)
    return root
