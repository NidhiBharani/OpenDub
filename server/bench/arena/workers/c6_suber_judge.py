"""C6 judge worker: SubER (Wilken et al., IWSLT 2022; github.com/apptek/SubER, pip
``subtitle-edit-rate``) between a candidate's cues and the reference subtitles.

SubER is a TER-style edit rate over words *and* line/block breaks with time-overlap
constraints — the primary metric of the IWSLT subtitling track. The worker writes both cue lists
as SRT and runs ``python -m suber -H hyp.srt -R ref.srt --metrics SubER SubER-cased`` from the
``c6_suber`` env. CPU only.

item.inputs: {hyp_cues: [{start, end, text}], ref_cues: [...]}.
payload: {metrics: {suber, suber_cased}} (percent; lower is better).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from _sdk import serve


def _ts(t: float) -> str:
    ms = round(float(t) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[dict]) -> str:
    blocks = [f"{i}\n{_ts(c['start'])} --> {_ts(c['end'])}\n{c['text'].strip() or '-'}\n"
              for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)


def load(params: dict, lang: str):
    return {"params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    inp = item["inputs"]
    with tempfile.TemporaryDirectory() as tmp:
        hyp, ref = Path(tmp) / "hyp.srt", Path(tmp) / "ref.srt"
        hyp.write_text(to_srt(inp["hyp_cues"]), encoding="utf-8")
        ref.write_text(to_srt(inp["ref_cues"]), encoding="utf-8")
        proc = subprocess.run([sys.executable, "-m", "suber", "-H", str(hyp), "-R", str(ref),
                               "--metrics", "SubER", "SubER-cased"], capture_output=True,
                              text=True, timeout=300, check=True)
    res = json.loads(proc.stdout)
    return {"metrics": {"suber": float(res["SubER"]), "suber_cased": float(res["SubER-cased"])}}


if __name__ == "__main__":
    serve(load, run)
