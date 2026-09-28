"""A5 worker: Montreal Forced Aligner 3.x (MIT; pretrained models CC BY 4.0), CPU.

``mfa align_one SOUND_FILE TEXT_FILE DICTIONARY ACOUSTIC_MODEL OUTPUT_PATH`` per item (MFA docs,
3.4.2), then the TextGrid "words" tier is read back. params: dictionary / acoustic_model
(pretrained names, default english_mfa / japanese_mfa by language — there is no hindi_mfa, so
hi needs a user-supplied model, e.g. from AI4Bharat IndicMFA), mfa (executable path, default
the env's ``bin/mfa``), beam, extra_args. Japanese needs the sudachi/spacy tokenizer extras in
the env. Run ``mfa model download acoustic <name>`` / ``dictionary <name>`` once per env (the
setup recipe does it for english_mfa and japanese_mfa).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _sdk import serve
from a5_common import project, tokenize

DEFAULTS = {"en": "english_mfa", "ja": "japanese_mfa"}


def load(params: dict, lang: str):
    mfa = params.get("mfa") or shutil.which("mfa") or str(Path(sys.executable).parent / "mfa")
    name = DEFAULTS.get(lang)
    dic, am = params.get("dictionary", name), params.get("acoustic_model", name)
    if not dic or not am:
        raise ValueError(f"no default MFA model for {lang!r}; set params.dictionary and "
                         "params.acoustic_model")
    return {"mfa": mfa, "dict": dic, "am": am, "params": params, "lang": lang}


def read_words(textgrid: Path) -> list[dict]:
    """Intervals of the "words" IntervalTier of a long-format TextGrid (non-empty labels)."""
    txt = textgrid.read_text(encoding="utf-8", errors="replace")
    tiers = re.split(r'\n\s*item \[\d+\]:', txt)
    tier = next((t for t in tiers if re.search(r'name = "words"', t)), txt)
    out = []
    for xmin, xmax, label in re.findall(
            r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*?)"', tier):
        if label.strip() and label.strip() not in ("<eps>", "sil", "sp", "spn"):
            out.append({"start": float(xmin), "end": float(xmax), "word": label})
    return out


def run(state: dict, item: dict, out: Path) -> dict:
    text = item["inputs"]["text"]
    with tempfile.TemporaryDirectory(prefix="mfa-") as tmp:
        t = Path(tmp)
        (t / "in.txt").write_text(text, encoding="utf-8")
        tg = t / "out.TextGrid"
        cmd = [state["mfa"], "align_one", item["inputs"]["audio"], str(t / "in.txt"),
               state["dict"], state["am"], str(tg), "--clean", "--quiet",
               *[str(a) for a in state["params"].get("extra_args", [])]]
        if state["params"].get("beam"):
            cmd += ["--beam", str(state["params"]["beam"])]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode or not tg.exists():
            raise RuntimeError(f"mfa align_one failed ({proc.returncode}): "
                               f"{(proc.stderr or proc.stdout)[-800:]}")
        units = read_words(tg)
        shutil.copy(tg, out.with_suffix(".TextGrid"))
    return {"words": project(tokenize(text, state["lang"]), units), "units": units}


if __name__ == "__main__":
    serve(load, run)
