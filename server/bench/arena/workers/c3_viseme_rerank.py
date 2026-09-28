"""C3 worker (experimental): PS-Comet-style N-best re-ranking of a translated dub line for
mouth-shape agreement with the original.

1. A generator LLM (any c_common backend: ``generator.backend`` = openai_compat params, or
   ``anthropic`` / ``openai`` / ``gemini`` with a model id) proposes ``n_candidates`` paraphrases.
2. Each candidate (and the input line itself) is scored locally — deterministic and free:
   ``alpha·vowel_dtw + beta·(1 − kept)``, where ``vowel_dtw`` is the length-normalised DTW
   distance between the source's and the candidate's open/rounded/closed vowel-class sequences
   plus bilabial-closure positions, from a grapheme-level approximation per language (en, hi, ja;
   no G2P dependency), and ``kept`` is the word overlap with the input line (a cheap semantic
   guard; the spec's QE-drop gate is the real one).
3. Candidates whose predicted duration leaves the slot by more than ``duration_tol`` (default
   ±10%) are dropped. The best is returned; ``n_candidates: 0`` is the "do nothing" baseline.

Hong et al. (ICPR 2026) weight α = 1.6, β = 0.4 over IPA vowel distances and COMET; this is a
cheaper proxy until G2P (C5) and G5 renders are wired in — which is why C3 is experimental.

item.inputs: {text (translated line), source_text, src_lang?, tgt_lang?, slot_s?}.
payload: {text, candidates: [{text, cost, dtw, kept, seconds}], chosen}.
"""
from __future__ import annotations

import re
from pathlib import Path

import c_common
import c_speech_rate as rate
from _sdk import serve

# vowel classes: A open, I spread, U rounded, E mid, O rounded-mid; B = bilabial closure
_EN = [("oo", "U"), ("ou", "O"), ("ee", "I"), ("ea", "I"), ("ai", "E"), ("ay", "E"),
       ("oa", "O"), ("a", "A"), ("e", "E"), ("i", "I"), ("o", "O"), ("u", "U"), ("y", "I"),
       ("m", "B"), ("b", "B"), ("p", "B")]
_KANA_VOWEL = {}
for row, v in (("あかさたなはまやらわがざだばぱゃ", "A"), ("いきしちにひみりぎじぢびぴ", "I"),
               ("うくすつぬふむゆるぐずづぶぷゅ", "U"), ("えけせてねへめれげぜでべぺ", "E"),
               ("おこそとのほもよろをごぞどぼぽょ", "O")):
    for ch in row:
        _KANA_VOWEL[ch] = v
_HI_MATRA = {"ा": "A", "ि": "I", "ी": "I", "ु": "U", "ू": "U",
             "े": "E", "ै": "E", "ो": "O", "ौ": "O"}
_HI_INDEP = {"अ": "A", "आ": "A", "इ": "I", "ई": "I", "उ": "U", "ऊ": "U", "ए": "E", "ऐ": "E",
             "ओ": "O", "औ": "O"}
_HI_BILABIAL = set("पफबभम")
_COST = {("A", "A"): 0, ("I", "I"): 0, ("U", "U"): 0, ("E", "E"): 0, ("O", "O"): 0,
         ("B", "B"): 0, ("U", "O"): 0.3, ("O", "U"): 0.3, ("I", "E"): 0.3, ("E", "I"): 0.3,
         ("A", "E"): 0.5, ("E", "A"): 0.5, ("A", "O"): 0.6, ("O", "A"): 0.6}


def mouth_sequence(text: str, lang: str) -> list[str]:
    lang = rate.base_lang(lang)
    seq: list[str] = []
    if lang == "ja":
        for ch in text:
            code = ord(ch)
            if 0x30A1 <= code <= 0x30F6:
                ch = chr(code - 0x60)                    # katakana → hiragana
            if ch in "まみむめもばびぶべぼぱぴぷぺぽ":
                seq.append("B")
            if ch in _KANA_VOWEL:
                seq.append(_KANA_VOWEL[ch])
        return seq
    if lang == "hi":
        prev_consonant = False
        for ch in text:
            if ch in _HI_BILABIAL:
                seq.append("B")
            if ch in _HI_INDEP:
                seq.append(_HI_INDEP[ch])
                prev_consonant = False
            elif ch in _HI_MATRA:
                seq.append(_HI_MATRA[ch])
                prev_consonant = False
            elif "क" <= ch <= "ह":
                if prev_consonant:
                    seq.append("A")                      # inherent schwa of the previous one
                prev_consonant = True
            else:
                prev_consonant = False
        return seq
    for word in re.findall(r"[a-z]+", text.lower()):
        i = 0
        while i < len(word):
            for g, cls in _EN:
                if word.startswith(g, i):
                    seq.append(cls)
                    i += len(g)
                    break
            else:
                i += 1
    return seq


def dtw(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 1.0
    inf = float("inf")
    prev = [0.0] + [inf] * len(b)
    for x in a:
        cur = [inf] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            c = _COST.get((x, y), 1.0)
            cur[j] = c + min(prev[j], prev[j - 1], cur[j - 1])
        prev = cur
    return prev[-1] / (len(a) + len(b))


def kept_ratio(original: str, cand: str) -> float:
    ow, cw = set(re.findall(r"\w+", original.lower())), set(re.findall(r"\w+", cand.lower()))
    if not ow:
        return 1.0
    if len(ow) == 1 and len(original) > 6:          # unspaced script: char bigram overlap
        og = {original[i:i + 2] for i in range(len(original) - 1)}
        cg = {cand[i:i + 2] for i in range(len(cand) - 1)}
        return len(og & cg) / max(1, len(og))
    return len(ow & cw) / len(ow)


def rerank(line: str, source: str, cands: list[str], src: str, tgt: str, slot: float | None,
           params: dict) -> tuple[str, list[dict]]:
    alpha, beta = float(params.get("alpha", 1.6)), float(params.get("beta", 0.4))
    tol = float(params.get("duration_tol", 0.10))
    ref_seq = mouth_sequence(source, src)
    scored = []
    for text in [line, *cands]:
        secs = rate.predicted_seconds(text, tgt)
        if slot and text != line and abs(secs / slot - 1) > tol:
            continue
        d, k = dtw(ref_seq, mouth_sequence(text, tgt)), kept_ratio(line, text)
        scored.append({"text": text, "dtw": round(d, 4), "kept": round(k, 3),
                       "seconds": round(secs, 2), "cost": round(alpha * d + beta * (1 - k), 4)})
    best = min(scored, key=lambda s: s["cost"])
    return best["text"], scored


def load(params: dict, lang: str):
    gen = params.get("generator")
    state = {"params": params, "lang": lang, "complete": None, "price": None}
    if gen and int(params.get("n_candidates", 16)) > 0:
        kind = gen.get("backend")
        if kind == "anthropic":
            import c2_anthropic as mod
        elif kind == "openai":
            import c2_openai as mod
        elif kind == "gemini":
            import c2_gemini as mod
        else:
            mod = None
        if mod is None:
            backend = c_common.OpenAICompatBackend(gen)
            state["complete"] = backend.complete
        else:
            sub = mod.load(gen, lang)
            state["complete"] = mod._complete(sub)
            state["price"] = gen.get("price") or getattr(mod, "PRICES", {}).get(gen.get("model"))
    return state


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    src, tgt = c_common.direction(state["lang"], item)
    inp = item["inputs"]
    cands: list[str] = []
    usage: dict = {}
    if state["complete"] is not None:
        cands = c_common.task_paraphrase(state["complete"], item, state["lang"],
                                         {**p, "n_candidates": p.get("n_candidates", 16)}, usage)
    text, scored = rerank(inp["text"], inp.get("source_text", ""), cands, src, tgt,
                          inp.get("slot_s"), p)
    return c_common.finish({"text": text, "candidates": scored, "chosen": text != inp["text"],
                            "_usage": usage}, state["price"])


if __name__ == "__main__":
    serve(load, run)
