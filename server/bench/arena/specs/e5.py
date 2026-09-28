"""E5 — watermarking + provenance. DEFERRED by the user: every candidate is registered
``enabled: false`` and no workers exist yet; this spec fixes how they will be judged.

Items (future ``e5-marks`` builder: FLEURS / D-phase takes, 48 kHz)::

    inputs: {audio, payload_bits?: "0101…" (16-bit message where supported)}
    refs:   {text}
    payload: {files: {audio: marked audio, manifest?: C2PA sidecar}, attacks: {name: {detected:
              bool, bits_ok: float?}}, detected_clean: bool}

A watermark worker embeds, then runs its own detector on the marked file and on a fixed attack
suite applied in the worker (MP3/AAC/Opus at 32–128 kb/s, resample 16 kHz, ±10 % time stretch,
EQ, reverb, polarity inversion, a neural codec round-trip) and on the *unmarked* input (false
positives). C2PA candidates report manifest validation before and after each re-encode.

Primary: mean detection rate over the attack suite at the vendor's 1 % FPR threshold. Gates:
imperceptibility (ViSQOL of marked vs original ≥ 4.0, via the E2 ViSQOL judge) and a false
positive rate ≤ 1 % on unmarked audio.
"""
from __future__ import annotations

from typing import Any

from ..judges import ModelJudge, Row, Spec, cost, speed
from ..packs import Item
from .e2 import visqol


def robustness(item: Item, out: dict[str, Any], row: Any) -> list[Row]:
    attacks = out.get("attacks") or {}
    rows: list[Row] = []
    if attacks:
        det = [float(bool(a.get("detected"))) for a in attacks.values()]
        rows.append(("detect_rate", sum(det) / len(det), float(len(det))))
        bits = [float(a["bits_ok"]) for a in attacks.values() if a.get("bits_ok") is not None]
        if bits:
            rows.append(("bit_acc", sum(bits) / len(bits), float(len(bits))))
    if "detected_clean" in out:
        rows.append(("false_positive", float(bool(out["detected_clean"])), 1.0))
    if "manifest_valid" in out:
        rows.append(("manifest_valid", float(bool(out["manifest_valid"])), 1.0))
    return rows


def _imperceptibility(lang: str) -> list[ModelJudge]:
    import dataclasses

    judge = visqol()

    def inputs(item: Item, out: dict[str, Any], prefix):
        marked = (out.get("files") or {}).get("audio")
        return {"audio": marked, "reference": item.inputs.get("audio")} if marked else None

    return [dataclasses.replace(judge, id="visqol.marked@1", inputs=inputs)]


SPEC = Spec(
    id="E5",
    title="Watermarking + provenance (deferred)",
    judges={"robustness@1": robustness, "speed@1": speed, "cost@1": cost},
    primary={"*": "detect_rate"},
    higher_is_better={"detect_rate": True, "bit_acc": True, "false_positive": False,
                      "manifest_valid": True, "visqol": True, "rtfx": True, "cost_usd": False},
    threshold={"detect_rate": 0.02},
    secondary=["bit_acc", "false_positive", "visqol", "manifest_valid", "rtfx"],
    model_judges=_imperceptibility,
    gates={"visqol": (">=", 4.0), "false_positive": ("<=", 0.01)},
    packs=[],
    io="""item.inputs: {audio, payload_bits?}; payload: {files: {audio, manifest?}, attacks:
{name: {detected, bits_ok?}}, detected_clean, manifest_valid?}. Deferred: no workers yet.""",
)
