"""B4 baseline worker: a constant label (server env, no model).

OpenDub today has no content-type gate: the operator picks ``lipsync.none`` by hand and nothing
stops a human-face lip-sync model from running on animation, i.e. every title is treated as live
action. params: label (default live_action).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str) -> dict:
    return {"label": params.get("label", "live_action")}


def run(state: dict, item: dict, out: Path) -> dict:
    return {"label": state["label"], "probs": {state["label"]: 1.0}, "confidence": 1.0}


if __name__ == "__main__":
    serve(load, run)
