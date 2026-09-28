"""Test judge worker: scores an output by the length of the text it was given (no model)."""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    return params


def run(state: dict, item: dict, out: Path) -> dict:
    words = len(str(item["inputs"].get("text", "")).split())
    return {"metrics": {"echo_len": words, "echo_len_x2": 2 * words}, "weights": {"echo_len": 1}}


if __name__ == "__main__":
    serve(load, run)
