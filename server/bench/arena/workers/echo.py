"""Test worker: echoes an input field (optionally mangled), so the arena runs without models."""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    return params


def run(state: dict, item: dict, out: Path) -> dict:
    if item["id"] in state.get("fail_ids", []):
        raise RuntimeError("scripted failure")
    text = str(item["inputs"].get(state.get("field", "text"), ""))
    drop = int(state.get("drop_words", 0))
    words = text.split()
    payload = {"text": " ".join(words[drop:])}
    if state.get("cost_per_item"):
        payload["_cost_usd"] = state["cost_per_item"]
    return payload


if __name__ == "__main__":
    serve(load, run)
