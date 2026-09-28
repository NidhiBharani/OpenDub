"""C2 worker (also C1/C3/C4/C5/C6 via ``params.task``): local LLMs behind an OpenAI-compatible
server — vLLM, llama.cpp or Ollama — that this worker starts and stops itself.

params (see ``c_common.OpenAICompatBackend`` for the server ones):
  backend      ollama | vllm | vllm_docker | llamacpp | llamacpp_docker | external
  model        Ollama tag / HF repo id / GGUF repo[:quant]
  task         translate (default) | segment | paraphrase | gemba | tn | g2p | kana | condense
  prompt       translate prompt style: dub (default) | opendub | plain | hunyuan_mt | seed_x |
               translategemma | tower
  temperature, max_tokens, n_candidates, nbest_temperature, extra_body, seed

Lifecycle: the model loads once per job (one server for every item) and is gone when the
process exits — a server we started is SIGTERM'd (docker containers stopped), and an Ollama
model is evicted with ``keep_alive: 0`` because Ollama otherwise keeps it resident and later GPU
stages run out of memory. Nothing is cached across jobs.
"""
from __future__ import annotations

from pathlib import Path

import c_common
from _sdk import serve


def load(params: dict, lang: str):
    c_common.check_task_lang(params, lang)
    backend = c_common.OpenAICompatBackend(params)
    return {"backend": backend, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]

    def complete(messages, *, temperature=None, json_mode=False, max_tokens=2048):
        return state["backend"].complete(messages, temperature=temperature, json_mode=json_mode,
                                         max_tokens=max_tokens, seed=p.get("seed"))

    return c_common.finish(c_common.run_task(complete, item, state["lang"], p), None)


def describe(state: dict) -> dict:
    b = state["backend"]
    return {"backend": b.kind, "model": b.model, "served_name": b.name,
            "task": state["params"].get("task", "translate"),
            "prompt": state["params"].get("prompt", "dub")}


if __name__ == "__main__":
    serve(load, run, describe)
