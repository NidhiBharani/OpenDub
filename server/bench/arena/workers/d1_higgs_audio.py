"""D1/D2/D4/D7 worker: Higgs Audio v3 TTS 4B (bosonai/higgs-tts-3-4b, Boson research/NC licence).

The documented local paths are servers (SGLang-Omni ``sgl-omni serve`` or vLLM-Omni
``vllm serve … --omni``) exposing OpenAI-style ``/v1/audio/speech`` with zero-shot cloning
(fields ``model, input, ref_audio (url or base64), ref_text, voice, response_format``, per the
model repo's AGENTS.md). ``load`` starts the server, ``run`` posts one request per item, and the
server dies with the worker process so its GPU memory is freed.

params:
  model        HF repo (bosonai/higgs-tts-3-4b); revision
  server       vllm-omni (default) | sglang-omni | external (use ``base_url``, start nothing)
  base_url     for server=external
  gpu_memory_utilization  vLLM fraction (default 0.85)
  temperature (0.8), top_k (50), max_new_tokens (1024)  — card sampling for cloning
  emotion_tags  prefix ``<|emotion:<x>|>`` from inputs.emotion (D2); PROMPTING.md has the catalog
  ref_format   base64 (default) | data_url

Response audio is 24 kHz (wav requested).
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import d_voice as dv
from _sdk import serve


def load(params: dict, lang: str):
    server = params.get("server", "vllm-omni")
    handle = None
    if server == "external":
        base = params["base_url"].rstrip("/")
    else:
        port = dv.free_port()
        base = f"http://127.0.0.1:{port}"
        model = params.get("model", "bosonai/higgs-tts-3-4b")
        bindir = Path(sys.executable).parent  # the env's own console scripts
        if server == "vllm-omni":
            cmd = [str(bindir / "vllm"), "serve", model, "--host", "127.0.0.1",
                   "--port", str(port), "--trust-remote-code", "--omni",
                   "--gpu-memory-utilization", str(params.get("gpu_memory_utilization", 0.85))]
            if params.get("revision"):
                cmd += ["--revision", params["revision"]]
        else:
            cmd = [str(bindir / "sgl-omni"), "serve", "--model-path", model, "--port", str(port)]
        handle = dv.LocalServer(cmd, f"{base}/health", timeout=float(params.get("boot_s", 1800)),
                                log_path=Path.home() / ".opendub" / f"higgs-{port}.log")
    return {"base": base, "server": handle, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    emo = dv.emotion_of(item)
    if p.get("emotion_tags") and emo:
        text = f"<|emotion:{emo.lower()}|>{text}"
    body = {"model": p.get("served_name", "higgs-audio-v3-tts"), "input": text,
            "response_format": "wav", "temperature": float(p.get("temperature", 0.8)),
            "top_k": int(p.get("top_k", 50)),
            "max_new_tokens": int(p.get("max_new_tokens", 1024)),
            "seed": dv.item_seed(item, p)}
    ref, ref_text = dv.ref_of(item, required=p.get("mode", "clone") == "clone")
    if ref:
        b64 = dv.b64file(ref)
        body["ref_audio"] = f"data:audio/wav;base64,{b64}" if p.get("ref_format") == \
            "data_url" else b64
        if ref_text:
            body["ref_text"] = ref_text
    elif p.get("voice"):
        body["voice"] = dv.pick_voice(p["voice"], state["lang"], item)
    data, headers = dv.post_json(f"{state['base']}/v1/audio/speech", body, timeout=600)
    if headers.get("Content-Type", "").startswith("application/json"):
        import json

        data = base64.b64decode(json.loads(data)["audio"])
    wav = dv.transcode_to_wav(data, dv.wav_path(out), ".wav")
    return dv.audio_payload(wav, seed=body["seed"])


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": p.get("model"), "revision": p.get("revision"), "server": p.get("server")}


if __name__ == "__main__":
    serve(load, run, describe)
