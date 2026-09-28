"""D1/D2/D4/D7 worker: Fish Audio S2 Pro (fishaudio/s2-pro, Fish Audio Research License — NC)
run locally through the fish-speech repo's HTTP server (``tools/api_server.py``), which is the
repo's documented serving path (docs/en/inference.md). ``load`` starts it from
``~/.opendub/src/fish-speech`` with this env's python; it dies with the worker.

params:
  checkpoint_dir   checkpoints/s2-pro (relative to the repo; the env recipe downloads it)
  server_args      extra args for api_server.py (e.g. ["--compile"], ["--half"])
  temperature (0.7), top_p (0.7), repetition_penalty (1.2), max_new_tokens (1024),
  chunk_length (200), normalize (true)
  emotion_tags     prefix "[<emotion>]" from inputs.emotion (S2 accepts free-form bracket tags)

Request: msgpack ServeTTSRequest {text, references: [{audio: bytes, text}], format: wav, seed …}
(inline reference audio is sent as msgpack bytes, as the Fish API requires).
Docs recommend ≥24 GB VRAM. Output 44.1 kHz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import d_voice as dv
from _sdk import serve


def load(params: dict, lang: str):
    repo = dv.src_dir("fish-speech")
    ckpt = params.get("checkpoint_dir", "checkpoints/s2-pro")
    port = dv.free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "tools/api_server.py", "--listen", f"127.0.0.1:{port}",
           "--llama-checkpoint-path", ckpt,
           "--decoder-checkpoint-path", f"{ckpt}/codec.pth", *params.get("server_args", [])]
    server = dv.LocalServer(cmd, f"{base}/v1/health", cwd=str(repo),
                            timeout=float(params.get("boot_s", 1200)),
                            log_path=Path.home() / ".opendub" / f"fish-speech-{port}.log")
    return {"base": base, "server": server, "params": params, "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    text = dv.text_of(item)
    emo = dv.emotion_of(item)
    if p.get("emotion_tags") and emo:
        text = f"[{emo.lower()}] {text}"
    seed = dv.item_seed(item, p)
    req = {"text": text, "format": "wav", "seed": seed, "streaming": False,
           "normalize": bool(p.get("normalize", True)), "use_memory_cache": "off",
           "chunk_length": int(p.get("chunk_length", 200)),
           "max_new_tokens": int(p.get("max_new_tokens", 1024)),
           "top_p": float(p.get("top_p", 0.7)), "temperature": float(p.get("temperature", 0.7)),
           "repetition_penalty": float(p.get("repetition_penalty", 1.2)), "references": []}
    ref, ref_text = dv.ref_of(item)
    req["references"] = [{"audio": Path(ref).read_bytes(), "text": ref_text}]
    data, _ = dv.http("POST", f"{state['base']}/v1/tts", body=dv.msgpack_dumps(req),
                      headers={"Content-Type": "application/msgpack"}, timeout=900)
    wav = dv.transcode_to_wav(data, dv.wav_path(out), ".wav")
    return dv.audio_payload(wav, seed=seed)


def describe(state: dict) -> dict:
    return {"model": "fishaudio/s2-pro", "checkpoint_dir": state["params"].get("checkpoint_dir")}


if __name__ == "__main__":
    serve(load, run, describe)
