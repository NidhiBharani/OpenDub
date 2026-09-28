"""F1 worker: LTX-2.3-22B IC-LoRA DubIt (Lightricks; formerly "LipDub"), via ``ltx-pipelines``.

Whole-frame generative redubbing: the pipeline regenerates lips *and* speech from the new line's
**text** — it takes no driving audio. The reference video's own audio is only a voice/identity
reference. So this candidate needs the spoken line (``item.meta.text``; items without it are
Unsupported), and its output carries *its own* generated audio, which is what ``files.audio``
points to (the sync panel then measures lips against the speech it actually produced). It
collapses D1+F1 into one stage; compare on the sync panel and identity, not on voice.

params:
  hf_repo / checkpoint: base checkpoint (default Lightricks/LTX-2.3,
      ltx-2.3-22b-distilled.safetensors; FP8: Lightricks/LTX-2.3-fp8,
      ltx-2.3-22b-distilled-fp8.safetensors)
  quantization: none | fp8-cast | fp8-scaled-mm | nvfp4-cast | nvfp4-prequant
  lora_repo / lora_file / lora_strength: Lightricks/LTX-2.3-22b-IC-LoRA-DubIt,
      ltx-2.3-22b-ic-lora-dubit-0.9.safetensors, 1.0
  upsampler_repo / upsampler_file: Lightricks/LTX-2.3, ltx-2.3-spatial-upscaler-x2-1.1.safetensors
  gemma_repo: google/gemma-3-12b-it-qat-q4_0-unquantized (gated: HF_TOKEN)
  reference_strength (1.0), seed (0), offload_mode (none), speaker ("The person")
  repo_dir: the LTX-2 checkout (default ~/.opendub/src/LTX-2)

Validated languages per the docs: en, fr, es, de, ru (single speaker, beta).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _sdk import Unsupported, serve
from f_common import cleanup, item_io, mux, repo_dir, to_wav, video_payload, workdir
from f_common import run as sh

LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "ru": "Russian",
              "hi": "Hindi", "ja": "Japanese"}


def load(params: dict, lang: str):
    from huggingface_hub import hf_hub_download, snapshot_download

    repo = repo_dir(params, "LTX-2")
    ck = hf_hub_download(params.get("hf_repo", "Lightricks/LTX-2.3"),
                         params.get("checkpoint", "ltx-2.3-22b-distilled.safetensors"))
    lora = hf_hub_download(params.get("lora_repo", "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt"),
                           params.get("lora_file", "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors"))
    ups = hf_hub_download(params.get("upsampler_repo", "Lightricks/LTX-2.3"),
                          params.get("upsampler_file",
                                     "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
    gemma = snapshot_download(params.get("gemma_repo",
                                         "google/gemma-3-12b-it-qat-q4_0-unquantized"))
    return {"repo": repo, "ckpt": ck, "lora": lora, "ups": ups, "gemma": gemma, "params": params,
            "lang": lang}


def run(state: dict, item: dict, out: Path) -> dict:
    text = (item.get("meta") or {}).get("text") or item["inputs"].get("text")
    if not text:
        raise Unsupported("DubIt is text-driven: the item has no meta.text for the spoken line")
    p = state["params"]
    video, _ = item_io(item)
    work = workdir(out)
    # Voice reference = the source clip's own speech (self-reenactment: the audio input).
    voice = item["inputs"].get("src_audio") or item["inputs"].get("audio")
    ref = mux(video, voice, work / "reference.mp4") if voice else video
    lang = LANG_NAMES.get(state["lang"], state["lang"])
    speaker = p.get("speaker", "The person")
    prompt = f'{speaker} is speaking {lang}, saying: "{text}"'
    produced = work / "dubit.mp4"
    cmd = [sys.executable, "-m", "ltx_pipelines.dubit", "--reference-video", ref, "--prompt",
           prompt, "--output-path", produced, "--spatial-upsampler-path", state["ups"],
           "--lora", state["lora"], str(p.get("lora_strength", 1.0)),
           "--distilled-checkpoint-path", state["ckpt"], "--gemma-root", state["gemma"],
           "--reference-strength", str(p.get("reference_strength", 1.0)),
           "--seed", str(p.get("seed", 0)), "--offload-mode", p.get("offload_mode", "none")]
    if p.get("quantization") and p["quantization"] != "none":
        cmd += ["--quantization", p["quantization"]]
    sh(cmd, cwd=state["repo"])
    gen_audio = to_wav(produced, out.with_suffix(".gen.wav"), sr=16000)
    payload = video_payload(out, produced, gen_audio, remux=False, generated_audio=True,
                            prompt=prompt)
    cleanup(work)
    return payload


def describe(state: dict) -> dict:
    p = state["params"]
    return {"model": "LTX-2.3-22b-IC-LoRA-DubIt", "checkpoint": Path(state["ckpt"]).name,
            "quantization": p.get("quantization", "none")}


if __name__ == "__main__":
    serve(load, run, describe)
