# Models covered by the benchmark

Generated 2026-09-28 by `python -m bench arena docs` from `server/bench/candidates/*.yaml` — edit the YAML, not this file.

**772 candidates** across 41 capabilities: 479 local models, 203 hosted APIs, 90 methods (compositions of other models); 249 registered but disabled (no released weights, gated, deferred, or a product without an API — the notes say which).

How to read the columns:

- **Kind** — `local` runs on your GPU/CPU in its own environment; `api` is a hosted service (needs a key, costs money per item); `method` composes other candidates.
- **Needs** — what the model requires to run (`requires` in the YAML): peak VRAM, CPU architecture, API key, tools.
- **Thalassa / Spark** — whether the hardware can run it: Thalassa = RTX 5060 Ti 16 GB, x86_64, 24 GB RAM; DGX Spark = GB10, aarch64, ~112 GB unified memory. API keys are not counted as hardware.
- **Licence** — `NC` = non-commercial weights (allowed as a default for now, by decision of 2026-09-27).
- **Smoke (thalassa)** — load/unload smoke test on this machine (`python -m bench arena smoke`): ✓ = loaded, produced an output, and gave all GPU memory back when it exited.
- **Get it** — model weights page, repository, or API documentation.
- 🔒 — uses a **gated Hugging Face repo**: needs `hf auth login` plus a one-time “Agree” on the repo page (see [Hugging Face access](#hugging-face-access)); 🔓 = gated but your token already has access.

## Summary

| Capability | Candidates | Local | API | Method | Disabled | Fit Thalassa | Fit Spark | Smoke ✓ (thalassa) |
|---|---|---|---|---|---|---|---|---|
| [A1](#a1) Dialogue / music-and-effects separation | 19 | 10 | 5 | 4 | 6 | 12 | 13 | 0 |
| [A2](#a2) Reference-clip enhancement | 19 | 13 | 4 | 2 | 4 | 14 | 13 | 0 |
| [A3](#a3) Speech / music / singing regions | 19 | 13 | 3 | 3 | 6 | 10 | 12 | 0 |
| [A4](#a4) Transcription with word-level timing | 54 | 36 | 17 | 1 | 8 | 43 | 43 | 2 |
| [A5](#a5) Forced alignment of known text | 14 | 7 | 3 | 4 | 7 | 7 | 5 | 0 |
| [A6](#a6) Diarization with overlap | 28 | 12 | 12 | 4 | 6 | 21 | 19 | 0 |
| [A7](#a7) Speaker embeddings / voice-bank matching | 18 | 15 | 2 | 1 | 5 | 13 | 13 | 0 |
| [A8](#a8) Paralinguistic / delivery tagging | 18 | 8 | 7 | 3 | 5 | 11 | 13 | 0 |
| [B1](#b1) Active speaker detection and face tracking | 16 | 10 | 4 | 2 | 6 | 10 | 10 | 0 |
| [B2](#b2) Shot boundary detection | 15 | 9 | 3 | 3 | 3 | 12 | 12 | 0 |
| [B3](#b3) On-screen text reading | 22 | 14 | 6 | 2 | 5 | 15 | 17 | 0 |
| [B4](#b4) Content type classification (live action / 2D / 3D / mixed) | 19 | 12 | 4 | 3 | 4 | 14 | 15 | 0 |
| [C1](#c1) Dub-line segmentation | 15 | 9 | 4 | 2 | 6 | 8 | 9 | 0 |
| [C2](#c2) Translation and dubbing adaptation | 38 | 24 | 12 | 2 | 9 | 18 | 26 | 0 |
| [C3](#c3) Viseme-aware paraphrase (experimental) | 12 | 1 | 5 | 6 | 5 | 6 | 7 | 0 |
| [C4](#c4) Reference-free MT quality estimation (meta-evaluation) | 22 | 15 | 6 | 1 | 5 | 12 | 17 | 0 |
| [C5](#c5) Text normalization, G2P and lexicon | 19 | 13 | 5 | 1 | 7 | 10 | 10 | 0 |
| [C6](#c6) Subtitle condensation | 15 | 7 | 5 | 3 | 6 | 6 | 9 | 0 |
| [D1](#d1) Zero-shot cross-lingual voice cloning | 28 | 18 | 8 | 2 | 4 | 19 | 22 | 0 |
| [D2](#d2) Expressive and style-transfer synthesis | 17 | 12 | 4 | 1 | 3 | 11 | 12 | 0 |
| [D3](#d3) Duration-controlled synthesis | 21 | 13 | 3 | 5 | 7 | 13 | 14 | 0 |
| [D4](#d4) Regional and low-resource language voices | 14 | 8 | 5 | 1 | 2 | 9 | 12 | 0 |
| [D5](#d5) Licensed non-cloning voices | 22 | 12 | 10 | 0 | 7 | 15 | 15 | 0 |
| [D6](#d6) Voice conversion | 14 | 10 | 3 | 1 | 3 | 10 | 11 | 0 |
| [D7](#d7) Non-verbal vocalizations | 15 | 9 | 3 | 3 | 5 | 8 | 9 | 0 |
| [D8](#d8) Direct speech-to-speech translation (evaluate only) | 10 | 7 | 2 | 1 | 7 | 2 | 3 | 0 |
| [E1](#e1) Timing fit + pause mapping | 16 | 9 | 2 | 5 | 9 | 5 | 5 | 0 |
| [E2](#e2) Bandwidth extension / 48 kHz restoration | 17 | 14 | 3 | 0 | 7 | 10 | 10 | 0 |
| [E3](#e3) Acoustic scene / room matching | 14 | 14 | 0 | 0 | 11 | 3 | 2 | 0 |
| [E4](#e4) Dialogue levelling + delivery loudness | 15 | 12 | 2 | 1 | 9 | 6 | 6 | 0 |
| [E5](#e5) Watermarking + provenance (deferred) | 12 | 7 | 3 | 2 | 12 | 0 | 0 | 0 |
| [F1](#f1) Live-action lip sync | 28 | 15 | 13 | 0 | 7 | 15 | 21 | 0 |
| [F2](#f2) Mouth-region restoration | 18 | 12 | 2 | 4 | 7 | 10 | 10 | 0 |
| [F3](#f3) 2D animation mouth retiming | 13 | 8 | 3 | 2 | 9 | 4 | 4 | 0 |
| [F4](#f4) On-screen text replacement | 23 | 14 | 7 | 2 | 6 | 14 | 16 | 0 |
| [G1](#g1) Intelligibility judge (ASR round-trip) — detection of defective takes | 18 | 13 | 3 | 2 | 3 | 14 | 15 | 10 |
| [G2](#g2) Speaker-similarity judge — agreement with human similarity ratings | 16 | 11 | 3 | 2 | 7 | 9 | 9 | 5 |
| [G3](#g3) Naturalness (MOS) predictor — agreement with human MOS | 17 | 12 | 3 | 2 | 5 | 12 | 12 | 2 |
| [G4](#g4) Emotion-consistency judge — agreement with human emotion labels | 13 | 8 | 3 | 2 | 5 | 6 | 8 | 0 |
| [G5](#g5) Lip-sync scorer — response to injected A/V offsets | 11 | 7 | 2 | 2 | 5 | 6 | 5 | 0 |
| [G6](#g6) Multimodal reviewer — per-defect detection on injected defects | 18 | 6 | 9 | 3 | 6 | 9 | 12 | 0 |

## Hugging Face access

18 gated Hugging Face repos are used by enabled candidates; **14 still need you to click “Agree”** on the repo page while logged in as the token's account (a one-time licence acceptance; there is no API for it). Log in once with `server/.venv/bin/hf auth login`; every worker env reads the saved token. Re-check with `python -m bench arena hf-access`, then regenerate these docs.

| ☐ | Repo | Your access | Needed by | Thalassa | Spark | Gate |
|---|---|---|---|---|---|---|
| ☐ | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | **click Agree** | A4 cohere-transcribe-2b | ✓ | ✓ | auto |
| ☐ | [Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) | **click Agree** | C4 xcomet-xl-qe | ✓ | ✓ | auto |
| ☐ | [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) | **click Agree** | C4 cometkiwi-da-22 | ✓ | ✓ | auto |
| ☐ | [Unbabel/wmt23-cometkiwi-da-xl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) | **click Agree** | C4 cometkiwi-da-xl | ✓ | ✓ | auto |
| ☐ | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | **click Agree** | D1 indicf5, D4 indicf5 | ✓ | ✓ | auto |
| ☐ | [pyannote/embedding](https://huggingface.co/pyannote/embedding) | **click Agree** | A6 pyannote-community-1, A6 pyannote-community-1-exclusive | ✓ | ✓ | auto |
| ☐ | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | **click Agree** | A6 pyannote-3.1, A6 pyannote-community-1, A6 pyannote-community-1-exclusive | ✓ | ✓ | auto |
| ☐ | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | **click Agree** | A6 pyannote-3.1 | ✓ | ✓ | auto |
| ☐ | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | **click Agree** | A6 pyannote-community-1, A6 pyannote-community-1-exclusive | ✓ | ✓ | auto |
| ☐ | [Lightricks/LTX-2.3-22b-IC-LoRA-DubIt](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt) | **click Agree** | F1 ltx-2.3-dubit-bf16, F1 ltx-2.3-dubit-fp8, F1 ltx-2.3-dubit-nvfp4 | — | ✓ | auto |
| ☐ | [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) | **click Agree** | C4 xcomet-xxl-qe | — | ✓ | auto |
| ☐ | [Unbabel/wmt23-cometkiwi-da-xxl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) | **click Agree** | C4 cometkiwi-da-xxl | — | ✓ | auto |
| ☐ | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | **click Agree** | F4 flux2-dev-32b | — | ✓ | auto |
| ☐ | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | **click Agree** | F4 flux2-klein-9b | — | ✓ | auto |
| ☑ | [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) | ✓ granted | C2 translategemma-4b | ✓ | ✓ | manual |
| ☑ | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) | ✓ granted | F1 ltx-2.3-dubit-bf16, F1 ltx-2.3-dubit-fp8, F1 ltx-2.3-dubit-nvfp4 | — | ✓ | manual |
| ☑ | [google/translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it) | ✓ granted | C2 translategemma-12b | — | ✓ | manual |
| ☑ | [google/translategemma-27b-it](https://huggingface.co/google/translategemma-27b-it) | ✓ granted | C2 translategemma-27b | — | ✓ | manual |

17 referenced ids are not Hugging Face repos (GitHub / torch.hub / ModelScope ids or paths inside a cloned repo) and need no token.


## A — Source analysis · audio

### A1

**Dialogue / music-and-effects separation**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **htdemucs** (baseline) | local | 0.042 B | MIT | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/adefossez/demucs](https://github.com/adefossez/demucs) | `server` |
| **auk-flash** | local | 1.5 B | MIT | all (✓ en hi ja) | 26 GB VRAM | ✗ | ✓ | not run | [tencent/AuK-Flash](https://huggingface.co/tencent/AuK-Flash) | `a1_auk` |
| **bandit-plus-dnr** | local | 0.046 B | MIT repo; checkpoint licence not stated | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [github.com/ZFTurbo/Music-Source-Separation-Tr…](https://github.com/ZFTurbo/Music-Source-Separation-Training) | `a1_msst` |
| **bandit-v2-eng** | local | 0.046 B | Apache-2.0 code; CC-BY-SA-4.0 weights | en | 6 GB VRAM | ✓ | ✓ | not run | [zenodo.org/records/12701995](https://zenodo.org/records/12701995) | `a1_msst` |
| **bandit-v2-multi** | local | 0.046 B | Apache-2.0 code; CC-BY-SA-4.0 weights | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [github.com/kwatcharasupat/bandit-v2](https://github.com/kwatcharasupat/bandit-v2) | `a1_msst` |
| **htdemucs-ft** | local | 0.168 B | MIT | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [adefossez/HTDemucs-ft](https://huggingface.co/adefossez/HTDemucs-ft) | `server` |
| **melband-roformer-kim-vocals** | local | — | MIT | all (✓ en hi ja) | 5 GB VRAM | ✓ | ✓ | not run | [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) | `a1_msst` |
| **audioshake-dialogue** | api | — | proprietary API | all (✓ en hi ja) | key AUDIOSHAKE | ✓ | ✓ | not run | [developer.audioshake.ai/models](https://developer.audioshake.ai/models) | `server` |
| **centre-channel** | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| **lalal-voice** | api | — | proprietary API | all (✓ en hi ja) | key LALAL | ✓ | ✓ | not run | [lalal.ai/api/v1/openapi.json](https://www.lalal.ai/api/v1/openapi.json) | `server` |
| **mvsep-bandit-v2** | api | — | proprietary API (hosts CC-BY-SA BandIt v2) | all (✓ en hi ja) | key MVSEP_API_TOKEN | ✓ | ✓ | not run | [mvsep.com/full_api](https://mvsep.com/full_api) | `server` |
| **mvsep-dnr-v3-ensemble** | api | — | proprietary API | all (✓ en hi ja) | key MVSEP_API_TOKEN | ✓ | ✓ | not run | [mvsep.com/full_api](https://mvsep.com/full_api) | `server` |
| **passthrough** | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| ~~auk~~ (disabled) | local | 1.5 B | MIT | all (✓ en hi ja) | 26 GB VRAM | — | — | not run | [tencent/AuK](https://huggingface.co/tencent/AuK) | `a1_auk` |
| ~~av-cass~~ (disabled) | local | — | MIT code (weights terms not stated) | all (✓ en hi ja) | 16 GB VRAM, x86_64 | — | — | not run | [github.com/pantheon5100/AVCASS](https://github.com/pantheon5100/AVCASS) | `a1_msst` |
| ~~banquet~~ (disabled) | local | 0.025 B | MIT code; CC-BY-NC-SA-4.0 weights · NC | all (✓ en hi ja) | 4 GB VRAM | — | — | not run | [github.com/kwatcharasupat/query-bandit](https://github.com/kwatcharasupat/query-bandit) | `a1_msst` |
| ~~audioshake-multi-voice~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key AUDIOSHAKE | — | — | not run | [developer.audioshake.ai/models](https://developer.audioshake.ai/models) | `server` |
| ~~bandit-v2-roformer-blend~~ (disabled) | method | — | mixed | all (✓ en hi ja) | 8 GB VRAM | — | — | not run | [mvsep.com/algorithms](https://mvsep.com/algorithms) | `a1_msst` |
| ~~dnr-nonverbal~~ (disabled) | method | — | dataset (Zenodo) | all (✓ en hi ja) | CPU | — | — | not run | [zenodo.org/search?q=DnR-nonverbal](https://zenodo.org/search?q=DnR-nonverbal) | `a1_msst` |

<details><summary>Notes</summary>

- **htdemucs**: Current OpenDub separator (vocals stem = dialogue, bed = mix − dialogue). demucs 4.1 needs sphn, which has no aarch64 wheel (Rust build on the Spark).
- **htdemucs-ft**: bag of 4 fine-tuned HTDemucs models (4× slower than htdemucs)
- **passthrough**: OpenDub's separation.passthrough default (dialogue = bed = mix) — the floor
- **centre-channel**: Classical supplied-stem / 5.1 centre-channel extraction. Needs multichannel input: every mono/stereo pack item is Unsupported (DnR is mono) — scored only on 5.1 in-house items.
- **bandit-v2-multi**: Spark pick. Multilingual DnR v3 checkpoint (48 kHz, speech/music/sfx) run through MSST's bandit_v2 model type; that the Zenodo checkpoints load unmodified in MSST is unverified (native repo pins torch 2.0 + internal nflx packages). Share-alike weights.
- **bandit-v2-eng**: English-dialogue DnR v3 checkpoint (one of seven: eng/deu/fra/spa/cmn/fao/multi)
- **bandit-plus-dnr**: DnR v2 (English LibriSpeech dialogue) BandIt Plus, 44.1 kHz, DnR SDR 11.47
- **melband-roformer-kim-vocals**: Music vocal separator used as a dialogue extractor (the "MSST ensemble" alternate); like Demucs it will also pull singing into the dialogue stem.
- **bandit-v2-roformer-blend**: blended ensemble (demixing-challenge practice); needs a composition worker — add once the single models are ranked
- **auk-flash**: Tencent AuK-Flash (released 2026-09-09): instruction-driven generative separation, 24 kHz output; ~25 GiB bf16 on A800 (17 GiB with cpu_offload, still above 16 GB) → Spark. The bed is mix − regenerated dialogue, so any drift shows as bleed. Plus Qwen2.5-Omni-3B.
- **auk**: full (non-Flash) AuK — checkpoint file names not verified; set params.config/checkpoint and enable
- **av-cass**: Unconstrained pick (CVPR 2026 flow matching; +0.22 dB SI-SDRi over BandIt on DnR v3). Weights are on Google Drive, but inference expects the AVDnR dataset layout (+ CAVP/TalkNet for the AV model, mmcv/xformers) — needs its own adapter; audio-only mode fits A1 packs.
- **banquet**: query-based single-decoder separator trained on MoisesDB music stems — no dialogue query; research-only weights
- **audioshake-dialogue**: API pick (dialogue model + residual = bed; 1.5 credits/min). USD per credit is dashboard-only: set usd_per_credit to record spend. The asset-upload route is unverified.
- **audioshake-multi-voice**: separates overlapping voices (10 credits/min) — belongs upstream of A6, not a DME split
- **lalal-voice**: existing OpenDub provider (separation.lalalai); price from the 750 min / $50 top-up
- **mvsep-bandit-v2**: hosted BandIt v2 multi; credits per minute, USD not published (usd_per_min param)
- **mvsep-dnr-v3-ensemble**: MVSep DnR v3 (SCNet / Mel-RoFormer ensemble) hosted algorithm
- **dnr-nonverbal**: a dataset (non-verbal vocalisation augmentation), not a model — use it as an extra A1 pack source (a1-dnr-v3 source=<dir>)

</details>

### A2

**Reference-clip enhancement**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **auk-flash-enhance** | local | 1.5 B | MIT | all (✓ en hi ja) | 26 GB VRAM | ✗ | ✓ | not run | [github.com/Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK) | `a1_auk` |
| **clearvoice-frcrn-16k** | local | — | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [github.com/modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | `a2_clearvoice` |
| **clearvoice-mossformer2-48k** | local | — | Apache-2.0 | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [alibabasglab/MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K) | `a2_clearvoice` |
| **deepfilternet3** | local | 0.002 B | MIT OR Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | `a2_deepfilter` |
| **melband-roformer-denoise-aufr33** | local | — | not stated | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [github.com/ZFTurbo/Music-Source-Separation-Tr…](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v.1.0.7) | `a1_msst` |
| **melband-roformer-dereverb-anvuew** | local | — | GPL-3.0 · NC | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [anvuew/dereverb_mel_band_roformer](https://huggingface.co/anvuew/dereverb_mel_band_roformer) | `a1_msst` |
| **resemble-denoise** | local | — | MIT | all (✓ en hi ja) | 2 GB VRAM, x86_64 | ✓ | ✗ | not run | [github.com/resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | `a2_resemble` |
| **resemble-enhance** | local | — | MIT | all (✓ en hi ja) | 4 GB VRAM, x86_64 | ✓ | ✗ | not run | [github.com/resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | `a2_resemble` |
| **unipase-16k** | local | 0.55 B | MIT code; Apache-2.0 weights (HF card) | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [github.com/xiaobin-rong/unipase](https://github.com/xiaobin-rong/unipase) | `a2_unipase` |
| **unipase-48k** | local | 0.55 B | MIT code; Apache-2.0 weights (HF card) | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [Xiaobin-Rong/unipase](https://huggingface.co/Xiaobin-Rong/unipase) | `a2_unipase` |
| **identity** (baseline) | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| **audioshake-speech-denoise** | api | — | proprietary API | all (✓ en hi ja) | key AUDIOSHAKE | ✓ | ✓ | not run | [developer.audioshake.ai/models](https://developer.audioshake.ai/models) | `server` |
| **audioshake-speech-dereverb** | api | — | proprietary API | all (✓ en hi ja) | key AUDIOSHAKE | ✓ | ✓ | not run | [developer.audioshake.ai/models](https://developer.audioshake.ai/models) | `server` |
| **mvsep-denoise** | api | — | proprietary API | all (✓ en hi ja) | key MVSEP_API_TOKEN | ✓ | ✓ | not run | [mvsep.com/full_api](https://mvsep.com/full_api) | `server` |
| **resemble-api-enhance** | api | — | proprietary API | all (✓ en hi ja) | key RESEMBLE | ✓ | ✓ | not run | [docs.resemble.ai/audio-tools/audio-enhancement](https://docs.resemble.ai/audio-tools/audio-enhancement) | `server` |
| ~~anyenhance~~ (disabled) | local | 0.36 B | CC-BY-NC-ND-4.0 · NC | all (✓ en hi ja) | 4 GB VRAM | — | — | not run | [arxiv.org/abs/2501.15417](https://arxiv.org/abs/2501.15417) | `a2_unipase` |
| ~~miipher-2~~ (disabled) | local | — | unreleased (Google); unofficial port CC-BY-NC-4.0 · NC | all (✓ en hi ja) | 4 GB VRAM | — | — | not run | [Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1](https://huggingface.co/Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1) | `a2_unipase` |
| ~~usemamba~~ (disabled) | local | — | CC BY 4.0 (paper) | all (✓ en hi ja) | GPU | — | — | not run | [arxiv.org/abs/2505.21198](https://arxiv.org/abs/2505.21198) | `a2_unipase` |
| ~~best-of-n-rerank~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |

<details><summary>Notes</summary>

- **identity**: OpenDub's A2 default is off (raw reference clip); every enhancer must beat leaving it alone
- **unipase-16k**: Spark pick (URGENT 2026 winner, TASLP 2026). Generative (DeWavLM-Omni + adapter + vocoder): not sample-aligned, so SI-SDR undersells it. ~3.2 GB for 1 s input per the paper.
- **unipase-48k**: 16 kHz core + PostNet bandwidth extension to 48 kHz
- **resemble-enhance**: denoiser + CFM enhancer, 44.1 kHz; old torch/deepspeed pins → installed --no-deps, x86_64 only; VRAM estimated
- **resemble-denoise**: denoiser stage only (discriminative, cannot hallucinate timbre)
- **clearvoice-mossformer2-48k**: 48 kHz discriminative enhancement; VRAM estimated
- **deepfilternet3**: 48 kHz, real-time on CPU (aarch64 recipe is CPU torch)
- **melband-roformer-dereverb-anvuew**: full-band discriminative dereverb (44.1 kHz); GPL weights → ship_ok false until reviewed
- **auk-flash-enhance**: instruction "remove noise and reverberation"; generative, 24 kHz; Spark only
- **audioshake-speech-denoise**: API pick (voice isolation; 1.5 credits/min; USD per credit not public)
- **audioshake-speech-dereverb**: 2.0 credits/min
- **resemble-api-enhance**: $0.045 per started minute; the job GET path is inferred from the create docs
- **mvsep-denoise**: hosted aufr33 Mel-RoFormer denoise; output stem name fragment "dry" unverified
- **anyenhance**: Unconstrained pick, but no official weights (no Amphion release). The CCF-AATC 2025 AnyEnhance-v1 challenge baseline (MIT, 45.7M) is a different, smaller model.
- **miipher-2**: Google declined to release code/weights; only Atotti's mHuBERT-147 port (NC, hydra CLI) exists — adapter not written
- **usemamba**: no public code or checkpoints
- **best-of-n-rerank**: best-of-N over enhancers re-ranked by DNSMOS/NISQA/speaker similarity — a composition method, add after single enhancers are ranked

</details>

### A3

**Speech / music / singing regions**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **audio-flamingo-3-regions** | local | 8 B | NVIDIA non-commercial · NC | all (✓ en hi ja) | 20 GB VRAM | ✗ | ✓ | not run | [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf) | `a4_transformers` |
| **audio-flamingo-next-think-regions** | local | 8 B | NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 20 GB VRAM | ✗ | ✓ | not run | [nvidia/audio-flamingo-next-think-hf](https://huggingface.co/nvidia/audio-flamingo-next-think-hf) | `a4_transformers` |
| **ced-base** | local | 0.086 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [mispeech/ced-base](https://huggingface.co/mispeech/ced-base) | `a3_audiotag` |
| **ced-mini** | local | 0.0096 B | Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [mispeech/ced-mini](https://huggingface.co/mispeech/ced-mini) | `a3_audiotag` |
| **ced-small** | local | 0.022 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [mispeech/ced-small](https://huggingface.co/mispeech/ced-small) | `a3_audiotag` |
| **ced-tiny** | local | 0.0055 B | Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [mispeech/ced-tiny](https://huggingface.co/mispeech/ced-tiny) | `a3_audiotag` |
| **inaspeechsegmenter-smn** | local | — | MIT | all (✓ en hi ja) | x86_64 | ✓ | ✗ | not run | [github.com/ina-foss/inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter) | `a3_ina` |
| **qwen3-omni-30b-a3b-regions** | local | 30 B | Apache-2.0 | all (✓ en hi ja) | 80 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `a8_qwen_omni` |
| **whisper-at-large-v1** | local | 1.55 B | BSD (code); MIT (Whisper weights) | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [github.com/YuanGongND/whisper-at](https://github.com/YuanGongND/whisper-at) | `a3_whisperat` |
| **stem-energy** (baseline) | method | — | MIT (Demucs) | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| **gemini-3.8-flash-regions** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/audio](https://ai.google.dev/gemini-api/docs/audio) | `server` |
| **qwen3.5-omni-plus-regions** | api | — | proprietary API | all (✓ en hi ja) | key DASHSCOPE | ✓ | ✓ | not run | [alibabacloud.com/help/en/model-studio/qwen-omni](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) | `server` |
| **scribe-v2-events** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| ~~beats-iter3-as2m~~ (disabled) | local | 0.09 B | MIT (repo) | all (✓ en hi ja) | 1 GB VRAM | — | — | not run | [github.com/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats) | `a3_audiotag` |
| ~~dasheng-1.2b~~ (disabled) | local | 1.2 B | Apache-2.0 | all (✓ en hi ja) | 6 GB VRAM | — | — | not run | [github.com/XiaoMi/dasheng](https://github.com/XiaoMi/dasheng) | `a3_audiotag` |
| ~~openbeats-large~~ (disabled) | local | 0.3 B | see release | all (✓ en hi ja) | 2 GB VRAM | — | — | not run | [arxiv.org/abs/2507.14129](https://arxiv.org/abs/2507.14129) | `a3_audiotag` |
| ~~sr-sad~~ (disabled) | local | 0.00087 B | not stated | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2512.09713](https://arxiv.org/abs/2512.09713) | `a3_audiotag` |
| ~~audioshake-stem-energy~~ (disabled) | method | — | proprietary API | all (✓ en hi ja) | key AUDIOSHAKE | — | — | not run | [developer.audioshake.ai/models](https://developer.audioshake.ai/models) | `server` |
| ~~qwen3-asr-adjudicator~~ (disabled) | method | 1.7 B | Apache-2.0 | en hi ja | 5 GB VRAM | — | — | not run | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `qwen3asr` |

<details><summary>Notes</summary>

- **stem-energy**: cross-stem energy heuristic on the Demucs split (OpenDub's builtin A3 plan); cannot tell singing from speech
- **ced-base**: Spark pick; AudioSet 527 multi-label (Speech / Music / Singing groups), trust_remote_code
- **inaspeechsegmenter-smn**: speech/music backstop; labels singing as music (the failure this pack measures). TF + onnxruntime-gpu → x86_64
- **whisper-at-large-v1**: AudioSet tags from a Whisper encoder at 0.4 s resolution; label names parsed in English (language='en' — verify)
- **qwen3-omni-30b-a3b-regions**: Unconstrained pick (open weights) as a region labeller; the plan's adjudicator over dense CED posteriors is a composition to add later. ~79 GB bf16 for a 15 s clip → Spark.
- **qwen3.5-omni-plus-regions**: API-only Qwen3.5-Omni-Plus; price not verified (usd_per_call param)
- **audio-flamingo-3-regions**: 8B bf16 (~18–24 GB estimated) → Spark; eval-only licence
- **audio-flamingo-next-think-regions**: stronger audio reasoning (MMAU-Pro 58.7) but non-commercial; transformers class availability in 5.17 unverified
- **gemini-3.8-flash-regions**: API pick family (Gemini 3-class audio); audio price for 3.8 Flash not captured — set the params
- **scribe-v2-events**: speech spans from word timings + inline audio-event tags (music / singing when tagged)
- **dasheng-1.2b**: no packaged AudioSet head for 1.2B; the Zenodo 49.7-mAP tagging checkpoint needs a hand-written wrapper
- **beats-iter3-as2m**: AudioSet-finetuned checkpoints are OneDrive links only and there is no pip package — manual download + adapter
- **openbeats-large**: encoder release (ESPnet); no AudioSet tagging head packaged for inference
- **sr-sad**: singing-robust speech activity detection — no code or weights released
- **qwen3-asr-adjudicator**: singing-aware ASR as an adjudicator of ambiguous CED windows — a composition method, not a region detector on its own
- **audioshake-stem-energy**: stem-energy heuristic on AudioShake stems — needs the stem heuristic to take API stems; tiebreaker only

</details>

### A4

**Transcription with word-level timing**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **fw-large-v3-turbo** (baseline) | local | 0.81 B | MIT | all (✓ en hi ja) | 2.5 GB VRAM | ✓ | ✓ | ✓ 2.0 GB, 3s load | [mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | `server` |
| **canary-1b-flash** | local | 0.883 B | CC-BY-4.0 | en +3 | 4 GB VRAM | ✓ | ✓ | not run | [nvidia/canary-1b-flash](https://huggingface.co/nvidia/canary-1b-flash) | `a4_nemo` |
| **canary-1b-v2** | local | 0.978 B | CC-BY-4.0 | en +24 | 5 GB VRAM | ✓ | ✓ | not run | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | `a4_nemo` |
| **canary-qwen-2.5b** | local | 2.5 B | CC-BY-4.0 | en | 8 GB VRAM | ✓ | ✓ | not run | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | `a4_nemo` |
| **cohere-transcribe-2b** 🔒 | local | 2 B | Apache-2.0 | en ja +12 | 6 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | `a4_transformers` |
| **crisperwhisper** | local | 1.55 B | CC-BY-NC-4.0 · NC | en +1 | 6 GB VRAM | ✓ | ✓ | not run | [nyrahealth/CrisperWhisper](https://huggingface.co/nyrahealth/CrisperWhisper) | `a4_crisper` |
| **fw-distil-large-v3.5** | local | 0.756 B | MIT | en | 2 GB VRAM | ✓ | ✓ | not run | [distil-whisper/distil-large-v3.5-ct2](https://huggingface.co/distil-whisper/distil-large-v3.5-ct2) | `server` |
| **fw-large-v2** | local | 1.55 B | MIT | all (✓ en hi ja) | 4.5 GB VRAM | ✓ | ✓ | not run | [Systran/faster-whisper-large-v2](https://huggingface.co/Systran/faster-whisper-large-v2) | `server` |
| **fw-large-v3** | local | 1.55 B | MIT | all (✓ en hi ja) | 4.5 GB VRAM | ✓ | ✓ | not run | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | `server` |
| **fw-medium** | local | 0.77 B | MIT | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) | `server` |
| **granite-speech-4.1-2b** | local | 2 B | Apache-2.0 | en ja +4 | 7 GB VRAM | ✓ | ✓ | not run | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | `a4_transformers` |
| **granite-speech-4.1-2b-plus** | local | 2 B | Apache-2.0 | en +4 | 7 GB VRAM | ✓ | ✓ | not run | [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | `a4_transformers` |
| **hf-whisper-large-v3** | local | 1.55 B | Apache-2.0 | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | `a4_transformers` |
| **kotoba-whisper-v2** | local | 0.76 B | Apache-2.0 | ja | 2 GB VRAM | ✓ | ✓ | not run | [kotoba-tech/kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) | `server` |
| **kotoba-whisper-v2.2** | local | 0.756 B | Apache-2.0 | ja | 3 GB VRAM | ✓ | ✓ | not run | [kotoba-tech/kotoba-whisper-v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | `a4_transformers` |
| **mms-1b-all** | local | 1 B | CC-BY-NC-4.0 · NC | en hi ja +8 | 5 GB VRAM | ✓ | ✓ | not run | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | `a4_transformers` |
| **moss-transcribe-diarize** | local | 0.9 B | Apache-2.0 | en ja +1 | 8 GB VRAM | ✓ | ✓ | not run | [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | `a4_moss` |
| **nemotron-3.5-asr-streaming-0.6b** | local | 0.6 B | OpenMDW-1.1 | en hi ja | 3 GB VRAM | ✓ | ✓ | not run | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | `a4_nemo` |
| **omniasr-llm-1b-v2** | local | 1 B | Apache-2.0 | en hi ja +9 | 5 GB VRAM, x86_64 | ✓ | ✗ | not run | [github.com/facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) | `a4_omniasr` |
| **omniasr-llm-7b-v2** | local | 7 B | Apache-2.0 | en hi ja +9 | 17 GB VRAM, x86_64 | ✗ | ✗ | not run | [facebook/omniASR-LLM-7B](https://huggingface.co/facebook/omniASR-LLM-7B) | `a4_omniasr` |
| **parakeet-tdt-0.6b-v3** | local | 0.6 B | CC-BY-4.0 | en +24 | 3 GB VRAM | ✓ | ✓ | not run | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | `a4_nemo` |
| **parakeet-tdt-ctc-0.6b-ja** | local | 0.6 B | CC-BY-4.0 | ja | 3 GB VRAM | ✓ | ✓ | not run | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | `a4_nemo` |
| **qwen3-asr-0.6b** | local | 0.9 B | Apache-2.0 | en hi ja | 5 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `qwen3asr` |
| **qwen3-asr-1.7b** | local | 1.7 B | Apache-2.0 | en hi ja | 7 GB VRAM | ✓ | ✓ | ✓ 5.9 GB, 17s load | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `qwen3asr` |
| **qwen3-omni-30b-a3b-asr** | local | 30 B | Apache-2.0 | en ja +8 | 80 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `a8_qwen_omni` |
| **sensevoice-small** | local | 0.234 B | FunASR model licence | en ja +3 | 2 GB VRAM | ✓ | ✓ | not run | [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | `a4_funasr` |
| **vibevoice-asr** | local | 8 B | MIT | en ja +1 | 24 GB VRAM | ✗ | ✓ | not run | [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) | `a4_transformers` |
| **vibevoice-asr-int8** | local | 8 B | MIT | en ja +1 | 13 GB VRAM | ✓ | ✓ | not run | [Dubedo/VibeVoice-ASR-HF-INT8](https://huggingface.co/Dubedo/VibeVoice-ASR-HF-INT8) | `a4_transformers` |
| **voxtral-mini-3b** | local | 4.7 B | Apache-2.0 | en hi +6 | 10 GB VRAM | ✓ | ✓ | not run | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | `a4_transformers` |
| **whisper-hindi-large-v2** | local | 1.55 B | unknown | hi | 6 GB VRAM | ✓ | ✓ | not run | [vasista22/whisper-hindi-large-v2](https://huggingface.co/vasista22/whisper-hindi-large-v2) | `a4_transformers` |
| **whisper-large-v3-vaani-hindi** | local | 1.55 B | unknown | hi | 6 GB VRAM | ✓ | ✓ | not run | [ARTPARK-IISc/whisper-large-v3-vaani-hindi](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi) | `a4_transformers` |
| **whisperx-large-v3** | local | 1.55 B | BSD-2-Clause (code); MIT (Whisper) | all (✓ en hi ja) | 6 GB VRAM, x86_64 | ✓ | ✗ | not run | [github.com/m-bain/whisperX](https://github.com/m-bain/whisperX) | `a4_whisperx` |
| **assemblyai-universal-2** | api | — | proprietary API | all (✓ en hi ja) | key ASSEMBLYAI | ✓ | ✓ | not run | [assemblyai.com/docs/api-reference/transcripts…](https://www.assemblyai.com/docs/api-reference/transcripts/submit) | `server` |
| **assemblyai-universal-3-5-pro** | api | — | proprietary API | en hi ja +9 | key ASSEMBLYAI | ✓ | ✓ | not run | [assemblyai.com/docs/api-reference/transcripts…](https://www.assemblyai.com/docs/api-reference/transcripts/submit) | `server` |
| **azure-mai-transcribe-2** | api | — | proprietary API | all (✓ en hi ja) | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/azure/ai-services/speech-…](https://learn.microsoft.com/azure/ai-services/speech-service/fast-transcription-create) | `server` |
| **deepgram-nova-3** | api | — | proprietary API | all (✓ en hi ja) | key DEEPGRAM | ✓ | ✓ | not run | [developers.deepgram.com/docs/models-languages…](https://developers.deepgram.com/docs/models-languages-overview) | `server` |
| **gemini-3.5-transcribe** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/transcribe](https://ai.google.dev/gemini-api/docs/transcribe) | `server` |
| **mistral-voxtral-mini-transcribe** | api | — | proprietary API | en hi ja +10 | key MISTRAL | ✓ | ✓ | not run | [docs.mistral.ai/studio/audio/speech_to_text/o…](https://docs.mistral.ai/studio/audio/speech_to_text/offline_transcription) | `server` |
| **openai-gpt-4o-mini-transcribe** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/guides/speech-…](https://developers.openai.com/api/docs/guides/speech-to-text) | `server` |
| **openai-gpt-4o-transcribe** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/guides/speech-…](https://developers.openai.com/api/docs/guides/speech-to-text) | `server` |
| **openai-gpt-transcribe** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/guides/speech-…](https://developers.openai.com/api/docs/guides/speech-to-text) | `server` |
| **openai-whisper-1** | api | 1.55 B | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/guides/speech-…](https://developers.openai.com/api/docs/guides/speech-to-text) | `server` |
| **sarvam-saaras-v3** | api | — | proprietary API | en hi +9 | key SARVAM | ✓ | ✓ | not run | [docs.sarvam.ai/api-reference-docs/speech-to-t…](https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe) | `server` |
| **scribe-v1** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| **scribe-v2** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| **speechmatics-enhanced** | api | — | proprietary API | all (✓ en hi ja) | key SPEECHMATICS | ✓ | ✓ | not run | [docs.speechmatics.com](https://docs.speechmatics.com) | `server` |
| ~~ark-asr-3b~~ (disabled) | local | 3 B | Apache-2.0 | en ja +1 | 9 GB VRAM | — | — | not run | [AutoArk-AI/ARK-ASR-3B](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) | `a4_transformers` |
| ~~calm-whisper~~ (disabled) | local | — | not stated | all (✓ en hi ja) | GPU | — | — | not run | [arxiv.org/abs/2505.12969](https://arxiv.org/abs/2505.12969) | `server` |
| ~~crisperwhisper-2.0~~ (disabled) | local | — | unknown | all (✓ en hi ja) | 6 GB VRAM | — | — | not run | [nyralabs/CrisperWhisper2.0_large](https://huggingface.co/nyralabs/CrisperWhisper2.0_large) | `a4_crisper` |
| ~~indicconformer-600m~~ (disabled) | local | 0.6 B | MIT | hi | 3 GB VRAM, key HF_TOKEN | — | — | not run | [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | `indicconformer` |
| ~~aws-transcribe~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key AWS_ACCESS_ID | — | — | not run | [aws.amazon.com/transcribe/](https://aws.amazon.com/transcribe/) | `server` |
| ~~muse-voice-transcribe~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [dev.meta.ai/models/muse-voice-transcribe](https://dev.meta.ai/models/muse-voice-transcribe) | `server` |
| ~~rover-fusion~~ (disabled) | method | — | mixed | en ja | GPU | — | — | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `qwen3asr` |
| ~~stepaudio-3-asr-max~~ (disabled) | api | — | proprietary API | en ja +4 | CPU | — | — | not run | [platform.stepfun.ai/docs/en/guides/models/ste…](https://platform.stepfun.ai/docs/en/guides/models/stepaudio-3-asr) | `server` |

<details><summary>Notes</summary>

- **fw-large-v3-turbo**: current asr.faster_whisper default
- **kotoba-whisper-v2**: distil-whisper fine-tuned on ReazonSpeech (ja only)
- **qwen3-asr-1.7b**: qwen-asr transformers backend, language forced; word timestamps from Qwen3-ForcedAligner-0.6B for en/ja (the aligner does not cover hi, so hi outputs are text-only). Measured peak 6.7 GB with the aligner (en/ja), 4.7 GB without (hi)
- **indicconformer-600m**: Gated repo (accept the conditions on HF and set HF_TOKEN before running). The card lists 22 Indic languages; only hi is in our set. CTC greedy decoding.
- **fw-large-v2**: CTranslate2 GPU wheels are x86_64 only (aarch64 PyPI wheels are CPU-only)
- **fw-distil-large-v3.5**: English-only distil model; word timestamps off (distil alignment heads crash CTranslate2, as with kotoba)
- **hf-whisper-large-v3**: transformers Whisper with word timestamps — the Whisper route on the Spark, where CTranslate2 has no GPU build
- **whisperx-large-v3**: VAD-batched faster-whisper + wav2vec2 alignment; languages without an align model fail alignment
- **crisperwhisper**: verbatim (fillers kept) with retuned word timing; needs nyrahealth's transformers fork; repo now nyralabs/CrisperWhisper
- **crisperwhisper-2.0**: 2.0 "works across most Whisper languages" — licence and language list unconfirmed
- **calm-whisper**: hallucination-targeted decoder-head fine-tune — paper only, no released weights
- **qwen3-asr-0.6b**: 30 languages per the card (the core worker's language map covers 11); word timestamps from the aligner except hi. Revision not pinned yet.
- **parakeet-tdt-0.6b-v3**: 25 European languages, no hi/ja (runs only on en in our set); no language switch (auto)
- **parakeet-tdt-ctc-0.6b-ja**: Japanese Parakeet (ReazonSpeech v2, 35k h); NeMo word timestamps not documented on the card
- **canary-1b-v2**: language forced (source_lang = target_lang)
- **canary-1b-flash**: word timestamps marked experimental
- **canary-qwen-2.5b**: Open ASR Leaderboard leader (English only), no timestamps; VRAM estimated
- **nemotron-3.5-asr-streaming-0.6b**: 2026-06 streaming FastConformer-RNNT; 19 "transcription-ready" languages incl. hi and ja (only those in our set listed here). Language forcing and timestamps not documented.
- **vibevoice-asr**: 50+ languages claimed (hi not confirmed, so not declared); no language switch — the language goes in as prompt context. Segment-level times only. VRAM estimated → Spark.
- **vibevoice-asr-int8**: community INT8 (LLM part only) — the plan's Thalassa variant; needs bitsandbytes (not in the env yet), unverified
- **moss-transcribe-diarize**: 50+ languages claimed; sources disagree on Hindi (Urdu is listed), so hi is not declared — add it after checking. No language switch; segment-level times. VRAM estimated.
- **granite-speech-4.1-2b**: no language switch (prompted transcription); VRAM estimated
- **granite-speech-4.1-2b-plus**: word end-times via [T:N] tags (start = previous end)
- **cohere-transcribe-2b**: gated repo; no timestamps
- **ark-asr-3b**: 19 languages incl. ja (no hi); trust_remote_code inference API not verified — add a family to a4_hf_asr first
- **voxtral-mini-3b**: language forced via apply_transcription_request; no timestamps (G1 panel member in the plan)
- **whisper-hindi-large-v2**: IITM / Bhashini Hindi fine-tune
- **kotoba-whisper-v2.2**: plain Whisper pipeline (the repo's custom pyannote-diarization pipeline is not used)
- **sensevoice-small**: non-autoregressive, very fast; no word timestamps
- **mms-1b-all**: 1,162 languages via adapters (only the mapped ISO codes listed); CTC with no LM — the plan's "does not repair misreadings" judge family
- **omniasr-llm-1b-v2**: clips < 40 s; language codes hin_Deva / jpn_Jpan assumed
- **omniasr-llm-7b-v2**: ~17 GB → neither Thalassa (16 GB) nor the Spark (x86-only fairseq2) today; kept for a bigger x86 box
- **qwen3-omni-30b-a3b-asr**: 19 speech-input languages (no Hindi; subset listed); prompted transcription; Spark only
- **rover-fusion**: unconstrained pick (ROVER of Qwen3-ASR + Canary-Qwen + MOSS, LLM-arbitrated, re-aligned) — composition, add after the parts are ranked
- **scribe-v2**: API pick (90+ languages; hi/ja in the "excellent" tier); $0.22/h read 2026-09-28 — lower than expected, re-check
- **scribe-v1**: previous Scribe generation; price param not re-read
- **openai-gpt-transcribe**: released 2026-07-28, $0.0045/min; language as a languages[] hint; no timestamps
- **openai-gpt-4o-transcribe**: $0.006/min; no timestamps
- **openai-gpt-4o-mini-transcribe**: $0.003/min; no timestamps
- **openai-whisper-1**: hosted Whisper large-v2; the only OpenAI model with word timestamps; $0.006/min
- **gemini-3.5-transcribe**: launched 2026-08-26; 85+ locales incl. hi-IN / ja-JP; word timestamps; 30 min cap with timestamps
- **assemblyai-universal-3-5-pro**: $0.21/h; 18 languages (list abbreviated); words in ms
- **assemblyai-universal-2**: $0.15/h; 99 languages
- **deepgram-nova-3**: hi and ja covered by nova-3; price from the official page (third-party quotes lower)
- **speechmatics-enhanced**: batch jobs API; hi/ja support and price unverified
- **azure-mai-transcribe-2**: Microsoft MAI-Transcribe-2 (2026-09-03; 2.0 % AA-WER) through Azure fast transcription; $0.10/h launch price to end of 2026
- **mistral-voxtral-mini-transcribe**: word timestamps cannot be combined with a forced language, so text only
- **sarvam-saaras-v3**: Indian languages only; REST clips < 30 s; ₹30/h
- **stepaudio-3-asr-max**: 2026-09 accuracy leader (1.7 % WER); HTTP+SSE endpoint, auth details not fetched — worker not written; no Hindi
- **muse-voice-transcribe**: Meta (2026-09-01, $0.18/h) — endpoint and word timestamps not documented; weights not released
- **aws-transcribe**: batch jobs need S3 + boto3 — worker not written

</details>

### A5

**Forced alignment of known text**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **mfa-3.4** | local | — | MIT code; CC BY 4.0 models | en ja | 4 GB RAM, x86_64 | ✓ | ✗ | not run | [montreal-forced-aligner.readthedocs.io](https://montreal-forced-aligner.readthedocs.io) | `a5_mfa` |
| **mms-ctc-forced-aligner** | local | 0.3 B | BSD code; default checkpoint CC-BY-NC-4.0 · NC | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [github.com/MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | `a5_ctc` |
| **qwen3-forced-aligner-0.6b** | local | 0.6 B | Apache-2.0 | en ja +9 | 3 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | `qwen3asr` |
| **whisperx-wav2vec2-align** | local | — | BSD-2 (code); align models per checkpoint | en hi ja +8 | 2 GB VRAM, x86_64 | ✓ | ✗ | not run | [github.com/m-bain/whisperX](https://github.com/m-bain/whisperX) | `a4_whisperx` |
| **asr-words-fw-turbo** (baseline) | method | 0.81 B | MIT | all (✓ en hi ja) | 2.5 GB VRAM | ✓ | ✓ | not run | [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `server` |
| **elevenlabs-forced-alignment** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/forced-align…](https://elevenlabs.io/docs/api-reference/forced-alignment/create) | `server` |
| **proportional** | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| ~~bfa~~ (disabled) | local | — | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/search/?query=Bournemouth+Forced+Al…](https://arxiv.org/search/?query=Bournemouth+Forced+Aligner) | `a5_ctc` |
| ~~charsiu~~ (disabled) | local | 0.095 B | unknown | en +1 | 1 GB VRAM | — | — | not run | [github.com/lingjzhu/charsiu](https://github.com/lingjzhu/charsiu) | `a5_ctc` |
| ~~mwa~~ (disabled) | local | — | CC BY 4.0 (paper) | all (✓ en hi ja) | GPU | — | — | not run | [arxiv.org/search/?query=Multilingual+Word+Ali…](https://arxiv.org/search/?query=Multilingual+Word+Aligner+Keshet) | `a5_ctc` |
| ~~gemini-3.5-transcribe-words~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | — | — | not run | [ai.google.dev/gemini-api/docs/transcribe](https://ai.google.dev/gemini-api/docs/transcribe) | `server` |
| ~~gradient-saliency-align~~ (disabled) | method | — | CC BY-SA 4.0 (paper) | all (✓ en hi ja) | GPU | — | — | not run | [arxiv.org/search/?query=Zeyer+Schl%C3%BCter+N…](https://arxiv.org/search/?query=Zeyer+Schl%C3%BCter+Ney+saliency+alignment) | `server` |
| ~~scribe-v2-words~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | — | — | not run | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| ~~three-way-ensemble~~ (disabled) | method | — | mixed | en ja | CPU | — | — | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `a5_mfa` |

<details><summary>Notes</summary>

- **asr-words-fw-turbo**: what OpenDub does today — A4 faster-whisper word timings projected onto the given text (no aligner)
- **proportional**: builtin proportional word interpolation over the voiced span — the floor
- **mfa-3.4**: Spark pick (CPU, Kaldi). english_mfa / japanese_mfa pretrained models; no hindi_mfa exists (a user-trained or IndicMFA model can be set via params.dictionary/acoustic_model). No linux-aarch64 conda build → x86_64 (Docker image is the Spark route). One align_one per item.
- **qwen3-forced-aligner-0.6b**: co-primary on the Spark (27.8 ms AAS); no Hindi; clips ≤ 5 min
- **mms-ctc-forced-aligner**: MMS-300m CTC aligner via torchaudio forced_align (uroman romanisation for hi/ja). The default checkpoint is NC — the torchaudio MMS_FA bundle (deprecated) is the commercial route.
- **whisperx-wav2vec2-align**: WhisperX's per-language wav2vec2 aligners (hi theainerd/Wav2Vec2-large-xlsr-hindi, ja jonatasgrosman xlsr-53) on the given text
- **elevenlabs-forced-alignment**: API pick; 29 languages incl. en/hi/ja; billed at the STT rate (character timings used for ja)
- **gradient-saliency-align**: RWTH/AppTek 2026 gradient-saliency alignment over a loaded ASR — research code, no packaged release
- **charsiu**: phone-level aligner (en/zh checkpoints); no hi/ja models; licence unverified
- **mwa**: code and checkpoints promised on acceptance — not released
- **bfa**: fast phoneme-CTC aligner that trades boundary precision for speed; adapter not written
- **three-way-ensemble**: unconstrained pick (MFA + Qwen3-ForcedAligner + MMS_FA with per-word agreement) — a composition across three envs, add after the parts are ranked
- **gemini-3.5-transcribe-words**: transcribes instead of aligning the approved script — ranked in A4
- **scribe-v2-words**: word timestamps from transcription — ranked in A4

</details>

### A6

**Diarization with overlap**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **pyannote-3.1** (baseline) 🔒 | local | — | MIT | all (✓ en hi ja) | 2 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | `a6_pyannote3` |
| **diarizen-base-s80** | local | — | MIT code; CC-BY-NC-4.0 weights · NC | all (✓ en hi ja) | 3 GB VRAM, x86_64 | ✓ | ✗ | not run | [BUT-FIT/diarizen-wavlm-base-s80-md](https://huggingface.co/BUT-FIT/diarizen-wavlm-base-s80-md) | `a6_diarizen` |
| **diarizen-large-s80-v2** | local | 0.063 B | MIT code; CC-BY-NC-4.0 weights · NC | all (✓ en hi ja) | 4 GB VRAM, x86_64 | ✓ | ✗ | not run | [BUT-FIT/diarizen-wavlm-large-s80-md-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) | `a6_diarizen` |
| **moss-transcribe-diarize** | local | 0.9 B | Apache-2.0 | all (✓ en hi ja) | 8 GB VRAM | ✓ | ✓ | not run | [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | `a4_moss` |
| **picovoice-falcon** | local | — | proprietary engine (Apache-2.0 SDK) | all (✓ en hi ja) | x86_64, key PICOVOICE_ACCESS | ✓ | ✗ | not run | [github.com/Picovoice/falcon](https://github.com/Picovoice/falcon) | `a6_falcon` |
| **pyannote-community-1** 🔒 | local | — | CC-BY-4.0 weights; MIT code | all (✓ en hi ja) | 2 GB VRAM, key HF_TOKEN, +ffmpeg | ✓ | ✓ | not run | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | `a6_pyannote4` |
| **pyannote-community-1-exclusive** 🔒 | local | — | CC-BY-4.0 weights; MIT code | all (✓ en hi ja) | 2 GB VRAM, key HF_TOKEN, +ffmpeg | ✓ | ✓ | not run | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | `a6_pyannote4` |
| **sortformer-4spk-v1** | local | 0.123 B | CC-BY-NC-4.0 · NC | all (✓ en hi ja) | 14 GB VRAM | ✓ | ✓ | not run | [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | `a4_nemo` |
| **sortformer-streaming-4spk-v2** | local | 0.117 B | CC-BY-4.0 | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | `a4_nemo` |
| **sortformer-streaming-4spk-v2.1** | local | 0.117 B | NVIDIA Open Model License | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | `a4_nemo` |
| **vibevoice-asr** | local | 8 B | MIT | all (✓ en hi ja) | 24 GB VRAM | ✗ | ✓ | not run | [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) | `a4_transformers` |
| **assemblyai-universal-3-5-pro-diarize** | api | — | proprietary API | en hi ja +9 | key ASSEMBLYAI | ✓ | ✓ | not run | [assemblyai.com/docs/speaker-diarization](https://www.assemblyai.com/docs/speaker-diarization) | `server` |
| **azure-mai-transcribe-2-diarize** | api | — | proprietary API | all (✓ en hi ja) | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/azure/ai-services/speech-…](https://learn.microsoft.com/azure/ai-services/speech-service/fast-transcription-create) | `server` |
| **deepgram-nova-3-diarize** | api | — | proprietary API | all (✓ en hi ja) | key DEEPGRAM | ✓ | ✓ | not run | [developers.deepgram.com/docs/diarization](https://developers.deepgram.com/docs/diarization) | `server` |
| **gemini-3.5-transcribe-diarize** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/transcribe](https://ai.google.dev/gemini-api/docs/transcribe) | `server` |
| **openai-gpt-4o-transcribe-diarize** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/guides/speech-…](https://developers.openai.com/api/docs/guides/speech-to-text) | `server` |
| **pyannoteai-community-1** | api | — | proprietary API (hosted CC-BY-4.0 model) | all (✓ en hi ja) | key PYANNOTEAI | ✓ | ✓ | not run | [docs.pyannote.ai/api-reference/diarize](https://docs.pyannote.ai/api-reference/diarize) | `server` |
| **pyannoteai-precision-2** | api | — | proprietary API | all (✓ en hi ja) | key PYANNOTEAI | ✓ | ✓ | not run | [docs.pyannote.ai/api-reference/diarize](https://docs.pyannote.ai/api-reference/diarize) | `server` |
| **pyannoteai-precision-3** | api | — | proprietary API | all (✓ en hi ja) | key PYANNOTEAI | ✓ | ✓ | not run | [docs.pyannote.ai/api-reference/diarize](https://docs.pyannote.ai/api-reference/diarize) | `server` |
| **scribe-v2-diarize** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| **single-speaker** | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| **speechmatics-enhanced-diarize** | api | — | proprietary API | all (✓ en hi ja) | key SPEECHMATICS | ✓ | ✓ | not run | [docs.speechmatics.com/speech-to-text/features…](https://docs.speechmatics.com/speech-to-text/features/diarization) | `server` |
| ~~dicow~~ (disabled) | local | 1 B | Apache-2.0 (SE-DiCoW) | all (✓ en hi ja) | 6 GB VRAM | — | — | not run | [BUT-FIT/SE-DiCoW](https://huggingface.co/BUT-FIT/SE-DiCoW) | `a4_transformers` |
| ~~aws-transcribe-diarize~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key AWS_ACCESS_ID | — | — | not run | [aws.amazon.com/transcribe/pricing/](https://aws.amazon.com/transcribe/pricing/) | `server` |
| ~~diarizen-domain-finetune~~ (disabled) | method | — | CC-BY-NC-4.0 · NC | all (✓ en hi ja) | GPU | — | — | not run | [github.com/BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen) | `a6_diarizen` |
| ~~dover-lap-fusion~~ (disabled) | method | — | mixed | all (✓ en hi ja) | GPU | — | — | not run | [github.com/desh2608/dover-lap](https://github.com/desh2608/dover-lap) | `a6_pyannote4` |
| ~~muse-voice-transcribe~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [dev.meta.ai/models/muse-voice-transcribe](https://dev.meta.ai/models/muse-voice-transcribe) | `server` |
| ~~target-speaker-extraction~~ (disabled) | method | — | varies | all (✓ en hi ja) | GPU | — | — | not run | [arxiv.org/search/?query=REAL-TSE+2026](https://arxiv.org/search/?query=REAL-TSE+2026) | `a6_diarizen` |

<details><summary>Notes</summary>

- **pyannote-3.1**: OpenDub's diarization.pyannote provider; gated repo (accept terms, HF_TOKEN). Card DER VoxConverse 11.3
- **single-speaker**: OpenDub's default diarization.single_speaker (energy VAD, one speaker) — the DER floor
- **pyannote-community-1**: pyannote.audio 4 (torchcodec + system ffmpeg); torchcodec 0.16 now has aarch64 wheels
- **pyannote-community-1-exclusive**: exclusive (no-overlap) view — what word-level attribution consumes; overlap DER shows the cost
- **diarizen-large-s80-v2**: Spark pick (AMI 13.9 / VoxConverse 9.1 DER, collar 0). Repo pins torch 2.1.1+cu121 → x86_64 recipe; on the Spark relax the pin (then drop the arch limit).
- **sortformer-4spk-v1**: offline end-to-end, max 4 speakers; memory grows with length (~12 min max on a 48 GB A6000 per the card) — long recordings will OOM on 16 GB
- **sortformer-streaming-4spk-v2**: streaming (bounded memory) with the card's chunk / speaker-cache settings; max 4 speakers
- **sortformer-streaming-4spk-v2.1**: DIHARD3 ≤4-spk 15.09 / CALLHOME-2 6.65 on the card; licence terms not reviewed
- **moss-transcribe-diarize**: joint ASR + diarization over up to 90 min; attention memory grows quadratically in eager mode (VRAM estimated for short files)
- **vibevoice-asr**: joint ASR + speaker segments (60 min context); VRAM estimated (bf16 weights + KV) → Spark
- **pyannoteai-precision-3**: API pick; €0.112/h (recorded at ≈ $0.13/h)
- **pyannoteai-precision-2**: prior primary; also OpenDub's diarization.pyannote_api provider; price not listed separately
- **pyannoteai-community-1**: hosted community-1 (€0.035/h) — isolates hosting from model differences
- **scribe-v2-diarize**: word-level speaker ids merged into turns (up to 32 speakers)
- **assemblyai-universal-3-5-pro-diarize**: utterance-level speakers; $0.21/h + $0.02/h diarization; the 18-language list is abbreviated here
- **speechmatics-enhanced-diarize**: price unclear on the pricing page (param)
- **openai-gpt-4o-transcribe-diarize**: diarized_json segments; $0.006/min
- **gemini-3.5-transcribe-diarize**: the diarization flag name in transcription_config is not verified
- **picovoice-falcon**: on-device CPU engine with an online AccessKey check; enterprise pricing unpublished
- **aws-transcribe-diarize**: StartTranscriptionJob needs an S3 bucket + boto3 — worker not written; price conflicting ($0.006 vs $0.024/min)
- **muse-voice-transcribe**: Meta Muse Voice Transcribe (2026-09-01, $0.18/h, 20+ speakers) — endpoint and auth not documented on the model page
- **dicow**: diarization-conditioned ASR (takes diarization as input) — belongs to a cascaded A4/A6 evaluation, not a diarizer
- **target-speaker-extraction**: enrolment-based TSE on overlapped regions (REAL-TSE / USEF-TSE) — an overlap branch after diarization, not a diarizer
- **dover-lap-fusion**: unconstrained pick (DiariZen + community-1 + Sortformer fused) — composition across envs, add after the parts are ranked
- **diarizen-domain-finetune**: per-domain fine-tuning (16.4 → 12.7 % DER out of domain) — a training job, out of the arena's scope

</details>

### A7

**Speaker embeddings / voice-bank matching**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **ecapa-speechbrain** (baseline) | local | 0.021 B | Apache-2.0 code; VoxCeleb-derived weights (CC BY 4.0) | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | `a7_speechbrain` |
| **campplus-voxceleb** | local | 0.0072 B | CC BY 4.0 (VoxCeleb-derived) | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [modelscope.cn/models/iic/speech_campplus_sv_e…](https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k) | `a7_3dspeaker` |
| **eres2netv2-zh-common** | local | 0.0178 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [modelscope.cn/models/iic/speech_eres2netv2_sv…](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common) | `a7_3dspeaker` |
| **pyannote-wespeaker-resnet34-lm** | local | 0.0063 B | CC-BY-4.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) | `a6_pyannote3` |
| **redimnet-b6-v1** | local | 0.015 B | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [github.com/IDRnD/ReDimNet](https://github.com/IDRnD/ReDimNet) | `a7_redimnet` |
| **redimnet2-b3-lm** | local | 0.003 B | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [github.com/PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) | `a7_redimnet` |
| **redimnet2-b6-lm** | local | 0.0123 B | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [github.com/PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) | `a7_redimnet` |
| **titanet-large** | local | 0.023 B | CC-BY-4.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) | `a4_nemo` |
| **wavlm-base-plus-sv** | local | 0.095 B | MIT | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | not run | [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) | `server` |
| **wespeaker-campplus-lm** | local | 0.0072 B | Apache-2.0 code; CC BY 4.0 weights | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [Wespeaker/wespeaker-voxceleb-campplus-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM) | `a7_wespeaker` |
| **wespeaker-ecapa-tdnn512-lm** | local | 0.0064 B | Apache-2.0 code; CC BY 4.0 weights | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [Wespeaker/wespeaker-ecapa-tdnn512-LM](https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM) | `a7_wespeaker` |
| **wespeaker-resnet293-lm** | local | 0.0286 B | Apache-2.0 code; CC BY 4.0 weights | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | not run | [Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) | `a7_wespeaker` |
| **wespeaker-simam-resnet34-vb2** | local | — | Apache-2.0 code; CC BY 4.0 weights | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | not run | [Wespeaker/wespeaker-voxlink2-samresnet34_ft](https://huggingface.co/Wespeaker/wespeaker-voxlink2-samresnet34_ft) | `a7_wespeaker` |
| ~~mhubert-147-probe~~ (disabled) | local | 0.094 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 1 GB VRAM | — | — | not run | [utter-project/mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) | `server` |
| ~~redimnet-mrl~~ (disabled) | local | — | unknown | all (✓ en hi ja) | 1 GB VRAM | — | — | not run | [sappho192/redimnet-mrl](https://huggingface.co/sappho192/redimnet-mrl) | `a7_redimnet` |
| ~~azure-speaker-recognition~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key AZURE_SPEECH | — | — | not run | [learn.microsoft.com/azure/ai-services/speech-…](https://learn.microsoft.com/azure/ai-services/speech-service/speaker-recognition-overview) | `server` |
| ~~pyannoteai-voiceprint~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key PYANNOTEAI | — | — | not run | [docs.pyannote.ai/api-reference/voiceprint](https://docs.pyannote.ai/api-reference/voiceprint) | `server` |
| ~~score-fusion-qmf~~ (disabled) | method | — | mixed | all (✓ en hi ja) | GPU | — | — | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `a7_redimnet` |

<details><summary>Notes</summary>

- **ecapa-speechbrain**: the encoder OpenDub's quality stage loads today (Vox1-O-clean EER 0.80 on the card)
- **redimnet2-b6-lm**: Spark pick (Vox1-O 0.26 % EER). Also a G2 similarity judge (redimnet2), so A2/D rankings and this candidate share a lineage. Same weights as sarvam/redimnet2-b6-vb2vox2-lm-mit and Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM (not registered twice).
- **redimnet-b6-v1**: prior-generation ReDimNet (the 3 September primary)
- **eres2netv2-zh-common**: short-clip / multi-device specialist (200k Mandarin speakers); pipeline output_emb field unverified
- **campplus-voxceleb**: model id from the 3D-Speaker model list (not re-verified); WeSpeaker's CAM++ below is the fallback
- **wespeaker-resnet293-lm**: Vox1-O-clean EER 0.447 with AS-norm (raw cosine here)
- **wespeaker-simam-resnet34-vb2**: the research named SimAM-ResNet100 (VoxBlink2), but only the ResNet34 SimAM fine-tune is on HF (checked 2026-09-28)
- **titanet-large**: also what Riva / NIM speaker models serve (the API alternate)
- **pyannote-wespeaker-resnet34-lm**: the embedding inside pyannote 3.1 / community-1
- **wavlm-base-plus-sv**: WavLM SV front-end family; also the default G2 judge (wavlm-sv) — lineage shared with A2/D judging
- **mhubert-147-probe**: self-supervised features, not a speaker model — needs a trained speaker probe first
- **redimnet-mrl**: Matryoshka multi-resolution ReDimNet (community) — loading code and licence unverified
- **score-fusion-qmf**: unconstrained pick (ReDimNet2 + ERes2NetV2 + ResNet293 fusion with QMF calibration) — composition across envs, add after the parts are ranked
- **pyannoteai-voiceprint**: API pick; voiceprints are opaque (no cosine) and identify runs inside diarization jobs — needs its own trial adapter (€0.015 per voiceprint)
- **azure-speaker-recognition**: retired 2025-09-30 — kept for the ledger only

</details>

### A8

**Paralinguistic / delivery tagging**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **audeering-msp-dim** | local | 0.165 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [audeering/wav2vec2-large-robust-12-ft-emotion…](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) | `a4_transformers` |
| **audio-flamingo-next-think-delivery** | local | 8 B | NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 20 GB VRAM | ✗ | ✓ | not run | [nvidia/audio-flamingo-next-think-hf](https://huggingface.co/nvidia/audio-flamingo-next-think-hf) | `a4_transformers` |
| **ced-base-nonverbal** | local | 0.086 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [mispeech/ced-base](https://huggingface.co/mispeech/ced-base) | `a3_audiotag` |
| **emotion2vec-plus-base** | local | 0.09 B | MIT code; model-license (read the card) | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [emotion2vec/emotion2vec_plus_base](https://huggingface.co/emotion2vec/emotion2vec_plus_base) | `a4_funasr` |
| **emotion2vec-plus-large** | local | 0.3 B | MIT code; model-license (read the card) | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [emotion2vec/emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) | `a4_funasr` |
| **hubert-large-superb-er** | local | 0.316 B | Apache-2.0 | en | 2 GB VRAM | ✓ | ✓ | not run | [superb/hubert-large-superb-er](https://huggingface.co/superb/hubert-large-superb-er) | `a4_transformers` |
| **qwen3-omni-30b-a3b-delivery** | local | 30 B | Apache-2.0 | all (✓ en hi ja) | 80 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `a8_qwen_omni` |
| **prosody-dsp** (baseline) | method | — | n/a (own code) | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `server` |
| **gemini-2.5-pro-delivery** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/audio](https://ai.google.dev/gemini-api/docs/audio) | `server` |
| **gemini-3.1-pro-delivery** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `server` |
| **gpt-audio-1.5-delivery** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models/gpt-aud…](https://developers.openai.com/api/docs/models/gpt-audio-1.5) | `server` |
| **qwen3.5-omni-plus-delivery** | api | — | proprietary API | all (✓ en hi ja) | key DASHSCOPE | ✓ | ✓ | not run | [alibabacloud.com/help/en/model-studio/qwen-omni](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) | `server` |
| **scribe-v2-nonverbal** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| ~~voc2vec~~ (disabled) | local | 0.095 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | — | — | not run | [alkiskoudounas/voc2vec](https://huggingface.co/alkiskoudounas/voc2vec) | `a4_transformers` |
| ~~gemini-3.8-live-delivery~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | — | — | not run | [ai.google.dev/gemini-api/docs/live](https://ai.google.dev/gemini-api/docs/live) | `server` |
| ~~gpt-live-1~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | — | — | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `server` |
| ~~specialist-prior-llm~~ (disabled) | method | — | mixed | all (✓ en hi ja) | GPU | — | — | not run | [github.com/NidhiBharani/OpenDub](https://github.com/NidhiBharani/OpenDub) | `a8_qwen_omni` |
| ~~wavlm-text-fusion~~ (disabled) | method | — | per recipe | all (✓ en hi ja) | GPU | — | — | not run | [arxiv.org/search/?query=Interspeech+2025+natu…](https://arxiv.org/search/?query=Interspeech+2025+naturalistic+speech+emotion+recognition+challenge) | `a4_transformers` |

<details><summary>Notes</summary>

- **prosody-dsp**: tier one — OpenDub's acoustic_prosody (rate/loudness/F0/pauses) with a transparent rule label + arousal proxy (a floor)
- **hubert-large-superb-er**: the English IEMOCAP 4-class classifier OpenDub's emotion module loads today (neu/hap/ang/sad)
- **emotion2vec-plus-large**: 9 classes (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown)
- **audeering-msp-dim**: dimensional only (arousal / dominance / valence) → scores CCC, not macro-F1
- **qwen3-omni-30b-a3b-delivery**: Spark pick tier two (delivery-note writer); FP8/NVFP4 would free memory but is not wired
- **qwen3.5-omni-plus-delivery**: unconstrained + API pick; price not verified (usd_per_call param)
- **gemini-2.5-pro-delivery**: the 3 September primary ("Gemini 2.5/3 Pro class"); prices are params, not re-read for 2.5 Pro
- **gemini-3.1-pro-delivery**: "Gemini 3.1 Pro for batch" — model id string and prices unverified
- **gemini-3.8-live-delivery**: Gemini 3.8 Live Extended Thinking is a Live (WebSocket) API — no batch worker
- **gpt-audio-1.5-delivery**: current successor of the ledger's "GPT-4o Audio"; audio in $32 / 1M tokens
- **gpt-live-1**: realtime (session) model released 2026-09-17 — no batch chat-completions worker
- **audio-flamingo-next-think-delivery**: "fits in ~16 GB" per compute-tiers but 8B bf16 needs ~18–24 GB (estimate) → Spark; eval-only
- **ced-base-nonverbal**: AudioSet laughter / crying / sigh / scream … events → nonverbal event F1 (no emotion label)
- **scribe-v2-nonverbal**: inline audio-event tags (laughter …) aligned to words
- **voc2vec**: non-verbal-vocalisation encoder without a classification head — needs a trained head
- **wavlm-text-fusion**: challenge recipes (WavLM-large + text branch) — training code, no packaged checkpoint
- **specialist-prior-llm**: emotion2vec+ / audEERING priors feeding a large LLM — composition, add after the parts are ranked

</details>

## B — Source analysis · picture

### B1

**Active speaker detection and face tracking**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **loconet-talknce** | local | 0.035 B | MIT (TalkNCE repo + checkpoint); LoCoNet upstream has no LICENSE file | all (✓ en hi ja) | 4 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/kaistmm/TalkNCE](https://github.com/kaistmm/TalkNCE) | `b1_asd` |
| **lr-asd** | local | 0.001 B | MIT | all (✓ en hi ja) | 2 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/Junhua-Liao/LR-ASD](https://github.com/Junhua-Liao/LR-ASD) | `b1_asd` |
| **lr-asd-ava** | local | 0.001 B | MIT | all (✓ en hi ja) | 2 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/Junhua-Liao/LR-ASD](https://github.com/Junhua-Liao/LR-ASD) | `b1_asd` |
| **lr-asd-scrfd** | local | 0.001 B | MIT (LR-ASD); insightface buffalo_l (SCRFD-10GF) weights non-commercial research only · NC | all (✓ en hi ja) | 2 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/Junhua-Liao/LR-ASD](https://github.com/Junhua-Liao/LR-ASD) | `b1_asd` |
| **lr-asd-yunet** | local | 0.001 B | MIT (LR-ASD, YuNet) | all (✓ en hi ja) | 2 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/Junhua-Liao/LR-ASD](https://github.com/Junhua-Liao/LR-ASD) | `b1_asd` |
| **talknet-asd** | local | — | MIT | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/TaoRuijie/TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) | `b1_asd` |
| **largest-face-s3fd** (baseline) | method | — | MIT (TalkNet-ASD code incl. S3FD port) | all (✓ en hi ja) | 2 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/TaoRuijie/TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) | `b1_asd` |
| **claude-opus-5** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC, +ffmpeg | ✓ | ✓ | not run | [docs.claude.com/en/docs/about-claude/models/o…](https://docs.claude.com/en/docs/about-claude/models/overview) | `b_api` |
| **gemini-3.1-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI, +ffmpeg | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `b_api` |
| **gpt-6-sol** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI, +ffmpeg | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `b_api` |
| ~~c3asd~~ (disabled) | local | — | MIT | all (✓ en hi ja) | CPU | — | — | not run | [github.com/jisoo-o/C3ASD](https://github.com/jisoo-o/C3ASD) | `b1_asd` |
| ~~gatefusion~~ (disabled) | local | — | none (repo has no LICENSE) · NC | all (✓ en hi ja) | 12 GB VRAM | — | — | not run | [github.com/Orpheusown/GateFusion](https://github.com/Orpheusown/GateFusion) | `b1_asd` |
| ~~loconet-talknce-sam31~~ (disabled) | local | — | MIT (TalkNCE) + SAM License (facebook/sam3.1) · NC | all (✓ en hi ja) | 10 GB VRAM, key HF_TOKEN | — | — | not run | [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) | `b1_asd` |
| ~~mused~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2404.00861](https://arxiv.org/abs/2404.00861) | `b1_asd` |
| ~~aws-rekognition-faces~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key AWS_ACCESS_ID | — | — | not run | [docs.aws.amazon.com/rekognition/latest/dg/fac…](https://docs.aws.amazon.com/rekognition/latest/dg/faces.html) | `b_api` |
| ~~loconet-talknce-ensemble~~ (disabled) | method | — | MIT (TalkNCE) + SAM License · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/kaistmm/TalkNCE](https://github.com/kaistmm/TalkNCE) | `b1_asd` |

<details><summary>Notes</summary>

- **largest-face-s3fd**: OpenDub has no B1 today (lip sync runs on whatever face is on screen); this face-only heuristic — S3FD detection, shot-cut-aware IoU tracks, the largest face "speaks" — is that behaviour made explicit, and the floor every ASD head must beat. VRAM estimated.
- **loconet-talknce**: kaistmm/TalkNCE @4295996 (LoCoNet trained with TalkNCE, 95.5 AVA val mAP). The repo has no inference script for arbitrary video; the worker's LoCoNetHead re-implements val_loader + evaluate_network (3 speaker slots, 200-frame clips, VGGish log-mel) and stubs the missing utils.distributed module — first run should be checked against AVA val. ~3 GB per the compute-tiers card; 4 GB budgeted.
- **lr-asd**: Junhua-Liao/LR-ASD @1b6dcd2 (IJCV 2025; ~1M params, 94.45 AVA mAP). Weights in-repo; the TalkSet fine-tune is the in-the-wild choice (pretrain_AVA.model also ships).
- **lr-asd-ava**: LR-ASD's AVA-trained checkpoint (vs the TalkSet fine-tune above).
- **talknet-asd**: TaoRuijie/TalkNet-ASD @6d68214, the reference end-to-end demo (92.3 AVA val mAP). Same front-end as the others, so differences are the ASD head.
- **lr-asd-yunet**: Front-end variant: OpenCV YuNet (MIT, opencv_zoo) instead of S3FD — the permissive detector option, since SCRFD's weights are non-commercial. Download the ONNX to yunet_model first.
- **lr-asd-scrfd**: Front-end variant with SCRFD-10GF (the research's recommended detector) via insightface FaceAnalysis(buffalo_l, detection only). Needs `uv pip install insightface onnxruntime-gpu` in b1_asd (not in the default recipe). ByteTrack-class association is approximated by the TalkNet IoU tracker with hard cuts.
- **gatefusion**: Code at github.com/Orpheusown/GateFusion but no released checkpoints and no licence (checked 2026-09-28); built on Whisper-large-v3 + AV-HuBERT encoders. Enable once weights appear.
- **mused**: MuSED (arXiv 2404.00861) — "code in due course"; no repository found 2026-09-28.
- **c3asd**: C3ASD (ECCV 2026) code at github.com/jisoo-o/C3ASD (MIT, built on Light-ASD); no released weights found. Would slot into b1_asd like LR-ASD once a checkpoint exists.
- **loconet-talknce-sam31**: SAM 3.1 Object Multiplex (HF facebook/sam3.1, gated, custom SAM License) as the tracker instead of IoU chaining (supersedes the research's SAM 2 / Seg2Track note). Tracker integration not written.
- **loconet-talknce-ensemble**: Unconstrained alternate: LoCoNet+TalkNCE over multi-crop/multi-scale inputs with SAM 3.1 tracks (~+0.5 mAP for ~5x compute). Needs the SAM tracker above; not written.
- **aws-rekognition-faces**: AWS Rekognition / Azure Video Indexer face tracking cover only the tracking half — no speaking decision — so they cannot answer the B1 question alone.
- **gemini-3.1-pro**: Native video+audio input: the whole clip goes in with the line windows. Price: ai.google.dev pricing 2026-09-28, $2/$12 per M tokens (≤200k prompt; $4/$18 above — not modelled). Not frame-accurate per the research.
- **claude-opus-5**: Frames only (the Messages API takes images, not video/audio): stills across each line plus its text — lip reading from stills, expected weak. $5/$25 per M tokens (Anthropic pricing, 2026-09-28). No refusal fallback, so the scored model is always Opus 5.
- **gpt-6-sol**: Not named in the artifacts for B1; added as the GPT counterpart of the frames-only API baseline. $2/$10 per M tokens (developers.openai.com pricing, 2026-09-28). Vision support of gpt-6-sol assumed from the family, not checked on a model card.

</details>

### B2

**Shot boundary detection**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **omnishotcut-paper** | local | 0.0345 B | MIT | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [uva-cv-lab/OmniShotCut_paper](https://huggingface.co/uva-cv-lab/OmniShotCut_paper) | `b2_omnishotcut` |
| **omnishotcut-v1.5** | local | 0.0528 B | MIT (code and weights per repo LICENSE; the arXiv paper is CC BY-NC-ND) | all (✓ en hi ja) | 4 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [uva-cv-lab/OmniShotCut_v1.5](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5) | `b2_omnishotcut` |
| **pyscenedetect-adaptive** | local | — | BSD-3-Clause | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [scenedetect.com/](https://www.scenedetect.com/) | `b2_scenedetect` |
| **pyscenedetect-content** | local | — | BSD-3-Clause | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [scenedetect.com/](https://www.scenedetect.com/) | `b2_scenedetect` |
| **pyscenedetect-threshold** | local | — | BSD-3-Clause | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [scenedetect.com/](https://www.scenedetect.com/) | `b2_scenedetect` |
| **transnetv2** | local | — | MIT | all (✓ en hi ja) | 1.5 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [pypi.org/project/transnetv2-pytorch/](https://pypi.org/project/transnetv2-pytorch/) | `b2_transnetv2` |
| **transvlm-4b** | local | 4 B | Apache-2.0 | all (✓ en hi ja) | 14 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct](https://huggingface.co/HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct) | `b2_transvlm` |
| **ffmpeg-scene-0.3** (baseline) | method | — | LGPL-2.1+ (ffmpeg, invoked as a binary) | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [ffmpeg.org/ffmpeg-filters.html#select_002c-as…](https://ffmpeg.org/ffmpeg-filters.html#select_002c-aselect) | `server` |
| **aws-rekognition-segments** | api | — | proprietary API | all (✓ en hi ja) | key AWS_ACCESS_ID, +ffmpeg | ✓ | ✓ | not run | [docs.aws.amazon.com/rekognition/latest/dg/seg…](https://docs.aws.amazon.com/rekognition/latest/dg/segments.html) | `b_api` |
| **gemini-3.1-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI, +ffmpeg | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `b_api` |
| **google-vi-shot-change** | api | — | proprietary API | all (✓ en hi ja) | key GOOGLE_APPLICATION_CREDENTIALS, +ffmpeg | ✓ | ✓ | not run | [cloud.google.com/video-intelligence/docs/anal…](https://cloud.google.com/video-intelligence/docs/analyze-shots) | `b_api` |
| **transnetv2+scenedetect-threshold** | method | — | MIT + BSD-3-Clause | all (✓ en hi ja) | 1.5 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) | `b2_transnetv2` |
| ~~autoshot~~ (disabled) | local | — | MIT | all (✓ en hi ja) | 2 GB VRAM | — | — | not run | [github.com/wentaozhu/AutoShot](https://github.com/wentaozhu/AutoShot) | `b2_transnetv2` |
| ~~persist~~ (disabled) | local | — | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [github.com/linty5/PERSIST](https://github.com/linty5/PERSIST) | `b2_persist` |
| ~~fassold-motion-ncc~~ (disabled) | method | — | paper CC BY-SA 4.0; no code · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2502.09202](https://arxiv.org/abs/2502.09202) | `server` |

<details><summary>Notes</summary>

- **ffmpeg-scene-0.3**: What OpenDub does today: ffmpeg `scale=320:-2,select='gt(scene,0.3)'` (app/media/ffmpeg.py detect_scene_changes). CPU only.
- **transnetv2**: transnetv2-pytorch 1.0.5 wheel (MIT fork allenday/transnetv2_pytorch that bundles converted weights; weak provenance — pass params.weights for a locally converted upstream file). Published F1 96.2 BBC / 93.9 RAI / 77.9 ClipShots. ~1.5 GB per compute-tiers.
- **transnetv2+scenedetect-threshold**: The compute-tiers Spark pick as a whole: TransNetV2 cuts plus PySceneDetect's ThresholdDetector (fades / black frames) as a second channel, merged within 0.5 s.
- **omnishotcut-v1.5**: UVA-Computer-Vision-Lab/OmniShotCut @40ecf12, v1.5 weights (52.8M params). The research assumed NC-ND and no code; both are resolved. VRAM estimated.
- **omnishotcut-paper**: The paper checkpoint (34.5M params, reproduces Table 1); compare with v1.5.
- **transvlm-4b**: heygen-com/TransVLM @7880c90, HF HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct (RGB + NeuFlow v2 optical flow, 6-channel input; only the repo's infer_video.py is correct). The research said CC BY-NC-SA; the repo and weights are Apache-2.0. VRAM estimated for a 4B VLM in bf16 plus flow — tight on 16 GB; rtfx includes a model load per item (subprocess per clip).
- **pyscenedetect-content**: PySceneDetect 0.7.1 ContentDetector at its default threshold; CPU.
- **pyscenedetect-adaptive**: AdaptiveDetector (rolling-average content score; fewer false cuts on camera motion).
- **pyscenedetect-threshold**: ThresholdDetector — fades / black frames only; expected low recall on hard cuts.
- **autoshot**: wentaozhu/AutoShot: the checkpoint is only on Baidu Pan (passcode sfkq); the repo's inference path is commented out. Download ckpt_0_200_0.pth manually, then a worker around TransNetV2Supernet (supernet_flattransf_3_8_8_8_13_12_0_16_60.py) can be added.
- **persist**: linty5/PERSIST (BMVC 2026): checkpoints on GitHub releases, but evaluation only through an mmaction2 config (torch 2.4.1 / mmcv 2.1.0 source build) with no arbitrary-video script.
- **fassold-motion-ncc**: Classical sparse motion-field + NCC detector (arXiv 2502.09202); no code released.
- **google-vi-shot-change**: Video Intelligence SHOT_CHANGE_DETECTION, $0.05/min (pricing page 2026-09-28). The API was deprecated 2026-09-14 and shuts down 2027-09-14. Inline upload (input_content); large files may exceed the inline limit (unverified).
- **aws-rekognition-segments**: StartSegmentDetection needs the video in S3: set OPENDUB_AWS_BUCKET (objects are deleted after each item). Shots $0.05/min + technical cues $0.05/min (us-east-1, 2026-09-28).
- **gemini-3.1-pro**: Semantic fallback (act breaks), not a frame-accurate detector; clips over 10 min are skipped. $2/$12 per M tokens (≤200k prompt), ai.google.dev pricing 2026-09-28.

</details>

### B3

**On-screen text reading**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **easyocr** (baseline) | local | — | Apache-2.0 | en hi ja +6 | 2 GB VRAM | ✓ | ✓ | not run | [github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) | `b3_classic_ocr` |
| **paddleocr-vl-1.6** | local | 0.9 B | Apache-2.0 | en hi ja +1 | 4 GB VRAM | ✓ | ✓ | not run | [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | `b3_paddleocr_vl` |
| **pp-ocrv5-devanagari** | local | — | Apache-2.0 | hi +2 | CPU | ✓ | ✓ | not run | [paddleocr.ai/latest/en/version3.x/pipeline_us…](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) | `b3_paddleocr` |
| **pp-ocrv6-medium** | local | 0.0345 B | Apache-2.0 (code); CC BY 4.0 (released artefacts) | en ja +6 | CPU | ✓ | ✓ | not run | [blog/PaddlePaddle/pp-ocrv6](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6) | `b3_paddleocr` |
| **pp-ocrv6-small** | local | 0.0077 B | Apache-2.0 / CC BY 4.0 | en ja +6 | CPU | ✓ | ✓ | not run | [blog/PaddlePaddle/pp-ocrv6](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6) | `b3_paddleocr` |
| **pp-ocrv6-tiny** | local | 0.0015 B | Apache-2.0 / CC BY 4.0 | en +1 | CPU | ✓ | ✓ | not run | [blog/PaddlePaddle/pp-ocrv6](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6) | `b3_paddleocr` |
| **qwen3-vl-32b** | local | 32 B | Apache-2.0 | all (✓ en hi ja) | 75 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | `b_hf` |
| **qwen3-vl-8b** | local | 8 B | Apache-2.0 | all (✓ en hi ja) | 22 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `b_hf` |
| **qwen3-vl-8b-fp8** | local | 8 B | Apache-2.0 | all (✓ en hi ja) | 12 GB VRAM, SM ≥ 8.9 | ✓ | ✓ | not run | [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) | `b_hf` |
| **rapidocr** | local | — | Apache-2.0 | en ja +1 | CPU | ✓ | ✓ | not run | [github.com/RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) | `b3_classic_ocr` |
| **surya** | local | 0.65 B | code Apache-2.0; weights modified AI Pubs Open RAIL-M (research/personal/<$5M orgs) · NC | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [github.com/datalab-to/surya](https://github.com/datalab-to/surya) | `b3_surya` |
| **claude-opus-5** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC, +ffmpeg | ✓ | ✓ | not run | [docs.claude.com/en/docs/about-claude/models/o…](https://docs.claude.com/en/docs/about-claude/models/overview) | `b_api` |
| **gemini-3.1-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI, +ffmpeg | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `b_api` |
| **gemini-3.8-flash** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI, +ffmpeg | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `b_api` |
| **google-vi-text** | api | — | proprietary API | all (✓ en hi ja) | key GOOGLE_APPLICATION_CREDENTIALS, +ffmpeg | ✓ | ✓ | not run | [cloud.google.com/video-intelligence/docs/text…](https://cloud.google.com/video-intelligence/docs/text-detection) | `b_api` |
| **gpt-4o** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI, +ffmpeg | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `b_api` |
| **gpt-6-sol** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI, +ffmpeg | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `b_api` |
| ~~dots-ocr~~ (disabled) | local | — | unconfirmed · NC | all (✓ en hi ja) | 8 GB VRAM | — | — | not run | [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | `b3_dots_ocr` |
| ~~glm-5.3-flash~~ (disabled) | local | 320 B | MIT | all (✓ en hi ja) | 170 GB VRAM | — | — | not run | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `b_hf` |
| ~~qwen3-vl-235b-a22b~~ (disabled) | local | 235 B | Apache-2.0 | all (✓ en hi ja) | 480 GB VRAM | — | — | not run | [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) | `b_hf` |
| ~~gomatching-pp~~ (disabled) | method | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2505.22228](https://arxiv.org/abs/2505.22228) | `b3_paddleocr` |
| ~~ppocrv6+paddleocr-vl+gomatching~~ (disabled) | method | — | Apache-2.0 / CC BY 4.0 | all (✓ en hi ja) | CPU | — | — | not run | [blog/PaddlePaddle/pp-ocrv6](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6) | `b3_paddleocr` |

<details><summary>Notes</summary>

- **easyocr**: The floor (0.507 CER on the VideoDB video-frame benchmark). Baseline because OpenDub has no OCR today and this is what "first reach" looks like. Reader langs per pack language.
- **rapidocr**: RapidOCR default (PP-OCR ONNX, CPU onnxruntime). Default models read zh/en; ja coverage via config (Rec.lang_type) unverified. 0.430 CER on VideoDB.
- **pp-ocrv6-medium**: Spark pick (34.5M params, ~1 GB). onnxruntime engine so no PaddlePaddle GPU build is needed (none exists for aarch64). Covers zh/en/ja + 46 Latin-script languages; no Devanagari — hi is served by the v5 script model below. GPU when onnxruntime-gpu is present, else CPU.
- **pp-ocrv6-small**: 7.7M-param tier.
- **pp-ocrv6-tiny**: 1.5M-param tier. The HF blog says the 50-language coverage is for the medium and small tiers; tiny is registered for en only here (ja unverified).
- **pp-ocrv5-devanagari**: Plan §7: PP-OCRv5 script models for Devanagari (v6 has none). PP-OCRv6 medium detector + the ultra-light v5 Devanagari recogniser.
- **paddleocr-vl-1.6**: 0.9B document VLM (96.33% OmniDocBench v1.6), transformers ≥ 5. The "OCR:" prompt returns text without boxes, so it scores on text_f / char_f only; the "Spotting:" prompt (boxes) has an undocumented output format and is not wired. hi coverage unverified.
- **dots-ocr**: dots.ocr loads as AutoModelForCausalLM + trust_remote_code with qwen_vl_utils, which the shared LocalVlm loader does not do; licence unconfirmed in the research. Not wired.
- **surya**: ~650M-param single VLM (91-language internal benchmark). Worker written against the surya-ocr ≥ 0.14 predictor API; confirm Surya 2's API before running. VRAM estimated.
- **qwen3-vl-8b-fp8**: Thalassa-sized Qwen3-VL (research: local semantic tier). FP8 checkpoint id and transformers FP8 loading on sm_120 unverified; VRAM estimated (8B × 1 byte + vision tower + KV).
- **qwen3-vl-8b**: bf16 8B — Spark (≈17 GB weights + activations; estimated).
- **qwen3-vl-32b**: bf16 32B — Spark only (≈66 GB weights; estimated). Leads OCRBench-v2 (en) at 32B.
- **qwen3-vl-235b-a22b**: Does not fit one Spark even at 4-bit (~120 GB+); PP-OCRv6 beats it on recognition per the PP-OCRv6 paper. Registered for the ledger.
- **glm-5.3-flash**: 320B-A18B MoE; ~160–170 GB at 4-bit — does not fit one 128 GB Spark.
- **gomatching-pp**: GoMatching++ video text spotting (arXiv 2505.22228): scene-text research code, licence and en/hi/ja checkpoints unconfirmed; a design reference for temporal dedup, not wired.
- **ppocrv6+paddleocr-vl+gomatching**: Unconstrained pick (PP-OCRv6 + PaddleOCR-VL consensus with GoMatching++ track voting). Compose once the members are ranked; needs GoMatching++.
- **gemini-3.1-pro**: API pick; box_2d output (0–1000, [ymin, xmin, ymax, xmax]). $2/$12 per M tokens (≤200k), ai.google.dev pricing 2026-09-28.
- **gemini-3.8-flash**: Volume tier ("Gemini 3.x Flash" in the research). $0.75/$3.75 per M tokens through 2026-12-31, then $1.50/$7.50 (ai.google.dev pricing 2026-09-28).
- **claude-opus-5**: $5/$25 per M tokens (Anthropic pricing 2026-09-28); images up to 2576 px long edge. Asked for the same 0–1000 box_2d convention as Gemini.
- **gpt-4o**: The VideoDB video-frame benchmark anchor (0.238 CER) named in the research. $2.50/$10 per M tokens (developers.openai.com pricing 2026-09-28).
- **gpt-6-sol**: Current-generation GPT counterpart of gpt-4o (not in the artifacts). $2/$10 per M tokens, 2026-09-28; vision support assumed from the family.
- **google-vi-text**: Video-native tracked text with boxes; $0.15/min, billed per started minute (so a still image costs a full minute). Deprecated 2026-09-14, shutdown 2027-09-14.

</details>

### B4

**Content type classification (live action / 2D / 3D / mixed)**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **deepghs-caformer** | local | — | MIT | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [deepghs/anime_classification](https://huggingface.co/deepghs/anime_classification) | `b4_deepghs` |
| **deepghs-mobilenet** | local | — | MIT | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [deepghs/anime_classification](https://huggingface.co/deepghs/anime_classification) | `b4_deepghs` |
| **pe-core-l14-336-probe** | local | — | Apache-2.0 (verify the variant) | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [facebook/PE-Core-L14-336](https://huggingface.co/facebook/PE-Core-L14-336) | `b_hf` |
| **prithiv-anime-classification-v1** | local | 0.0929 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [prithivMLmods/Anime-Classification-v1.0](https://huggingface.co/prithivMLmods/Anime-Classification-v1.0) | `b_hf` |
| **qwen3-vl-8b** | local | 8 B | Apache-2.0 | all (✓ en hi ja) | 22 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `b_hf` |
| **qwen3-vl-8b-fp8** | local | 8 B | Apache-2.0 | all (✓ en hi ja) | 12 GB VRAM, SM ≥ 8.9 | ✓ | ✓ | not run | [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) | `b_hf` |
| **siglip2-base-probe** | local | 0.086 B | Apache-2.0 | all (✓ en hi ja) | 1.5 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | `b_hf` |
| **siglip2-so400m-probe** | local | 0.4 B | Apache-2.0 | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | `b_hf` |
| **siglip2-so400m-zeroshot** | local | 0.4 B | Apache-2.0 | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | `b_hf` |
| **constant-live-action** (baseline) | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [builtin](builtin) | `server` |
| **claude-opus-5** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC, +ffmpeg | ✓ | ✓ | not run | [docs.claude.com/en/docs/about-claude/models/o…](https://docs.claude.com/en/docs/about-claude/models/overview) | `b_api` |
| **gemini-3.1-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI, +ffmpeg | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `b_api` |
| **gemini-3.8-flash** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI, +ffmpeg | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `b_api` |
| **gpt-6-sol** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI, +ffmpeg | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `b_api` |
| **siglip2-so400m-probe-conformal** | method | 0.4 B | Apache-2.0 | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | `b_hf` |
| ~~forensic-cgi-vs-photo~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2308.07279](https://arxiv.org/abs/2308.07279) | `b_hf` |
| ~~glm-5.3-flash~~ (disabled) | local | 320 B | MIT | all (✓ en hi ja) | 170 GB VRAM | — | — | not run | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `b_hf` |
| ~~qwen3-vl-235b-a22b~~ (disabled) | local | 235 B | Apache-2.0 | all (✓ en hi ja) | 480 GB VRAM | — | — | not run | [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) | `b_hf` |
| ~~ensemble-probe-deepghs-vlm~~ (disabled) | method | — | mixed | all (✓ en hi ja) | CPU | — | — | not run | [builtin](builtin) | `b_hf` |

<details><summary>Notes</summary>

- **constant-live-action**: No gate today — a human-face lip-sync model may run on anything.
- **siglip2-so400m-probe**: Spark pick: frozen SigLIP 2 so400m + multinomial logistic head fitted on the pack's dev titles (never test). Needs a dev split with ≥ 2 pure classes (b4_titles builder writes one).
- **siglip2-so400m-probe-conformal**: The research's abstention mechanism (split conformal prediction, α = 0.1): half the dev titles fit the head, half calibrate; a title abstains unless its prediction set is a singleton. Scored on misroute_cost / sel_acc95 as well as macro-F1.
- **siglip2-so400m-zeroshot**: Zero-shot text prompts per class (no labelled frames) — the control that shows what the linear probe buys.
- **siglip2-base-probe**: The 86M base encoder (research lists all four SigLIP 2 scales); cost floor of the probe.
- **pe-core-l14-336-probe**: Perception Encoder as the alternative frozen backbone. The open_clip hub id and licence are unverified (the research says "Meta research licence — verify").
- **deepghs-caformer**: dghs-imgutils anime_classify (3d / bangumi / comic / illustration / not_painting) + anime_real (anime / real) for the live-action share. CPU. The 97.23% figure the research quoted is not reproducible per compute-tiers.
- **deepghs-mobilenet**: The lightweight defaults of both imgutils classifiers (the "day-one gate").
- **prithiv-anime-classification-v1**: SigLIP 2 base fine-tune, 4 classes, no live-action class — it can never say live_action (research: "cannot make the gate decision alone"). Label names assumed to match the card.
- **qwen3-vl-8b-fp8**: Plan §7's Thalassa entry (Qwen3-VL-8B FP8). FP8 checkpoint id and transformers FP8 loading on sm_120 unverified; VRAM estimated.
- **qwen3-vl-8b**: bf16 on the Spark (~16 GB weights per compute-tiers; 22 GB budgeted).
- **qwen3-vl-235b-a22b**: Does not fit one Spark even at 4-bit; registered for the ledger.
- **glm-5.3-flash**: Unconstrained pick (320B-A18B); ~160–170 GB at 4-bit, a hard wall on one Spark.
- **forensic-cgi-vs-photo**: Multi-colourspace ViT / Swin CGI-vs-photo classifiers (arXiv 2308.07279, 2409.04742): code and licences unconfirmed, cross-dataset collapse documented, and the axis does not change routing. Registered only.
- **ensemble-probe-deepghs-vlm**: SigLIP probe + deepghs + VLM, abstain on disagreement. Compose from the members' cached outputs once they are ranked; composition worker not written.
- **gemini-3.1-pro**: API pick; 8 evenly spaced frames with an explicit abstain option (a whole title as video would exceed any sane token budget). $2/$12 per M tokens, 2026-09-28.
- **gemini-3.8-flash**: $0.75/$3.75 per M tokens to 2026-12-31, then $1.50/$7.50 (ai.google.dev, 2026-09-28).
- **claude-opus-5**: Second opinion on the ambiguous tail; $5/$25 per M tokens (2026-09-28).
- **gpt-6-sol**: GPT counterpart (not in the artifacts); $2/$10 per M tokens (2026-09-28).

</details>

## C — Text

### C1

**Dub-line segmentation**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **pause-splitter** (baseline) | local | — | MIT (OpenDub) | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/app/pipeline/segmentation.py](server/app/pipeline/segmentation.py) | `server` |
| **qwen3-14b-awq-segment** | local | 14.8 B | Apache-2.0 | en hi ja | 14 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) | `c2_vllm` |
| **qwen3.8-27b-fp8-segment** | local | 27 B | Apache-2.0 | en hi ja | 44 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `c2_vllm` |
| **sat-12l-sm** | local | 0.3 B | MIT | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | not run | [segment-any-text/sat-12l-sm](https://huggingface.co/segment-any-text/sat-12l-sm) | `c1_sat` |
| **sat-12l-ted-lora** | local | 0.3 B | MIT | en hi ja | 1.5 GB VRAM | ✓ | ✓ | not run | [segment-any-text/sat-12l](https://huggingface.co/segment-any-text/sat-12l) | `c1_sat` |
| **sat-3l-sm** | local | — | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | not run | [segment-any-text/sat-3l-sm](https://huggingface.co/segment-any-text/sat-3l-sm) | `c1_sat` |
| **claude-opus-5-segment** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **gemini-3.1-pro-segment** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **gpt-6-astra-segment** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| ~~glm-5.3-segment~~ (disabled) | local | 744 B | GLM-5.3 licence | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/blog/glm-5-3-specs-benchmarks-api-ho…](https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/) | `server` |
| ~~qwen3.8-2.4t-a95b-segment~~ (disabled) | local | 2400 B | unknown | all (✓ en hi ja) | CPU | — | — | not run | [en.wikipedia.org/wiki/Qwen](https://en.wikipedia.org/wiki/Qwen) | `server` |
| ~~shas~~ (disabled) | local | 0.3 B | MIT | en | CPU | — | — | not run | [github.com/mt-upc/SHAS](https://github.com/mt-upc/SHAS) | `server` |
| ~~apptek-line-segmenter~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.iwslt-1.21/](https://aclanthology.org/2025.iwslt-1.21/) | `server` |
| ~~frontier-joint-seg-translate~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [docs/model-candidates-by-compute.md](docs/model-candidates-by-compute.md) | `server` |
| ~~hwtsc-parse-pause~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/events/iwslt-2024/](https://aclanthology.org/events/iwslt-2024/) | `server` |

<details><summary>Notes</summary>

- **pause-splitter**: segmentation.pause: split on pauses ≥ 0.35 s or sentence ends, cap 12 s.
- **sat-12l-sm**: 0.3B, 85 languages; boundary probs drive the pause/duration DP search.
- **sat-3l-sm**: Fast variant (runs on CPU with device: cpu).
- **sat-12l-ted-lora**: ASR-style (TED, corrupted text) LoRA; loras/ted2020-corrupted has hi and ja. Language taken from the pack.
- **hwtsc-parse-pause**: HW-TSC IWSLT 2024 parsing-informed pause selector: needs a constituency/dependency parser per language (not in any env yet); reimplementable.
- **shas**: SHAS audio segmentation (MIT): checkpoints only for en/es/fr/it/pt (no hi/ja) and it needs an audio-input worker; not written.
- **apptek-line-segmenter**: AppTek neural line segmenter with hard constraints: proprietary.
- **qwen3-14b-awq-segment**: AWQ int4 (~9.5 GB weights) on vLLM with thinking disabled; vLLM reserves 85% of the card. Card: 100+ languages incl. hi/ja. Plan seed '14B LLM'.
- **qwen3.8-27b-fp8-segment**: Dense 27B VL (hybrid Gated DeltaNet), 262K context; Spark (FP8 official). The artifact's NVFP4 recipe is faster but NVFP4 on GB10 is immature. The card lists no languages: hi/ja unverified. enable_thinking assumed to follow Qwen3.
- **claude-opus-5-segment**: List price $5/$25 per 1M tokens (Anthropic list price).
- **gemini-3.1-pro-segment**: List price $2/$12 per 1M tokens (≤200K prompt), 2026-09-28.
- **gpt-6-astra-segment**: List price $10/$50 per 1M tokens, 2026-09-28.
- **frontier-joint-seg-translate**: Joint segmentation + translation + length budget in one scene pass with QE best-of-N: a cascade method whose output is C2, not C1; measured in the cascade view.
- **glm-5.3-segment**: GLM-5.3 (744B): cluster-scale.
- **qwen3.8-2.4t-a95b-segment**: Qwen3.8-2.4T-A95B open flagship: cluster-scale.

</details>

### C2

**Translation and dubbing adaptation**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **qwen2.5-14b-ollama-opendub** (baseline) | local | 14.7 B | Apache-2.0 | en hi ja +5 | 10.5 GB VRAM, +ollama | ✗ | ✗ | not run | [ollama.com/library/qwen2.5](https://ollama.com/library/qwen2.5) | `server` |
| **gemma3-12b-ollama** | local | 12.2 B | Gemma Terms of Use | en hi ja +5 | 9.5 GB VRAM, +ollama | ✗ | ✗ | not run | [ollama.com/library/gemma3](https://ollama.com/library/gemma3) | `server` |
| **gpt-oss-120b** | local | 117 B | Apache-2.0 | en hi ja +5 | 70 GB VRAM | ✗ | ✓ | not run | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | `c2_vllm` |
| **gpt-oss-20b** | local | 21 B | Apache-2.0 | en hi ja +5 | 15 GB VRAM | ✓ | ✓ | not run | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | `c2_vllm` |
| **hunyuan-mt-7b-fp8** | local | 8.03 B | Tencent Hunyuan Community License · NC | en hi ja +5 | 12 GB VRAM | ✓ | ✓ | not run | [tencent/Hunyuan-MT-7B-fp8](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8) | `c2_vllm` |
| **hy-mt1.5-1.8b** | local | 2.04 B | Tencent HY Community License · NC | en hi ja +5 | 8 GB VRAM | ✓ | ✓ | not run | [tencent/HY-MT1.5-1.8B](https://huggingface.co/tencent/HY-MT1.5-1.8B) | `c2_vllm` |
| **hy-mt1.5-7b-fp8** | local | 8.03 B | Tencent HY Community License · NC | en hi ja +5 | 12 GB VRAM | ✓ | ✓ | not run | [tencent/HY-MT1.5-7B-FP8](https://huggingface.co/tencent/HY-MT1.5-7B-FP8) | `c2_vllm` |
| **hy-mt2-1.8b** | local | 1.8 B | Apache-2.0 | en hi ja +5 | 8 GB VRAM | ✓ | ✓ | not run | [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) | `c2_vllm` |
| **hy-mt2-30b-a3b-fp8** | local | 30.06 B | Apache-2.0 | en hi ja +5 | 40 GB VRAM | ✗ | ✓ | not run | [tencent/Hy-MT2-30B-A3B-FP8](https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8) | `c2_vllm` |
| **hy-mt2-7b-fp8** | local | 8.03 B | Apache-2.0 | en hi ja +5 | 12 GB VRAM | ✓ | ✓ | not run | [tencent/Hy-MT2-7B-FP8](https://huggingface.co/tencent/Hy-MT2-7B-FP8) | `c2_vllm` |
| **qwen2.5-14b-ollama-dub** | local | 14.7 B | Apache-2.0 | en hi ja +5 | 10.5 GB VRAM, +ollama | ✗ | ✗ | not run | [ollama.com/library/qwen2.5](https://ollama.com/library/qwen2.5) | `server` |
| **qwen3-14b-awq** | local | 14.8 B | Apache-2.0 | en hi ja +5 | 14 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) | `c2_vllm` |
| **qwen3.8-27b-fp8** | local | 27 B | Apache-2.0 | en hi ja +5 | 44 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `c2_vllm` |
| **seed-x-instruct-7b** | local | 7 B | OpenMDW | en ja +2 | 18 GB VRAM | ✗ | ✓ | not run | [ByteDance-Seed/Seed-X-Instruct-7B](https://huggingface.co/ByteDance-Seed/Seed-X-Instruct-7B) | `c2_vllm` |
| **seed-x-ppo-7b** | local | 7 B | OpenMDW | en ja +2 | 18 GB VRAM | ✗ | ✓ | not run | [ByteDance-Seed/Seed-X-PPO-7B](https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B) | `c2_vllm` |
| **tower-plus-9b** | local | 9 B | CC-BY-NC-SA-4.0 · NC | en hi ja +5 | 22 GB VRAM | ✗ | ✓ | not run | [Unbabel/Tower-Plus-9B](https://huggingface.co/Unbabel/Tower-Plus-9B) | `c2_vllm` |
| **translategemma-12b** 🔓 | local | 12 B | Gemma Terms of Use | en hi ja +5 | 30 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [google/translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it) | `c2_vllm` |
| **translategemma-27b** 🔓 | local | 27 B | Gemma Terms of Use | en hi ja +5 | 62 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [google/translategemma-27b-it](https://huggingface.co/google/translategemma-27b-it) | `c2_vllm` |
| **translategemma-4b** 🔓 | local | 4 B | Gemma Terms of Use | en hi ja +5 | 11 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) | `c2_vllm` |
| **claude-fable-5-1** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **claude-opus-5** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **claude-sonnet-5** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **deepl-quality** | api | — | proprietary API | en hi ja +5 | key DEEPL | ✓ | ✓ | not run | [developers.deepl.com/docs](https://developers.deepl.com/docs) | `server` |
| **gemini-2.5-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **gemini-3.1-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **google-translate-nmt** | api | — | proprietary API | en hi ja +5 | key GOOGLE | ✓ | ✓ | not run | [cloud.google.com/translate/docs](https://cloud.google.com/translate/docs) | `server` |
| **gpt-5.5** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| **gpt-5.6-sol** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| **gpt-6-astra** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| ~~glm-5.3~~ (disabled) | local | 744 B | GLM-5.3 licence | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/blog/glm-5-3-specs-benchmarks-api-ho…](https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/) | `server` |
| ~~homura-qwen3-8b~~ (disabled) | local | 8.2 B | research-only | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2601.10187](https://arxiv.org/abs/2601.10187) | `server` |
| ~~hunyuan-mt-chimera-7b-fp8~~ (disabled) | local | 8.03 B | Tencent Hunyuan Community License · NC | en hi ja +5 | CPU | — | — | not run | [tencent/Hunyuan-MT-Chimera-7B-fp8](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8) | `server` |
| ~~kimi-k3~~ (disabled) | local | 2800 B | unknown | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/news/best-open-weight-ai-models-in-2…](https://kingy.ai/news/best-open-weight-ai-models-in-2026-glm-5-2-vs-deepseek-v4-vs-kimi-k2-6-vs-qwen-vs-mistral/) | `server` |
| ~~tower-plus-72b~~ (disabled) | local | 72 B | CC-BY-NC-4.0 · NC | en hi ja +5 | CPU | — | — | not run | [Unbabel/Tower-Plus-72B](https://huggingface.co/Unbabel/Tower-Plus-72B) | `server` |
| ~~apptek-lcmt~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.iwslt-1.21/](https://aclanthology.org/2025.iwslt-1.21/) | `server` |
| ~~hy-mt2-30b-bestofn-metricx~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [docs/model-candidates-by-compute.md](docs/model-candidates-by-compute.md) | `server` |
| ~~microsoft-lsst-labs~~ (disabled) | method | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~microsoft-translator~~ (disabled) | api | — | proprietary API | en hi ja +5 | CPU | — | — | not run | [learn.microsoft.com/azure/ai-services/transla…](https://learn.microsoft.com/azure/ai-services/translator/) | `server` |

<details><summary>Notes</summary>

- **qwen2.5-14b-ollama-opendub**: Ollama Q4_K_M tag (9 GB). Qwen2.5 does not list Hindi among its 29+ languages; kept in every direction because it is today's default. `ollama pull qwen2.5:14b` first; the worker unloads it with keep_alive 0 at the end of the job. Baseline: today's translation.openai_compatible default with today's prompt (15 English chars/s budget for every language).
- **qwen2.5-14b-ollama-dub**: Ollama Q4_K_M tag (9 GB). Qwen2.5 does not list Hindi among its 29+ languages; kept in every direction because it is today's default. `ollama pull qwen2.5:14b` first; the worker unloads it with keep_alive 0 at the end of the job. Same model, syllable/mora budget prompt: isolates the prompt change.
- **qwen3-14b-awq**: AWQ int4 (~9.5 GB weights) on vLLM with thinking disabled; vLLM reserves 85% of the card. Card: 100+ languages incl. hi/ja.
- **gemma3-12b-ollama**: Ollama Q4_K_M (8.1 GB; measured 8.9 GB resident). 140+ languages. `ollama pull gemma3:12b` first; unloaded with keep_alive 0.
- **hunyuan-mt-7b-fp8**: WMT25 best constrained system (as Shy-hunyuan-MT). Tencent Hunyuan Community License: not licensed in the EU, UK or South Korea; needs Tencent's permission above 100M MAU. Model-card prompt template; no budget control (duration gate decides).
- **hunyuan-mt-chimera-7b-fp8**: Fusion model: takes ~6 candidate translations and fuses them; needs a method worker that feeds it other candidates' outputs (not written). tencent/Hunyuan-MT-Chimera-7B-fp8. Tencent Hunyuan Community License: not licensed in the EU, UK or South Korea; needs Tencent's permission above 100M MAU.
- **hy-mt1.5-7b-fp8**: 36 languages; terminology/context/format controls not used by the arena prompt. Same territory exclusion as Hunyuan-MT. Superseded by Hy-MT2 per its card.
- **hy-mt1.5-1.8b**: bf16 (2.0B params). Same territory exclusion as Hunyuan-MT.
- **hy-mt2-7b-fp8**: Hy-MT2 family (2026-05-21). Prompt template assumed to match Hunyuan-MT (unverified).
- **hy-mt2-1.8b**: bf16. Prompt template assumed (unverified).
- **hy-mt2-30b-a3b-fp8**: Spark pick: 30B/3B-active MoE, FP8 ~32 GB. New HYV3ForCausalLM architecture: at release it needed transformers>=5.6 and vLLM from source — if vllm 0.30.0 cannot load it use backend vllm_docker with a nightly image. Card recommends ≤4096 generated tokens.
- **qwen3.8-27b-fp8**: Dense 27B VL (hybrid Gated DeltaNet), 262K context; Spark (FP8 official). The artifact's NVFP4 recipe is faster but NVFP4 on GB10 is immature. The card lists no languages: hi/ja unverified. enable_thinking assumed to follow Qwen3. Artifact: the long-context adaptation model on the Spark.
- **gpt-oss-20b**: 21B/3.6B-active MXFP4 (~13 GB): fits a 16 GB card only just (vLLM at 0.90). Fallback: Ollama gpt-oss:20b.
- **gpt-oss-120b**: 117B/5.1B-active MXFP4 (~63 GB); Spark only. English-centric, no published multilingual MT numbers (artifact: better as critic than translator).
- **translategemma-4b**: Gated (accept the Gemma terms). 55 languages (hi/ja stated, list not seen). Structured chat content item per the card — vLLM must pass source_lang_code/target_lang_code through to the template (unverified). Trained with MetricX-QE in its RL loop: its MetricX rank is optimistic.
- **translategemma-12b**: bf16 24 GB: Spark (or a quantised build on 16 GB — none official). Same caveats as translategemma-4b.
- **translategemma-27b**: bf16 ~54 GB: Spark. Same caveats as translategemma-4b.
- **seed-x-ppo-7b**: 28 languages, no Hindi. bf16 ~15 GB: Spark (official GPTQ-8bit / AWQ-4bit builds would fit 16 GB; repo ids not verified). Needs the <ja>/<en> target tag, which the seed_x prompt adds.
- **seed-x-instruct-7b**: As seed-x-ppo-7b.
- **tower-plus-9b**: Gemma-2 base, 22 languages incl. hi/ja; non-commercial (header says CC-BY-NC-SA-4.0, text CC-BY-NC-4.0). bf16 18 GB: Spark.
- **tower-plus-72b**: Unbabel/Tower-Plus-72B (Qwen2.5 base, CC-BY-NC-4.0): ~145 GB in bf16 and no official quantised checkpoint, so it exceeds the Spark.
- **gemini-3.1-pro**: List price $2/$12 per 1M tokens (≤200K prompt), 2026-09-28. Preview model id. Strongest published MT evidence (WMT25 lineage).
- **gemini-2.5-pro**: List price $1.25/$10 per 1M tokens (≤200K prompt). WMT25 best system overall (thinking mode).
- **claude-opus-5**: List price $5/$25 per 1M tokens (Anthropic list price).
- **claude-fable-5-1**: List price $10/$50 per 1M tokens.
- **claude-sonnet-5**: List price $2/$10 per 1M tokens.
- **gpt-6-astra**: List price $10/$50 per 1M tokens, 2026-09-28.
- **gpt-5.6-sol**: List price $4/$20 per 1M tokens, 2026-09-28.
- **gpt-5.5**: List price $5/$30 per 1M tokens (<272K context). The OpenAI version with published FLORES-200 MT numbers (Hy-MT2 report).
- **deepl-quality**: Dedicated MT, no duration budget. HI and JA are supported targets (HI: basic translation only). Price per 1M characters unverified (API Pro reportedly replaced by Developer/Growth plans in 2026-07): set price_per_mchar.
- **google-translate-nmt**: ONLINE-G-style reference; Cloud Translation Basic v2, $20 per 1M chars (search snippet of the pricing page).
- **microsoft-translator**: ONLINE-B-style commercial MT: no worker written yet.
- **homura-qwen3-8b**: HOMURA syllable-budget GRPO on Qwen3-8B (Bilibili): code and the Sand-Glass benchmark are research-use only; no weights released.
- **apptek-lcmt**: AppTek length-controlled NMT with iterative re-translation: proprietary, no public API.
- **microsoft-lsst-labs**: Length-Sensitive Speech Translation + Length-Aware Beam Search (Interspeech 2025): paper only, no weights or API.
- **hy-mt2-30b-bestofn-metricx**: Best-of-N with QE reranking (artifact's scaling lever): a method worker composing a generator and judge_mt_qe; not written, and MetricX as reranker + judge is circular (plan §4.7) — rerank with CometKiwi if built.
- **kimi-k3**: Moonshot Kimi K3 (2.8T open weights): cluster-scale, beyond the Spark.
- **glm-5.3**: Zhipu GLM-5.3 (744B-A40B, 756 GB FP8): cluster-scale, beyond the Spark; bespoke licence.

</details>

### C3

**Viseme-aware paraphrase (experimental)**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **do-nothing** (baseline) | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/c3_viseme_rerank.py](server/bench/arena/workers/c3_viseme_rerank.py) | `server` |
| **psc-claude-opus-5** | api | — | proprietary API | en hi ja +5 | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **psc-gemini-3.8-flash** | api | — | proprietary API | en hi ja +5 | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **psc-gpt-5.6-sol** | api | — | proprietary API | en hi ja +5 | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| **psc-gpt-6-astra** | api | — | proprietary API | en hi ja +5 | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| **psc-gpt-oss-120b** | method | 117 B | Apache-2.0 | en hi ja +5 | 70 GB VRAM | ✗ | ✓ | not run | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | `c2_vllm` |
| **psc-qwen3-14b** | method | 14.8 B | Apache-2.0 | en hi ja +5 | 14 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) | `c2_vllm` |
| ~~psc-glm-kimi~~ (disabled) | local | — | various | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/blog/glm-5-3-specs-benchmarks-api-ho…](https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/) | `server` |
| ~~ipa-lip-shape-selection~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~syncnet-rerank~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [robots.ox.ac.uk/~vgg/software/lipsync/](https://www.robots.ox.ac.uk/~vgg/software/lipsync/) | `server` |
| ~~vaq-phoneme-viseme~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~video-modifying-vendors~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |

<details><summary>Notes</summary>

- **do-nothing**: Returns the input line unchanged (C3 off, today's behaviour).
- **psc-qwen3-14b**: Plan seed: small generator + vowel-DTW scorer. Experimental (see specs/c3.py).
- **psc-gpt-oss-120b**: Spark pick generator.
- **psc-claude-opus-5**: API pick generator; $5/$25 per 1M tokens.
- **psc-gpt-6-astra**: $10/$50 per 1M tokens.
- **psc-gpt-5.6-sol**: $4/$20 per 1M tokens.
- **psc-gemini-3.8-flash**: Flash-tier generator; $0.75/$3.75 per 1M tokens until 2026-12-31.
- **syncnet-rerank**: SyncNet LSE-D/LSE-C reranking on rendered candidates: needs every candidate rendered through D1 + F1; offline calibration tool, not in the loop. Wav2Lip-bundled SyncNet is research-only.
- **vaq-phoneme-viseme**: Gupta et al. (WACV 2024 workshop) phoneme-viseme agreement score: paper only, licence unknown.
- **ipa-lip-shape-selection**: Lip-shape-aware word selection from the IPA chart (lyric translation, LNCS): paper only.
- **video-modifying-vendors**: Deepfake re-rendering vendors: changes the video (phase F), not the text.
- **psc-glm-kimi**: GLM-5.3 / Kimi K3 as generator: cluster-scale; the cost function, not the generator, decides the outcome.

</details>

### C4

**Reference-free MT quality estimation (meta-evaluation)**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **rules** (baseline) | local | — | MIT (OpenDub) | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/c4_rules.py](server/bench/arena/workers/c4_rules.py) | `server` |
| **cometkiwi-da-22** 🔒 | local | 0.56 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 3 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) | `judge_mt_qe` |
| **cometkiwi-da-xl** 🔒 | local | 3.5 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 9 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [Unbabel/wmt23-cometkiwi-da-xl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl) | `judge_mt_qe` |
| **cometkiwi-da-xxl** 🔒 | local | 10.7 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 24 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [Unbabel/wmt23-cometkiwi-da-xxl](https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl) | `judge_mt_qe` |
| **gemba-gpt-oss-120b** | local | 117 B | Apache-2.0 | all (✓ en hi ja) | 70 GB VRAM | ✗ | ✓ | not run | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | `c2_vllm` |
| **gemba-qwen3-14b-awq** | local | 14.8 B | Apache-2.0 | all (✓ en hi ja) | 14 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) | `c2_vllm` |
| **gemba-qwen3.8-27b-fp8** | local | 27 B | Apache-2.0 | all (✓ en hi ja) | 44 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `c2_vllm` |
| **metricx-24-hybrid-large-qe** | local | 1.2 B | Apache-2.0 | all (✓ en hi ja) | 5 GB VRAM | ✓ | ✓ | not run | [google/metricx-24-hybrid-large-v2p6](https://huggingface.co/google/metricx-24-hybrid-large-v2p6) | `judge_mt_qe` |
| **metricx-24-hybrid-xl-qe** | local | 3.7 B | Apache-2.0 | all (✓ en hi ja) | 9 GB VRAM | ✓ | ✓ | not run | [google/metricx-24-hybrid-xl-v2p6](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6) | `judge_mt_qe` |
| **metricx-24-hybrid-xxl-qe** | local | 13 B | Apache-2.0 | all (✓ en hi ja) | 28 GB VRAM | ✗ | ✓ | not run | [google/metricx-24-hybrid-xxl-v2p6-bfloat16](https://huggingface.co/google/metricx-24-hybrid-xxl-v2p6-bfloat16) | `judge_mt_qe` |
| **xcomet-xl-qe** 🔒 | local | 3.5 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 9 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [Unbabel/XCOMET-XL](https://huggingface.co/Unbabel/XCOMET-XL) | `judge_mt_qe` |
| **xcomet-xxl-qe** 🔒 | local | 10.7 B | CC-BY-NC-SA-4.0 · NC | all (✓ en hi ja) | 24 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) | `judge_mt_qe` |
| **gemba-claude-fable-5-1** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **gemba-claude-opus-5** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **gemba-gemini-2.5-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **gemba-gemini-3.1-pro** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **gemba-gpt-6-astra** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| ~~gemba-glm-kimi~~ (disabled) | local | — | various | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/blog/glm-5-3-specs-benchmarks-api-ho…](https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/) | `server` |
| ~~gemspaneval~~ (disabled) | local | — | Gemma Terms of Use | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.wmt-1.70/](https://aclanthology.org/2025.wmt-1.70/) | `server` |
| ~~metricx-25-qe~~ (disabled) | local | — | Gemma Terms of Use | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.wmt-1.70/](https://aclanthology.org/2025.wmt-1.70/) | `server` |
| ~~autorank-ensemble~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.wmt-1.22/](https://aclanthology.org/2025.wmt-1.22/) | `server` |
| ~~taser-no-ref-o3~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.wmt-1.22/](https://aclanthology.org/2025.wmt-1.22/) | `server` |

<details><summary>Notes</summary>

- **rules**: Empty/copy/wrong-script/repetition/length-ratio checks; the floor to beat (OpenDub has no QE today).
- **metricx-24-hybrid-large-qe**: google/metricx-24-hybrid-large-v2p6 in QE mode (1.2B).
- **metricx-24-hybrid-xl-qe**: google/metricx-24-hybrid-xl-v2p6 in QE mode (3.7B); also C2's primary judge.
- **metricx-24-hybrid-xxl-qe**: Spark pick: -bfloat16 checkpoint (~13B, 26 GB); measure its rank agreement with XL once on the Spark.
- **metricx-25-qe**: MetricX-25 (Gemma 3 12B encoder): Google-internal WMT25 submission, no public checkpoint.
- **cometkiwi-da-22**: Unbabel/wmt22-cometkiwi-da (InfoXLM-large). Gated (auto-accept); eval-only.
- **cometkiwi-da-xl**: Unbabel/wmt23-cometkiwi-da-xl (3.5B). Gated; eval-only.
- **cometkiwi-da-xxl**: Unbabel/wmt23-cometkiwi-da-xxl (10.7B): Spark. Gated; eval-only.
- **xcomet-xl-qe**: Unbabel/XCOMET-XL without reference (QE mode). Gated; eval-only; Unbabel require contact for commercial use.
- **xcomet-xxl-qe**: Unbabel/XCOMET-XXL (10.7B, ~21 GB bf16) QE mode: Spark. Gated; eval-only.
- **gemspaneval**: GemSpanEval (Gemma 3 27B fine-tuned for error spans): weights not released; a recipe to reproduce.
- **gemba-gemini-3.1-pro**: List price $2/$12 per 1M tokens (≤200K prompt), 2026-09-28. GEMBA-ESA (single-call variant: spans + 0-100 score).
- **gemba-gemini-2.5-pro**: List price $1.25/$10 per 1M tokens (≤200K prompt).
- **gemba-claude-opus-5**: List price $5/$25 per 1M tokens (Anthropic list price).
- **gemba-claude-fable-5-1**: List price $10/$50 per 1M tokens.
- **gemba-gpt-6-astra**: List price $10/$50 per 1M tokens, 2026-09-28.
- **gemba-qwen3-14b-awq**: AWQ int4 (~9.5 GB weights) on vLLM with thinking disabled; vLLM reserves 85% of the card. Card: 100+ languages incl. hi/ja. Plan seed: local GEMBA with a 12–14B model.
- **gemba-qwen3.8-27b-fp8**: Dense 27B VL (hybrid Gated DeltaNet), 262K context; Spark (FP8 official). The artifact's NVFP4 recipe is faster but NVFP4 on GB10 is immature. The card lists no languages: hi/ja unverified. enable_thinking assumed to follow Qwen3.
- **gemba-gpt-oss-120b**: 117B/5.1B-active MXFP4 (~63 GB); Spark only. English-centric, no published multilingual MT numbers (artifact: better as critic than translator).
- **taser-no-ref-o3**: TASER-No-Ref (WMT25): reasoning-model prompting over OpenAI o3; prompt not reproduced here.
- **autorank-ensemble**: AUTORANK-style ensemble (MetricX + span model + GEMBA, rescaled and averaged): a method over other C4 outputs; not written.
- **gemba-glm-kimi**: GLM-5.3 / Kimi K3 as judge: cluster-scale.

</details>

### C5

**Text normalization, G2P and lexicon**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **charsiu-byt5-small** | local | 0.3 B | unknown (code MIT) | — +3 | 1.5 GB VRAM | ✓ | ✓ | not run | [charsiu/g2p_multilingual_byT5_small_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100) | `c5_charsiu` |
| **epitran** | local | — | MIT | — +2 | CPU | ✓ | ✓ | not run | [github.com/dmort27/epitran](https://github.com/dmort27/epitran) | `c5_epitran` |
| **espeak-ng** | local | — | GPL-3.0 | — +3 | +espeak-ng | ✗ | ✗ | not run | [github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) | `server` |
| **misaki** | local | — | Apache-2.0 | — +2 | CPU | ✓ | ✓ | not run | [github.com/hexgrad/misaki](https://github.com/hexgrad/misaki) | `c5_misaki` |
| **nemo-tn** | local | — | Apache-2.0 | — +3 | 4 GB RAM, x86_64 | ✓ | ✗ | not run | [github.com/NVIDIA/NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) | `c5_nemo` |
| **pyopenjtalk** | local | — | MIT (OpenJTalk: Modified BSD) | — +1 | CPU | ✓ | ✓ | not run | [github.com/r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk) | `c5_misaki` |
| **qwen3-14b-awq-arbiter** | local | 14.8 B | Apache-2.0 | — +7 | 14 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) | `c2_vllm` |
| **qwen3.8-27b-fp8-arbiter** | local | 27 B | Apache-2.0 | — +7 | 44 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `c2_vllm` |
| **claude-fable-5-1-c5** | api | — | proprietary API | — +7 | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **claude-opus-5-c5** | api | — | proprietary API | — +7 | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **gemini-3.1-pro-c5** | api | — | proprietary API | — +7 | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **gpt-6-astra-c5** | api | — | proprietary API | — +7 | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| ~~c5-glm-kimi~~ (disabled) | local | — | various | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/blog/glm-5-3-specs-benchmarks-api-ho…](https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/) | `server` |
| ~~crf-dict-g2p~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2609.19805](https://arxiv.org/abs/2609.19805) | `server` |
| ~~nemo-g2p-conformer-ctc~~ (disabled) | local | — | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [docs.nvidia.com/nemo/speech/nightly/tts/g2p.html](https://docs.nvidia.com/nemo/speech/nightly/tts/g2p.html) | `server` |
| ~~openphonemizer~~ (disabled) | local | — | BSD-3-Clause-Clear | all (✓ en hi ja) | CPU | — | — | not run | [pypi.org/project/openphonemizer/](https://pypi.org/project/openphonemizer/) | `server` |
| ~~ur-bert~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2606.11681](https://arxiv.org/abs/2606.11681) | `server` |
| ~~backend-lexicons~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [learn.microsoft.com/en-us/azure/ai-services/s…](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation) | `server` |
| ~~orthographic-respelling~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [elevenlabs.io/docs/best-practices/prompting/p…](https://elevenlabs.io/docs/best-practices/prompting/pronunciation) | `server` |

<details><summary>Notes</summary>

- **nemo-tn**: nemo-text-processing 1.2.0 WFST (CPU); hi TN new in 1.2.0. x86_64 only (pynini wheels); grammars compile on first use.
- **misaki**: Fallback disabled (OOV = error). No Hindi. Unmaintained since 2025-08.
- **espeak-ng**: Shelled out, never linked (GPL). Install espeak-ng (not on Thalassa yet).
- **charsiu-byt5-small**: ~300M ByT5; 100 languages incl. hi. Model card states no licence. Frozen (weights 2022). Language codes eng-us/hin/jpn assumed.
- **epitran**: Rule-based hin-Deva / jpn-Hira.
- **pyopenjtalk**: misaki's Japanese backend; katakana readings.
- **qwen3-14b-awq-arbiter**: AWQ int4 (~9.5 GB weights) on vLLM with thinking disabled; vLLM reserves 85% of the card. Card: 100+ languages incl. hi/ja. Local 8-30B arbiter (here: every item, not only flagged homographs).
- **qwen3.8-27b-fp8-arbiter**: Dense 27B VL (hybrid Gated DeltaNet), 262K context; Spark (FP8 official). The artifact's NVFP4 recipe is faster but NVFP4 on GB10 is immature. The card lists no languages: hi/ja unverified. enable_thinking assumed to follow Qwen3.
- **claude-opus-5-c5**: List price $5/$25 per 1M tokens (Anthropic list price).
- **claude-fable-5-1-c5**: List price $10/$50 per 1M tokens.
- **gemini-3.1-pro-c5**: List price $2/$12 per 1M tokens (≤200K prompt), 2026-09-28.
- **gpt-6-astra-c5**: List price $10/$50 per 1M tokens, 2026-09-28.
- **openphonemizer**: Dropped by the 2026-09-18 pass: English-only, last release 2024-03.
- **nemo-g2p-conformer-ctc**: NeMo G2P-Conformer-CTC: needs the full NeMo toolkit; English checkpoints only; worker not written.
- **backend-lexicons**: Azure PLS / ElevenLabs pronunciation dictionaries / Polly PLS: delivery mechanisms for a lexicon, not predictors; exercised in D-phase.
- **orthographic-respelling**: Respelling (the only override F5-TTS accepts): a delivery technique, evaluated with D1 round-trip, not here.
- **ur-bert**: UR-BERT universal romanization (Interspeech 2026): weights not verified.
- **crf-dict-g2p**: Dictionary-constrained CRF G2P (2026 paper): no released model.
- **c5-glm-kimi**: GLM-5.3 / Kimi K3 as arbiter: cluster-scale.

</details>

### C6

**Subtitle condensation**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **wrap-only** (baseline) | local | — | MIT (OpenDub) | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/c6_rules.py](server/bench/arena/workers/c6_rules.py) | `server` |
| **hwtsc-gpt-oss-120b** | local | 117 B | Apache-2.0 | en hi ja +5 | 70 GB VRAM | ✗ | ✓ | not run | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | `c2_vllm` |
| **hwtsc-qwen3-14b-awq** | local | 14.8 B | Apache-2.0 | en hi ja +5 | 14 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) | `c2_vllm` |
| **hwtsc-qwen3-32b-fp8** | local | 32.8 B | Apache-2.0 | en hi ja +5 | 44 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8) | `c2_vllm` |
| **hwtsc-qwen3-coder-30b-a3b-fp8** | local | 30.5 B | Apache-2.0 | en hi ja +5 | 40 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) | `c2_vllm` |
| **claude-opus-5-condense** | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | ✓ | ✓ | not run | [platform.claude.com/docs/en/about-claude/mode…](https://platform.claude.com/docs/en/about-claude/models/overview) | `c2_api` |
| **gemini-3.8-flash-condense** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `c2_api` |
| **gpt-5.6-sol-condense** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| **gpt-6-astra-condense** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models](https://developers.openai.com/api/docs/models) | `c2_api` |
| ~~c6-glm-kimi~~ (disabled) | local | — | various | all (✓ en hi ja) | CPU | — | — | not run | [kingy.ai/blog/glm-5-3-specs-benchmarks-api-ho…](https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/) | `server` |
| ~~sbaam~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [github.com/hlt-mt/FBK-fairseq](https://github.com/hlt-mt/FBK-fairseq) | `server` |
| ~~apptek-lcmt-retranslate~~ (disabled) | method | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2025.iwslt-1.21/](https://aclanthology.org/2025.iwslt-1.21/) | `server` |
| ~~apptek-subtitling-stack~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [aclanthology.org/2026.iwslt-1.39/](https://aclanthology.org/2026.iwslt-1.39/) | `server` |
| ~~delivery-spec-profiles~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [iwslt.org/2026/subtitling](https://iwslt.org/2026/subtitling) | `server` |
| ~~suber-metric~~ (disabled) | method | — | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [github.com/apptek/SubER](https://github.com/apptek/SubER) | `server` |

<details><summary>Notes</summary>

- **wrap-only**: Greedy re-wrap to CPL, no condensation (no C6 provider today).
- **hwtsc-qwen3-32b-fp8**: HW-TSC IWSLT 2026 two-pass condenser (T=0 then T=0.3, violating cues only). FP8 instead of the artifact's Q8_0.
- **hwtsc-qwen3-14b-awq**: AWQ int4 (~9.5 GB weights) on vLLM with thinking disabled; vLLM reserves 85% of the card. Card: 100+ languages incl. hi/ja. Plan seed: Qwen3-14B two-pass condenser.
- **hwtsc-gpt-oss-120b**: 117B/5.1B-active MXFP4 (~63 GB); Spark only. English-centric, no published multilingual MT numbers (artifact: better as critic than translator).
- **hwtsc-qwen3-coder-30b-a3b-fp8**: Artifact's headroom option (44 tok/s on a Spark). Repo id unverified.
- **claude-opus-5-condense**: List price $5/$25 per 1M tokens (Anthropic list price).
- **gpt-6-astra-condense**: List price $10/$50 per 1M tokens, 2026-09-28.
- **gpt-5.6-sol-condense**: List price $4/$20 per 1M tokens, 2026-09-28.
- **gemini-3.8-flash-condense**: List price $0.75/$3.75 per 1M tokens until 2026-12-31, then $1.50/$7.50.
- **apptek-subtitling-stack**: AppTek full stack (length-class MT + ILS line segmentation + LLM APE): proprietary; IWSLT 2026 winner.
- **apptek-lcmt-retranslate**: AppTek-style length-class-conditioned MT with iterative re-translation: needs training.
- **sbaam**: SBAAM direct speech-to-subtitle (FBK): audio input, different task; FBK-fairseq licence unverified.
- **suber-metric**: SubER is the C6 primary metric (c6_suber_judge), not a candidate.
- **delivery-spec-profiles**: Netflix/IWSLT constraint profiles: configuration, implemented as specs/c6.py LIMITS.
- **c6-glm-kimi**: GLM-5.3 / Kimi K3 as condenser: cluster-scale.

</details>

## D — Voice

### D1

**Zero-shot cross-lingual voice cloning**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **f5-hindi-small** (baseline) | local | 0.16 B | CC-BY-4.0 | hi | 2 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **f5-v1-base** (baseline) | local | 0.34 B | CC-BY-NC-4.0 · NC | en +1 | 3 GB VRAM | ✓ | ✓ | not run | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | `server` |
| **chatterbox-multilingual** | local | 0.5 B | MIT | en hi ja +20 | 5 GB VRAM | ✓ | ✓ | not run | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | `d1_chatterbox` |
| **cosyvoice3-0.5b** | local | 0.5 B | Apache-2.0 | en ja +7 | 5 GB VRAM | ✓ | ✓ | not run | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | `d2_cosyvoice3` |
| **f5-hindi-small-translit** | local | 0.16 B | CC-BY-4.0 | hi | 2 GB VRAM, +ollama | ✗ | ✗ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **fish-s2-pro** | local | 4.56 B | Fish-Audio-Research-License (NC) · NC | en hi ja +80 | 24 GB VRAM | ✗ | ✓ | not run | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | `d1_fish_speech` |
| **higgs-audio-v3-4b** | local | 4 B | Boson Higgs TTS 3 Research and Non-Commercial License · NC | en hi ja +12 | 24 GB VRAM | ✗ | ✓ | not run | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | `d1_higgs_audio` |
| **indextts-2.5** | local | 0.8 B | bilibili-model-license · NC | en ja +3 | 8 GB VRAM | ✓ | ✓ | not run | [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | `d1_indextts` |
| **indicf5** 🔒 | local | 0.4 B | MIT | hi +10 | 3 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | `d4_indicf5` |
| **moss-tts-local-v1.5-4b** | local | 4.55 B | Apache-2.0 | en hi ja +28 | 11 GB VRAM | ✓ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | `d1_moss_tts` |
| **moss-tts-v1.5-8b** | local | 8.49 B | Apache-2.0 | en hi ja +28 | 20 GB VRAM | ✗ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | `d1_moss_tts` |
| **omnivoice** | local | 0.61 B | code Apache-2.0; weights CC-BY-NC · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | `d4_omnivoice` |
| **qwen3-tts-0.6b-base** | local | 0.6 B | Apache-2.0 | en ja +8 | 3.5 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | `d1_qwen3_tts` |
| **qwen3-tts-1.7b-base** | local | 1.7 B | Apache-2.0 | en ja +8 | 6.5 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | `d1_qwen3_tts` |
| **step-audio-editx** | local | 3 B | Apache-2.0 | en ja +2 | 15 GB VRAM, x86_64 | ✓ | ✗ | not run | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | `d2_step_audio_editx` |
| **voxcpm2** | local | 2 B | Apache-2.0 | en hi ja +27 | 9 GB VRAM | ✓ | ✓ | not run | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `d1_voxcpm` |
| **cartesia-sonic-3.6** | api | — | proprietary API | en hi ja +12 | key CARTESIA | ✓ | ✓ | not run | [docs.cartesia.ai/build-with-cartesia/tts-mode…](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) | `server` |
| **elevenlabs-multilingual-v2** | api | — | proprietary API | en hi ja +26 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/text-to-spee…](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | `server` |
| **elevenlabs-v3** | api | — | proprietary API | en hi ja +24 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/text-to-spee…](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | `server` |
| **f5-v1-bestof5** | method | 0.34 B | CC-BY-NC-4.0 · NC | en +1 | 5 GB VRAM | ✓ | ✓ | not run | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | `server` |
| **fish-audio-api-s2.1-pro** | api | — | proprietary API | en hi ja +80 | key FISH_AUDIO | ✓ | ✓ | not run | [docs.fish.audio/api-reference/endpoint/openap…](https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech) | `server` |
| **fish-s2-pro-bestof32** | method | 4.56 B | Fish-Audio-Research-License (NC) · NC | en hi ja +80 | 27 GB VRAM | ✗ | ✓ | not run | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | `d1_fish_speech` |
| **minimax-speech-2.8-hd** | api | — | proprietary API | en hi ja +9 | key MINIMAX | ✓ | ✓ | not run | [platform.minimax.io/docs/api-reference/speech…](https://platform.minimax.io/docs/api-reference/speech-t2a-http) | `server` |
| **minimax-speech-2.8-turbo** | api | — | proprietary API | en hi ja +9 | key MINIMAX | ✓ | ✓ | not run | [platform.minimax.io/docs/api-reference/speech…](https://platform.minimax.io/docs/api-reference/speech-t2a-http) | `server` |
| ~~confucius4-tts~~ (disabled) | local | — | CC-BY-4.0 | all (✓ en hi ja) | CPU | — | — | not run | [github.com/netease-youdao](https://github.com/netease-youdao) | `server` |
| ~~dots-tts~~ (disabled) | local | 2 B | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [models?search=dots.tts](https://huggingface.co/models?search=dots.tts) | `server` |
| ~~qwen-audio-3.0-tts-plus~~ (disabled) | api | — | proprietary API | en ja +8 | key DASHSCOPE | — | — | not run | [openrouter.ai/qwen/qwen-audio-3.0-tts-plus](https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus) | `server` |
| ~~resemble-clone~~ (disabled) | api | — | proprietary API | en | key RESEMBLE | — | — | not run | [docs.resemble.ai](https://docs.resemble.ai) | `server` |

<details><summary>Notes</summary>

- **f5-v1-base**: Stock F5-TTS v1 Base (en/zh vocab); single take. The app provider adds Whisper verification + speed retries (see the best-of-N row).
- **f5-hindi-small**: SPRINGLab F5-Hindi-24KHz (F5TTS_Small config, as in the app provider's placeholder). Non-Devanagari reference transcripts are outside its vocab: see the translit row.
- **f5-hindi-small-translit**: SPRINGLab F5-Hindi-24KHz (F5TTS_Small config, as in the app provider's placeholder). Non-Devanagari reference transcripts are outside its vocab: see the translit row. Reference transcript spelled in Devanagari by a local LLM (Ollama gemma3:12b), the provider's translit option; needs `ollama` running.
- **f5-v1-bestof5**: Method: N takes (item seed + k) re-ranked by a large-v3-turbo verifier (not the reporting large-v3/Qwen3/MMS panel) + duration fit. VRAM = inner + ~2 GB verifier. Closest arena analogue of the app's verified F5 (5 takes, Whisper re-read).
- **qwen3-tts-1.7b-base**: No Hindi. SDPA attention (no FA3 on sm_120; FA2 optional). VRAM estimated from params (compute-tiers: ~6 GB bf16 for 1.7B). Research primary (Sept 3); the only D1 model with published GB10 numbers.
- **qwen3-tts-0.6b-base**: No Hindi. SDPA attention (no FA3 on sm_120; FA2 optional). VRAM estimated from params (compute-tiers: ~6 GB bf16 for 1.7B).
- **voxcpm2**: ~8 GB on a 4090 per the README; 48 kHz output; no language argument (the model reads the script). prompt_wav + prompt_text + reference_wav (README max-similarity setting).
- **fish-s2-pro**: fishaudio/s2-pro @1de9996. Docs recommend ≥24 GB → Spark. SGLang fast path unverified on aarch64/sm_121: plain PyTorch api_server. Compute-tiers Spark pick.
- **fish-s2-pro-bestof32**: Method: N takes (item seed + k) re-ranked by a large-v3-turbo verifier (not the reporting large-v3/Qwen3/MMS panel) + duration fit. VRAM = inner + ~2 GB verifier. Compute-tiers 'Unconstrained' pick (S2 at N=32 with cross-family re-ranking).
- **confucius4-tts**: Weights reported released (CC BY 4.0, NetEase Youdao, 2026-08) but no inference API was verified; add a worker once the repo is checked.
- **indextts-2.5**: IndexTTS 2.5 (2026-08-10, 0.8B): lang ZH/EN/JA/ES/AR, no Hindi. bilibili Model Use License: commercial terms need reading → ship_ok false. VRAM estimated (T2S + S2M + BigVGAN + w2v-BERT + Qwen emo).
- **dots-tts**: dots.tts (2B, Apache-2.0, evaluate-only in the research ledger): repo and inference API not verified; no worker yet.
- **minimax-speech-2.8-hd**: T2A fields verified on platform.minimax.io; clone endpoints and 2.8 prices not re-verified.
- **minimax-speech-2.8-turbo**: T2A fields verified on platform.minimax.io; clone endpoints and 2.8 prices not re-verified.
- **elevenlabs-v3**: Mirrors app/providers/tts/elevenlabs.py; IVC voices are deleted at job end. Prices in _cost_usd are estimates.
- **elevenlabs-multilingual-v2**: Mirrors app/providers/tts/elevenlabs.py; IVC voices are deleted at job end. Prices in _cost_usd are estimates. Today's tts.elevenlabs default model.
- **moss-tts-v1.5-8b**: 8.49B params (HF safetensors total); VRAM = bf16 weights + codec, estimated. ~17 GB bf16 → Spark. Loses to smaller siblings on SIM per the artifact.
- **moss-tts-local-v1.5-4b**: 4.55B params (HF safetensors total); VRAM = bf16 weights + codec, estimated.
- **higgs-audio-v3-4b**: 100+ languages per the card (listed: the in-scope subset). Served by vLLM-Omni (the card cites ≥24 GB reported, 40 GB for SGLang-Omni) → Spark. Request fields from the repo's AGENTS.md; not run yet.
- **chatterbox-multilingual**: chatterbox-tts 0.1.7 installed --no-deps over torch 2.11 (it pins torch 2.6, no sm_120 kernels).
- **cosyvoice3-0.5b**: Cross-lingual path for foreign references, zero-shot for same-language; ja text converted to katakana (pyopenjtalk) as the example requires. No Hindi.
- **indicf5**: Gated repo (HF_TOKEN).
- **omnivoice**: 600+ languages per the card; language inferred from text (no documented language arg).
- **step-audio-editx**: x86_64 only (onnxruntime-gpu/deepspeed deps). 12–16 GB: tight on Thalassa.
- **qwen-audio-3.0-tts-plus**: Compute-tiers API pick. DashScope model id and voice-enrollment target are unverified (OpenRouter exposes it without cloning) → disabled until checked in the console.
- **cartesia-sonic-3.6**: Sonic 3.6 (44 languages, $49/M chars per Cartesia, 2026-08). Clone endpoint not re-verified.
- **fish-audio-api-s2.1-pro**: Inline references are msgpack-only (docs.fish.audio); price not re-verified.
- **resemble-clone**: Per-item instant cloning over the API is not wired (rapid clones are made in the Resemble app); disabled.

</details>

### D2

**Expressive and style-transfer synthesis**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **f5-hindi-small** (baseline) | local | 0.16 B | CC-BY-4.0 | hi | 2 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **f5-v1-base** (baseline) | local | 0.34 B | CC-BY-NC-4.0 · NC | en +1 | 3 GB VRAM | ✓ | ✓ | not run | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | `server` |
| **cosyvoice3-instruct** | local | 0.5 B | Apache-2.0 | en ja +7 | 5 GB VRAM | ✓ | ✓ | not run | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | `d2_cosyvoice3` |
| **fish-s2-pro-tags** | local | 4.56 B | Fish-Audio-Research-License (NC) · NC | en hi ja +80 | 24 GB VRAM | ✗ | ✓ | not run | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | `d1_fish_speech` |
| **higgs-audio-v3-4b-tags** | local | 4 B | Boson Higgs TTS 3 Research and Non-Commercial License · NC | en hi ja +12 | 24 GB VRAM | ✗ | ✓ | not run | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | `d1_higgs_audio` |
| **indextts-2-emo-audio** | local | 1.5 B | bilibili-model-license · NC | en +1 | 10 GB VRAM | ✓ | ✓ | not run | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | `d1_indextts` |
| **indextts-2.5-emo-audio** | local | 0.8 B | bilibili-model-license · NC | en ja +3 | 8 GB VRAM | ✓ | ✓ | not run | [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | `d1_indextts` |
| **moss-tts-v1.5-8b** | local | 8.49 B | Apache-2.0 | en hi ja +28 | 20 GB VRAM | ✗ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | `d1_moss_tts` |
| **step-audio-editx-awq-emotion** | local | 3 B | Apache-2.0 | en ja +2 | 8 GB VRAM, x86_64 | ✓ | ✗ | not run | [stepfun-ai/Step-Audio-EditX-AWQ-4bit](https://huggingface.co/stepfun-ai/Step-Audio-EditX-AWQ-4bit) | `d2_step_audio_editx` |
| **step-audio-editx-emotion** | local | 3 B | Apache-2.0 | en ja +2 | 15 GB VRAM, x86_64 | ✓ | ✗ | not run | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | `d2_step_audio_editx` |
| **voxcpm2-instruct** | local | 2 B | Apache-2.0 | en hi ja +27 | 9 GB VRAM | ✓ | ✓ | not run | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `d1_voxcpm` |
| **cartesia-sonic-3.6** | api | — | proprietary API | en hi ja +12 | key CARTESIA | ✓ | ✓ | not run | [docs.cartesia.ai/build-with-cartesia/tts-mode…](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) | `server` |
| **elevenlabs-v3-tags** | api | — | proprietary API | en hi ja +24 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/text-to-spee…](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | `server` |
| **hume-octave-2** | api | — | proprietary API | en hi ja +8 | key HUME | ✓ | ✓ | not run | [dev.hume.ai/docs/text-to-speech-tts/overview](https://dev.hume.ai/docs/text-to-speech-tts/overview) | `server` |
| ~~emovoice~~ (disabled) | local | 1.5 B | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2504.12867](https://arxiv.org/abs/2504.12867) | `server` |
| ~~emosteer-ted-tts~~ (disabled) | method | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2508.03543](https://arxiv.org/abs/2508.03543) | `server` |
| ~~qwen-audio-3.0-tts-plus~~ (disabled) | api | — | proprietary API | en ja +8 | key OPENROUTER | — | — | not run | [openrouter.ai/qwen/qwen-audio-3.0-tts-plus](https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus) | `server` |

<details><summary>Notes</summary>

- **f5-v1-base**: Stock F5-TTS v1 Base (en/zh vocab); single take. The app provider adds Whisper verification + speed retries (see the best-of-N row). Conditions on the line's own source audio (the app's segment_style).
- **f5-hindi-small**: SPRINGLab F5-Hindi-24KHz (F5TTS_Small config, as in the app provider's placeholder). Non-Devanagari reference transcripts are outside its vocab: see the translit row.
- **indextts-2.5-emo-audio**: IndexTTS 2.5 (2026-08-10, 0.8B): lang ZH/EN/JA/ES/AR, no Hindi. bilibili Model Use License: commercial terms need reading → ship_ok false. VRAM estimated (T2S + S2M + BigVGAN + w2v-BERT + Qwen emo). Source line doubles as the emotion prompt (emo_audio_prompt).
- **indextts-2-emo-audio**: IndexTTS2: zh/en only. bilibili Model Use License: commercial terms need reading → ship_ok false. VRAM estimated (T2S + S2M + BigVGAN + w2v-BERT + Qwen emo).
- **step-audio-editx-emotion**: Clone, then an emotion edit pass when inputs.emotion is one of its 14 labels. x86 only.
- **step-audio-editx-awq-emotion**: AWQ 4-bit (6–8 GB per the README): the Thalassa-safe entry (checkpoint fetched by the env recipe).
- **fish-s2-pro-tags**: fishaudio/s2-pro @1de9996. Docs recommend ≥24 GB → Spark. SGLang fast path unverified on aarch64/sm_121: plain PyTorch api_server. [<emotion>] free-form tags.
- **elevenlabs-v3-tags**: Mirrors app/providers/tts/elevenlabs.py; IVC voices are deleted at job end. Prices in _cost_usd are estimates.
- **voxcpm2-instruct**: ~8 GB on a 4090 per the README; 48 kHz output; no language argument (the model reads the script). (<emotion>) instruction prefix from inputs.emotion.
- **cosyvoice3-instruct**: Compute-tiers says drop (no expressive gain from scale); kept as the niche row.
- **emovoice**: EmoVoice (1.5B, ACM MM 2025): licence unknown, no worker.
- **emosteer-ted-tts**: EmoSteer-TTS / TED-TTS: training-free steering layers over a base model; code availability unverified — a method worker over a D1 model later.
- **higgs-audio-v3-4b-tags**: 100+ languages per the card (listed: the in-scope subset). Served by vLLM-Omni (the card cites ≥24 GB reported, 40 GB for SGLang-Omni) → Spark. Request fields from the repo's AGENTS.md; not run yet. <\|emotion:x\|> inline control tokens.
- **moss-tts-v1.5-8b**: 8.49B params (HF safetensors total); VRAM = bf16 weights + codec, estimated.
- **qwen-audio-3.0-tts-plus**: OpenRouter exposes two preset voices and no instruction field: it cannot take the source line's emotion → disabled for D2 (kept in D5/D7).
- **cartesia-sonic-3.6**: Sonic 3.6 (44 languages, $49/M chars per Cartesia, 2026-08). Clone endpoint not re-verified. Infers emotional subtext from the transcript (no tags).
- **hume-octave-2**: Acting instructions from inputs.emotion; designed voice (no clone). Endpoint and languages not re-verified.

</details>

### D3

**Duration-controlled synthesis**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **f5-hindi-free** (baseline) | local | 0.16 B | CC-BY-4.0 | hi | 2 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **f5-v1-free** (baseline) | local | 0.34 B | CC-BY-NC-4.0 · NC | en +1 | 3 GB VRAM | ✓ | ✓ | not run | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | `server` |
| **f5-hindi-fixdur** | local | 0.16 B | CC-BY-4.0 | hi | 2 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **f5-v1-fixdur** | local | 0.34 B | CC-BY-NC-4.0 · NC | en +1 | 3 GB VRAM | ✓ | ✓ | not run | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | `server` |
| **indextts-2.5-duration** | local | 0.8 B | bilibili-model-license · NC | en ja +3 | 8 GB VRAM | ✓ | ✓ | not run | [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | `d1_indextts` |
| **moss-tts-local-v1.0-1.7b-tokens** | local | 3.06 B | Apache-2.0 | en ja +18 | 8 GB VRAM | ✓ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-Local-Transformer](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) | `d1_moss_tts` |
| **moss-tts-local-v1.5-4b-tokens** | local | 4.55 B | Apache-2.0 | en hi ja +28 | 11 GB VRAM | ✓ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | `d1_moss_tts` |
| **moss-tts-v1.5-8b-tokens** | local | 8.49 B | Apache-2.0 | en hi ja +28 | 20 GB VRAM | ✗ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | `d1_moss_tts` |
| **omnivoice-duration** | local | 0.61 B | code Apache-2.0; weights CC-BY-NC · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | `d4_omnivoice` |
| **azure-mai-voice-2-rate** | api | — | proprietary API | en hi | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/en-us/azure/ai-services/s…](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices) | `server` |
| **chirp3-hd-rate** | api | — | proprietary API | en hi ja | key GOOGLE | ✓ | ✓ | not run | [cloud.google.com/text-to-speech/docs/chirp3-hd](https://cloud.google.com/text-to-speech/docs/chirp3-hd) | `server` |
| **f5-hindi-fixdur-bestof4** | method | 0.16 B | CC-BY-4.0 | hi | 4 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **f5-v1-fixdur-bestof4** | method | 0.34 B | CC-BY-NC-4.0 · NC | en +1 | 5 GB VRAM | ✓ | ✓ | not run | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | `server` |
| **indextts-2.5-bestof16** | method | 0.8 B | bilibili-model-license · NC | en ja +3 | 10 GB VRAM | ✓ | ✓ | not run | [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) | `d1_indextts` |
| ~~magic-tts~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/](https://arxiv.org/) | `server` |
| ~~maskgct~~ (disabled) | local | — | CC-BY-NC-4.0 · NC | en ja +4 | CPU | — | — | not run | [amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT) | `server` |
| ~~ps-tts~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/](https://arxiv.org/) | `server` |
| ~~voicestar~~ (disabled) | local | — | unknown · NC | en | CPU | — | — | not run | [github.com/jasonppy/VoiceStar](https://github.com/jasonppy/VoiceStar) | `server` |
| ~~elevenlabs-dubbing-control~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [elevenlabs.io/docs/api-reference/dubbing/create](https://elevenlabs.io/docs/api-reference/dubbing/create) | `server` |
| ~~phoneme-tsm~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [github.com/NidhiBharani/OpenDub/blob/main/doc…](https://github.com/NidhiBharani/OpenDub/blob/main/docs/plans/model-ranking.md) | `server` |
| ~~ted-tts~~ (disabled) | method | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/](https://arxiv.org/) | `server` |

<details><summary>Notes</summary>

- **f5-v1-free**: Stock F5-TTS v1 Base (en/zh vocab); single take. The app provider adds Whisper verification + speed retries (see the best-of-N row). No duration control (use_target_duration false): today's path, where E1 stretches the take afterwards.
- **f5-v1-fixdur**: Stock F5-TTS v1 Base (en/zh vocab); single take. The app provider adds Whisper verification + speed retries (see the best-of-N row). F5 fix_duration = prompt + target_s.
- **f5-hindi-free**: SPRINGLab F5-Hindi-24KHz (F5TTS_Small config, as in the app provider's placeholder). Non-Devanagari reference transcripts are outside its vocab: see the translit row.
- **f5-hindi-fixdur**: SPRINGLab F5-Hindi-24KHz (F5TTS_Small config, as in the app provider's placeholder). Non-Devanagari reference transcripts are outside its vocab: see the translit row.
- **f5-v1-fixdur-bestof4**: Method: N takes (item seed + k) re-ranked by a large-v3-turbo verifier (not the reporting large-v3/Qwen3/MMS panel) + duration fit. VRAM = inner + ~2 GB verifier. Rejection sampling on the duration window + verifier CER.
- **f5-hindi-fixdur-bestof4**: Method: N takes (item seed + k) re-ranked by a large-v3-turbo verifier (not the reporting large-v3/Qwen3/MMS panel) + duration fit. VRAM = inner + ~2 GB verifier. Rejection sampling on the duration window + verifier CER.
- **indextts-2.5-duration**: IndexTTS 2.5 (2026-08-10, 0.8B): lang ZH/EN/JA/ES/AR, no Hindi. bilibili Model Use License: commercial terms need reading → ship_ok false. VRAM estimated (T2S + S2M + BigVGAN + w2v-BERT + Qwen emo). Two passes: measure, then duration_factor = d0/target (README speed factor).
- **indextts-2.5-bestof16**: Method: N takes (item seed + k) re-ranked by a large-v3-turbo verifier (not the reporting large-v3/Qwen3/MMS panel) + duration fit. VRAM = inner + ~2 GB verifier. Compute-tiers Unconstrained pick (N=16–32 with duration-window rejection).
- **moss-tts-local-v1.5-4b-tokens**: 4.55B params (HF safetensors total); VRAM = bf16 weights + codec, estimated. tokens = round(target_s × 12.5).
- **moss-tts-v1.5-8b-tokens**: 8.49B params (HF safetensors total); VRAM = bf16 weights + codec, estimated.
- **moss-tts-local-v1.0-1.7b-tokens**: 3.06B params (HF safetensors total); VRAM = bf16 weights + codec, estimated. v1.0 (superseded by v1.5 4B per compute-tiers); kept for the version delta.
- **omnivoice-duration**: generate(duration=target_s): fixed output length.
- **maskgct**: MaskGCT (Amphion, CC-BY-NC weights) takes an explicit target length; worker not wired yet (models/tts/maskgct in Amphion).
- **magic-tts**: MAGIC-TTS (2026-04 arXiv): no repository or licence found.
- **voicestar**: VoiceStar (VoiceCraft lineage, duration-controllable, English): code at jasonppy/VoiceStar, worker not wired.
- **ted-tts**: TED-TTS training-free duration steering: code availability unverified; a method over a base model.
- **phoneme-tsm**: Non-isoelastic phoneme-level time-scale modification: a DSP method that belongs to E1 (timing fit) — ranked there.
- **ps-tts**: PS-TTS (ICPR 2026): no code release identified.
- **chirp3-hd-rate**: Stock voice + speakingRate from a first pass (two calls when off-target).
- **azure-mai-voice-2-rate**: SSML <prosody rate> from a first pass. MAI-Voice-2 has no ja-JP voice.
- **elevenlabs-dubbing-control**: ElevenLabs Dubbing does its own isochrony with no per-line duration handle and needs source audio (D3 items are text): compared in D8 instead.

</details>

### D4

**Regional and low-resource language voices**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **f5-hindi-small** (baseline) | local | 0.16 B | CC-BY-4.0 | hi | 2 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **chatterbox-multilingual** | local | 0.5 B | MIT | en hi ja +20 | 5 GB VRAM | ✓ | ✓ | not run | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | `d1_chatterbox` |
| **fish-s2-pro** | local | 4.56 B | Fish-Audio-Research-License (NC) · NC | en hi ja +80 | 24 GB VRAM | ✗ | ✓ | not run | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | `d1_fish_speech` |
| **higgs-audio-v3-4b** | local | 4 B | Boson Higgs TTS 3 Research and Non-Commercial License · NC | en hi ja +12 | 24 GB VRAM | ✗ | ✓ | not run | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | `d1_higgs_audio` |
| **indicf5** 🔒 | local | 0.4 B | MIT | hi +10 | 3 GB VRAM, key HF_TOKEN | ✓ | ✓ | not run | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | `d4_indicf5` |
| **moss-tts-v1.5-8b** | local | 8.49 B | Apache-2.0 | en hi ja +28 | 20 GB VRAM | ✗ | ✓ | not run | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | `d1_moss_tts` |
| **omnivoice** | local | 0.61 B | code Apache-2.0; weights CC-BY-NC · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | `d4_omnivoice` |
| **voxcpm2** | local | 2 B | Apache-2.0 | en hi ja +27 | 9 GB VRAM | ✓ | ✓ | not run | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `d1_voxcpm` |
| **azure-mai-voice-2** | api | — | proprietary API | en hi | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/en-us/azure/ai-services/s…](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices) | `server` |
| **cartesia-sonic-3.6** | api | — | proprietary API | en hi ja +12 | key CARTESIA | ✓ | ✓ | not run | [docs.cartesia.ai/build-with-cartesia/tts-mode…](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) | `server` |
| **sarvam-bulbul-v2** | api | — | proprietary API | en hi +9 | key SARVAM | ✓ | ✓ | not run | [docs.sarvam.ai/api-reference-docs/text-to-spe…](https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert) | `server` |
| **sarvam-bulbul-v3** | api | — | proprietary API | en hi +9 | key SARVAM | ✓ | ✓ | not run | [docs.sarvam.ai/api-reference-docs/text-to-spe…](https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert) | `server` |
| ~~intron-sahara~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [intron.io](https://www.intron.io) | `server` |
| ~~jaitts-pattern~~ (disabled) | method | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/](https://arxiv.org/) | `server` |

<details><summary>Notes</summary>

- **f5-hindi-small**: SPRINGLab F5-Hindi-24KHz (F5TTS_Small config, as in the app provider's placeholder). Non-Devanagari reference transcripts are outside its vocab: see the translit row.
- **voxcpm2**: ~8 GB on a 4090 per the README; 48 kHz output; no language argument (the model reads the script).
- **indicf5**: Gated repo (HF_TOKEN). The Indic specialist with the cleanest licence.
- **chatterbox-multilingual**: chatterbox-tts 0.1.7 installed --no-deps over torch 2.11 (it pins torch 2.6, no sm_120 kernels).
- **higgs-audio-v3-4b**: 100+ languages per the card (listed: the in-scope subset). Served by vLLM-Omni (the card cites ≥24 GB reported, 40 GB for SGLang-Omni) → Spark. Request fields from the repo's AGENTS.md; not run yet.
- **fish-s2-pro**: fishaudio/s2-pro @1de9996. Docs recommend ≥24 GB → Spark. SGLang fast path unverified on aarch64/sm_121: plain PyTorch api_server.
- **omnivoice**: 600+ languages; NC weights: a coverage/evaluation tool.
- **sarvam-bulbul-v2**: Stock speakers (no cloning). Endpoint/speakers as known for bulbul:v2; not re-verified.
- **sarvam-bulbul-v3**: Model id and v3 speaker names unverified; Bulbul V4 announced 2026-07-30 (set model when public).
- **jaitts-pattern**: Per-language adaptation of a strong base (JaiTTS on VoxCPM, Thai): an approach, and Thai/African/Vietnamese are outside en/hi/ja.
- **intron-sahara**: Intron Sahara v2 (African languages, commercial API): out of scope.
- **cartesia-sonic-3.6**: Sonic 3.6 (44 languages, $49/M chars per Cartesia, 2026-08). Clone endpoint not re-verified.
- **azure-mai-voice-2**: Stock voices (cloning is gated). No ja-JP MAI voice.
- **moss-tts-v1.5-8b**: 8.49B params (HF safetensors total); VRAM = bf16 weights + codec, estimated.

</details>

### D5

**Licensed non-cloning voices**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **kokoro-82m** | local | 0.082 B | Apache-2.0 | en hi ja +5 | 3 GB RAM | ✓ | ✓ | not run | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | `d5_kokoro` |
| **moss-voicegenerator-1.7b** | local | 2.11 B | Apache-2.0 | en +1 | 6 GB VRAM | ✓ | ✓ | not run | [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) | `d1_moss_tts` |
| **omnivoice-design** | local | 0.61 B | code Apache-2.0; weights CC-BY-NC · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | `d4_omnivoice` |
| **qwen3-tts-0.6b-customvoice** | local | 0.6 B | Apache-2.0 | en ja +8 | 3.5 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | `d1_qwen3_tts` |
| **qwen3-tts-1.7b-customvoice** | local | 1.7 B | Apache-2.0 | en ja +8 | 6.5 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `d1_qwen3_tts` |
| **qwen3-tts-1.7b-voicedesign** | local | 1.7 B | Apache-2.0 | en ja +8 | 6.5 GB VRAM | ✓ | ✓ | not run | [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | `d1_qwen3_tts` |
| **voxcpm2-design** | local | 2 B | Apache-2.0 | en hi ja +27 | 9 GB VRAM | ✓ | ✓ | not run | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `d1_voxcpm` |
| **azure-mai-voice-2** | api | — | proprietary API | en hi +13 | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/en-us/azure/ai-services/s…](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices) | `server` |
| **azure-mai-voice-2-flash** | api | — | proprietary API | en hi +13 | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/en-us/azure/ai-services/s…](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices) | `server` |
| **azure-neural** | api | — | proprietary API | en hi ja | key AZURE_SPEECH, AZURE_SPEECH_REGION | ✓ | ✓ | not run | [learn.microsoft.com/en-us/azure/ai-services/s…](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support) | `server` |
| **chirp3-hd** | api | — | proprietary API | en hi ja | key GOOGLE | ✓ | ✓ | not run | [cloud.google.com/text-to-speech/docs/chirp3-hd](https://cloud.google.com/text-to-speech/docs/chirp3-hd) | `server` |
| **elevenlabs-voice-library** | api | — | proprietary API | en hi ja +26 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/text-to-spee…](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | `server` |
| **hume-octave-2-design** | api | — | proprietary API | en hi ja +8 | key HUME | ✓ | ✓ | not run | [dev.hume.ai/docs/text-to-speech-tts/overview](https://dev.hume.ai/docs/text-to-speech-tts/overview) | `server` |
| **openai-gpt-4o-mini-tts** | api | — | proprietary API | en hi ja +12 | key OPENAI | ✓ | ✓ | not run | [platform.openai.com/docs/guides/text-to-speech](https://platform.openai.com/docs/guides/text-to-speech) | `server` |
| **qwen-audio-3.0-tts-plus** | api | — | proprietary API | en ja +8 | key OPENROUTER | ✓ | ✓ | not run | [openrouter.ai/qwen/qwen-audio-3.0-tts-plus](https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus) | `server` |
| ~~breeze-tts-2~~ (disabled) | local | — | BreezeBlue Research and Non-Commercial License · NC | all (✓ en hi ja) | CPU | — | — | not run | [models?search=breeze](https://huggingface.co/models?search=breeze) | `server` |
| ~~chatterbox-multilingual-library~~ (disabled) | local | 0.5 B | MIT | en hi ja +20 | 5 GB VRAM | — | — | not run | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | `d1_chatterbox` |
| ~~orpheus-3b~~ (disabled) | local | 3 B | Llama 3.2 Community License · NC | en | CPU | — | — | not run | [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | `server` |
| ~~piper~~ (disabled) | local | — | GPL-3.0 · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) | `server` |
| ~~raon-opentts-1b~~ (disabled) | local | 1 B | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [models?search=raon](https://huggingface.co/models?search=raon) | `server` |
| ~~cartesia-sonic-3.6-stock~~ (disabled) | api | — | proprietary API | en hi ja +12 | key CARTESIA | — | — | not run | [docs.cartesia.ai/build-with-cartesia/tts-mode…](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) | `server` |
| ~~resemble-stock~~ (disabled) | api | — | proprietary API | en | key RESEMBLE | — | — | not run | [docs.resemble.ai](https://docs.resemble.ai) | `server` |

<details><summary>Notes</summary>

- **azure-mai-voice-2**: Voice names from the MAI-Voice doc (2026-07-23); no ja-JP voice.
- **azure-mai-voice-2-flash**: Low-latency tier; no ja-JP voice.
- **azure-neural**: Standard neural catalogue (covers ja-JP); HD V3 voices can be swapped in via voices.
- **kokoro-82m**: v1.0 voices (af_heart/am_michael, hf_alpha/hm_omega, jf_alpha/jm_kumo); runs on CPU. 'Kokoro v3' claims unverified.
- **cartesia-sonic-3.6-stock**: Sonic 3.6 (44 languages, $49/M chars per Cartesia, 2026-08). Clone endpoint not re-verified. Needs preset voice ids per language in params.voices (not filled).
- **chirp3-hd**: No cloning offered at all: the cleanest non-cloning guarantee. Voice names follow <locale>-Chirp3-HD-<Name> (defaults Aoede/Charon, unverified per locale).
- **chatterbox-multilingual-library**: chatterbox-tts 0.1.7 installed --no-deps over torch 2.11 (it pins torch 2.6, no sm_120 kernels). Driven from a licensed reference library: set params.stock_voices {lang: {female, male}: wav}. Disabled until the library exists.
- **elevenlabs-voice-library**: Mirrors app/providers/tts/elevenlabs.py; IVC voices are deleted at job end. Prices in _cost_usd are estimates. Premade voices (Rachel / Adam ids); Voice Library Addendum governs shared voices.
- **orpheus-3b**: Orpheus TTS 3B: English-first (multilingual research preview), Llama 3.2 licence; compute-tiers says superseded.
- **piper**: Piper (GPL-3.0 engine, per-voice licences): evaluate-only; no worker (CPU, espeak-ng).
- **breeze-tts-2**: Breeze TTS 2 (research/NC weights, 2026-08): inference API and languages not verified.
- **voxcpm2-design**: ~8 GB on a 4090 per the README; 48 kHz output; no language argument (the model reads the script).
- **qwen3-tts-1.7b-voicedesign**: No Hindi. SDPA attention (no FA3 on sm_120; FA2 optional). VRAM estimated from params (compute-tiers: ~6 GB bf16 for 1.7B).
- **qwen3-tts-1.7b-customvoice**: No Hindi. SDPA attention (no FA3 on sm_120; FA2 optional). VRAM estimated from params (compute-tiers: ~6 GB bf16 for 1.7B). 9 preset timbres; Ono_Anna is the only Japanese one (used for both genders).
- **qwen3-tts-0.6b-customvoice**: No Hindi. SDPA attention (no FA3 on sm_120; FA2 optional). VRAM estimated from params (compute-tiers: ~6 GB bf16 for 1.7B).
- **moss-voicegenerator-1.7b**: Instruction-only voice design; language list not stated on the card (zh/en examples).
- **raon-opentts-1b**: Raon-OpenTTS 1B (open models + open data): release and inference API not verified.
- **qwen-audio-3.0-tts-plus**: OpenRouter: two preset voices; languages and price not stated there (unverified).
- **openai-gpt-4o-mini-tts**: Stock voices (coral/ash); no language parameter.
- **hume-octave-2-design**: Voice designed from a description; not re-verified.
- **resemble-stock**: Needs voice uuids per language in params.voices; disabled until filled.
- **omnivoice-design**: instruct= attribute voice design; NC weights.

</details>

### D6

**Voice conversion**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **chatterbox-vc** | local | 0.5 B | MIT | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | `d1_chatterbox` |
| **fish-s2-pro-resynth** | local | 4.56 B | Fish-Audio-Research-License (NC) · NC | en hi ja +80 | 24 GB VRAM | ✗ | ✓ | not run | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | `d1_fish_speech` |
| **freevc** | local | — | MIT | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [github.com/OlaWod/FreeVC](https://github.com/OlaWod/FreeVC) | `d6_freevc` |
| **freevc-24** | local | — | MIT | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [github.com/OlaWod/FreeVC](https://github.com/OlaWod/FreeVC) | `d6_freevc` |
| **knn-vc** | local | 0.3 B | MIT | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/bshall/knn-vc](https://github.com/bshall/knn-vc) | `d6_knn_vc` |
| **seed-vc-v1** | local | — | GPL-3.0 · NC | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [github.com/Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | `d6_seed_vc` |
| **vevo2-ar-fm** | local | 0.87 B | CC-BY-NC-ND-4.0 (checkpoints); Amphion code MIT · NC | en ja +4 | 6 GB VRAM | ✓ | ✓ | not run | [RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2) | `d6_vevo2` |
| **vevo2-fm** | local | 0.87 B | CC-BY-NC-ND-4.0 (checkpoints); Amphion code MIT · NC | en ja +4 | 4 GB VRAM | ✓ | ✓ | not run | [RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2) | `d6_vevo2` |
| **voxcpm2-resynth** | local | 2 B | Apache-2.0 | en hi ja +27 | 9 GB VRAM | ✓ | ✓ | not run | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `d1_voxcpm` |
| **cartesia-resynth** | api | — | proprietary API | en hi ja +12 | key CARTESIA | ✓ | ✓ | not run | [docs.cartesia.ai/build-with-cartesia/tts-mode…](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) | `server` |
| **elevenlabs-voice-changer** | api | — | proprietary API | en hi ja +26 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/text-to-spee…](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | `server` |
| ~~x-vc~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/Jerrister/X-VC](https://github.com/Jerrister/X-VC) | `server` |
| ~~respeecher-marketplace~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [respeecher.com](https://www.respeecher.com) | `server` |
| ~~vevo2-bestof16-sim~~ (disabled) | method | — | CC-BY-NC-ND-4.0 · NC | all (✓ en hi ja) | CPU | — | — | not run | [RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2) | `server` |

<details><summary>Notes</summary>

- **vevo2-fm**: Timbre conversion, style preserved (inference_fm). ~3 GB per compute-tiers.
- **vevo2-ar-fm**: Style-converted VC via inference_ar_and_fm with the source transcript (inputs.text).
- **elevenlabs-voice-changer**: Mirrors app/providers/tts/elevenlabs.py; IVC voices are deleted at job end. Prices in _cost_usd are estimates.
- **x-vc**: X-VC (2026-04): code at Jerrister/X-VC, licence not stated and checkpoints unverified.
- **chatterbox-vc**: chatterbox-tts 0.1.7 installed --no-deps over torch 2.11 (it pins torch 2.6, no sm_120 kernels). S3Gen VC is language-agnostic.
- **seed-vc-v1**: Evaluate-only (GPL). CLI per item (model reloads each item).
- **knn-vc**: Non-parametric floor; a single short reference is its worst case.
- **respeecher-marketplace**: Respeecher Marketplace: subscription product, no public per-request API confirmed.
- **freevc**: 16 kHz; checkpoints fetched from the OlaWod/FreeVC Space (unverified).
- **freevc-24**: 24 kHz output variant.
- **voxcpm2-resynth**: ~8 GB on a 4090 per the README; 48 kHz output; no language argument (the model reads the script). Re-synthesis instead of conversion: the source transcript (inputs.text) spoken in the target voice (compute-tiers Unconstrained pick).
- **fish-s2-pro-resynth**: fishaudio/s2-pro @1de9996. Docs recommend ≥24 GB → Spark. SGLang fast path unverified on aarch64/sm_121: plain PyTorch api_server. Re-synthesis with the source transcript.
- **vevo2-bestof16-sim**: Vevo2 at N=16 re-ranked by a WavLM SIM judge: needs a SIM verifier in the best-of-N worker (only an ASR verifier is wired).
- **cartesia-resynth**: Sonic 3.6 (44 languages, $49/M chars per Cartesia, 2026-08). Clone endpoint not re-verified. Pseudo-VC: 10 s clone of the target + the source transcript.

</details>

### D7

**Non-verbal vocalizations**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **chatterbox-turbo** | local | 0.5 B | MIT | en | 5 GB VRAM | ✓ | ✓ | not run | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | `d1_chatterbox` |
| **fish-s2-pro** | local | 4.56 B | Fish-Audio-Research-License (NC) · NC | en hi ja +80 | 24 GB VRAM | ✗ | ✓ | not run | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | `d1_fish_speech` |
| **higgs-audio-v3-4b** | local | 4 B | Boson Higgs TTS 3 Research and Non-Commercial License · NC | en hi ja +12 | 24 GB VRAM | ✗ | ✓ | not run | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | `d1_higgs_audio` |
| **step-audio-editx-paralinguistic** | local | 3 B | Apache-2.0 | en ja +2 | 15 GB VRAM, x86_64 | ✓ | ✗ | not run | [stepfun-ai/Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) | `d2_step_audio_editx` |
| **cartesia-sonic-3.6** | api | — | proprietary API | en hi ja +12 | key CARTESIA | ✓ | ✓ | not run | [docs.cartesia.ai/build-with-cartesia/tts-mode…](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) | `server` |
| **elevenlabs-v3-tags** | api | — | proprietary API | en hi ja +24 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/text-to-spee…](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) | `server` |
| **qwen-audio-3.0-tts-plus** | api | — | proprietary API | en ja +8 | key OPENROUTER | ✓ | ✓ | not run | [openrouter.ai/qwen/qwen-audio-3.0-tts-plus](https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus) | `server` |
| **splice-f5** | method | 0.34 B | CC-BY-NC-4.0 (inner F5) · NC | en +1 | 3 GB VRAM | ✓ | ✓ | not run | [github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) | `server` |
| **splice-f5-hindi** | method | 0.16 B | CC-BY-4.0 (inner F5-Hindi) | hi | 2 GB VRAM | ✓ | ✓ | not run | [SPRINGLab/F5-Hindi-24KHz](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz) | `server` |
| **splice-voxcpm2** | method | 2 B | Apache-2.0 (inner VoxCPM2) | en hi ja +27 | 9 GB VRAM | ✓ | ✓ | not run | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | `d1_voxcpm` |
| ~~ced-beats-tagging~~ (disabled) | local | — | Apache-2.0 / MIT | all (✓ en hi ja) | CPU | — | — | not run | [mispeech/ced-base](https://huggingface.co/mispeech/ced-base) | `server` |
| ~~dia-1.6b~~ (disabled) | local | 1.6 B | Apache-2.0 | en | CPU | — | — | not run | [nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) | `server` |
| ~~moss-soundeffect-v2~~ (disabled) | local | 1.3 B | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) | `server` |
| ~~multilinguahah~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/sofia-callejas/Multilinguahah](https://github.com/sofia-callejas/Multilinguahah) | `server` |
| ~~nvspeech-sensevoice~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2508.04195](https://arxiv.org/abs/2508.04195) | `server` |

<details><summary>Notes</summary>

- **splice-f5**: Pass-through splice of the source vocalisation around F5 speech (research primary); the CED/BEATs gate is upstream (A8) — here the pack says where events are.
- **splice-f5-hindi**: Splice method over F5-Hindi.
- **splice-voxcpm2**: Splice method over VoxCPM2.
- **ced-beats-tagging**: CED / BEATs are detectors, not generators: CED is the D7 judge (nv_events.ced) and the A8 tagger that gates the splice.
- **step-audio-editx-paralinguistic**: Clone the untagged text, then a paralinguistic edit pass with the tags ([laugh], [sigh] …). x86 only.
- **nvspeech-sensevoice**: NVSpeech paralinguistic ASR (inline event tokens): a detector for the router (A8), not a generator.
- **multilinguahah**: MultiLinguahah laughter segmentation: a detector (A8).
- **elevenlabs-v3-tags**: Mirrors app/providers/tts/elevenlabs.py; IVC voices are deleted at job end. Prices in _cost_usd are estimates. Script tags passed through as v3 audio tags.
- **cartesia-sonic-3.6**: Sonic 3.6 (44 languages, $49/M chars per Cartesia, 2026-08). Clone endpoint not re-verified. Infers non-verbals from the transcript; literal [tags] may be read out.
- **dia-1.6b**: Dia-1.6B (English, Apache-2.0): compute-tiers says superseded; no worker.
- **higgs-audio-v3-4b**: 100+ languages per the card (listed: the in-scope subset). Served by vLLM-Omni (the card cites ≥24 GB reported, 40 GB for SGLang-Omni) → Spark. Request fields from the repo's AGENTS.md; not run yet. Script tags passed as text; Higgs' own control tokens are <\|…\|>.
- **moss-soundeffect-v2**: MOSS-SoundEffect v2.0 generates sound effects, not voice (E/F-phase material).
- **fish-s2-pro**: fishaudio/s2-pro @1de9996. Docs recommend ≥24 GB → Spark. SGLang fast path unverified on aarch64/sm_121: plain PyTorch api_server. Bracket tags are native.
- **qwen-audio-3.0-tts-plus**: Preset voices; tag vocabulary (86 inline tags) not mapped from ours — unverified.
- **chatterbox-turbo**: chatterbox-tts 0.1.7 installed --no-deps over torch 2.11 (it pins torch 2.6, no sm_120 kernels). English Turbo with native [laugh]/[cough] tags.

</details>

### D8

**Direct speech-to-speech translation (evaluate only)**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **seamless-m4t-v2-large** | local | 2.3 B | CC-BY-NC-4.0 · NC | en hi ja | 8 GB VRAM | ✓ | ✓ | not run | [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | `d8_seamless` |
| **step-audio-2-mini** | local | 8.3 B | Apache-2.0 | en +1 | 20 GB VRAM | ✗ | ✓ | not run | [stepfun-ai/Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) | `d8_step_audio2` |
| **elevenlabs-dubbing** | api | — | proprietary API | en hi ja +26 | key ELEVENLABS | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/dubbing/create](https://elevenlabs.io/docs/api-reference/dubbing/create) | `server` |
| ~~hibiki-2b~~ (disabled) | local | 2.7 B | CC-BY-4.0 | en | 10 GB VRAM | — | — | not run | [kyutai/hibiki-2b-pytorch-bf16](https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16) | `d8_hibiki` |
| ~~hibiki-zero-3b~~ (disabled) | local | 3 B | CC-BY-4.0 | en | 12 GB VRAM | — | — | not run | [kyutai/hibiki-zero-3b-pytorch-bf16](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) | `d8_hibiki` |
| ~~seamless-expressive~~ (disabled) | local | — | Seamless research licence (NC) · NC | all (✓ en hi ja) | CPU | — | — | not run | [facebook/seamless-expressive](https://huggingface.co/facebook/seamless-expressive) | `server` |
| ~~transvip~~ (disabled) | local | — | unknown · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2405.17809](https://arxiv.org/abs/2405.17809) | `server` |
| ~~uniss~~ (disabled) | local | 1.86 B | CC-BY-4.0 | en +1 | 10 GB VRAM | — | — | not run | [cmots/UniSS](https://huggingface.co/cmots/UniSS) | `d8_uniss` |
| ~~cascade-opendub~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [github.com/NidhiBharani/OpenDub/blob/main/doc…](https://github.com/NidhiBharani/OpenDub/blob/main/docs/plans/model-ranking.md) | `server` |
| ~~seed-liveinterpret-2~~ (disabled) | api | — | proprietary · NC | all (✓ en hi ja) | CPU | — | — | not run | [seed.bytedance.com](https://seed.bytedance.com) | `server` |

<details><summary>Notes</summary>

- **seamless-m4t-v2-large**: Speech output covers eng/hin/jpn (36 targets); no voice preservation (vocoder speakers). The only open direct S2ST here that covers ja→en/hi and en→hi.
- **seamless-expressive**: SeamlessExpressive: gated research weights, eng↔fra/deu/ita/spa/cmn only — no ja/hi pair.
- **uniss**: en↔zh only: no ja/hi source or target → disabled for the en/hi/ja scope.
- **hibiki-zero-3b**: fr/es/pt/de→en only (no ja/hi source) → disabled; uses its own hibiki-zero package (set params.module).
- **cascade-opendub**: Control condition = the OpenDub cascade itself (A4→C2→D1), measured through the ladder/cascade view on the same clips, not as a D8 worker.
- **hibiki-2b**: fr→en only → disabled for en/hi/ja.
- **seed-liveinterpret-2**: Seed LiveInterpret 2.0 / Seed-Live 2.0: private beta, no public API.
- **step-audio-2-mini**: S2ST output en/zh per the repo examples (source ja allowed via S2TT support); ~8B speech LLM bf16 → Spark. Source clip used as the token2wav voice prompt.
- **transvip**: TransVIP (Microsoft research, 2024): weights not released/verified.
- **elevenlabs-dubbing**: Productised cascade (control condition). Per-minute price is an estimate.

</details>

## E — Audio post-production

### E1

**Timing fit + pause mapping**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **atempo-app** (baseline) | local | — | LGPL-2.1-or-later (ffmpeg) | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [server/app/pipeline/audio.py (fit_to_duration…](server/app/pipeline/audio.py (fit_to_duration; ffmpeg atempo)) | `server` |
| **atempo-uniform** | local | — | LGPL-2.1-or-later (ffmpeg) | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [ffmpeg.org/ffmpeg-filters.html#atempo](https://ffmpeg.org/ffmpeg-filters.html#atempo) | `server` |
| **rubberband-r3-uniform** | local | — | GPL-2.0-or-later (commercial licence available) | all (✓ en hi ja) | +rubberband | ✗ | ✗ | not run | [breakfastquay.com/rubberband/](https://breakfastquay.com/rubberband/) | `server` |
| **signalsmith-uniform** | local | — | MIT | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/Signalsmith-Audio/signalsmith-stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) | `e1_signalsmith` |
| **pause-dp+atempo** | method | — | Apache-2.0 (in-house DP) + LGPL ffmpeg | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [arxiv.org/abs/2204.02530](https://arxiv.org/abs/2204.02530) | `server` |
| **pause-dp+rubberband-r3** | method | — | Apache-2.0 (in-house DP) + GPL-2.0-or-later CLI | all (✓ en hi ja) | +rubberband | ✗ | ✗ | not run | [arxiv.org/abs/2204.02530](https://arxiv.org/abs/2204.02530) | `server` |
| **pause-dp+signalsmith** | method | — | Apache-2.0 (in-house DP) + MIT | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [arxiv.org/abs/2204.02530](https://arxiv.org/abs/2204.02530) | `e1_signalsmith` |
| ~~diffatsm~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~dubwise~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2406.08802](https://arxiv.org/abs/2406.08802) | `server` |
| ~~holidubber~~ (disabled) | local | — | unstated | all (✓ en hi ja) | CPU | — | — | not run | [holidubber.github.io/](https://holidubber.github.io/) | `server` |
| ~~stsm-film~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~zplane-elastique-pro~~ (disabled) | local | — | proprietary SDK · NC | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~duration-controlled-regen~~ (disabled) | method | — | per paper (Dub-S2ST, duration-based translation) · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2505.20899](https://arxiv.org/abs/2505.20899) | `server` |
| ~~elevenlabs-dubbing-v2~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | — | — | not run | [elevenlabs.io/docs/capabilities/dubbing](https://elevenlabs.io/docs/capabilities/dubbing) | `server` |
| ~~heygen-video-translate~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key HEYGEN | — | — | not run | [docs.heygen.com/](https://docs.heygen.com/) | `server` |
| ~~ps-tts-isochronic-paraphrase~~ (disabled) | method | — | method (ICPR 2026, arXiv 2604.09111) | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2604.09111](https://arxiv.org/abs/2604.09111) | `server` |

<details><summary>Notes</summary>

- **atempo-app**: timing_fit.atempo as shipped: app.pipeline.audio.fit_to_duration (trim outer silence, speed up ≤1.15×, never slow down, pad). Beyond 1.15× the app regenerates; the arena keeps the clamped take and scores it as an overflow.
- **atempo-uniform**: scalar stretch of the speech span onto the source speech span (both directions)
- **rubberband-r3-uniform**: Rubber Band CLI R3 (-3 --formant -D), shelled out and never linked, so the GPL stays outside the Apache-2.0 artefact. Install: apt install rubberband-cli (x86_64 and arm64 packages).
- **signalsmith-uniform**: Signalsmith Stretch via the third-party python-stretch 0.3.1 binding (sdist build on aarch64)
- **pause-dp+atempo**: in-house prosodic-alignment DP (no reference implementation exists) with WSOLA residual
- **pause-dp+signalsmith**: the research's Spark pick (pause-structure DP + Signalsmith residual stretch)
- **duration-controlled-regen**: Regenerating the take at the slot length (Dub-S2ST, best-of-N) replaces E1 rather than post-processing a take; it is ranked under D3 (duration control) / D8 (S2ST), not here.
- **diffatsm**: paywalled paper (CSL 2025), no verified public checkpoint
- **stsm-film**: arXiv Oct 2025 neural TSM; code and licence not confirmed
- **zplane-elastique-pro**: closed per-product commercial SDK; not obtainable for an open-source build
- **ps-tts-isochronic-paraphrase**: LLM duration-constrained paraphrase + vowel DTW: a C2 translation-loop strategy (reuses the C2 model); benchmark it there as the escalation for lines needing >10 % stretch.
- **holidubber**: arXiv 2606.09098; code "coming soon", no weights
- **dubwise**: Sony AI Interspeech 2024 video-guided duration control inside TTS; no public weights found
- **elevenlabs-dubbing-v2**: End-to-end dubbing product; fit is by regeneration (no per-clip duration knob; editing and regeneration via API are Enterprise-only), so it cannot be isolated as an E1 stage — see D8.
- **heygen-video-translate**: closed end-to-end video translation; regeneration-based fit, not an E1 stage

</details>

### E2

**Bandwidth extension / 48 kHz restoration**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **resample-48k** (baseline) | local | — | LGPL-2.1-or-later (ffmpeg) | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [ffmpeg.org/ffmpeg-resampler.html](https://ffmpeg.org/ffmpeg-resampler.html) | `server` |
| **ap-bwe-48k** | local | 0.0298 B | MIT (code and weights) | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | not run | [github.com/yxlu-0102/AP-BWE](https://github.com/yxlu-0102/AP-BWE) | `e2_ap_bwe` |
| **audiosr-speech** | local | — | MIT | all (✓ en hi ja) | 9 GB VRAM | ✓ | ✓ | not run | [github.com/haoheliu/versatile_audio_super_res…](https://github.com/haoheliu/versatile_audio_super_resolution) | `e2_audiosr` |
| **clearvoice-mossformer2-sr48k** | local | — | Apache-2.0 (toolkit; checkpoint terms not stated separately) | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | `e2_clearvoice` |
| **flowhigh** | local | 0.0488 B | MIT | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [ResembleAI/FlowHigh](https://huggingface.co/ResembleAI/FlowHigh) | `e2_flowhigh` |
| **resemble-enhance** | local | — | MIT | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [github.com/resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | `e2_resemble_enhance` |
| **sidon-v0.1** | local | 0.65 B | MIT | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) | `e2_sidon` |
| **unipase** | local | 0.5457 B | MIT (code) / Apache-2.0 (HF weights card) | all (✓ en hi ja) | 5 GB VRAM | ✓ | ✓ | not run | [Xiaobin-Rong/unipase](https://huggingface.co/Xiaobin-Rong/unipase) | `e2_unipase` |
| **voicefixer** | local | — | MIT | all (✓ en hi ja) | 2.5 GB VRAM | ✓ | ✓ | not run | [github.com/haoheliu/voicefixer](https://github.com/haoheliu/voicefixer) | `e2_voicefixer` |
| **elevenlabs-voice-isolator** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS, +ffmpeg | ✓ | ✓ | not run | [elevenlabs.io/docs/api-reference/audio-isolat…](https://elevenlabs.io/docs/api-reference/audio-isolation/convert) | `server` |
| ~~cogsr~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2512.16304](https://arxiv.org/abs/2512.16304) | `server` |
| ~~flashsr~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2501.10807](https://arxiv.org/abs/2501.10807) | `server` |
| ~~miipher-2~~ (disabled) | local | 2 B | closed (Google DeepMind) · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2505.04457](https://arxiv.org/abs/2505.04457) | `server` |
| ~~nvidia-re-use~~ (disabled) | local | 0.00961 B | NVIDIA One-Way Noncommercial License (NSCLv1) · NC | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~open-miipher-2~~ (disabled) | local | 0.6 B | MIT | all (✓ en hi ja) | CPU | — | — | not run | [github.com/yukara-ikemiya/Open-Miipher-2](https://github.com/yukara-ikemiya/Open-Miipher-2) | `server` |
| ~~adobe-podcast-enhance~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [podcast.adobe.com/enhance](https://podcast.adobe.com/enhance) | `server` |
| ~~dolby-io-media-enhance~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [optiview.dolby.com/docs/](https://optiview.dolby.com/docs/) | `server` |

<details><summary>Notes</summary>

- **resample-48k**: today's behaviour — no BWE, the take is resampled to 48 kHz (ffmpeg.to_std_wav)
- **ap-bwe-48k**: 29.8M params; one checkpoint per input rate (8/12/16/24 kHz → 48 kHz, discovered from each config.json); unsupported input rates are skipped. Weights via gdown from the authors' Drive.
- **flowhigh**: Packaged fork loads FLowHigh_basic_400k (basic CFM) + BigVGAN-48k, not the paper's main indep_adaptive model (Drive-only). Output peak-matched to the input.
- **sidon-v0.1**: w2v-BERT 2.0 feature cleanser (143-language pretraining; fine-tuned on LibriTTS-R + FLEURS-R) + 48 kHz vocoder, TorchScript; a restorer that resynthesises the voice (ΔSIM gate).
- **unipase**: 545.7M universal restorer (URGENT 2026 winner backbone); the paper reports CER/SpkSim regressions vs a predictive baseline — research says use it conditionally, the gates decide.
- **clearvoice-mossformer2-sr48k**: The ledger's "HiFi-SR via ClearerVoice-Studio" row: the released toolkit's 48 kHz SR model is MossFormer2_SR_48K (HiFi-SR is not exposed by name). Trained on VCTK-style English.
- **resemble-enhance**: 44.1 kHz output; repo quiet since 2024-12; dropped from the Sep 18 shortlist (kept as a candidate)
- **audiosr-speech**: latent diffusion; slow; literature reports excess sibilance / hallucination (demoted Sep 18)
- **voicefixer**: 44.1 kHz vocoder restoration (2021); better suited to A2 than to delivered takes
- **elevenlabs-voice-isolator**: a hosted denoiser, not BWE (no hosted BWE API exists); $0.12/min list price (2026-09-28)
- **nvidia-re-use**: 9.6M bi-Mamba universal SE (2026); NC is allowed but release location / inference entry point were not confirmed in this pass, so no worker yet.
- **miipher-2**: weights not released
- **open-miipher-2**: training harness only, no pretrained checkpoints; 24 kHz output (does not solve BWE)
- **flashsr**: one-step AudioSR distillation; code availability not confirmed from a primary source
- **cogsr**: LALM-guided flow matching for severe degradation (archival); code/weights unverified
- **adobe-podcast-enhance**: no public Enhance Speech API as of 2026-09 (web app / partner integrations only)
- **dolby-io-media-enhance**: docs.dolby.io media-processing now redirects to Dolby OptiView; availability unconfirmed

</details>

### E3

**Acoustic scene / room matching**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **passthrough** (baseline) | local | — | Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/e3_dsp.py (in-house)](server/bench/arena/workers/e3_dsp.py (in-house)) | `server` |
| **blind-parametric-dsp** | local | — | Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/e3_dsp.py (in-house)](server/bench/arena/workers/e3_dsp.py (in-house)) | `server` |
| **rec-rir** | local | — | MIT (code + checkpoint in repo) | all (✓ en hi ja) | 3 GB VRAM, x86_64 | ✓ | ✗ | not run | [github.com/Audio-WestlakeU/Rec-RIR](https://github.com/Audio-WestlakeU/Rec-RIR) | `e3_rec_rir` |
| ~~accentize-chameleon~~ (disabled) | local | — | proprietary plugin · NC | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~av-soundscape-stylisation~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~buddy~~ (disabled) | local | — | none stated (no LICENSE in sp-uhh/buddy) · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/sp-uhh/buddy](https://github.com/sp-uhh/buddy) | `server` |
| ~~deepafx-st~~ (disabled) | local | — | NOASSERTION (adobe-research, treat as non-commercial) · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/adobe-research/DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) | `server` |
| ~~fins~~ (disabled) | local | — | research code | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~flamo-fdn-matching~~ (disabled) | local | — | paper CC BY 4.0; FLAMO framework MIT | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2510.23158](https://arxiv.org/abs/2510.23158) | `server` |
| ~~gencho~~ (disabled) | local | — | not stated | all (✓ en hi ja) | CPU | — | — | not run | [linjac.github.io/Gencho/](https://linjac.github.io/Gencho/) | `server` |
| ~~izotope-dialogue-match~~ (disabled) | local | — | proprietary (RX 12 Advanced, AAX AudioSuite) · NC | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~reverbmiipher~~ (disabled) | local | — | closed | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2505.05077](https://arxiv.org/abs/2505.05077) | `server` |
| ~~rirflow~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [pubs.aip.org/asa/jasa/article/159/6/5527/3395…](https://pubs.aip.org/asa/jasa/article/159/6/5527/3395888/Solving-room-impulse-response-inverse-problems) | `server` |
| ~~st-ito~~ (disabled) | local | — | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [github.com/csteinmetz1/st-ito](https://github.com/csteinmetz1/st-ito) | `server` |

<details><summary>Notes</summary>

- **passthrough**: today's behaviour — no room matching, the dry take is mixed as is
- **blind-parametric-dsp**: Classical free-decay T60 estimate + DRR prior + exponential-noise RIR (in-house). The synthetic test rooms share the exponential-tail family, which flatters it slightly.
- **rec-rir**: Blind RIR identification (16 kHz); needs mamba-ssm/causal-conv1d CUDA extensions (x86_64 only here). Estimation runs once per job in load(), so rtfx excludes it.
- **flamo-fdn-matching**: Götz et al. (arXiv 2510.23158) VAE-embedding + differentiable FDN; FLAMO (pip flamo) is the framework, but the trained acoustic-embedding VAE is not released — no runnable matcher yet.
- **accentize-chameleon**: GUI/DAW plugin, no headless API; reference ceiling only
- **izotope-dialogue-match**: plugin only (Pro Tools); no scripting
- **av-soundscape-stylisation**: Li & Owens ECCV 2024 latent diffusion; also transfers ambience (duplicates the A1 bed) and resynthesises the voice; evaluate-only, worker not written.
- **buddy**: Code + VCTK checkpoint exist (Google Drive) but no licence and bash-script-driven diffusion posterior sampling (slow, per scene); worker deferred while E3 is parked.
- **st-ito**: run_optim.py searches a VST effect chain (plugins not bundled; Linux/aarch64 unverified); matches colour, not decay
- **deepafx-st**: production-style transfer; licence to clear; no reverberation-time model
- **gencho**: Adobe/UIUC ICASSP 2026 DiT RIR generator; demos only, no code or weights
- **fins**: Steinmetz et al. WASPAA 2021 filtered-noise RIR estimation; no released weights found
- **reverbmiipher**: Google; not open-sourced
- **rirflow**: JASA 2026 training-free flow matching for RIR inverse problems; code not confirmed

</details>

### E4

**Dialogue levelling + delivery loudness**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **static-chain** (baseline) | local | — | Apache-2.0 + LGPL ffmpeg | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [server/app/pipeline/audio.py (mix_tracks)](server/app/pipeline/audio.py (mix_tracks)) | `server` |
| **bs1770-levelled** | local | — | MIT (pyloudnorm) + Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | `e4_pyloudnorm` |
| **bs1770-static** | local | — | MIT (pyloudnorm) + Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | `e4_pyloudnorm` |
| **ffmpeg-loudnorm-2pass** | local | — | LGPL-2.1-or-later (ffmpeg) | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [ffmpeg.org/ffmpeg-filters.html#loudnorm](https://ffmpeg.org/ffmpeg-filters.html#loudnorm) | `server` |
| **static-chain+duck** | local | — | Apache-2.0 + LGPL ffmpeg | all (✓ en hi ja) | +ffmpeg | ✓ | ✓ | not run | [ffmpeg.org/ffmpeg-filters.html#sidechaincompress](https://ffmpeg.org/ffmpeg-filters.html#sidechaincompress) | `server` |
| **auphonic** | api | — | proprietary SaaS | all (✓ en hi ja) | key AUPHONIC | ✓ | ✓ | not run | [auphonic.com/help/api/simple_api.html](https://auphonic.com/help/api/simple_api.html) | `server` |
| ~~deepafx-st~~ (disabled) | local | — | NOASSERTION (treat as non-commercial) · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/adobe-research/DeepAFx-ST](https://github.com/adobe-research/DeepAFx-ST) | `server` |
| ~~dialog-plus-remix~~ (disabled) | local | — | proprietary (Fraunhofer IIS) · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2112.09494](https://arxiv.org/abs/2112.09494) | `server` |
| ~~dolby-dialogue-gating-reference~~ (disabled) | local | — | Dolby reference-code terms (not redistributable in Apache-2.0) · NC | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~harmony-search-automix~~ (disabled) | local | — | paper only (Liu & Reiss, JAES 2025) | all (✓ en hi ja) | CPU | — | — | not run | [joshreiss.github.io/documents/2025/Liu2025JAE…](https://joshreiss.github.io/documents/2025/Liu2025JAESAutomaticMixingSpeech.pdf) | `server` |
| ~~neural-automix~~ (disabled) | local | — | varies | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~nugen-lm-correct~~ (disabled) | local | — | proprietary plugin · NC | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~st-ito~~ (disabled) | local | — | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [github.com/csteinmetz1/st-ito](https://github.com/csteinmetz1/st-ito) | `server` |
| ~~audio-llm-mix-critic~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~dolby-io-media-enhance~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [optiview.dolby.com/docs/](https://optiview.dolby.com/docs/) | `server` |

<details><summary>Notes</summary>

- **static-chain**: loudness.static as shipped (app.pipeline.audio.mix_tracks): programme-gated static gains, −1.5 dBFS sample-peak alimiter, the delivery preset only sets the integrated target.
- **static-chain+duck**: the app's optional sidechain duck of the bed (sidechaincompress, ratio 3, 25/250 ms)
- **bs1770-levelled**: Two-pass BS.1770 static chain with per-deliverable targets, dialogue gating on the known line windows (replacing the relative gate), per-line levelling to the original, 4× TP limiter.
- **bs1770-static**: same chain without per-line levelling (isolates the levelling contribution)
- **ffmpeg-loudnorm-2pass**: two-pass linear loudnorm on the premix; programme gating; falls back to dynamic mode on TP
- **auphonic**: Adaptive Leveler + loudness target via the Simple API. Set params.preset to a preset with 48 kHz WAV output; per-hour price is plan-dependent (usd_per_hour unverified).
- **dolby-io-media-enhance**: speech levelling + loudness API; docs.dolby.io now redirects to Dolby OptiView, availability unconfirmed
- **dolby-dialogue-gating-reference**: the in-house timing-gated variant (bs1770-levelled) is preferred, as the research advises
- **st-ito**: learned parameter search over a VST chain (plugins not bundled); cannot hit a loudness number
- **deepafx-st**: production-style transfer; licence to clear; no loudness target
- **neural-automix**: no public model trained on delivery levelling; "the wrong tool" per the ledger
- **dialog-plus-remix**: dialogue separation + remix; only its 1–4 kHz dialogue-boost finding is reusable
- **harmony-search-automix**: classical optimisation over level/EQ/DRC; no released code
- **audio-llm-mix-critic**: research idea, untested
- **nugen-lm-correct**: plugin/standalone, no REST API

</details>

### E5

**Watermarking + provenance (deferred)**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| ~~audioseal~~ (disabled) | local | — | MIT (code and weights) | all (✓ en hi ja) | CPU | — | — | not run | [github.com/facebookresearch/audioseal](https://github.com/facebookresearch/audioseal) | `server` |
| ~~aware~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2510.17512](https://arxiv.org/abs/2510.17512) | `server` |
| ~~c2pa-c2patool~~ (disabled) | local | — | MIT OR Apache-2.0 (c2pa-rs / c2patool) | all (✓ en hi ja) | +c2patool | — | — | not run | [github.com/contentauth/c2pa-rs](https://github.com/contentauth/c2pa-rs) | `server` |
| ~~latent-mark~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2603.05310](https://arxiv.org/abs/2603.05310) | `server` |
| ~~silentcipher~~ (disabled) | local | — | MIT | all (✓ en hi ja) | CPU | — | — | not run | [github.com/sony/silentcipher](https://github.com/sony/silentcipher) | `server` |
| ~~voicemark~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~wavmark~~ (disabled) | local | — | MIT | all (✓ en hi ja) | CPU | — | — | not run | [github.com/wavmark/wavmark](https://github.com/wavmark/wavmark) | `server` |
| ~~adobe-content-authenticity~~ (disabled) | api | — | proprietary (maintains c2pa-rs) | all (✓ en hi ja) | CPU | — | — | not run | [opensource.contentauthenticity.org/](https://opensource.contentauthenticity.org/) | `server` |
| ~~generation-time-watermarks~~ (disabled) | method | — | papers CC BY; code unconfirmed | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2606.21365](https://arxiv.org/abs/2606.21365) | `server` |
| ~~provcheck-panel~~ (disabled) | method | — | Apache-2.0 (provcheck pattern) | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~synthid-audio~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [deepmind.google/technologies/synthid/](https://deepmind.google/technologies/synthid/) | `server` |
| ~~truepic-c2pa~~ (disabled) | api | — | proprietary managed signing | all (✓ en hi ja) | CPU | — | — | not run | [truepic.com/](https://truepic.com/) | `server` |

<details><summary>Notes</summary>

- **c2pa-c2patool**: deferred. C2PA 2.4 manifests on WAV/MP3/M4A/MP4; needs a signing certificate (Trust List)
- **audioseal**: deferred. Tamper-evident, not robust (flag removed training-free, arXiv 2608.16566)
- **silentcipher**: deferred; dropped from the Sep 18 shortlist (payload erased by the same attack)
- **synthid-audio**: deferred. Embedder not released; verification only via issuing vendors' APIs
- **wavmark**: deferred; unmaintained since 2024-01; detector role only
- **generation-time-watermarks**: deferred. LambdaMark (arXiv 2606.21365) etc.; requires control of the generator
- **provcheck-panel**: deferred. C2PA hard binding + AudioSeal + latent mark + perceptual hash, cross-checked
- **aware**: deferred. Latent-domain mark (arXiv 2510.17512); survived the 2026 structural attacks
- **voicemark**: deferred. Speaker-specific RVQ-latent mark, hardened against voice cloning
- **latent-mark**: deferred; watch item (arXiv 2603.05310, robust to neural codecs)
- **truepic-c2pa**: deferred; managed C2PA signing
- **adobe-content-authenticity**: deferred; hosted signing on the same C2PA stack

</details>

## F — Picture out

### F1

**Live-action lip sync**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **wav2lip-gan** (baseline) | local | 0.04 B | Wav2Lip weights personal/research/non-commercial only (LRS2); code MIT-style · NC | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | `f1_wav2lip` |
| **highsync** | local | — | MIT (code and HF weights saeed-5959/high_sync) | all (✓ en hi ja) | 12 GB VRAM | ✓ | ✓ | not run | [github.com/saeed5959/high_sync](https://github.com/saeed5959/high_sync) | `f1_highsync` |
| **infinitetalk-480-bf16** | local | 14 B | Apache-2.0 | all (✓ en hi ja) | 48 GB VRAM | ✗ | ✓ | not run | [MeiGen-AI/InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk) | `f1_infinitetalk` |
| **infinitetalk-480-fp8** | local | 14 B | Apache-2.0 | all (✓ en hi ja) | 32 GB VRAM | ✗ | ✓ | not run | [MeiGen-AI/InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk) | `f1_infinitetalk` |
| **keysync** | local | — | Apache-2.0 (code and HF weights toninio19/keysync) | all (✓ en hi ja) | 14 GB VRAM | ✓ | ✓ | not run | [toninio19/keysync](https://huggingface.co/toninio19/keysync) | `f1_keysync` |
| **latentsync-1.5** | local | 0.9 B | openrail++ (weights, ByteDance/LatentSync-1.5); code Apache-2.0 | all (✓ en hi ja) | 9 GB VRAM | ✓ | ✓ | not run | [ByteDance/LatentSync-1.5](https://huggingface.co/ByteDance/LatentSync-1.5) | `f1_latentsync` |
| **latentsync-1.6** | local | 0.9 B | openrail++ (weights, ByteDance/LatentSync-1.6); code Apache-2.0 | all (✓ en hi ja) | 15 GB VRAM | ✓ | ✓ | not run | [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) | `f1_latentsync` |
| **ltx-2.3-dubit-bf16** 🔒 | local | 22 B | LTX-2 Community License (free below $10M ARR; revenue-conditional, not open source) · NC | en +4 | 80 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [Lightricks/LTX-2.3-22b-IC-LoRA-DubIt](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt) | `f1_ltx2` |
| **ltx-2.3-dubit-fp8** 🔒 | local | 22 B | LTX-2 Community License (free below $10M ARR) · NC | en +4 | 50 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8) | `f1_ltx2` |
| **ltx-2.3-dubit-nvfp4** 🔒 | local | 22 B | LTX-2 Community License (free below $10M ARR) · NC | en +4 | 40 GB VRAM, SM ≥ 12.0, key HF_TOKEN | ✗ | ✓ | not run | [Lightricks/LTX-2.3-22b-IC-LoRA-DubIt](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt) | `f1_ltx2` |
| **musetalk-1.5** | local | — | MIT (code and weights; sub-models keep their own licences) | all (✓ en hi ja) | 5 GB VRAM | ✓ | ✓ | not run | [TMElyralab/MuseTalk](https://huggingface.co/TMElyralab/MuseTalk) | `f1_musetalk` |
| **wan2.2-s2v-14b** | local | 16 B | Apache-2.0 | all (✓ en hi ja) | 80 GB VRAM | ✗ | ✓ | not run | [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) | `f1_wan22` |
| **heygen-lipsync-precision** | api | — | proprietary API (HeyGen) | all (✓ en hi ja) | key HEYGEN | ✓ | ✓ | not run | [developers.heygen.com/](https://developers.heygen.com/) | `server` |
| **heygen-lipsync-speed** | api | — | proprietary API (HeyGen) | all (✓ en hi ja) | key HEYGEN | ✓ | ✓ | not run | [developers.heygen.com/](https://developers.heygen.com/) | `server` |
| **lipsync-2** | api | — | proprietary API (sync. labs) | all (✓ en hi ja) | key SYNC | ✓ | ✓ | not run | [docs.sync.so/models/lipsync](https://docs.sync.so/models/lipsync) | `server` |
| **lipsync-2-pro** | api | — | proprietary API (sync. labs) | all (✓ en hi ja) | key SYNC | ✓ | ✓ | not run | [docs.sync.so/models/lipsync](https://docs.sync.so/models/lipsync) | `server` |
| **replicate-kling-lip-sync** | api | — | proprietary API (Kuaishou Kling via Replicate) | all (✓ en hi ja) | key REPLICATE_API_TOKEN | ✓ | ✓ | not run | [replicate.com/kwaivgi/kling-lip-sync](https://replicate.com/kwaivgi/kling-lip-sync) | `server` |
| **replicate-latentsync** | api | 0.9 B | proprietary hosting of LatentSync (openrail++ weights) | all (✓ en hi ja) | key REPLICATE_API_TOKEN | ✓ | ✓ | not run | [replicate.com/bytedance/latentsync](https://replicate.com/bytedance/latentsync) | `server` |
| **replicate-pixverse-lipsync** | api | — | proprietary API (PixVerse via Replicate) | all (✓ en hi ja) | key REPLICATE_API_TOKEN | ✓ | ✓ | not run | [replicate.com/pixverse/lipsync](https://replicate.com/pixverse/lipsync) | `server` |
| **replicate-video-retalking** | api | — | proprietary hosting (VideoReTalking code Apache-2.0; weights research) · NC | all (✓ en hi ja) | key REPLICATE_API_TOKEN | ✓ | ✓ | not run | [replicate.com/chenxwh/video-retalking](https://replicate.com/chenxwh/video-retalking) | `server` |
| **sync-3** | api | — | proprietary API (sync. labs) | all (✓ en hi ja) | key SYNC | ✓ | ✓ | not run | [docs.sync.so/models/lipsync](https://docs.sync.so/models/lipsync) | `server` |
| ~~just-dub-it~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [github.com/justdubit/just-dub-it](https://github.com/justdubit/just-dub-it) | `f1_ltx2` |
| ~~ltx-2.3-dubit-gguf-q4~~ (disabled) | local | 22 B | LTX-2 Community License (free below $10M ARR) · NC | en +4 | 30 GB VRAM | — | — | not run | [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) | `f1_ltx2` |
| ~~omnisync~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2505.21448](https://arxiv.org/abs/2505.21448) | `server` |
| ~~omnihuman-1.5~~ (disabled) | api | — | proprietary API (ByteDance/BytePlus) | all (✓ en hi ja) | CPU | — | — | not run | [omnihuman-lab.github.io/v1_5/](https://omnihuman-lab.github.io/v1_5/) | `server` |
| ~~replicate-musetalk~~ (disabled) | api | — | proprietary hosting of MuseTalk (MIT) | all (✓ en hi ja) | key REPLICATE_API_TOKEN | — | — | not run | [replicate.com/tmappdev/lipsync](https://replicate.com/tmappdev/lipsync) | `server` |
| ~~runway-act-two~~ (disabled) | api | — | proprietary API (Runway) | all (✓ en hi ja) | CPU | — | — | not run | [docs.dev.runwayml.com/](https://docs.dev.runwayml.com/) | `server` |
| ~~veed-lipsync~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [help.elevenlabs.io/hc/en-us/articles/23793433…](https://help.elevenlabs.io/hc/en-us/articles/23793433149073-Do-you-offer-lip-sync-in-Dubbing) | `server` |

<details><summary>Notes</summary>

- **wav2lip-gan**: Today's local baseline (app/providers/lipsync/wav2lip.py). 96x96 mouth; trained against a SyncNet expert, so never rank it by SyncNet LSE. Repo HEAD bac9a81e; GAN checkpoint from the authors' Google Drive (the env recipe fetches it with gdown).
- **latentsync-1.6**: 512 px face crop composited back. Measured 15.8 GB process peak on a 16 GB card (README says 18 GB); vram_gb is set to 15.0 so the runner's free-VRAM check (needs vram+0.5 GB free) admits it on an idle 16 GB card — run it with nothing else resident. Upstream frozen since June 2025; repo commit a229c394. The user's working clone (~/sources/LatentSync, torch 2.11+cu128) can be used by pointing the env python at its .venv and params.repo_dir at it.
- **latentsync-1.5**: 256 px crop (blurrier teeth than 1.6). README states 8 GB; +10% headroom. Runs from the 1.6 repo checkout with the 1.5 UNet in checkpoints/1.5/ (unverified that the 1.6 code path is byte-identical to the 1.5 release for this checkpoint).
- **keysync**: Leakage-free lower-face masking (LipLeak metric), occlusion handling. VRAM is an estimate (SVD-class keyframe + interpolation UNets, WavLM, SAM2.1-L; not stated upstream) — if it exceeds 16 GB it becomes Spark-only. Repo HEAD 5e2503b1. Model reloads per item.
- **musetalk-1.5**: 256x256 latent inpainting, fastest open option (preview renderer). README: fp16 runs on a 4 GB laptop GPU; 5 GB with headroom. MM stack (mmcv 2.x) is the install risk on torch 2.9+ / sm_120 (see envs/f1_musetalk.yaml). Repo HEAD 0a89dec4.
- **highsync**: Named in the F1 research text (HDTF FID 7.36 / CSIM 0.85 / LSE-C 7.72; silence test). Released with weights (github.com/saeed5959/high_sync). VRAM is an estimate for an SD-1.5-class latent lip-sync model; unverified.
- **ltx-2.3-dubit-bf16**: Renamed from LipDub. TEXT-driven: regenerates lips and speech from item.meta.text (items without it are unsupported); output carries its own generated audio. Single speaker, beta. VRAM estimate: 22B BF16 (~44 GB) + Gemma-3-12B (~24 GB) + VAEs/activations. Gemma is gated (HF_TOKEN). ship_ok false = revenue-conditional licence, not NC.
- **ltx-2.3-dubit-fp8**: FP8 distilled checkpoint (~29 GB file for dev-fp8; distilled similar) + Gemma BF16. Would need block-swap offload to fit 16 GB, which Thalassa does not do for 14B+ video models.
- **ltx-2.3-dubit-nvfp4**: Compute-tiers alternate ("fits the Spark at ~20 GB after NVFP4"; that figure excludes the Gemma text encoder). No distilled NVFP4 file exists (only dev), so the BF16 distilled checkpoint is cast at load (--quantization nvfp4-cast). NVFP4 saves memory, not speed, on GB10. Blackwell-only kernels.
- **ltx-2.3-dubit-gguf-q4**: GGUF quants exist (unsloth/LTX-2.3-GGUF, QuantStack/LTX-2.3-GGUF) but only ComfyUI loads them; ltx-pipelines (the DubIt pipeline) has no GGUF loader. Disabled until a non-ComfyUI path exists.
- **infinitetalk-480-bf16**: Wan2.1-I2V-14B-480P + InfiniteTalk adapter, video-to-video (regenerates head/body; CSIM 0.775 on HDTF in its own paper, so the CSIM gate should reject it for dubbing). VRAM estimate (14B BF16 + umT5-XXL + CLIP + wav2vec); README gives no figure. Max 40 s default.
- **infinitetalk-480-fp8**: FP8 DiT (~14 GB) still needs the T5/CLIP/wav2vec encoders and activations; it only fits 16 GB with --num_persistent_param_in_dit 0 (block swap), which Thalassa does not do.
- **wan2.2-s2v-14b**: Single reference image + audio → new video (not video-to-video); a whole-frame comparator expected to lose on identity. README: ≥80 GB on one GPU. Repo HEAD 1ea34ff4.
- **sync-3**: 4K-native, whole-shot model (launched 2026-08-14). $0.107–0.133/s of output at 25 fps (checked 2026-09-28; the higher pay-as-you-go end is billed in _cost_usd). Multipart upload < 20 MB per file.
- **lipsync-2-pro**: 512 px + diffusion super-resolution; $0.067–0.083/s (2026-09-28). Needs natural speaking motion in the input.
- **lipsync-2**: The cheaper sync. labs rung named in the pricing evidence ($0.04–0.05/s, 2026-09-28).
- **heygen-lipsync-precision**: v3 /lipsyncs with our own audio (not Video Translate, which would also replace C2/D1). $0.0667/s (rate seen on its Replicate listing; heygen.com shows no table). Asset upload field name is assumed ("file").
- **heygen-lipsync-speed**: Cheaper HeyGen mode, $0.0333/s (2026-09-28).
- **replicate-latentsync**: The app's lipsync.replicate default model. Which LatentSync version Replicate serves is not stated; cost ≈ $0.10/run on L40S (GPU-seconds billed; estimate).
- **replicate-video-retalking**: Community model: the worker posts to /models/…/predictions, which Replicate only accepts for official models — pin params.version before running. Price is a GPU-time estimate.
- **replicate-musetalk**: Duplicate of the local musetalk-1.5 via a community host; disabled (enable only for an API-only machine, pin version).
- **replicate-kling-lip-sync**: Video 2–10 s, 720p–1080p, < 100 MB; audio < 5 MB. Price not shown on Replicate (estimate).
- **replicate-pixverse-lipsync**: Official Replicate model; price not shown (estimate).
- **omnisync**: NeurIPS 2025; no code or weights released (only the AIGC-LipSync benchmark and a Kling demo).
- **just-dub-it**: Weights released 2026-02-10 (justdubit/justdubit) but the repo is archived in favour of the LTX-2 DubIt pipeline, i.e. the ltx-2.3-dubit-* candidates.
- **omnihuman-1.5**: Single-image-to-avatar generator (~$0.16/s, ~30 s clips); edits no existing footage — evaluate-only shelf for synthetic presenters, not dubbing.
- **runway-act-two**: Performance transfer onto a character reference (driven by a performer video, not audio); not an F1 task.
- **veed-lipsync**: What ElevenLabs Dubbing routes footage to; no public developer API verified.

</details>

### F2

**Mouth-region restoration**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **codeformer-w0.7** | local | 0.09 B | NTU S-Lab License 1.0 (non-commercial) · NC | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [github.com/sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | `f2_codeformer` |
| **codeformer-w1.0** | local | 0.09 B | NTU S-Lab License 1.0 (non-commercial) · NC | all (✓ en hi ja) | 4 GB VRAM | ✓ | ✓ | not run | [github.com/sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | `f2_codeformer` |
| **dicface** | local | — | NTU S-Lab License 1.0 per README (HF card says MIT; no LICENSE file) — treated as NC · NC | all (✓ en hi ja) | 10 GB VRAM | ✓ | ✓ | not run | [fudan-generative-ai/DicFace_model](https://huggingface.co/fudan-generative-ai/DicFace_model) | `f2_dicface` |
| **gfpgan-1.4** | local | 0.08 B | Apache-2.0 except third-party components (commercial status unclear) · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | `f2_gfpgan` |
| **gpen-bfr-512** | local | — | no licence file; Alibaba academic / non-commercial per the research brief · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/yangxy/GPEN](https://github.com/yangxy/GPEN) | `f2_gpen` |
| **keep** | local | — | NTU S-Lab License 1.0 (non-commercial) · NC | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [github.com/jnjaby/KEEP](https://github.com/jnjaby/KEEP) | `f2_keep` |
| **pgtformer** | local | — | PGTFormer licence (BSD-style, non-commercial clause; code and weights) · NC | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | not run | [kepeng/pgtformer-base](https://huggingface.co/kepeng/pgtformer-base) | `f2_pgtformer` |
| **restoreformer** | local | — | unconfirmed (RestoreFormer weights served from the GFPGAN release) · NC | all (✓ en hi ja) | 3 GB VRAM | ✓ | ✓ | not run | [github.com/TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | `f2_gfpgan` |
| **svfr** | local | 1.5 B | code MIT; weights non-commercial research only · NC | all (✓ en hi ja) | 16 GB VRAM, x86_64 | ✗ | ✗ | not run | [github.com/wangzhiyaoo/SVFR](https://github.com/wangzhiyaoo/SVFR) | `f2_svfr` |
| **passthrough** (baseline) | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/f_passthrough.py](server/bench/arena/workers/f_passthrough.py) | `server` |
| **annulus-classical** | method | — | n/a (own implementation) | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/f2_annulus.py](server/bench/arena/workers/f2_annulus.py) | `server` |
| ~~dvface~~ (disabled) | local | 1.3 B | MIT (repo), weights not released | all (✓ en hi ja) | 12 GB VRAM | — | — | not run | [github.com/zhengchen1999/DVFace](https://github.com/zhengchen1999/DVFace) | `server` |
| ~~ip-fvr~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2507.10293](https://arxiv.org/abs/2507.10293) | `server` |
| ~~rgfvr~~ (disabled) | local | 1.3 B | no licence file | all (✓ en hi ja) | CPU | — | — | not run | [github.com/batuhanntosun/RG-FVR](https://github.com/batuhanntosun/RG-FVR) | `server` |
| ~~dvface-large-ensemble~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [github.com/zhengchen1999/DVFace](https://github.com/zhengchen1999/DVFace) | `server` |
| ~~heygen-internal-restoration~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [developers.heygen.com/](https://developers.heygen.com/) | `server` |
| ~~lipsync-2-pro-plus-dvface~~ (disabled) | method | — | proprietary API + research weights | all (✓ en hi ja) | CPU | — | — | not run | [docs.sync.so/models/lipsync](https://docs.sync.so/models/lipsync) | `server` |
| ~~sync-3-builtin-sr~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | CPU | — | — | not run | [docs.sync.so/models/lipsync](https://docs.sync.so/models/lipsync) | `server` |

<details><summary>Notes</summary>

- **passthrough**: Today's behaviour — the raw F1 output is muxed as final.
- **annulus-classical**: Colour/luma transfer + grain re-synthesis from the untouched ring + feathered seam; CPU. Compute-tiers records it as a mandatory layer before any model; here it competes alone (a stacked annulus+model method can be added once a model wins).
- **pgtformer**: 512x512 face video in/out; the worker crops a square around the F1 edit region. Best NIQE (4.97) in DVFace's table. VRAM estimate. conda dlib dependency (aarch64: conda-forge build).
- **dicface**: ICCV 2025 Highlight; weights HF fudan-generative-ai/DicFace_model. VRAM estimate (A800 in the paper).
- **svfr**: SVD-based; README asks for ≥16 GB, so on a 15.9 GB Thalassa card it is borderline (the arena reports "needs 16 GB"); runs on larger x86 cards. x86_64 only: decord has no aarch64 wheel (eva-decord might work — untested).
- **keep**: Kalman-inspired video face SR (ECCV 2024); worst FVD/VIDD in DVFace's table. VRAM estimate.
- **codeformer-w0.7**: Single-frame codebook prior (flickers); the brief keeps it as a floor at high fidelity weight.
- **codeformer-w1.0**: Maximum fidelity weight ("far toward fidelity" per the research).
- **gfpgan-1.4**: Legacy single-image GAN prior, per frame (images only; the worker extracts frames).
- **restoreformer**: RestoreFormer (v1, via GFPGAN's -v RestoreFormer) stands in for the research's RestoreFormer++, which has no inference path in this repo.
- **gpen-bfr-512**: Images only; output file naming undocumented (matched by frame stem). Compiles CUDA ops at first run (needs ninja + nvcc).
- **dvface**: Research/compute-tiers primary (one-step Wan2.1-1.3B restorer, VFHQ PSNR 31.81), but the repo (zhengchen1999/DVFace, 290ad72b) still lists "release code and pretrained models" as TODO. Add a worker when weights land.
- **dvface-large-ensemble**: DVFace on a larger Wan backbone ensembled with DicFace, re-ranked by ArcFace CSIM — blocked on DVFace weights.
- **rgfvr**: Reference-guided restoration (github.com/batuhanntosun/RG-FVR, Wan2.1-T2V-1.3B based); code public but weights "will be uploaded to MediaTUM" (page not reachable 2026-09-28) and no licence. The formulation fits F2 best (untouched frames as the identity reference).
- **ip-fvr**: "Show and Polish" (ACM MM 2025) reference-guided restoration; no public weights found.
- **sync-3-builtin-sr**: Compute-tiers API "pick" is sync-3's built-in 4K generation, which makes F2 unnecessary — it is an F1 candidate (sync-3), not a restorer.
- **lipsync-2-pro-plus-dvface**: Hosted 512 px generation + self-hosted DVFace; blocked on DVFace weights.
- **heygen-internal-restoration**: HeyGen's restoration is internal to its lipsync output and not exposed separately.

</details>

### F3

**2D animation mouth retiming**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **latentsync-1.6-negative-control** | local | 0.9 B | openrail++ (weights) | all (✓ en hi ja) | 15 GB VRAM | ✓ | ✓ | not run | [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6) | `f1_latentsync` |
| **musetalk-1.5-negative-control** | local | — | MIT | all (✓ en hi ja) | 5 GB VRAM | ✓ | ✓ | not run | [TMElyralab/MuseTalk](https://huggingface.co/TMElyralab/MuseTalk) | `f1_musetalk` |
| **rhubarb-1.14** | local | — | MIT | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [github.com/DanielSWolf/rhubarb-lip-sync/relea…](https://github.com/DanielSWolf/rhubarb-lip-sync/releases/tag/v1.14.0) | `f3_rhubarb` |
| **decline-export-cues** (baseline) | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/f3_decline.py](server/bench/arena/workers/f3_decline.py) | `server` |
| ~~anisora-v3.2~~ (disabled) | local | 14 B | Apache-2.0 | all (✓ en hi ja) | CPU | — | — | not run | [github.com/bilibili/Index-anisora](https://github.com/bilibili/Index-anisora) | `server` |
| ~~cel-dit-retrained~~ (disabled) | local | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | — | `server` |
| ~~ltx-2.3-dubit-on-animation~~ (disabled) | local | 22 B | LTX-2 Community License (free below $10M ARR) · NC | en +4 | 80 GB VRAM, key HF_TOKEN | — | — | not run | [Lightricks/LTX-2.3-22b-IC-LoRA-DubIt](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt) | `f1_ltx2` |
| ~~tooncrafter-toncomposer~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2405.17933](https://arxiv.org/abs/2405.17933) | `server` |
| ~~unisync~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2603.03882](https://arxiv.org/abs/2603.03882) | `server` |
| ~~adobe-character-animator~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [adobe.com/products/character-animator.html](https://www.adobe.com/products/character-animator.html) | `server` |
| ~~domoai~~ (disabled) | api | — | proprietary | all (✓ en hi ja) | CPU | — | — | not run | [domoai.app/](https://domoai.app/) | `server` |
| ~~flap-state-resequencing~~ (disabled) | method | — | n/a | all (✓ en hi ja) | CPU | — | — | not run | [server/bench/arena/workers/f3_decline.py](server/bench/arena/workers/f3_decline.py) | `server` |
| ~~sync-3-3d-only~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key SYNC | — | — | not run | [docs.sync.so/models/lipsync](https://docs.sync.so/models/lipsync) | `server` |

<details><summary>Notes</summary>

- **decline-export-cues**: Picture unchanged (today's behaviour) + the dub's speech track exported as open/closed cues for C3 flap fitting. mouth_open = the original line timing (src_speech) or a VAD of the source audio — a proxy for the drawn flaps.
- **rhubarb-1.14**: CPU viseme track (A–F + G/H/X) from the dub audio; pocketSphinx for en, phonetic otherwise (compute-tiers: drive hi from A4/A5 phonemes instead — not wired). A cue exporter: picture unchanged, scored by cue_f1; flap_f1 equals the decline baseline by construction. v1.14.0 has no arm64 binary: the aarch64 recipe builds from source (CMake + Boost).
- **latentsync-1.6-negative-control**: Research bench plan: run a human-face model on anime as a negative control to price a B4 gating failure. No flap track (face detection mostly fails → unsupported items); judged by changed_frac, the sync panel and the user's audit.
- **musetalk-1.5-negative-control**: Second negative control named in the research bench plan.
- **ltx-2.3-dubit-on-animation**: "For the record, not recommended": unevaluated on hand-drawn 2D, near-certain to break line art; text-driven, and the SCENE items carry no dub line text. Enable as a Spark negative control once meta.text exists.
- **sync-3-3d-only**: Vendor claims 3D/CG and AI-generated sources, never hand-drawn 2D; the F3 pack is hand-drawn, so disabled (enable for a 3D pack).
- **flap-state-resequencing**: Classify existing drawn mouths into flap states and re-sequence them to the dub; no published implementation and no public mouth-state classifier for anime.
- **unisync**: Mango TV 2026 paper trained on 2D/3D cartoons; no code, weights or licence.
- **anisora-v3.2**: Anime-native generator (Wan2.2 base); generates animation, does not edit mouths to new dialogue. A fine-tune base, not a candidate.
- **tooncrafter-toncomposer**: Cartoon inbetweening / post-keyframing; no audio conditioning or mouth editing.
- **cel-dit-retrained**: Hypothetical DiT retrained on a cel-animation corpus; no such corpus or model exists.
- **domoai**: Markets stylised-character lip sync; no metrics, papers or verified developer API.
- **adobe-character-animator**: Viseme-driven retiming only for rigs authored inside it; no API, cannot retime delivered footage.

</details>

### F4

**On-screen text replacement**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **clear-render** | local | 1.3 B | Apache-2.0 (CLEAR code + LoRA; base Wan2.1-Fun-V1.1-1.3B-Control Apache-2.0) | all (✓ en hi ja) | 14 GB VRAM | ✓ | ✓ | not run | [charlesw09/CLEAR-mask-free-video-subtitle-rem…](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal) | `f4_clear` |
| **diffueraser-render** | local | — | Apache-2.0 except third-party parts; the ProPainter prior is NTU S-Lab (NC) · NC | all (✓ en hi ja) | 13 GB VRAM | ✓ | ✓ | not run | [lixiaowen/diffuEraser](https://huggingface.co/lixiaowen/diffuEraser) | `f4_diffueraser` |
| **flux2-dev-32b** 🔒 | local | 32 B | FLUX.2-dev Non-Commercial License (gated) · NC | all (✓ en hi ja) | 100 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | `f4_diffusers` |
| **flux2-klein-4b** | local | 4 B | Apache-2.0 | all (✓ en hi ja) | 14.5 GB VRAM | ✓ | ✓ | not run | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `f4_diffusers` |
| **flux2-klein-9b** 🔒 | local | 9 B | FLUX Non-Commercial License (gated) · NC | all (✓ en hi ja) | 32 GB VRAM, key HF_TOKEN | ✗ | ✓ | not run | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `f4_diffusers` |
| **propainter-render** | local | — | NTU S-Lab License 1.0 (non-commercial) · NC | all (✓ en hi ja) | 13 GB VRAM | ✓ | ✓ | not run | [github.com/sczhou/ProPainter](https://github.com/sczhou/ProPainter) | `f4_diffueraser` |
| **qwen-image-edit-2511-bf16** | local | 20 B | Apache-2.0 | all (✓ en hi ja) | 64 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | `f4_diffusers` |
| **qwen-image-edit-2511-nunchaku-fp4** | local | 20 B | Apache-2.0 (base); third-party SVDQuant conversion | all (✓ en hi ja) | 14.5 GB VRAM, 22 GB RAM, x86_64, SM ≥ 12.0 | ✓ | ✗ | not run | [stuqiu/nunchaku-qwen-image-edit-2511](https://huggingface.co/stuqiu/nunchaku-qwen-image-edit-2511) | `f4_nunchaku` |
| **qwen-image-edit-2511-q4-gguf** | local | 20 B | Apache-2.0 | all (✓ en hi ja) | 14.5 GB VRAM, 22 GB RAM | ✓ | ✓ | not run | [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) | `f4_diffusers` |
| **passthrough** (baseline) | method | — | n/a | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/f_passthrough.py](server/bench/arena/workers/f_passthrough.py) | `server` |
| **flux-2-pro-api** | api | — | proprietary API (Black Forest Labs) | all (✓ en hi ja) | key BFL | ✓ | ✓ | not run | [docs.bfl.ai/](https://docs.bfl.ai/) | `server` |
| **flux-kontext-pro** | api | — | proprietary API (Black Forest Labs) | all (✓ en hi ja) | key BFL | ✓ | ✓ | not run | [docs.bfl.ai/kontext/kontext_image_editing](https://docs.bfl.ai/kontext/kontext_image_editing) | `server` |
| **nano-banana-2** | api | — | proprietary API (Google) | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/image-generation](https://ai.google.dev/gemini-api/docs/image-generation) | `server` |
| **nano-banana-pro** | api | — | proprietary API (Google) | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/image-generation](https://ai.google.dev/gemini-api/docs/image-generation) | `server` |
| **overlay-plate-template** | method | — | n/a (own implementation; Noto fonts SIL OFL 1.1) | all (✓ en hi ja) | CPU | ✓ | ✓ | not run | [server/bench/arena/workers/f4_overlay.py](server/bench/arena/workers/f4_overlay.py) | `server` |
| **qwen-image-2.0-api** | api | 7 B | proprietary API (Alibaba Model Studio; no open weights) | all (✓ en hi ja) | key DASHSCOPE | ✓ | ✓ | not run | [alibabacloud.com/help/en/model-studio/qwen-im…](https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit) | `server` |
| **qwen-image-edit-plus-api** | api | — | proprietary API (Alibaba Model Studio) | all (✓ en hi ja) | key DASHSCOPE | ✓ | ✓ | not run | [alibabacloud.com/help/en/model-studio/qwen-im…](https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit) | `server` |
| ~~flux-kontext-dev~~ (disabled) | local | 12 B | FLUX.1 [dev] Non-Commercial | all (✓ en hi ja) | CPU | — | — | not run | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | `server` |
| ~~sedit~~ (disabled) | local | 2.4 B | unknown | all (✓ en hi ja) | 10 GB VRAM | — | — | not run | [zheng222.github.io/SEDiT_project](https://zheng222.github.io/SEDiT_project) | `server` |
| ~~strive~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [github.com/striveiccv2021/STRIVE-ICCV2021](https://github.com/striveiccv2021/STRIVE-ICCV2021) | `server` |
| ~~textctrl~~ (disabled) | local | — | no licence file | en +1 | CPU | — | — | not run | [github.com/weichaozeng/TextCtrl](https://github.com/weichaozeng/TextCtrl) | `server` |
| ~~textwand~~ (disabled) | local | — | paper CC BY 4.0; code unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2606.05730](https://arxiv.org/abs/2606.05730) | `server` |
| ~~vozo-visual-translate~~ (disabled) | api | — | proprietary SaaS API (Vozo) | en hi ja | key VOZO | — | — | not run | [vozo.ai/docs/api_reference/get_started](https://www.vozo.ai/docs/api_reference/get_started) | `server` |

<details><summary>Notes</summary>

- **passthrough**: Today's behaviour — nothing is replaced (the floor for ocr_cer / src_leak).
- **overlay-plate-template**: Research verdict: ship this first. Colour-matched opaque plate + rendered translation, and a JSON cue template for a finisher. Deterministic; looks like a localisation plate.
- **clear-render**: Mask-free subtitle eraser (zero-shot en/ko/fr/ja/ru/de) + the overlay renderer without a plate. 4.86 s/frame in the paper. VRAM is an estimate for a 1.3B Wan2.1 control model at the pack's 1280x720 (not stated upstream). Repo 3cd19a2c; the README's base-model line is wrong (the code loads Wan2.1-Fun-V1.1-1.3B-Control).
- **flux2-klein-4b**: Model card ~13 GB (+10%). Sub-second edits. diffusers ≥ 0.37 (Flux2KleinPipeline).
- **flux2-klein-9b**: Model card ~29 GB (+10%) → Spark (or a 32 GB+ card).
- **flux2-dev-32b**: Unconstrained pick (with best-of-N OCR re-ranking, not implemented). BF16 estimate: 32B transformer (~64 GB) + Mistral-3 24B text encoder (~48 GB) exceeds ~112 GB; the worker would need the FP8/bnb-4bit variants — unverified that it fits the Spark as configured.
- **qwen-image-edit-2511-bf16**: Spark pick for re-render. Weights: 40.9 GB transformer + 16.6 GB text encoder (~58 GB); peak not stated, +10%. The research ledger's "Qwen-Image-Edit" (2508) is pinned to 2511 per the compute-tiers correction.
- **qwen-image-edit-2511-q4-gguf**: Q4_K_M transformer (13.24 GB file). The 16.6 GB Qwen2.5-VL text encoder cannot share a 16 GB card, so cpu_offload moves whole models (model-level offload, not block swap) — tight on Thalassa's 24 GB RAM; unverified. This is an image model, not a 14B+ video DiT.
- **qwen-image-edit-2511-nunchaku-fp4**: No official nunchaku-ai 2511 checkpoint; this third-party FP4 (r128) quant's exact file name is unverified. Nunchaku v1.2.1 wheels are x86_64 only (cu13 torch 2.9–2.11). FP4 needs Blackwell.
- **diffueraser-render**: Generic mask-based video inpainting baseline (README 12 GB at 640x360); box masks come from the pack.
- **propainter-render**: The ProPainter prior DiffuEraser writes along the way; flow-propagation inpainting.
- **nano-banana-pro**: $0.134/image at 1K–2K + $0.0011/input image (2026-09-28). Id without "-preview" per current docs.
- **nano-banana-2**: $0.067/image at 2K (2026-09-28).
- **flux-kontext-pro**: $0.04/image (2026-09-28). Compute-tiers says it is superseded by FLUX.2; kept as the ledger's comparison point.
- **flux-2-pro-api**: FLUX.2 [pro] editing from $0.045/image (per megapixel; 2026-09-28).
- **qwen-image-edit-plus-api**: $0.03/image, Singapore (2026-09-28).
- **qwen-image-2.0-api**: API-only 7B (compute-tiers correction); $0.035/image (2026-09-28).
- **vozo-visual-translate**: Public API (frame_translate) but URL-only input: set params.url_prefix/pack_root to a public mirror of data/eval/F4 and enable. It translates by itself (our translation is passed as guidance); billed in plan points (dollar rate unverified).
- **sedit**: One-step mask-free eraser (ICML 2026) — compute-tiers' Spark erase pick — but only a project page and paper; no code or weights.
- **textctrl**: Best specialist on ScenePair, but crop-level scene-text editing (i_s/i_t folder protocol), Latin/Chinese only (no hi/ja), torch 1.13 + cu116 (no sm_120 build), no licence. Worker not written; would need the crop → edit → paste wrapper.
- **strive**: ICCV 2021 architecture reference; the repo has dataset scripts only (no model code or weights).
- **textwand**: Unified image-level erase/generate/replace (June 2026); no code found.
- **flux-kontext-dev**: Open Kontext weights superseded by FLUX.2 [klein]/[dev] (compute-tiers correction); not registered as a separate runnable row.

</details>

## G — Judges

### G1

**Intelligibility judge (ASR round-trip) — detection of defective takes**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **whisper-large-v3** (baseline) | local | 1.55 B | MIT | all (✓ en hi ja) | 4.5 GB VRAM | ✓ | ✓ | ✓ 3.7 GB, 4s load | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | `server` |
| **canary-1b-v2** | local | 0.978 B | CC-BY-4.0 | en +24 | 5 GB VRAM | ✓ | ✓ | ✓ 9.8 GB, 454s load | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | `a4_nemo` |
| **canary-qwen-2.5b** | local | 2.5 B | CC-BY-4.0 | en | 8 GB VRAM | ✓ | ✓ | ✗ load_failed | [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | `a4_nemo` |
| **kotoba-whisper-v2** | local | 0.756 B | Apache-2.0 | ja | 2 GB VRAM | ✓ | ✓ | ✓ 2.0 GB, 2s load | [kotoba-tech/kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) | `server` |
| **mms-1b-all-ctc** | local | 1 B | CC-BY-NC-4.0 · NC | en hi ja +12 | 5 GB VRAM | ✓ | ✓ | ✓ 2.2 GB, 136s load | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | `server` |
| **mms-1b-all-forced** | local | 1 B | CC-BY-NC-4.0 · NC | en hi ja +12 | 5 GB VRAM | ✓ | ✓ | ✓ 2.3 GB, 12s load | [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | `server` |
| **parakeet-tdt-0.6b-v3** | local | 0.6 B | CC-BY-4.0 | en +24 | 3 GB VRAM | ✓ | ✓ | ✓ 5.0 GB, 16s load | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | `a4_nemo` |
| **qwen3-asr-1.7b** | local | 1.7 B | Apache-2.0 | en hi ja +8 | 5 GB VRAM | ✓ | ✓ | ✓ 4.6 GB, 14s load | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `qwen3asr` |
| **voxtral-small-24b** | local | 24 B | Apache-2.0 | en hi +6 | 56 GB VRAM | ✗ | ✓ | skipped | [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | `a4_transformers` |
| **wav2vec2-large-960h-lv60-self** | local | 0.317 B | Apache-2.0 | en | 2 GB VRAM | ✓ | ✓ | ✓ 0.8 GB, 44s load | [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) | `server` |
| **whisper-large-v3-turbo** | local | 0.809 B | MIT | all (✓ en hi ja) | 2.5 GB VRAM | ✓ | ✓ | ✓ 2.0 GB, 4s load | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | `server` |
| **assemblyai-universal-3-5-pro** | api | — | proprietary API | en hi ja +15 | key ASSEMBLYAI | ✓ | ✓ | skipped | [assemblyai.com/docs/api-reference/transcripts…](https://www.assemblyai.com/docs/api-reference/transcripts/submit) | `server` |
| **elevenlabs-scribe-v2** | api | — | proprietary API | all (✓ en hi ja) | key ELEVENLABS | ✓ | ✓ | skipped | [elevenlabs.io/docs/api-reference/speech-to-te…](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) | `server` |
| **gemini-3.5-transcribe** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | skipped | [ai.google.dev/gemini-api/docs/transcribe](https://ai.google.dev/gemini-api/docs/transcribe) | `server` |
| **whisper-mms-panel** | method | 2.55 B | MIT + CC-BY-NC-4.0 · NC | en hi ja | 10 GB VRAM | ✓ | ✓ | ✓ 6.0 GB, 18s load | [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `server` |
| ~~cohere-transcribe~~ (disabled) | local | — | unknown | en | 6 GB VRAM | — | — | skipped | [CohereLabs](https://huggingface.co/CohereLabs) | `a4_transformers` |
| ~~indicconformer-600m~~ (disabled) | local | 0.6 B | MIT | hi | 3 GB VRAM, key HF_TOKEN | — | — | skipped | [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | `indicconformer` |
| ~~cross-family-rank-ensemble~~ (disabled) | method | — | mixed | en | CPU | — | — | skipped | [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `server` |

<details><summary>Notes</summary>

- **whisper-large-v3**: The judgelib ASR-panel default (asr_roundtrip "whisper"); same params, so its G1 rank is the rank of the judge every TTS spec uses. Seq2seq with a language prior: may repair misreadings.
- **whisper-large-v3-turbo**: OpenDub's current ASR (A4 baseline); cheaper Whisper-family judge.
- **qwen3-asr-1.7b**: judgelib panel member "qwen3" (A4 winner for hi). Strong LLM prior: the research warns it recovers surface-correct text from wrong readings (false negatives).
- **kotoba-whisper-v2**: Plan G1 row names kotoba-whisper for the ja panel. Distil-Whisper lineage (Whisper family).
- **mms-1b-all-ctc**: The "strict" judge: wav2vec2 CTC with per-language adapters (ISO 639-3), greedy, no LM, so misreadings stay visible. judgelib panel member "mms". Japanese output script of the jpn adapter not checked (CER assumes native script).
- **mms-1b-all-forced**: Forced-alignment posterior instead of free decoding: defect_score = CTC NLL of the intended line per label (cannot hallucinate). Characters outside the adapter vocabulary are dropped.
- **wav2vec2-large-960h-lv60-self**: Apache-licensed English CTC (third lineage, no LM); upper-case character vocabulary.
- **whisper-mms-panel**: Two lineage-disjoint recognisers (Whisper seq2seq + MMS CTC) gated on agreement: defect score = max CER of the two. Stand-in for the unconstrained "cross-family rank ensemble".
- **parakeet-tdt-0.6b-v3**: Spark verifier pick (TDT decoder cannot free-hallucinate). 25 European languages: no hi/ja, so it only judges en here. Reuses the A4 NeMo worker. Never pair with Canary as the two opinions (same FastConformer lineage).
- **canary-1b-v2**: Research #1 primary (NeMo lineage, disjoint from Whisper). European languages only.
- **canary-qwen-2.5b**: English-only SALM (LLM decoder: expect more "repair" than Canary-1B-v2).
- **voxtral-small-24b**: Strongest open multilingual recogniser (audio-LLM). BF16 ~48 GB weights + activations (estimate): Spark only. No ja.
- **elevenlabs-scribe-v2**: API pick; use as the reported evaluator, never as a verifier optimised into.
- **assemblyai-universal-3-5-pro**: Compute-tiers names "Universal 3 Pro"; the A4 worker's current model is universal-3-5-pro (18 languages incl. hi, ja per the worker docstring; list above not individually checked).
- **gemini-3.5-transcribe**: Released 2026-08-26, 85+ languages (compute-tiers). Reuses the A4 worker.
- **cohere-transcribe**: Research lists it as an evaluate-only commercial API; the A4 worker has a "cohere" family. Model id / licence not verified here: copy the A4 registration and enable.
- **indicconformer-600m**: Plan's Indic panel member; disabled per user decision 2026-09-28 (gated repo).
- **cross-family-rank-ensemble**: Three-lineage rank ensemble (Yu & Kang, ICML 2026 workshop) needs recognisers from different envs in one judgement; compose it from G1 outputs as a derived metric once the single judges are ranked. whisper-mms-panel is the implemented two-lineage version.

</details>

### G2

**Speaker-similarity judge — agreement with human similarity ratings**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **ecapa-voxceleb** (baseline) | local | 0.0208 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | ✗ env_setup_failed | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | `judge_speaker` |
| **eres2netv2** | local | 0.0178 B | Apache-2.0 | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | ✗ load_failed | [modelscope.cn/models/iic/speech_eres2netv2_sv…](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common) | `judge_speaker` |
| **redimnet2-b3-lm** | local | 0.0041 B | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | ✓ 0.4 GB, 140s load | [github.com/PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) | `judge_speaker` |
| **redimnet2-b6-lm** | local | 0.0123 B | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | ✓ 0.3 GB, 9s load | [github.com/PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) | `judge_speaker` |
| **redimnet2-b6-lm-cnc** | local | 0.0123 B | MIT | all (✓ en hi ja) | 1 GB VRAM | ✓ | ✓ | ✓ 0.0 GB, 143s load | [github.com/PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) | `judge_speaker` |
| **resemblyzer** | local | 0.0014 B | Apache-2.0 | all (✓ en hi ja) | CPU | ✓ | ✓ | ✓ 0.2 GB, 1s load | [github.com/resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | `judge_speaker` |
| **wavlm-base-plus-sv** | local | 0.095 B | MIT | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | ✓ 0.7 GB, 23s load | [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv) | `judge_speaker` |
| **wespeaker-resnet293-lm** | local | 0.028 B | Apache-2.0 | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | ✗ load_failed | [Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) | `judge_speaker` |
| **gemini-3.7-flash-same-voice** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | skipped | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| ~~titanet-large~~ (disabled) | local | 0.023 B | CC-BY-4.0 | all (✓ en hi ja) | 1 GB VRAM | — | — | skipped | [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) | `a4_nemo` |
| ~~voxsim-wavlm-ecapa~~ (disabled) | local | 0.4 B | code unknown; VoxSim data CC BY 4.0 | all (✓ en hi ja) | 3 GB VRAM | — | — | skipped | [github.com/kaistmm/voxsim_trainer](https://github.com/kaistmm/voxsim_trainer) | `judge_speaker` |
| ~~wavlm-large-ecapa-seedtts~~ (disabled) | local | 0.317 B | MIT (UniSpeech) | all (✓ en hi ja) | 2 GB VRAM | — | — | skipped | [github.com/microsoft/UniSpeech/tree/main/down…](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) | `judge_speaker` |
| ~~azure-speaker-recognition~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key AZURE_SPEECH, AZURE_SPEECH_REGION | — | — | skipped | [learn.microsoft.com/azure/ai-services/speech-…](https://learn.microsoft.com/azure/ai-services/speech-service/speaker-recognition-overview) | `judge_speaker` |
| ~~redimnet2-sweep-ensemble~~ (disabled) | method | — | MIT | all (✓ en hi ja) | CPU | — | — | skipped | [github.com/PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) | `judge_speaker` |
| ~~riva-nim-titanet~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key NGC | — | — | skipped | [build.nvidia.com/nvidia](https://build.nvidia.com/nvidia) | `judge_speaker` |
| ~~vmc2026-t04-ensemble~~ (disabled) | method | — | mixed | all (✓ en hi ja) | CPU | — | — | skipped | [arxiv.org/abs/2609.13792](https://arxiv.org/abs/2609.13792) | `judge_speaker` |

<details><summary>Notes</summary>

- **ecapa-voxceleb**: The encoder OpenDub's tts.speaker_similarity slot names (app/pipeline/quality.py). VoxSim paper: pretrained ECAPA LCC 0.768 / SRCC 0.758 (better than WavLM-ECAPA despite worse EER).
- **wavlm-base-plus-sv**: transformers WavLMForXVector; card threshold 0.86 for same speaker. Licence = UniSpeech repo licence.
- **redimnet2-b6-lm**: Spark identity-score pick (0.287% VoxCeleb1-O EER). torch.hub pinned to commit c5bbe0b.
- **redimnet2-b3-lm**: The Sept-3 research's working judge (B3); compute-tiers moved to B6.
- **redimnet2-b6-lm-cnc**: B6 trained with CN-Celeb added (non-English speakers): the cross-lingual variant. Whether the lm train type exists for this dataset tag is not verified (README lists the dataset for B3/B6).
- **eres2netv2**: 3D-Speaker (Mandarin-heavy 200k-speaker training): the disjoint lineage. CAM++-family judges favour CosyVoice (plan §4.7) — do not rank CosyVoice with 3D-Speaker models alone.
- **wespeaker-resnet293-lm**: Best-validated production recipe (VoxCeleb + CN-Celeb); HF repo layout (avg_model.pt + config.yaml) assumed.
- **resemblyzer**: Listed to rule out (GE2E d-vector, hobby-pipeline floor); CPU-viable.
- **gemini-3.7-flash-same-voice**: LLM same-speaker rating (1..5), asked in both orders and averaged (open LALMs were 0.97-1.00 position-locked). Tie-breaker only per compute-tiers. Prices checked 2026-09-28.
- **voxsim-wavlm-ecapa**: Spark perceptual-score pick (WavLM-ECAPA regressor fine-tuned on VoxSim, LCC 0.835). The checkpoint is on Google Drive and the repo only prints corpus Pearson; a per-pair scoring wrapper around its SpeakerNet is not written yet. Note: evaluating it on the VoxSim pack is train/test-adjacent (it was fine-tuned on VoxSim train).
- **wavlm-large-ecapa-seedtts**: WavLM-Large + ECAPA head (the Seed-TTS-eval SIM model). Needs the UniSpeech code and a Google-Drive checkpoint; not wired into the worker yet.
- **titanet-large**: Local stand-in for the API pick (Riva/NIM TitaNet endpoint, used by Hume's benchmark). The NeMo env has TitaNet, but no worker returns speaker-sim payloads from it yet.
- **riva-nim-titanet**: Hosted TitaNet embedding endpoint; no worker (compute-tiers: nothing hosted beats a local 12M model).
- **azure-speaker-recognition**: No worker; Azure Speaker Recognition was announced for retirement (verify availability).
- **vmc2026-t04-ensemble**: VoiceMOS 2026 T3 winner (8-model ensemble + listener conditioning); no released weights.
- **redimnet2-sweep-ensemble**: Rank ensemble over all ReDimNet2 sizes; needs a multi-model mode in the worker (not written).

</details>

### G3

**Naturalness (MOS) predictor — agreement with human MOS**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **audiobox-aesthetics-ce** | local | — | CC-BY-4.0 | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | ✗ run_failed | [facebook/audiobox-aesthetics](https://huggingface.co/facebook/audiobox-aesthetics) | `judge_mos` |
| **audiobox-aesthetics-pq** | local | — | CC-BY-4.0 | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | ✗ run_failed | [facebook/audiobox-aesthetics](https://huggingface.co/facebook/audiobox-aesthetics) | `judge_mos` |
| **distill-mos** | local | 0.0043 B | MIT | all (✓ en hi ja) | 0.5 GB VRAM | ✓ | ✓ | ✓ 0.3 GB, 1s load | [github.com/microsoft/Distill-MOS](https://github.com/microsoft/Distill-MOS) | `judge_mos` |
| **dnsmos-p808** | local | — | MIT | all (✓ en hi ja) | CPU | ✓ | ✓ | ✗ load_failed | [github.com/microsoft/DNS-Challenge/tree/maste…](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | `judge_mos` |
| **nisqa-tts** | local | — | MIT code; weights CC BY-NC-SA 4.0 · NC | all (✓ en hi ja) | CPU | ✓ | ✓ | ✓ 0.2 GB, 2s load | [github.com/gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) | `judge_mos` |
| **nisqa-v2** | local | — | MIT code; weights CC BY-NC-SA 4.0 · NC | all (✓ en hi ja) | CPU | ✓ | ✓ | ✗ load_failed | [github.com/gabrielmittag/NISQA](https://github.com/gabrielmittag/NISQA) | `judge_mos` |
| **utmosv2** | local | — | MIT | all (✓ en hi ja) | 2.5 GB VRAM | ✓ | ✓ | ✗ env_setup_failed | [github.com/sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | `judge_mos` |
| **xls-r-sqa-2b** | local | 2 B | MIT | all (✓ en hi ja) | 6 GB VRAM | ✓ | ✓ | ✗ load_failed | [github.com/lcn-kul/xls-r-analysis-sqa](https://github.com/lcn-kul/xls-r-analysis-sqa) | `judge_mos` |
| **xls-r-sqa-300m** | local | 0.3 B | MIT | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | ✗ load_failed | [github.com/lcn-kul/xls-r-analysis-sqa](https://github.com/lcn-kul/xls-r-analysis-sqa) | `judge_mos` |
| **gemini-3.1-pro-mos** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | skipped | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gemini-3.7-flash-mos** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | skipped | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gpt-audio-1.5-mos** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | skipped | [developers.openai.com/api/docs/models/gpt-aud…](https://developers.openai.com/api/docs/models/gpt-audio-1.5) | `g6_api` |
| ~~indicmos~~ (disabled) | local | — | unknown | hi | CPU | — | — | skipped | [github.com/AI4Bharat](https://github.com/AI4Bharat) | `judge_mos` |
| ~~mambarate~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | skipped | [arxiv.org/search/?query=MambaRate&searchtype=all](https://arxiv.org/search/?query=MambaRate&searchtype=all) | `judge_mos` |
| ~~sq-llm~~ (disabled) | local | — | unknown | all (✓ en hi ja) | CPU | — | — | skipped | [arxiv.org/abs/2510.14664](https://arxiv.org/abs/2510.14664) | `judge_mos` |
| ~~ttsds2~~ (disabled) | method | — | open source (verify) | en hi ja +11 | CPU | — | — | skipped | [github.com/ttsds/ttsds](https://github.com/ttsds/ttsds) | `judge_mos` |
| ~~vmc2026-t09-ensemble~~ (disabled) | method | — | mixed | all (✓ en hi ja) | CPU | — | — | skipped | [arxiv.org/abs/2609.13792](https://arxiv.org/abs/2609.13792) | `judge_mos` |

<details><summary>Notes</summary>

- **utmosv2**: VMC2024 track-1 winner (high-quality synthetic speech); per-take ranker pick. Param count differs between the artifacts (~120M vs ~400M class): left unset. Within-language only.
- **distill-mos**: XLS-R-SQA student (different lineage from UTMOSv2); cheap second opinion.
- **audiobox-aesthetics-pq**: Production Quality axis (the artefact gate). Weights are CC-BY-4.0 per the repo (the research guessed NC). All four axes are in the payload metrics.
- **audiobox-aesthetics-ce**: Content Enjoyment axis, to check which Audiobox axis tracks naturalness MOS.
- **nisqa-v2**: NISQA v2.0 via torchmetrics (48 kHz input); noisiness/coloration/discontinuity/loudness sub-scores kept.
- **nisqa-tts**: The dedicated TTS-naturalness head (weights/nisqa_tts.tar in the repo; API args unverified).
- **dnsmos-p808**: Listed to rule out (noise-suppression metric); ONNX on CPU via torchmetrics.
- **xls-r-sqa-2b**: The Distill-MOS teacher family (XLS-R truncated at layer 10 + transformer head; params_b is the untruncated backbone). VRAM is an estimate.
- **xls-r-sqa-300m**: Smaller teacher variant for the size/quality curve.
- **gemini-3.7-flash-mos**: LALM MOS judge (VMC2026 organisers' baseline class); prices checked 2026-09-28.
- **ttsds2**: Research #1, but distributional and system-level (needs a system's whole output set vs references): it cannot give a per-utterance prediction, so utterance SRCC does not apply. Use it inside D-phase system comparisons instead.
- **mambarate**: AudioMOS 2025 T3 system (sampling-rate robust); code/weights release not verified.
- **vmc2026-t09-ensemble**: VoiceMOS 2026 winners (29-model ensemble T09 / 64-listener WavLM T02); not released.
- **sq-llm**: SpeechLLM-as-Judge fine-tuned quality rater; weight release not verified.
- **indicmos**: The plan names IndicMOS for Indic naturalness; checkpoint not located/verified.

</details>

### G4

**Emotion-consistency judge — agreement with human emotion labels**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **hubert-large-superb-er** (baseline) | local | 0.317 B | Apache-2.0 | all (✓ en hi ja) | 1.5 GB VRAM | ✓ | ✓ | not run | [superb/hubert-large-superb-er](https://huggingface.co/superb/hubert-large-superb-er) | `judge_emotion` |
| **audio-flamingo-3-emotion** | local | 8 B | NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 20 GB VRAM | ✗ | ✓ | not run | [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf) | `g6_audio_flamingo` |
| **emotion2vec-plus-large** | local | 0.3 B | emotion2vec model licence (see card) | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [emotion2vec/emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) | `judge_emotion` |
| **emotion2vec-plus-large-posterior** | local | 0.3 B | emotion2vec model licence (see card) | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [emotion2vec/emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) | `judge_emotion` |
| **odyssey-wavlm-avd** | local | 0.317 B | MIT | all (✓ en hi ja) | 2 GB VRAM | ✓ | ✓ | not run | [3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes) | `judge_emotion` |
| **qwen3-omni-30b-a3b-emotion** | local | 30.5 B | Apache-2.0 | all (✓ en hi ja) | 80 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `g6_qwen3_omni` |
| **gemini-3.7-flash-emotion** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gpt-audio-1.5-emotion** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models/gpt-aud…](https://developers.openai.com/api/docs/models/gpt-audio-1.5) | `g6_api` |
| ~~audio-flamingo-next-emotion~~ (disabled) | local | 8 B | NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 20 GB VRAM | — | — | not run | [arxiv.org/abs/2604.10905](https://arxiv.org/abs/2604.10905) | `g6_audio_flamingo` |
| ~~cross-lingual-ser~~ (disabled) | local | — | varies | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/list/eess.AS/recent](https://arxiv.org/list/eess.AS/recent) | `judge_emotion` |
| ~~emobox-backbones~~ (disabled) | method | — | open source (verify) | all (✓ en hi ja) | CPU | — | — | not run | [github.com/emo-box/EmoBox](https://github.com/emo-box/EmoBox) | `judge_emotion` |
| ~~hume-expression-prosody~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key HUME | — | — | not run | [dev.hume.ai/docs/expression-measurement/overview](https://dev.hume.ai/docs/expression-measurement/overview) | `g6_api` |
| ~~vmc2026-t02-ensemble~~ (disabled) | method | — | mixed | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2609.13792](https://arxiv.org/abs/2609.13792) | `judge_emotion` |

<details><summary>Notes</summary>

- **hubert-large-superb-er**: What OpenDub uses today (4-class IEMOCAP SER); posterior cosine of take vs source.
- **emotion2vec-plus-large**: Utterance-embedding cosine. The 2026 "False Resonance" audit found this near chance across a content/language change; ranked here to confirm on our data.
- **emotion2vec-plus-large-posterior**: The categorical second opinion (9-class posterior cosine) rather than the embedding.
- **odyssey-wavlm-avd**: Primary pick: arousal/dominance/valence regression (MSP-Podcast); score = −Euclidean A/V/D distance. Output order per card: arousal, dominance, valence (~0..1).
- **audio-flamingo-3-emotion**: Best audio-only emotion accuracy of six audited LALM judges (0.68). "Same feeling" 1..5, both orders averaged, clips joined with silence (one audio per request). VRAM estimated.
- **qwen3-omni-30b-a3b-emotion**: Spark only (BF16). Audit: 0.35 audio-only emotion accuracy (Instruct).
- **gemini-3.7-flash-emotion**: The VMC2026 organisers' EMOS baseline class; both A/B orders averaged.
- **gpt-audio-1.5-emotion**: Audit: 0.29 audio-only emotion accuracy; independent-vendor reference.
- **hume-expression-prosody**: API pick (prosody emotion vectors → cosine); worker not written (API shape unverified).
- **vmc2026-t02-ensemble**: VoiceMOS 2026 T2 winner (EMOS UTT-SRCC 0.758); no released weights.
- **emobox-backbones**: A benchmark harness (10 backbones), not a judge; its ensemble is a possible later method.
- **cross-lingual-ser**: Research direction (zero-shot cross-lingual SER, Deep-WCCN); no checkpoint to register.
- **audio-flamingo-next-emotion**: HF id and transformers support not verified; enable once the checkpoint is confirmed.

</details>

### G5

**Lip-sync scorer — response to injected A/V offsets**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **syncnet-lse** (baseline) | local | — | MIT code; VGG research weights · NC | all (✓ en hi ja) | 3 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/joonson/syncnet_python](https://github.com/joonson/syncnet_python) | `judge_lipsync_syncnet` |
| **peavs** | local | — | Apache-2.0 | all (✓ en hi ja) | 6 GB VRAM, x86_64, +ffmpeg | ✓ | ✗ | not run | [github.com/amazon-science/avgen-eval-toolkit](https://github.com/amazon-science/avgen-eval-toolkit) | `judge_lipsync_peavs` |
| **synchformer-audioset** | local | — | MIT | all (✓ en hi ja) | 4 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/v-iashin/Synchformer](https://github.com/v-iashin/Synchformer) | `judge_lipsync_synchformer` |
| **synchformer-lrs3** | local | — | MIT | all (✓ en hi ja) | 4 GB VRAM, +ffmpeg | ✓ | ✓ | not run | [github.com/v-iashin/Synchformer](https://github.com/v-iashin/Synchformer) | `judge_lipsync_synchformer` |
| **gemini-3.1-pro-sync** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/video-understan…](https://ai.google.dev/gemini-api/docs/video-understanding) | `g6_api` |
| **gemini-3.7-flash-sync** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/video-understan…](https://ai.google.dev/gemini-api/docs/video-understanding) | `g6_api` |
| ~~avhubert-sync-expert~~ (disabled) | local | — | CC-BY-NC-4.0 (AV-HuBERT backbone) · NC | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2405.04327](https://arxiv.org/abs/2405.04327) | `judge_lipsync_avhubert` |
| ~~imagebind-av~~ (disabled) | local | — | CC-BY-NC-4.0 · NC | all (✓ en hi ja) | CPU | — | — | not run | [github.com/facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind) | `judge_lipsync_imagebind` |
| ~~sparsesync~~ (disabled) | local | — | MIT (verify) | all (✓ en hi ja) | CPU | — | — | not run | [github.com/v-iashin/SparseSync](https://github.com/v-iashin/SparseSync) | `judge_lipsync_synchformer` |
| ~~per-face-track-panel~~ (disabled) | method | — | mixed | all (✓ en hi ja) | CPU | — | — | not run | [github.com/v-iashin/Synchformer](https://github.com/v-iashin/Synchformer) | `judge_lipsync_synchformer` |
| ~~theval~~ (disabled) | method | — | unknown | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/search/?query=THEval&searchtype=all](https://arxiv.org/search/?query=THEval&searchtype=all) | `judge_lipsync_synchformer` |

<details><summary>Notes</summary>

- **syncnet-lse**: The bench harness's declared slot; LSE-C/LSE-D for comparability only (not shift invariant, negatively correlated with preference). Offset sign convention unverified (params.sign).
- **synchformer-lrs3**: LRS3-trained (talking heads; 86.6% Acc@1 on LRS3). 0.2 s offset grid: sub-grid offsets come from the posterior mean. Upstream env is torch 2.0 / py3.8; torch 2.8 is unverified.
- **synchformer-audioset**: AudioSet-trained checkpoint (general AV sync, sparse cues); expected weaker on faces.
- **peavs**: Co-primary (human-calibrated 1..5; Pearson 0.79 set / 0.54 clip). I3D flow uses cupy PWC kernels: kept x86_64-only until cupy + OpenCV are checked on aarch64 (the env has an aarch64 recipe to try). No signed offset.
- **gemini-3.7-flash-sync**: Video-LLM "is the mouth obviously wrong" rater (probability, no offset).
- **avhubert-sync-expert**: AVSu/AVSm/AVSv metrics (shift-invariant); code/weight release not verified, no backend yet.
- **sparsesync**: Sparse-cue sibling of Synchformer; no backend in the worker yet.
- **imagebind-av**: Semantic AV correspondence (wrong voice on screen), not a sync scorer; near chance on offsets.
- **theval**: An evaluation framework to read, not a scorer.
- **per-face-track-panel**: Shot-segmented, per-face-track PEAVS + Synchformer (+ AV-HuBERT) panel with LLM triage: the structural change; build from the ranked single scorers.

</details>

### G6

**Multimodal reviewer — per-defect detection on injected defects**

| Candidate | Kind | Size | Licence | Languages | Needs | Thalassa | Spark | Smoke (thalassa) | Get it | Env |
|---|---|---|---|---|---|---|---|---|---|---|
| **audio-flamingo-3-decomposed** | local | 8 B | NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 20 GB VRAM | ✗ | ✓ | not run | [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf) | `g6_audio_flamingo` |
| **qwen3-omni-30b-a3b-instruct-decomposed** | local | 30.5 B | Apache-2.0 | all (✓ en hi ja) | 90 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `g6_qwen3_omni` |
| **qwen3-omni-30b-a3b-instruct-holistic** | local | 30.5 B | Apache-2.0 | all (✓ en hi ja) | 90 GB VRAM | ✗ | ✓ | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `g6_qwen3_omni` |
| **gemini-3.1-pro-decomposed** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gemini-3.5-flash-holistic** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gemini-3.7-flash-decomposed** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gemini-3.7-flash-holistic** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gemini-3.8-flash-decomposed** | api | — | proprietary API | all (✓ en hi ja) | key GEMINI | ✓ | ✓ | not run | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | `g6_api` |
| **gpt-audio-1.5-decomposed** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models/gpt-aud…](https://developers.openai.com/api/docs/models/gpt-audio-1.5) | `g6_api` |
| **gpt-audio-1.5-holistic** | api | — | proprietary API | all (✓ en hi ja) | key OPENAI | ✓ | ✓ | not run | [developers.openai.com/api/docs/models/gpt-aud…](https://developers.openai.com/api/docs/models/gpt-audio-1.5) | `g6_api` |
| **qwen3.5-omni-plus** | api | — | proprietary API | all (✓ en hi ja) | key DASHSCOPE | ✓ | ✓ | not run | [alibabacloud.com/help/en/model-studio/qwen-omni](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) | `g6_api` |
| **transcript-review-claude-opus-5** | method | — | proprietary API + MIT (Whisper) | all (✓ en hi ja) | 4.5 GB VRAM, key ANTHROPIC | ✓ | ✓ | not run | [docs.claude.com/en/api/overview](https://docs.claude.com/en/api/overview) | `g6_api` |
| ~~audio-flamingo-next~~ (disabled) | local | 8 B | NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 20 GB VRAM | — | — | not run | [arxiv.org/abs/2604.10905](https://arxiv.org/abs/2604.10905) | `g6_audio_flamingo` |
| ~~qwen3-omni-30b-a3b-thinking~~ (disabled) | local | 30.5 B | Apache-2.0 | all (✓ en hi ja) | 80 GB VRAM | — | — | not run | [Qwen/Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) | `g6_qwen3_omni` |
| ~~sq-llm-judge~~ (disabled) | local | — | varies | all (✓ en hi ja) | CPU | — | — | not run | [arxiv.org/abs/2510.14664](https://arxiv.org/abs/2510.14664) | `g6_audio_flamingo` |
| ~~claude-opus-5-direct~~ (disabled) | api | — | proprietary API | all (✓ en hi ja) | key ANTHROPIC | — | — | not run | [docs.claude.com/en/docs/about-claude/models/o…](https://docs.claude.com/en/docs/about-claude/models/overview) | `g6_api` |
| ~~qwen3-omni-captioner-text-judge~~ (disabled) | method | 30.5 B | Apache-2.0 | all (✓ en hi ja) | 80 GB VRAM | — | — | not run | [Qwen/Qwen3-Omni-30B-A3B-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) | `g6_qwen3_omni` |
| ~~qwen3-omni-self-consistency-af-next~~ (disabled) | method | — | Apache-2.0 + NVIDIA OneWay Noncommercial · NC | all (✓ en hi ja) | 100 GB VRAM | — | — | not run | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | `g6_qwen3_omni` |

<details><summary>Notes</summary>

- **gemini-3.7-flash-holistic**: API pick (audio + video natively). Promo price through 2026-12-31, then $1.50 / $7.50.
- **gemini-3.7-flash-decomposed**: AudioJudge-style decomposition (lexical / identity both orders / signal / sync sub-judges).
- **gemini-3.8-flash-decomposed**: Newer than both research artifacts (listed on the pricing page 2026-09-28).
- **gemini-3.1-pro-decomposed**: The model class with the published reliability study (calibration drift noted).
- **gemini-3.5-flash-holistic**: The Sept-3 primary (superseded by 3.7 Flash); kept to measure the change.
- **gpt-audio-1.5-holistic**: Independent vendor; audio only (off-sync items count as misses).
- **gpt-audio-1.5-decomposed**: Audit found GPT-Audio 0.97 position-locked on A/B; the identity sub-judge averages both orders.
- **qwen3.5-omni-plus**: API-only (no weights). OpenAI-compatible DashScope endpoint; request shape for video/audio data URLs and streaming follows the Model Studio docs but is unverified; no price pinned (no _cost_usd recorded).
- **qwen3-omni-30b-a3b-instruct-decomposed**: Spark pick (open weights, audio + video). Card: 78.85 GB peak for a 15 s video in BF16 (talker disabled here). transformers, not vLLM (vLLM sm_121 issue #36821).
- **audio-flamingo-3-decomposed**: Audio only (off-sync = miss); VRAM estimated (8B BF16 ≈ 17 GB weights) → Spark / 24 GB+.
- **transcript-review-claude-opus-5**: Transcript + language-ID + clipping measurements reviewed by Claude (text-only). Cannot judge voice identity or sync (reports 0 → misses by design). On aarch64 set asr device cpu.
- **claude-opus-5-direct**: Claude models take text + images only (no audio/video input), so no direct review is possible.
- **qwen3-omni-30b-a3b-thinking**: Excluded on evidence (100% position-locked in the 2026 judge audit; "never Thinking").
- **qwen3-omni-captioner-text-judge**: Captioner describing the take for a text judge; "captioner" protocol not implemented.
- **audio-flamingo-next**: 30-min audio context, timestamp-grounded CoT; HF id / transformers class not verified.
- **sq-llm-judge**: Fine-tuned SpeechLLM quality judge; weight release not verified.
- **qwen3-omni-self-consistency-af-next**: Unconstrained pick (16-64-way self-consistency + AF-Next ensemble); sampling mode not implemented.

</details>

## Load / unload smoke tests

What each status means: `ok` loaded, produced an output and returned GPU memory to the baseline after the worker exited; `leak` left GPU memory or a process (model server, Ollama model) behind; `load_failed` / `run_failed` see error; `env_missing` the environment was not set up; `skipped` not runnable on this machine (reason given).

| Host | Capability | Candidate | Status | Load s | Peak VRAM | Left after exit | Downloaded | Error / reason |
|---|---|---|---|---|---|---|---|---|
| thalassa | A4 | fw-large-v3-turbo | ok | 2.638 | 2.0 GB | 0 MiB | — |  |
| thalassa | A4 | qwen3-asr-1.7b | ok | 16.628 | 5.9 GB | 0 MiB | — |  |
| thalassa | G1 | assemblyai-universal-3-5-pro | skipped | — | — | — | — | missing API key(s): ASSEMBLYAI_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G1 | canary-1b-v2 | ok | 453.793 | 9.8 GB | 0 MiB | 5.9 GB |  |
| thalassa | G1 | canary-qwen-2.5b | load_failed | — | — | 0 MiB | — | worker exit -11 |
| thalassa | G1 | cohere-transcribe | skipped | — | — | — | — | disabled in registry |
| thalassa | G1 | cross-family-rank-ensemble | skipped | — | — | — | — | disabled in registry |
| thalassa | G1 | elevenlabs-scribe-v2 | skipped | — | — | — | — | missing API key(s): ELEVENLABS_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G1 | gemini-3.5-transcribe | skipped | — | — | — | — | missing API key(s): GEMINI_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G1 | indicconformer-600m | skipped | — | — | — | — | disabled in registry; missing API key(s): HF_TOKEN |
| thalassa | G1 | kotoba-whisper-v2 | ok | 1.869 | 2.0 GB | 0 MiB | — |  |
| thalassa | G1 | mms-1b-all-ctc | ok | 136.24 | 2.2 GB | 0 MiB | 3.6 GB |  |
| thalassa | G1 | mms-1b-all-forced | ok | 12.465 | 2.3 GB | 0 MiB | — |  |
| thalassa | G1 | parakeet-tdt-0.6b-v3 | ok | 15.76 | 5.0 GB | 0 MiB | — |  |
| thalassa | G1 | qwen3-asr-1.7b | ok | 13.985 | 4.6 GB | 0 MiB | — |  |
| thalassa | G1 | voxtral-small-24b | skipped | — | — | — | — | needs 56 GB VRAM, have 15.9 |
| thalassa | G1 | wav2vec2-large-960h-lv60-self | ok | 44.373 | 0.8 GB | 0 MiB | 2.4 GB |  |
| thalassa | G1 | whisper-large-v3 | ok | 3.709 | 3.7 GB | 0 MiB | — |  |
| thalassa | G1 | whisper-large-v3-turbo | ok | 3.721 | 2.0 GB | 0 MiB | — |  |
| thalassa | G1 | whisper-mms-panel | ok | 17.884 | 6.0 GB | 0 MiB | — |  |
| thalassa | G2 | azure-speaker-recognition | skipped | — | — | — | — | disabled in registry; missing API key(s): AZURE_SPEECH_KEY, AZURE_SPEECH_REGION; API candidate (pass --include-api; costs money) |
| thalassa | G2 | ecapa-voxceleb | env_setup_failed | — | — | — | — | env judge_speaker setup failed |
| thalassa | G2 | eres2netv2 | load_failed | — | — | 0 MiB | — | worker exit 1 |
| thalassa | G2 | gemini-3.7-flash-same-voice | skipped | — | — | — | — | missing API key(s): GEMINI_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G2 | redimnet2-b3-lm | ok | 139.58 | 0.4 GB | 0 MiB | — |  |
| thalassa | G2 | redimnet2-b6-lm | ok | 8.698 | 0.3 GB | 0 MiB | — |  |
| thalassa | G2 | redimnet2-b6-lm-cnc | ok | 142.786 | 0.0 GB | 0 MiB | — |  |
| thalassa | G2 | redimnet2-sweep-ensemble | skipped | — | — | — | — | disabled in registry |
| thalassa | G2 | resemblyzer | ok | 1.218 | 0.2 GB | 0 MiB | — |  |
| thalassa | G2 | riva-nim-titanet | skipped | — | — | — | — | disabled in registry; missing API key(s): NGC_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G2 | titanet-large | skipped | — | — | — | — | disabled in registry |
| thalassa | G2 | vmc2026-t04-ensemble | skipped | — | — | — | — | disabled in registry |
| thalassa | G2 | voxsim-wavlm-ecapa | skipped | — | — | — | — | disabled in registry |
| thalassa | G2 | wavlm-base-plus-sv | ok | 22.704 | 0.7 GB | 0 MiB | 0.8 GB |  |
| thalassa | G2 | wavlm-large-ecapa-seedtts | skipped | — | — | — | — | disabled in registry |
| thalassa | G2 | wespeaker-resnet293-lm | load_failed | — | — | 0 MiB | — | worker exit 1 |
| thalassa | G3 | audiobox-aesthetics-ce | run_failed | 2.487 | 0.4 GB | 0 MiB | — | ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function. |
| thalassa | G3 | audiobox-aesthetics-pq | run_failed | 20.001 | 0.4 GB | 0 MiB | 0.4 GB | ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function. |
| thalassa | G3 | distill-mos | ok | 1.3 | 0.3 GB | 0 MiB | — |  |
| thalassa | G3 | dnsmos-p808 | load_failed | — | — | 0 MiB | — | worker exit 1 |
| thalassa | G3 | gemini-3.1-pro-mos | skipped | — | — | — | — | missing API key(s): GEMINI_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G3 | gemini-3.7-flash-mos | skipped | — | — | — | — | missing API key(s): GEMINI_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G3 | gpt-audio-1.5-mos | skipped | — | — | — | — | missing API key(s): OPENAI_API_KEY; API candidate (pass --include-api; costs money) |
| thalassa | G3 | indicmos | skipped | — | — | — | — | disabled in registry |
| thalassa | G3 | mambarate | skipped | — | — | — | — | disabled in registry |
| thalassa | G3 | nisqa-tts | ok | 2.032 | 0.2 GB | 0 MiB | — |  |
| thalassa | G3 | nisqa-v2 | load_failed | — | — | 0 MiB | — | worker exit 1 |
| thalassa | G3 | sq-llm | skipped | — | — | — | — | disabled in registry |
| thalassa | G3 | ttsds2 | skipped | — | — | — | — | disabled in registry |
| thalassa | G3 | utmosv2 | env_setup_failed | — | — | — | — | env judge_mos setup failed |
| thalassa | G3 | vmc2026-t09-ensemble | skipped | — | — | — | — | disabled in registry |
| thalassa | G3 | xls-r-sqa-2b | load_failed | — | — | 0 MiB | — | worker exit 1 |
| thalassa | G3 | xls-r-sqa-300m | load_failed | — | — | 0 MiB | — | worker exit 1 |

### App model manager (server process)

Models the OpenDub server loads in-process, acquired through `app/providers/_runtime.py` and then unloaded. *Left after unload* is GPU memory above the level before loading (the CUDA context itself stays once created).

| Host | Model | Status | Load s | GPU while loaded | Left after unload | Note |
|---|---|---|---|---|---|---|
| thalassa | asr.faster_whisper large-v3 | ok | 3.91 | 3.7 GB | 0 MiB |  |
| thalassa | asr.faster_whisper large-v3-turbo | ok | 1.84 | 2.1 GB | 0 MiB |  |
| thalassa | tts.f5_tts F5-Hindi | ok | 1.98 | 0.7 GB | 0 MiB |  |
| thalassa | tts.f5_tts F5TTS_v1_Base | ok | 9.35 | 0.9 GB | 0 MiB |  |
