# Model candidates by compute budget

Companion to the 3 September 2026 "OpenDub Model Research" pass (41 categories, quality-only ranking) and to
`docs/plans/capability-map-implementation.md`. Prepared 18 September 2026 by seven parallel research passes,
one per phase. Objective: **best quality of output**; cost and compute are not limiters except where the box is fixed.

Every category is answered three ways:

- **Spark**: the best model that runs well on one NVIDIA DGX Spark (GB10; 128 GB unified LPDDR5x at ~273 GB/s; Blackwell SM 12.1; aarch64; CUDA 13).
- **Unconstrained**: what to run with unlimited compute (a B200/GB300 rack or any hosted cluster), and whether it beats the Spark pick by a meaningful margin.
- **API**: the best hosted model, with version and date.

Plus a **compute-sensitivity score** (1 = solved at small scale, the Spark pick is the ceiling; 5 = quality keeps scaling and one Spark is a real bottleneck) and a **delta** against the 3 September primary.

## What the DGX Spark means for this pipeline

- Every audio model in the map fits with room to spare; the natural way to spend the box is best-of-N generation with judge re-ranking (G1–G4).
- Large language models fit at 4–8 bit but decode at single-digit tokens per second per stream because the box is bandwidth-bound (rule of thumb: tokens/s ≈ 273 / model bytes). Batching restores throughput; latency does not improve.
- Diffusion video models (lip sync, text replacement) run but are compute-bound: practical per shot, not per film.
- Ecosystem: PyTorch aarch64 CUDA wheels are fine; flash-attention, xformers, bitsandbytes, TensorRT-LLM, vLLM and NIM support are checked per pick below and are the usual source of friction.

## Where more compute buys quality

Ranked by compute-sensitivity score. Spend first at the top.

| Score | ID | Category | Why |
|---|---|---|---|
| 5 | C2 | Translation and dubbing adaptation | The most compute-sensitive category in this phase - but the scaling lever is best-of-N search with QE reranking, not raw parameter count. |
| 5 | D4 | Regional and low-resource language voices | The one Phase D category where scale plainly pays: the tail is data- and capacity-starved in a way English and Chinese are not, so both parameters and RL compute keep buying quality long after they have stopped in D1. |
| 5 | F1 | Live-action lip sync | The most compute-sensitive category in the pipeline: quality scales along four independent axes at once - model size (0.9B UNet -> 14B -> 22B DiT), sampling steps, resolution (512x512 crop -> 4K full frame) and best-of-N re-ranking - and one Spark can pay for none of them. |
| 4 | A4 | Transcription with word-level timing | The text scales but the timing does not: more compute buys fewer wrong words, not better sync. |
| 4 | A8 | Paralinguistic and delivery tagging | The Phase A category where more compute buys the most - the best open model a Spark can hold is a 30B-A3B MoE while the API frontier is hundreds of billions - but tier-one measured prosody is compute-free and already captures most of the actionable value. |
| 4 | C4 | Reference-free translation quality estimation | Quality tracks the base model and the ensemble size more than raw parameter count, and the single biggest win - a judge that sees the whole scene - is a context-window property rather than a FLOPs property. |
| 4 | D2 | Expressive and style-transfer synthesis | The failure mode here is variance rather than capacity - roughly one take in eight lands the reading - so best-of-N against an emotion-consistency judge converts compute into quality more directly than in any other D category, and the judged metric is not one the model was trained against, so the headroom is not already consumed. |
| 4 | D3 | Duration-controlled synthesis | Not because larger models fit durations better, but because rejection sampling against a hard constraint is the textbook compute-for-quality trade, and on a Spark N=16 at RTF 0.4 costs roughly 6-7x realtime per line - acceptable offline and it eliminates the tempo stretch. |
| 4 | D8 | Direct speech-to-speech translation | S2ST is the one Phase D category where model scale clearly moves the needle, but more compute does not flip the adopt/don't-adopt decision - a rack of B200s buys a better comparator, not a better product. |
| 4 | G6 | Multimodal review | The frontier hosted tier is measurably better behaved as a judge than any open 24-30B model, but post-training rather than parameter count is what drives the difference, and the best system-level agreement comes from decomposition, which is free. |
| 3 | A1 | Dialogue, music and effects separation | Masking models are near saturation and generation-over-generation gains are small, but generative (flow-matching) CASS newly rewards sampling steps and ensembles. |
| 3 | A2 | Reference-clip enhancement | The ceiling is set by the source clip, not the model - picking a clean 8-second window out of a 2-hour feature beats any restorer, and that is free. |
| 3 | D1 | Zero-shot cross-lingual voice cloning | More parameters have stopped buying cloning quality, but more sampling and more post-training still do - and a 128 GB box does both, so the Spark is not the bottleneck. |
| 3 | D6 | Voice conversion | At under 1B parameters Vevo2 is nowhere near a capacity wall, and its own results show data and objectives moving the numbers rather than size; more compute buys judge-re-ranked sampling, which is a reliable but modest lift. |
| 3 | E3 | Acoustic scene matching | The parametric route is effectively free and gets most of the way, which caps the score; but unlike E1/E4/E5 there is a documented quality gap that compute closes, and the ceiling is set by the task definition (plausibly the same room) rather than by the model. |
| 3 | F4 | On-screen text replacement | Split by sub-problem, which is why it is neither a 4 nor a 2: erasure is solved at small scale and the Spark is genuinely at the ceiling, while re-render legibility in complex scripts still scales with model size and best-of-N sampling. |
| 2 | A3 | Speech, music and singing classification | Close to solved at small scale; the remaining errors are definitional (is sung dialogue speech or music?) rather than capacity-limited. |
| 2 | A6 | Diarization with overlap handling | Progress comes from training data, pruning recipes and calibration, not parameters. |
| 2 | B1 | Active speaker detection and face tracking | ASD is close to saturated on film-like footage, and the remaining quality comes from face-crop resolution, correct track association across shot cuts and domain-matched fine-tuning rather than from model scale. |
| 2 | B3 | On-screen text reading | More compute buys almost nothing here and can buy negative: the 2026 evidence is that tiny specialists beat frontier VLMs at OCR, and the real wins are temporal (tracking a string across frames and voting) and resolution/crop strategy rather than parameter count. |
| 2 | B4 | Content type classification (live action vs 2D animation vs 3D etc.) | A coarse 3-5-way visual-style decision is solved by a frozen encoder plus a linear head; more compute only moves the ambiguous tail, and even there the gain comes from multi-frame temporal evidence and calibrated abstention rather than parameter count. |
| 2 | C1 | Dub-line segmentation | This is a search problem with a cheap scoring model; the cost function and word-level timestamps dominate, not model size. |
| 2 | C3 | Viseme-aware phonetic adaptation | Quality is set by the vowel-distance/DTW cost and the duration gate, not by generator size. |
| 2 | C5 | Normalization, grapheme-to-phoneme and lexicon | Lexicon coverage and TN grammar coverage dominate; more compute halves an already-small error rate and a tiny discriminative model beats the LLMs it was distilled from. |
| 2 | C6 | Subtitle condensation | A 32B open model already recovers most of the available compliance while improving SubER; the residual error is line segmentation and timestamp assignment, not rewriting. |
| 2 | D5 | Licensed non-cloning voices | The binding constraint is provenance and casting depth, not model capacity; best-of-N applies once per character at casting time, not per line, so a 128 GB box buys a better audition process and little else. |
| 2 | D7 | Non-verbal vocalizations | The ceiling is a capability gap, not a compute gap - the winning strategy (splice the original) costs essentially nothing, and re-sampling does not fix a categorical failure where the model omits the vocalisation or renders it as a word. |
| 2 | E1 | Timing fit and pause mapping | The DP aligner and the stretcher are solved at essentially zero compute and are the parts that fix the audible defect; only the paraphrase/regeneration escalation scales, and its gain is capped by how much slack the line has. |
| 2 | E2 | Bandwidth extension and restoration | Bandwidth extension is close to solved at ~30M parameters and the scaling evidence points the other way: generative decisively beats deterministic at every size, but bigger generative does not beat smaller generative, and extra sampling steps buy nothing. |
| 2 | F2 | Mouth-region restoration | Bounded by the source footage rather than by model capacity: the output must blend invisibly into unrestored surrounding pixels, so headroom is small and excess capacity is a liability because over-restoration makes the annulus look better than the rest of the face. |
| 2 | G1 | Intelligibility judge | Within a family a bigger verifier helps modestly, but family diversity and N are far larger levers than judge size, and most of the oracle headroom stays unexploited. |
| 2 | G3 | Naturalness prediction | A 111x smaller student loses only 0.05 correlation against its teacher, and the 2026 challenge was won by ensembling and listener conditioning rather than scale. |
| 2 | G4 | Emotion consistency judge | Bigger is measurably worse here - 24B and 30B audio LLMs land near chance on audio-only emotion while a 300M WavLM head does the job; the score is 2 rather than 1 only because the ceiling is far from solved and ensembling still helps. |
| 2 | G5 | Lip-sync scoring | The ceiling is the metric's agreement with viewers, and no larger model has beaten the best small one; the documented failure of the incumbent is an architecture bug, not a capacity limit. |
| 1 | A5 | Text-to-audio forced alignment | Solved at small scale and the Spark pick is the ceiling; scaling the aligner cannot help because the error is already at the level of human labelling disagreement. |
| 1 | A7 | Speaker embeddings and voice-bank matching | Solved. ReDimNet2's explicit contribution is pushing the accuracy-per-GMAC Pareto front, not scaling. |
| 1 | B2 | Shot boundary detection | A sub-10M-parameter 3D CNN from 2020 is essentially at the ceiling on broadcast material; the residual errors are precision errors on flashes and overlays that better problem framing fixes, not more FLOPs. |
| 1 | E4 | Dialogue levelling and loudness | The measurement is a fixed algorithm defined in BS.1770-5; a two-pass implementation is exact, and no amount of compute improves exact. A Spark is four orders of magnitude more compute than this needs. |
| 1 | E5 | Watermarking and provenance | Signing is cryptography with zero learned parameters, and every credible watermarker is under ~100M parameters and runs faster than realtime on a CPU. Robustness is a property of the embedding domain, not of scale. |
| 1 | F3 | 2D animation mouth retiming | Solved-at-small-scale in the degenerate sense: the Spark pick IS the ceiling because the ceiling is a CPU-only phoneme recogniser and the frontier above it is empty. |
| 1 | G2 | Speaker similarity judge | A better verifier is measurably a worse perceptual judge; the entire open SOTA EER range is invisible to human listeners and the gain comes from human-rating supervision. |

## Summary tables

15 of 41 Spark picks differ from the 3 September primary (marked **changed**).

### Phase A · Source analysis · audio

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| A1 | Dialogue, music and effects separation | Bandit v2 (DnR v3 multilingual checkpoint) (tens of millions (band-split RNN; exact count not published, checkpoints <1 GB fp32) · FP32 or BF16 with overlap-add chunking · 6 GB) | AV-CASS (audio-visual cinematic source separation via conditional flow matching) | AudioShake Dialogue / Music / Effects (DME) separation (AudioShake · 2025-10-27 (latest dialogue-separation model post)) | 3/5 |
| A2 | Reference-clip enhancement | UniPASE (DeWavLM-Omni + adapter + vocoder) (~0.4-0.5B (WavLM-Large-class distilled encoder ~315M plus adapter and vocoder) · BF16 · 3 GB) | AnyEnhance | AudioShake voice isolator / dialogue separation API (AudioShake · 2025-10-27) | 3/5 |
| A3 | Speech, music and singing classification | CED-base (Consistent Ensemble Distillation) (86M · BF16 · 1 GB) | Audio-LLM adjudicator over dense CED posteriors (Qwen3.5-Omni-Plus, or Qwen3-Omni-30B-A3B-Instruct for open weights) | Gemini 3-class audio models (gemini-3.5-transcribe plus Gemini 3.8 Live Extended Thinking) (Google DeepMind · 2026-09-15 (Gemini 3.8 Audio model card); gemini-3.5-transcribe listed as New Stable) | 2/5 |
| A4 | Transcription with word-level timing | Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B (2.3B combined (1.7B ASR with a 300M AuT encoder, plus a 0.6B non-autoregressive aligner) · BF16 · 5 GB) | ROVER-style N-best fusion of Qwen3-ASR-1.7B + Canary-Qwen-2.5B + MOSS-Transcribe-Diarize, LLM-arbitrated on disagreement spans, then re-aligned with Qwen3-ForcedAligner | Scribe v2 (ElevenLabs · 2026-03-11) | 4/5 |
| A5 | Text-to-audio forced alignment | Montreal Forced Aligner 3.x (n/a (Kaldi GMM/DNN acoustic models, per-language) · CPU float32 · 4 GB) | Three-way aligner ensemble (MFA 3.x + Qwen3-ForcedAligner-0.6B + MMS_FA) with per-word agreement scoring, plus MFA speaker-adapted training | Forced Alignment API (ElevenLabs · current as of 2026-09 (no dated version published)) | 1/5 |
| A6 | Diarization with overlap handling **(changed)** | DiariZen-Large-s80-v2 (pruned WavLM-Large + Conformer, powerset classification, VBx clustering) (~315M before structural pruning · BF16 · 3 GB) | DOVER-Lap fusion of DiariZen + community-1 + Streaming Sortformer, with enrolment-based target speaker extraction on overlapped regions, over an unpruned WavLM-Large encoder | Precision-3 (pyannoteAI · 2026 (released after Precision-2; available on API and on-premise per the changelog)) | 2/5 |
| A7 | Speaker embeddings and voice-bank matching **(changed)** | ReDimNet2-B6 (12.3M (13 GMACs) · BF16 or FP32 - the model is too small for precision to matter · 1 GB) | Score-fusion ensemble of ReDimNet2-B6 + ERes2NetV2 + WeSpeaker ResNet293-LM with large-margin fine-tuning and quality-aware score calibration (QMF) | pyannoteAI voiceprints / speaker identification (Precision-3 tier) (pyannoteAI · 2026 (Precision-3 generation)) | 1/5 |
| A8 | Paralinguistic and delivery tagging | Measured prosody from A5 alignments (tier one) + Qwen3-Omni-30B-A3B-Instruct as delivery-note writer (tier two) (30B total / 3B active (MoE) · BF16, or FP8/NVFP4 to free memory for the rest of the pipeline · 62 GB) | Qwen3.5-Omni-Plus | Qwen3.5-Omni-Plus (Alibaba Cloud · 2026-04-21 (technical report arXiv 2604.15804v2); model released April 2026) | 4/5 |

### Phase B · Source analysis · picture

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| B1 | Active speaker detection and face tracking | LoCoNet trained with TalkNCE loss (+ SCRFD face detection and shot-cut-aware ByteTrack-class tracking) (~30-40M for the ASD network; SCRFD-10GF detector is ~3.9M · BF16 (FP32 also trivially affordable) · 3 GB) | GateFusion (Hierarchical Gated Cross-Modal Fusion) | Gemini 3.1 Pro (video input) (Google DeepMind · 2026-02-19) | 2/5 |
| B2 | Shot boundary detection | TransNetV2 (+ PySceneDetect threshold/fade detectors as a second channel) (<10M (roughly 41 GMACs per inference window) · FP32 or BF16 - the model is too small for precision to matter · 1.5 GB) | OmniShotCut (shot-query Transformer) | Cloud Video Intelligence API - SHOT_CHANGE_DETECTION (Google Cloud · GA service, no 2026 version bump verified) | 1/5 |
| B3 | On-screen text reading **(changed)** | PP-OCRv6_medium, run through the ONNX exports on onnxruntime-gpu (not on PaddlePaddle) (34.5M total (15.5M detection + 19M recognition) on the PPLCNetV4 backbone · FP16 ONNX (FP32 is also cheap at this size) · 1 GB) | PP-OCRv6_medium + PaddleOCR-VL-1.6 consensus with GoMatching++ track-level voting | Gemini 3.1 Pro (Google DeepMind · 2026-02-19) | 2/5 |
| B4 | Content type classification (live action vs 2D animation vs 3D etc.) | SigLIP 2 (so400m) frozen embeddings + calibrated linear head trained on OpenDub frames, with conformal abstention (~400M vision encoder; the head is a few thousand parameters · BF16 · 2 GB) | GLM-5.3-Flash (320B total / 18B active MoE, natively multimodal) multi-frame vote | Gemini 3.1 Pro (Google DeepMind · 2026-02-19) | 2/5 |

### Phase C · Text transformation

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| C1 | Dub-line segmentation | SaT sat-12l-sm (wtpsplit) + `ted` LoRA, driving a pause/parse/duration split search over faster-whisper word timestamps (0.3B · BF16 (or ONNX on CPU) · 0.7 GB) | Frontier LLM doing joint segmentation + translation + length budgeting in one scene-level pass, with best-of-N reranked by a QE model | Claude Opus 5 (claude-opus-5) (Anthropic · 2026-09 (current lineup)) | 2/5 |
| C2 | Translation and dubbing adaptation **(changed)** | Hy-MT2-30B-A3B-FP8 (vLLM), with Qwen3.8-27B for the long-context adaptation pass (30B total / ~3B active (MoE) · FP8 (official checkpoint) · 32 GB) | Gemini 3.1 Pro | Gemini 3.1 Pro (Google · 2026-02-19) | 5/5 |
| C3 | Viseme-aware phonetic adaptation | PS-Comet-style N-best reranking: gpt-oss-120b (MXFP4) generating paraphrase candidates, scored by vowel-level DTW over G2P output plus a semantic term (117B total / 5.1B active for the generator; the scorer is non-neural (DTW) plus a small G2P model · MXFP4 generator; FP32 CPU scorer · 64 GB) | Same PS-Comet method with a frontier LLM in the generator slot | Claude Opus 5 (claude-opus-5) (Anthropic · 2026-09 (current lineup)) | 2/5 |
| C4 | Reference-free translation quality estimation **(changed)** | MetricX-24-Hybrid-XXL (bfloat16) in QE / reference-free mode (~13B (mT5-XXL encoder-decoder) · bfloat16 · 26 GB) | AUTORANK-style ensemble: MetricX-25-class scalar QE + GemSpanEval-class fine-tuned span model + frontier LLM GEMBA-ESA judge | Gemini 3.1 Pro as a GEMBA-ESA error-span judge (Google · 2026-02-19) | 4/5 |
| C5 | Normalization, grapheme-to-phoneme and lexicon | nemo-text-processing 1.2.0 (WFST TN/ITN) + misaki (en/ja/zh/ko/vi) + espeak-ng 1.52 OOV fallback + project PLS lexicon, with a local 8-30B instruct model arbitrating flagged homographs (0 (WFST + rule-based) plus an 8-30B arbiter · CPU WFST; BF16/FP8 for the arbiter · 17 GB) | Frontier LLM in parse mode for homograph/polyphone/name decisions, with deterministic WFST TN/ITN underneath and dictionary-constrained decoding on top | Claude Opus 5 (claude-opus-5); Claude Fable 5.1 for the hardest cases (Anthropic · 2026-09 (current lineup)) | 2/5 |
| C6 | Subtitle condensation | HW-TSC IWSLT 2026 two-pass LLM condenser on Qwen3-32B (pass 1 greedy at temperature 0, pass 2 at 0.3), rewriting only the cues that violate CPS/CPL (32.8B dense · Q8_0 (or MXFP4 / 4-bit if you want headroom) · 33 GB) | AppTek's full subtitling stack: length-class-conditioned Transformer-Big MT + iterative shortening + ILS neural line segmentation + LLM automatic post-editing over ~20-sentence SRT chunks | Claude Opus 5 (claude-opus-5) (Anthropic · 2026-09 (current lineup)) | 2/5 |

### Phase D · Speech generation

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| D1 | Zero-shot cross-lingual voice cloning **(changed)** | Fish Audio S2 (4.4B (4B slow AR + 400M fast AR) · bf16 · 11 GB) | Fish Audio S2 at N=32 with cross-family ASR + SIM + MOS judge re-ranking | Qwen-Audio-3.0-TTS Plus (Alibaba Cloud Model Studio / DashScope · 2026-07-23 (Artificial Analysis Speech Arena snapshot where it ranked #1 at Elo 1234; Elo 1259 and #2 in the September 2026 snapshot)) | 3/5 |
| D2 | Expressive and style-transfer synthesis **(changed)** | IndexTTS 2.5 (0.8B · bf16 · 4 GB) | Fish Audio S2 | Eleven v3 (ElevenLabs · 2026-02-02 (GA)) | 4/5 |
| D3 | Duration-controlled synthesis **(changed)** | IndexTTS 2.5 (0.8B · bf16 · 4 GB) | IndexTTS 2.5 in token-count mode, N=16-32 per line with duration-window rejection and WER/SIM re-ranking | Chirp 3: HD (Google Cloud Text-to-Speech · GA as of September 2026 (57 locales, 28 voices; Punjabi-IN and zh-HK in Preview)) | 4/5 |
| D4 | Regional and low-resource language voices **(changed)** | VoxCPM2 (2B · bf16 · 8 GB) | Fish Audio S2 | Sonic 3.6 (Cartesia · 2026-08-27) | 5/5 |
| D5 | Licensed non-cloning voices **(changed)** | VoxCPM2 voice-design mode (2B · bf16 · 8 GB) | VoxCPM2 voice-design with best-of-N casting (N=64 candidate voices per role) | MAI-Voice-2 / MAI-Voice-2-Flash (Microsoft Azure AI Speech · 2026-09-09 (language-support docs last updated 9-10 September 2026)) | 2/5 |
| D6 | Voice conversion | Vevo2 (872M total (509M AR content-style + 363M flow-matching acoustic + 255M unified vocoder) · bf16 · 3 GB) | Re-synthesis instead of conversion: drive Fish Audio S2 or VoxCPM2 with the known transcript and the target speaker reference | Voice Changer (speech-to-speech) (ElevenLabs · 2026-02-02 (Eleven v3 generation, GA)) | 3/5 |
| D7 | Non-verbal vocalizations | Pass-through splicing gated by CED / BEATs audio tagging, with Step-Audio-EditX as the synthesis fallback (CED-base ~90M and BEATs ~90M for detection; Step-Audio-EditX 3B for synthesis · bf16 · 13 GB) | Fish Audio S2 | Qwen-Audio-3.0-TTS Plus (Alibaba Cloud Model Studio · 2026-07-23 (Artificial Analysis Speech Arena snapshot; Elo 1259 in the September 2026 snapshot)) | 2/5 |
| D8 | Direct speech-to-speech translation | UniSS (evaluate-only comparator) (Small speech-LLM scale (text-LLM backbone + speech semantic/style modelling) · bf16 · 16 GB) | Seed-Live 2.0 and Step-Audio 2 (speech-LLM class) | Dubbing API (productised cascade control condition) (ElevenLabs · 2026-02-02 (Eleven v3 generation, GA)) | 4/5 |

### Phase E · Audio post-production

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| E1 | Timing fit and pause mapping | Pause-structure prosodic alignment (DP over A4 word timings) + Signalsmith Stretch residual TSM (0 (dynamic programming + phase-locked STFT stretcher) · float64 DP, float32 audio; CPU only · 0.1 GB) | Dub-S2ST (discrete-diffusion speech-to-unit translation with explicit duration control + conditional flow matching synthesis) | ElevenLabs Dubbing v2 (ElevenLabs · docs verified 18 September 2026 (v2 current)) | 2/5 |
| E2 | Bandwidth extension and restoration | AP-BWE (48 kHz checkpoint) (29.76M · FP32 (FP16 fine; nothing here needs BF16) · 0.5 GB) | The same models - UniPASE (545.7M) or Sidon as the largest credible open restorers; Miipher-2 is the closed frontier but is not available as weights | Adobe Podcast Enhance Speech (Adobe · March 2026 (Advanced Source Separation update)) | 2/5 |
| E3 | Acoustic scene matching | Learned acoustic embeddings + feedback delay network parameter matching (VAE embedding of reverberant speech, differentiable FDN optimised to match it) (small VAE (single-digit millions) plus the FDN coefficient set; well under 50M total · FP32 · 0.5 GB) | Gencho, escalated to many diffusion samples per scene with re-ranking on reverberation-metric agreement | Accentize Chameleon (and Chameleon Surround) (Accentize · current product as of 2026; no Chameleon 2 release verified) | 3/5 |
| E4 | Dialogue levelling and loudness | BS.1770 two-pass measurement (libebur128 or pyloudnorm) + static delivery-preset chain + dialogue-gated measurement driven by OpenDub's own A4 word timings (0 · float64 measurement, float32 audio; CPU only · 0.05 GB) | The same DSP chain - unchanged | Dolby.io Media Enhance API (Dolby Laboratories · current docs as of 2026; platform migrating to Dolby OptiView) | 1/5 |
| E5 | Watermarking and provenance **(changed)** | C2PA Content Credentials via c2pa-rs / c2patool (0 (signing, hashing and manifest embedding) · n/a · 0.3 GB) | A panel, not a bigger model: C2PA hard binding + AudioSeal + a latent-domain mark + a perceptual-hash soft binding, all cross-checked at verification time | Google SynthID (audio), now a cross-vendor standard (Google DeepMind, with OpenAI, ElevenLabs and Kakao · 31 July 2026 (OpenAI extended SynthID to audio from ChatGPT and the OpenAI API, with verification API access); >100 billion items marked by May 2026; Lyria 3 carries the SynthID extension) | 1/5 |

### Phase F · Picture generation

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| F1 | Live-action lip sync **(changed)** | LatentSync 1.6 (~0.9B SD1.5 UNet + VAE + Whisper audio encoder · BF16 (no quantisation needed) · 18 GB) | LTX-2.3-22b IC-LoRA LipDub | sync-3 (sync. labs · 2026 (current default model, verified 18 September 2026)) | 5/5 |
| F2 | Mouth-region restoration | DVFace (Wan 2.1 T2V backbone (1.3B-class) with spatio-temporal dual-prior codebooks · BF16 · 12 GB) | DVFace on a larger Wan backbone with multi-step sampling, ensembled with DicFace and re-ranked by an ArcFace/CSIM identity judge | sync-3 (built-in 4K super-resolution) (sync. labs · 2026 (verified 18 September 2026)) | 2/5 |
| F3 | 2D animation mouth retiming | Rhubarb Lip Sync (used as a viseme/flap-count oracle feeding C3, not as a picture generator) (n/a - a classical recogniser, not a neural generator · n/a (CPU, no GPU path)) | None - decline to lip-sync animation and move the effort to C3 flap-aware script adaptation | sync-3 (weak, flagged - 3D/CG only, never hand-drawn 2D) (sync. labs · 2026 (verified 18 September 2026)) | 1/5 |
| F4 | On-screen text replacement **(changed)** | SEDiT (erase) + Qwen-Image-Edit-2511 (re-render on keyframes) + planar-tracked composite (SEDiT: LTX-Video-2B-0.9.6 backbone + rank-256 LoRA adding 381M trainable params (19% of base). Qwen-Image-Edit-2511: 20B. · BF16 for SEDiT; BF16 (~40 GB) or NVFP4 (~11 GB) for Qwen-Image-Edit-2511 · 8 GB) | SEDiT (erase) + FLUX.2 [dev] 32B (re-render) with best-of-N re-ranking by an OCR legibility judge | Nano Banana Pro (Gemini 3 Pro Image) (Google DeepMind · 17 November 2025) | 3/5 |

### Phase G · Automatic quality judges

| ID | Category | Spark pick | Unconstrained | API | Compute |
|---|---|---|---|---|---|
| G1 | Intelligibility judge | Parakeet-TDT-0.6B-v3 (verifier) + Whisper-large-v3 (disjoint evaluator) + wav2vec2/MMS CTC (third family) (600M + 1.55B + ~300M · BF16 / FP16 (faster-whisper int8_float16 for Whisper) · 6 GB) | Cross-family rank ensemble over 3+ disjoint ASR lineages (Whisper + wav2vec2 + Voxtral/Canary) | Scribe v2 (ElevenLabs · reported on ASR Leaderboard v4, 27 Mar 2026) | 2/5 |
| G2 | Speaker similarity judge | ReDimNet2-B6 (identity score) + VoxSim-fine-tuned WavLM-ECAPA regressor (perceptual score) (12.3M + ~400M · BF16 · 2 GB) | VoiceMOS-2026 Track-3 style 8-model embedding ensemble with listener-embedding conditioning and system-level calibration | Riva / NIM speaker recognition (TitaNet) embedding endpoint (NVIDIA · TitaNet cosine used as the objective similarity measure in Hume's Voice Replication Benchmark, 10 Sept 2026) | 1/5 |
| G3 | Naturalness prediction | UTMOSv2 (per-take ranker) + Distill-MOS (independent second opinion) + TTSDS2 (system-level bench) (~120M + 4.3M + TTSDS2 feature extractors (wav2vec2, HuBERT, WavLM, Whisper) · BF16 / FP32 for the small heads · 8 GB) | VoiceMOS-2026 style weighted ensemble with listener-identity conditioning (T09 / T02 recipe) | Gemini 3.7 Flash as an LALM MOS judge (Google · 13 Aug 2026) | 2/5 |
| G4 | Emotion consistency judge | Odyssey 2024 SER multi-attribute WavLM model (arousal/valence/dominance) + emotion2vec+ Large (categorical second opinion) (~315M + ~300M · BF16 · 3 GB) | VoiceMOS-2026 T02 ensemble at full scale, with Audio Flamingo Next 8B as a describer for disagreements | Expression Measurement API (prosody model) (Hume AI · Voice Replication Benchmark published 10 Sept 2026) | 2/5 |
| G5 | Lip-sync scoring **(changed)** | PEAVS (human-calibrated per-shot score) + Synchformer (millisecond offset), with SyncNet LSE-C/LSE-D reported alongside for comparability (small (<100M each); PEAVS built on an AV feature backbone, Synchformer on 0.64s segment extractors · FP16/FP32 · 4 GB) | Three-metric panel: PEAVS + Synchformer + AV-HuBERT-Large sync expert, with a frontier video LLM triaging disagreements | Gemini 3.7 Flash (agentic video understanding mode) (Google · model 13 Aug 2026; agentic video understanding announced 1 Sept 2026) | 2/5 |
| G6 | Multimodal review **(changed)** | Qwen3-Omni-30B-A3B-Instruct (30B total / ~3B active (MoE Thinker-Talker) · NVFP4 or FP8 via NVIDIA ModelOpt · 40 GB) | Qwen3-Omni-30B-A3B-Instruct at BF16 with AudioJudge-style decomposition and 16-64-way self-consistency, ensembled with AF-Next | Gemini 3.7 Flash (Google · 13 Aug 2026 (agentic video understanding added 1 Sept 2026)) | 4/5 |

## Per-category detail

Each section below is the research pass's own write-up: the three picks with footprint, speed and Spark gotchas, the compute-sensitivity reasoning, the delta against the prior pass, and numbered sources.

---

# Phase A · Source analysis · audio

## A1 Dialogue, music and effects separation

### 1. DGX Spark pick

**Primary — Bandit v2 (DnR v3 "multi" checkpoint), Karn Watcharasupat / Georgia Tech with Netflix.**
Band-split RNN, tens of millions of parameters (the model card does not state an exact count; the released
checkpoints are well under 1 GB in fp32). Run FP32 or BF16 with overlap-add chunking; resident footprint
~4–8 GB including activations for 10-second chunks at 44.1 kHz. Expect roughly 5–15× realtime for a 3-stem
pass on GB10 (estimate — no published Spark benchmark), i.e. a 2-hour feature in 10–25 minutes. Licence:
research/academic terms on the repo — check before commercial use. Gotchas: it is a recurrent/convolutional
model, so **no flash-attn dependency** and no sm_121 attention problem at all; the only real Spark risk is
`torchaudio` I/O (use `soundfile`). It is the only openly available model trained on DnR v3, whose dialogue
stem spans 30+ languages across Germanic, Romance, Indo-Aryan, Dravidian, Malayo-Polynesian and Bantu
families — which matters directly for a Hindi/multilingual dubbing pipeline.

**Alternates.** (a) **Tencent AuK / AuK-Flash** (1.5B, **MIT**, weights shipped 8 September 2026): one
instruction-driven model that does speech enhancement, speech separation, **music separation** and target
speaker extraction. 1.5B is trivial on a Spark and the MIT licence is strictly better than Bandit's.
Unbenchmarked against Bandit on DnR — trial it. (b) **Mel-Band RoFormer / MSST ensemble** for the dialogue
stem specifically; the community registry catalogues ~70–90 audited checkpoints (vocals, denoise, dereverb).

### 2. Unconstrained pick

**Primary — AV-CASS (audio-visual cinematic source separation via conditional flow matching), KAIST, CVPR
2026, CC BY 4.0, arXiv 2603.26113, 27 March 2026.** Published SI-SDRi: **DnRv2 8.10 dB vs BandIt 7.68 dB;
DnRv3 9.36 dB vs BandIt 9.14 dB** — and those are the *audio-only* ablations, before the visual conditioning
that is the paper's contribution. Being a flow-matching generative model it also scales with sampling steps
and best-of-N re-ranking, which BandIt's masking formulation cannot use. Margin over the Spark pick:
**+0.2 to +0.4 dB SI-SDRi on the audio-only configuration**, plus an unquantified gain from video. Training
cost was modest (600k steps, batch 8, 4×RTX 4090); *released weights were not confirmed* as of this pass —
code and a demo page exist. **Alternates:** a blended ensemble of Bandit v2 + Mel-Band RoFormer variants
(standard MVSEP/MSST practice, worth a few tenths of a dB); AudioShake's DME models run at scale.

### 3. API pick

**Primary — AudioShake Dialogue / Music / Effects (DME) separation**, via the REST API / SDK. Their latest
dialogue-separation model post is dated **27 October 2025** and claims better stereo field, better
speech-vs-singing distinction and context-aware separation; no public SDR numbers, which is the main
weakness of the pick. **Alternates:** AudioShake **multi-speaker separation** (separates overlapping voices
— useful upstream of A6); MVSEP hosted ensembles, which expose BandIt v2 among other algorithms and publish
a leaderboard.

### 4. Compute sensitivity

**3.** Generation-over-generation gains in CASS are ~0.2–0.5 dB SI-SDRi and the masking models are close to
saturation, but generative CASS (flow matching) newly rewards sampling steps and ensembles. Concretely,
AV-CASS buys +0.42 dB on DnRv2 over BandIt. For dubbing the dominant error is dialogue bleed into the bed,
and the engineering fix — build the bed as *mixture minus dialogue* rather than as the model's music+effects
sum — is worth more than any model swap.

### 5. Delta vs prior pass

**No change to the Spark pick: Bandit v2 still stands.** Two things are new. (i) **Tencent AuK** (8 Sept
2026, MIT) is a genuinely new alternate — one 1.5B model covering separation, TSE and enhancement under a
permissive licence, which would simplify A1 and A2 into one provider. (ii) AV-CASS is confirmed real
(CVPR 2026, arXiv 2603.26113) and is now the quality ceiling, but it needs video and its weights are
unconfirmed, so it does not displace Bandit v2 on a Spark.

### Sources
1. Bandit v2 repo — https://github.com/kwatcharasupat/bandit-v2
2. "Remastering Divide and Remaster" (DnR v3), ICASSP — https://ieeexplore.ieee.org/document/10704085/
3. AV-CASS, arXiv 2603.26113 (27 Mar 2026) — https://arxiv.org/html/2603.26113v1
4. AV-CASS project page — https://cass-flowmatching.github.io/
5. Tencent AuK model card — https://huggingface.co/tencent/AuK
6. AudioShake dialogue separation launch — https://www.audioshake.ai/post/audioshake-launches-latest-dialogue-separation-model
7. AudioShake multi-speaker separation — https://www.audioshake.ai/products/multi-speaker-separation
8. MVSEP BandIt v2 entry — https://mvsep.com/algorithms/45

---

## A2 Reference-clip enhancement

### 1. DGX Spark pick

**Primary — UniPASE (Nanjing University), arXiv 2604.14606, accepted IEEE TASLP 2026, code + weights at
github.com/xiaobin-rong/unipase and Xiaobin-Rong/unipase on HF.** Architecture: DeWavLM-Omni (a WavLM-Large
distillation, ~315M) → adapter → neural vocoder; total roughly **0.4–0.5B parameters**, BF16 ≈ **2–3 GB**
resident, comfortably faster than realtime on GB10 for 6–20 s reference clips. It took **1st place in the
objective evaluation of the URGENT 2026 Challenge** (ICASSP 2026; 80+ registrations, 29 valid entries), and
its design goal — "high fidelity and low hallucinations" — is exactly right for a clip that will be handed
to a voice cloner, where an invented timbre is worse than residual noise.

**The gotcha that decides the design: UniPASE reconstructs at 16 kHz.** OpenDub's TTS path wants 24 kHz+.
So do not run it unconditionally. Run a full-band, *discriminative* chain by default — Mel-Band RoFormer
dereverb + denoise checkpoints, which stay in the original band and cannot hallucinate timbre — and escalate
to UniPASE only for clips that DNSMOS/NISQA score below threshold, then band-extend. Other Spark notes: no
attention kernels at a size where flash-attn matters; WavLM loads through plain transformers on aarch64.

**Alternates.** (a) **Mel-Band RoFormer dereverb/denoise** (community checkpoints, full-band, ~30–50M) —
the safe default. (b) **Tencent AuK / AuK-Flash** (1.5B, MIT, 8 Sep 2026) — instruction-driven enhancement
in the same model as A1 separation; AuK-Flash does 4-step inference. (c) Resemble Enhance and Alibaba's
ClearerVoice-Studio (FRCRN, MossFormer2-SE-48K) remain reasonable, lower-risk baselines.

### 2. Unconstrained pick

**Primary — AnyEnhance (arXiv 2501.15417): 363.5M parameters, native 44.1 kHz full-band, one model for
denoising, dereverberation, declipping, super-resolution and target-speaker extraction, and it handles
*singing* as well as speech.** That combination is a better fit for film reference clips than UniPASE's
16 kHz output. It beats Voicefixer/MaskSR/AudioSR/FRCRN on DNSMOS and NISQA across DNS no-reverb and
with-reverb. Weights release is **not confirmed** in the paper — demos only — which is why it is not the
Spark pick. **Margin over the Spark pick:** not a parameter-count gap (364M vs ~450M); the gap is
*bandwidth* (44.1 kHz vs 16 kHz) and task coverage. The genuine unlimited-compute play here is not a bigger
model at all: it is **best-of-N — sample 16–32 restorations per reference window and re-rank with
DNSMOS + NISQA + a speaker-similarity gate against the raw clip.** **Alternates:** Miipher-2 (Google
DeepMind, USM-conditioned, 300+ languages) is the quality ceiling in the literature but **Google explicitly
declined to release code or checkpoints**; only an unofficial reimplementation (yukara-ikemiya/Open-Miipher-2)
exists.

### 3. API pick

This category is poorly served by APIs and is better self-hosted. **Primary — AudioShake voice isolator /
dialogue separation API** (27 October 2025 model), which is the only hosted option in this pass with a
credible film-audio track record. **Alternates:** Resemble AI's hosted enhancement (their Resemble Enhance
is the open counterpart); Alibaba ClearerVoice-Studio via ModelScope. I could not verify a dedicated hosted
"reference-clip restoration for voice cloning" product from a major vendor as of 18 September 2026.

### 4. Compute sensitivity

**3.** The ceiling is set by the source clip, not the model: picking a clean 8-second window out of a
2-hour feature beats any restoration model, and that selection is free. Above that floor, generative
restoration does reward test-time compute — best-of-N with DNSMOS/NISQA/speaker-similarity re-ranking is
the highest-yield knob — but the URGENT 2026 field clustered tightly, and hallucination risk *rises* with
generative capacity, so more compute is not monotonically better here.

### 5. Delta vs prior pass

**UniPASE still stands as the Spark primary, and is now much better evidenced** — TASLP acceptance,
public code and weights, and a verified URGENT 2026 1st place (objective track). Two corrections to the
prior verdict: (i) its **16 kHz output** is a hard constraint the prior pass did not flag, and it changes
the recommended topology (full-band discriminative default, UniPASE as escalation); (ii) **Tencent AuK**
(8 September 2026, MIT) is new since the prior pass and is the first permissively-licensed model that
covers A1 and A2 together — worth a trial before committing to two separate providers.

### Sources
1. UniPASE, arXiv 2604.14606 — https://arxiv.org/abs/2604.14606
2. UniPASE code — https://github.com/xiaobin-rong/unipase/
3. ICASSP 2026 URGENT Challenge, arXiv 2601.13531 — https://arxiv.org/abs/2601.13531
4. AnyEnhance, arXiv 2501.15417 — https://ar5iv.labs.arxiv.org/html/2501.15417
5. Miipher-2, arXiv 2505.04457 — https://arxiv.org/abs/2505.04457
6. Open-Miipher-2 (unofficial) — https://github.com/yukara-ikemiya/Open-Miipher-2
7. Tencent AuK — https://huggingface.co/tencent/AuK
8. Mel-Band RoFormer dereverb checkpoints — https://huggingface.co/anvuew/dereverb_mel_band_roformer

---

## A3 Speech, music and singing classification

### 1. DGX Spark pick

**Primary — CED-base (Xiaomi, Consistent Ensemble Distillation), 86M parameters, 50.0 mAP on AudioSet-2M,
Apache-2.0.** BF16 ≈ 0.3 GB; at a 1-second hop over the A1 dialogue and music stems it runs at hundreds of
times realtime on GB10, so it is effectively free. A ViT over mel patches with batch-normalised inputs and
variable-length support — no exotic kernels, no flash-attn, no aarch64 friction. The ~10M CED-tiny variant
still reaches 49.0 mAP if latency ever mattered, and a ggml port (ced.cpp) exists for CPU fallback.

**Alternates.** (a) **inaSpeechSegmenter (INA, MIT)** as the licence-clean speech/music backstop; note its
known limitation — **singing voice is labelled as music**, which is precisely the failure mode that hurts a
dubbing pipeline on musicals. (b) **SR-SAD** (AudioLabs Erlangen, arXiv 2512.09713), purpose-built for
speech activity detection in the presence of singing, reported AUC 0.919 while rejecting singing.
(c) **Dasheng-1.2B** (same Xiaomi lab, masked audio encoder trained on 272k hours) as a stronger frozen
encoder if CED's posteriors prove too coarse.

The right Spark configuration is not one model: CED-base dense posteriors + inaSpeechSegmenter + a
cross-stem energy heuristic on the A1 output as tiebreaker, with the ambiguous 5–10% escalated to an audio
LLM. **Qwen3-ASR-1.7B is an unusually good adjudicator here** because it was explicitly trained on singing
voice and full songs with backing music (M4Singer 5.98% WER, MIR-1k-vocal 6.25%, Opencpop 3.08%,
EntireSongs-en 14.60%) and does language identification in the same pass — so "is this sung, and in what
language" comes back from a model already resident for A4.

### 2. Unconstrained pick

**Primary — an audio-LLM adjudicator over dense CED posteriors: Qwen3.5-Omni-Plus (API) or, for open
weights, Qwen3-Omni-30B-A3B-Instruct (Apache-2.0).** Alternate: **Audio Flamingo Next-Think (8B, NVIDIA,
April 2026)**, which scores 75.01% on MMAU-v05.15.25 and 58.7 on MMAU-Pro, beating Gemini-2.5-Pro (57.4) —
but it ships under the **NVIDIA OneWay Noncommercial licence**, so it cannot go into OpenDub. **Margin over
the Spark pick: small.** Self-supervised scaling moved AudioSet mAP from 0.485 to 0.502 across BEATs → EAT →
SSLAM — about 1.7 mAP points for orders of magnitude more compute. The gain from an LLM adjudicator is not
on the easy 90%; it is on sung dialogue, musical numbers and whispered lines, where the label itself is
contested.

### 3. API pick

**Primary — Gemini 3-class audio models (`gemini-3.5-transcribe` for diarised transcript + word timestamps;
Gemini 3.8 Live Extended Thinking, model card published 15 September 2026, for reasoning over a segment).**
**Alternates:** **ElevenLabs Scribe v2** (11 March 2026), whose *dynamic audio tagging* detects non-speech
events such as laughter and footsteps inline with the transcript — the cheapest way to get event labels
aligned to words; **AudioShake stem energies** as an implicit classifier derived from A1.

### 4. Compute sensitivity

**2.** This is close to solved at small scale. CED-base at 86M hits 50.0 mAP; the entire SSL-scaling arms
race bought ~1.7 mAP points (0.485 → 0.502) on AudioSet. The residual errors are definitional — is sung
dialogue "speech" or "music"? — and a bigger model does not resolve a label your ontology has not defined.
Spend the compute on the escalation path for the ambiguous decile, not on the base classifier.

### 5. Delta vs prior pass

**No change. CED-base + inaSpeechSegmenter still stands.** I found **no CED successor** published in 2026;
the frontier moved to general audio encoders (Dasheng) and audio LLMs rather than to dedicated taggers.
One refinement: the prior pass's "add Whisper-AT if A4 stays on Whisper" clause is now moot, because A4's
pick is Qwen3-ASR, which brings singing/song recognition and language ID natively — so the adjudicator is
already in the pipeline for free.

### Sources
1. CED (Consistent Ensemble Distillation), arXiv 2308.11957 — https://huggingface.co/mispeech/ced-base
2. Dasheng, "Scaling up masked audio encoder learning", arXiv 2406.06992 — https://arxiv.org/pdf/2406.06992
3. inaSpeechSegmenter — https://github.com/ina-foss/inaSpeechSegmenter
4. SR-SAD, arXiv 2512.09713 — https://arxiv.org/html/2512.09713
5. Qwen3-ASR Technical Report, arXiv 2601.21337 — https://arxiv.org/html/2601.21337v1
6. Audio Flamingo Next — https://huggingface.co/nvidia/audio-flamingo-next-hf
7. ElevenLabs Scribe v2 — https://elevenlabs.io/blog/introducing-scribe-v2
8. Gemini 3.8 Audio model card (15 Sep 2026) — https://deepmind.google/models/model-cards/gemini-3-8-audio/

---

## A4 Transcription with word-level timing

### 1. DGX Spark pick

**Primary — Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B (Alibaba Qwen, Apache-2.0, released 29 January 2026;
native `transformers` support landed 26 June 2026).** 2.3B parameters combined, BF16 ≈ **5 GB** — nothing
on a 128 GB box. Verified numbers: LibriSpeech 1.63% / 3.38% WER (clean/other), WenetSpeech 4.97% net /
5.88% meeting, multilingual average 12.80% (beating Whisper large-v3), 52 languages and dialects (30
languages + 22 Chinese dialects) with language ID, and **singing/song recognition** — which matters for
film. The aligner is the part that changes OpenDub most: **27.8 ms accumulated average shift on raw test
sets vs NeMo Forced Aligner's 88.6 ms, and 24.8 ms vs 140.0 ms on 300-second concatenated segments** — a
67–77% relative reduction, and crucially it does *not* degrade on long form. Spark gotchas: at 1.7B you do
not need vLLM, TensorRT-LLM or flash-attn — run `transformers` with the SDPA/math backend and sidestep the
whole sm_121 kernel problem. The 0.6B ASR variant claims 2000× throughput at concurrency 128 if you ever
batch a whole catalogue.

**Alternates.** (a) **MOSS-Transcribe-Diarize-0.9B** (OpenMOSS, **Apache-2.0**, 9 July 2026): joint ASR +
diarization + timestamps + acoustic events in one pass over up to **90 minutes**, 50+ languages, and
notably **6.36% CER on movies / 5.97% on podcasts**, beating GPT-4o and Gemini variants on those sets. It
would collapse A4 and A6 into one provider — strongly worth a trial. (b) **NVIDIA Parakeet TDT 0.6B v3**
(25 European languages, word- and segment-level timestamps): the *most proven on Spark specifically*, with
NVIDIA forum walkthroughs and community Docker images. (c) **Canary-Qwen-2.5B** — top of the Hugging Face
Open ASR Leaderboard at 5.63% average WER, but English-centric.

### 2. Unconstrained pick

**Primary — N-best fusion: run Qwen3-ASR-1.7B, Canary-Qwen-2.5B and MOSS-Transcribe-Diarize in parallel,
reconcile with ROVER-style voting arbitrated by a large LLM over the disagreement spans, then re-align the
agreed text with Qwen3-ForcedAligner.** The largest open-weights single models are **VibeVoice-ASR (~9B,
Microsoft, MIT, open-sourced 21 January 2026)** — 60-minute single-pass, 64K context, joint ASR +
diarization + alignment, hotwords, 50+ languages, 7.77% average WER on the Open ASR Leaderboard — and
**Qwen3-Omni-30B-A3B**. **Margin over the Spark pick: real but bounded.** VibeVoice-ASR's 7.77% is *worse*
than Canary-Qwen's 5.63%, so scale alone does not win; the gain comes from fusion, which typically buys
10–20% relative WER. The honest statement is that the Spark runs every open model in this category
comfortably — the ceiling here is the closed APIs, not the hardware.

### 3. API pick

**Primary — ElevenLabs Scribe v2 (11 March 2026): 2.2% AA-WER, 99 languages, 98% speaker-label accuracy,
per-word `start`/`end`/`speaker_id`/`logprob`, up to 32 speakers, plus non-speech audio tags.** It is not
the lowest WER on the board, but it is the only top-tier API that returns *exactly* the structure OpenDub
consumes. **Alternates:** **StepAudio 3 ASR Max** (StepFun, **17 September 2026**, 1.7% WER — the current
accuracy leader, API-only); **MAI-Transcribe-2** (Microsoft AI, **3 September 2026**, 2.0% AA-WER, ~400×
realtime, $1.67/1,000 min); **Gemini 3.5 Transcribe** (diarization + word timestamps + utterance-level
language detection); **OpenAI `gpt-transcribe`** (28 July 2026; Common Voice 22-language WER 19.27% vs
whisper-1's 40.37%, with keyword hints and context biasing).

### 4. Compute sensitivity

**4 — but split.** The *text* scales: the best APIs sit at 1.7–2.2% AA-WER while the best open model on
the leaderboard is 5.63%, and fusion/re-ranking reliably adds 10–20% relative on top. The *timing* does
not: a 0.6B non-autoregressive aligner already lands at 27.8 ms AAS, inside the range where lip-sync
tolerance, not model capacity, is the binding constraint. So more compute buys you fewer wrong words, not
better sync.

### 5. Delta vs prior pass

**Spark pick unchanged and now fully verified** — Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B, Apache-2.0,
with published AAS numbers that beat MFA, NFA and WhisperX. Three things happened in the fortnight after
the prior pass, **all API-only, so none displace the Spark pick**: Meta **Muse Voice Transcribe** (1 Sept,
streaming ASR + diarization + endpointing in one model, 3.1% streaming AA-WER, **weights explicitly not
released**), **MAI-Transcribe-2** (3 Sept, 2.0%), **StepAudio 3 ASR Max** (17 Sept, 1.7%). They do move the
**API** pick's ceiling. On the open side, **MOSS-Transcribe-Diarize-0.9B** (July 2026) is the material new
candidate the prior pass missed — Apache-2.0, movie-domain CER 6.36%, and it merges A4 with A6.

### Sources
1. Qwen3-ASR Technical Report, arXiv 2601.21337 — https://arxiv.org/abs/2601.21337
2. Qwen3-ASR repo — https://github.com/QwenLM/Qwen3-ASR
3. Qwen3-ForcedAligner-0.6B — https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
4. MOSS-Transcribe-Diarize — https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
5. Microsoft VibeVoice-ASR — https://huggingface.co/microsoft/VibeVoice-ASR
6. Parakeet TDT 0.6B v3 — https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
7. Parakeet on DGX Spark (NVIDIA forum) — https://forums.developer.nvidia.com/t/multilingual-speech-to-text-stt-asr-with-nvidia-parakeet-tdt-0-6b-v3-for-the-dgx-spark/365554
8. ElevenLabs Scribe v2 — https://elevenlabs.io/blog/introducing-scribe-v2
9. Meta Muse Voice Transcribe — https://www.marktechpost.com/2026/09/01/meta-superintelligence-labs-releases-muse-voice-transcribe-one-real-time-model-for-streaming-asr-diarization-and-endpointing/
10. OpenAI transcription models — https://developers.openai.com/api/docs/models/gpt-transcribe
11. Open ASR Leaderboard — https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

---

## A5 Text-to-audio forced alignment

### 1. DGX Spark pick

**Primary — Montreal Forced Aligner 3.x (MIT, McGill / Montreal Corpus Tools).** The June 2026 benchmark
paper (arXiv 2606.18466, 16 June 2026) evaluates MFA 3.0 across English, Japanese and Korean on four
datasets and finds it achieves **state-of-the-art or near-SOTA with mean boundary errors below 15 ms**, and
that it **substantially outperforms all three neural ASR-based aligners tested, with the gap greatest at the
tight 10 ms and 25 ms thresholds** that actually matter for lip timing. It is a Kaldi GMM/DNN pipeline: it
runs on the Spark's 20 Arm cores, needs **no CUDA at all**, and therefore has zero sm_121 / flash-attn /
TorchCodec exposure — genuinely the least risky component in Phase A. Memory ~2–4 GB per worker. It also
returns **phone-level** boundaries, which word-level neural aligners do not, and which viseme planning
downstream will want.

**Alternates.** (a) **Qwen3-ForcedAligner-0.6B** (Apache-2.0) — co-primary in practice: 27.8 ms AAS beating
MFA, NFA and WhisperX by 67–77% relative, and near-flat degradation on 300-second segments (24.8 ms vs NFA's
140 ms). Note the apparent contradiction with MFA's <15 ms figure: the two papers use different metrics
(accumulated average shift vs mean boundary error), datasets and languages, so **they are not directly
comparable** — run both on OpenDub's own material before choosing. Qwen's coverage is **11 languages**
against MFA's very wide dictionary/acoustic-model zoo. (b) **MMS_FA via `torchaudio.functional.forced_align`**
as the coverage fallback for languages neither covers — but this is the one place where the **GB10 torchaudio
defect (pytorch/audio#4169)** bites, so validate on Spark first.

### 2. Unconstrained pick

**Primary — a three-way aligner ensemble (MFA 3.x + Qwen3-ForcedAligner-0.6B + MMS_FA) with per-word
agreement scoring**: take the median boundary where all three agree within a tolerance, and flag the
remainder for the reconciliation stage rather than trusting any single aligner. Unlimited compute also
enables **speaker-adapted training** in MFA (per-speaker acoustic-model adaptation over the whole feature,
rather than a global model) — historically the single largest quality lever in the Kaldi alignment
tradition. **Margin over the Spark pick: a few milliseconds.** Both components of the ensemble already run
on a Spark; the ensemble is 3× the cost for a change measured in single-digit milliseconds. This is the
least compute-limited category in Phase A. **Alternates:** gradient-saliency alignment over a large ASR
model (RWTH Aachen / AppTek); BFA, which claims up to 240× faster alignment at competitive recall but
trades away precision.

### 3. API pick

**Primary — ElevenLabs Forced Alignment API**, which is the only true hosted forced-alignment product
verified in this pass (returns character/word alignment for a supplied transcript). **Alternates:**
**Gemini 3.5 Transcribe** and **ElevenLabs Scribe v2** — both return per-word timestamps, but by
*transcribing*, not by aligning supplied text, so they cannot honour an approved subtitle script verbatim;
they belong in A4. There is no strong hosted-API quality argument in this category.

### 4. Compute sensitivity

**1.** Solved at small scale, and the Spark pick is the ceiling. Mean boundary error is already **under
15 ms** for a Kaldi-era GMM/DNN system and **27.8 ms AAS** for a 0.6B neural aligner — both inside the
range of human phone-boundary labelling disagreement. Scaling the aligner cannot help; what helps is the
surrounding machinery the prior pass identified (Needleman-Wunsch anchoring of the subtitle text against
the ASR hypothesis, RANSAC offset/rate fitting, per-word confidence).

### 5. Delta vs prior pass

**No change to the primary: MFA 3.x still stands, and is now backed by a dedicated June 2026 benchmark
paper** rather than reputation. One promotion: **Qwen3-ForcedAligner-0.6B moves from "strong" to
co-primary** — it is Apache-2.0, has published numbers against MFA/NFA/WhisperX, is already in memory for
A4, and its long-form stability is the specific property OpenDub needs on feature-length audio. Keep MFA
for phone-level output and for the 40+ languages Qwen does not cover.

### Sources
1. "Montreal Forced Aligner and the state of speech-to-text alignment in 2026", arXiv 2606.18466 — https://arxiv.org/abs/2606.18466
2. MFA repo — https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
3. MFA user guide — https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/index.html
4. Qwen3-ASR Technical Report (aligner section), arXiv 2601.21337 — https://arxiv.org/html/2601.21337v1
5. Qwen3-ForcedAligner-0.6B — https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
6. torchaudio GB10/aarch64 defect — https://github.com/pytorch/audio/issues/4169

---

## A6 Diarization with overlap handling

### 1. DGX Spark pick

**Primary — DiariZen-Large-s80-v2 (BUT Speech@FIT): structurally pruned WavLM-Large encoder + Conformer
backend with powerset classification + VBx clustering.** ~315M parameters before pruning, BF16 ≈ **1.5–3 GB**
— trivial on a Spark, and many times realtime. Published DER: **AMI-SDM 13.9, AISHELL-4 10.1, AliMeeting-far
10.8, NOTSOFAR-1 16.7, MSDWild 15.8, DIHARD3-full 14.5, RAMC 11.0, VoxConverse 9.1.** Set those against
pyannote community-1 on the same datasets (**AMI-SDM 19.9, AliMeeting 20.3, DIHARD3 20.2, MSDWild 22.8,
VoxConverse 11.2**) and DiariZen wins by **4–7 DER points**, putting it at or past the *commercial*
pyannoteAI **Precision-2** tier (AMI-SDM 15.6, AliMeeting 15.2, DIHARD3 14.7, MSDWild 17.3). Powerset
classification means overlap is modelled natively rather than bolted on. Spark gotchas: pure PyTorch,
no flash-attn need at this size; confirm the repo's exact licence terms before shipping.

**Alternates.** (a) **pyannote speaker-diarization-community-1** (CC BY 4.0) — the low-risk drop-in for the
3.1 weights already in OpenDub, better than 3.1 on eleven of twelve datasets, with an exclusive
single-speaker mode that simplifies reconciliation against A4 timestamps. **Spark gotcha: pyannote.audio 4.x
requires TorchCodec, which has no Linux-aarch64 PyPI wheel** — build it or patch the decode path; working
precedents exist (rappdw/transcribe-dgx, eleqtrizit/nvscribe). (b) **NVIDIA Streaming Sortformer 4spk
v2/v2.1** (NeMo, aarch64 containers, NVFP4 quantized inference now merged) — capped at 4 speakers.
(c) **MOSS-Transcribe-Diarize-0.9B** (Apache-2.0) if you want A4 and A6 from one model.

### 2. Unconstrained pick

**Primary — DOVER-Lap fusion of DiariZen + community-1 + Streaming Sortformer, with enrolment-based target
speaker extraction (BSRNN / TF-GridNet class) on the overlapped regions.** Plus an unpruned WavLM-Large
encoder instead of the s80-pruned one. **Margin over the Spark pick: small — and that is the finding.**
DiariZen's whole contribution is that a *structurally pruned* encoder matches heavier systems, and
pyannoteAI's own generational gain (Precision-2 → Precision-3, 16.02 → 14.35 average DER, **10.4%
relative**) came from data and calibration, not scale. **This category is not compute-limited on a Spark.**
**Alternates:** DiCoW / TS-ASR-Whisper (diarization-conditioned ASR) if you want diarization and
transcription jointly optimised; per-domain fine-tuning of DiariZen, which the literature shows takes it
from 16.4% to 12.7% DER on out-of-domain material.

### 3. API pick

**Primary — pyannoteAI Precision-3.** **14.35 average DER across 15 datasets** (AISHELL, ALLIES, AliMeeting,
AMI, AMI-SDM, AVA-AVD, CALLHOME, DIHARD, INA, MSDWILD, NOTSOFAR, RAMC, SBCSAE, VoxConverse, VoxSRC2023) —
**10.4% better than Precision-2**, with over half the gain from speaker attribution specifically (18.7%
more speech assigned to the right speaker). It also decouples `vadSensitivity` and `crosstalkSensitivity`
from the output scores and exposes `speakerProbability` / `speechProbability` / `crosstalkProbability`
separately, which is exactly the per-word confidence OpenDub needs, and it offers speaker identification
via voiceprints. Available on API **and on-premise**. **Alternates:** **ElevenLabs Scribe v2** (98%
speaker-label accuracy, up to 32 speakers, diarization fused with word timestamps); **Meta Muse Voice
Transcribe** (1 September 2026, 20+ speakers, streaming, closed weights, $0.18/hour).

### 4. Compute sensitivity

**2.** Progress here is coming from training data, pruning recipes and calibration, not parameters:
DiariZen's pruned WavLM beats a commercial system 20× its budget, and Precision-3's headline gain over
Precision-2 is 10.4% relative from attribution improvements. The residual errors are overlap regions and
speaker-count estimation, which respond to *fusion and TSE*, not to a bigger encoder.

### 5. Delta vs prior pass

**Changed.** The prior pass named **pyannote community-1** as the ship-first Spark pick. On the published
benchmarks **DiariZen-Large-s80-v2 beats community-1 by 4–7 DER points on every shared dataset** (AMI-SDM
13.9 vs 19.9; MSDWild 15.8 vs 22.8; DIHARD3 14.5 vs 20.2) at comparable cost, and it lands at Precision-2
quality without an API bill. **Recommend DiariZen as the Spark primary, community-1 as the low-risk
fallback.** Two other updates: **Precision-3 now exists** and supersedes Precision-2 as the API pick
(14.35 vs 16.02 average DER); and the prior pass did not flag that **pyannote.audio 4.x will not install
cleanly on aarch64** because of TorchCodec — a real deployment blocker on this exact box. The prior
verdict's consumption-layer rewrite (word-level attribution from A4 timings, explicit overlap mask) stands
and is still worth more than the model swap.

### Sources
1. DiariZen repo — https://github.com/BUTSpeechFIT/DiariZen
2. "DiariZen Explained", arXiv 2604.21507 — https://arxiv.org/abs/2604.21507
3. pyannote community-1 model card (benchmark table) — https://huggingface.co/pyannote/speaker-diarization-community-1
4. pyannoteAI Precision-3 — https://www.pyannote.ai/blog/precision-3
5. pyannoteAI Precision-3 availability changelog — https://www.pyannote.ai/changelog/precision-3
6. pyannote.audio aarch64 discussion — https://github.com/pyannote/pyannote-audio/discussions/1990
7. Streaming Sortformer 4spk v2.1 — https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
8. transcribe-dgx (pyannote on GB10) — https://github.com/rappdw/transcribe-dgx
9. MOSS-Transcribe-Diarize — https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize

---

## A7 Speaker embeddings and voice-bank matching

### 1. DGX Spark pick

**Primary — ReDimNet2-B6 (Interspeech 2026, arXiv 2603.11841; official weights at PalabraAI/redimnet2).**
**12.3M parameters, 13 GMACs, 0.287% EER on Vox1-O**, trained on VoxCeleb2-dev. BF16 ≈ **0.1 GB** — the
smallest component in the whole pipeline; embedding a 2-hour feature's worth of segments takes seconds.
Loads via `torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name="b6", train_type="lm",
pretrained=True)`. An MIT-licensed VoxBlink2+VoxCeleb2 large-margin checkpoint is mirrored at
`sarvam/redimnet2-b6-vb2vox2-lm-mit`, which is the one to use if licence cleanliness matters. The family
spans B0–B6 (1.1M–12.3M params, 0.33–13 GMACs: B0 1.04% EER, B1 0.78%, B2 0.57%, B6 0.287%), so you can
pick any point on the curve. Spark gotchas: none meaningful in PyTorch. **Avoid the ONNX Runtime path** —
`onnxruntime-gpu` for aarch64/sm_121 only exists as community-built wheels.

**Alternates.** (a) **ERes2NetV2 (Alibaba 3D-Speaker, Apache-2.0)** — wins on Mandarin, multi-device data
and **short clips**, which is the regime voice-bank matching actually operates in. (b) **WeSpeaker
ResNet293-LM / SimAMResNet100** for packaging, ONNX export and a well-tested scoring/calibration stack.

### 2. Unconstrained pick

**Primary — a score-fusion ensemble of ReDimNet2-B6 + ERes2NetV2 + WeSpeaker ResNet293-LM with
large-margin fine-tuning and quality-aware score calibration (QMF)**, i.e. the standard VoxSRC
challenge-winning recipe, whose systems land at EER <1.5% / minDCF ≪0.1 on the much harder challenge
evaluation sets. **Margin over the Spark pick: negligible in EER.** The Spark pick already sits at 0.287%
EER; an ensemble might reach ~0.2%, which is below the noise floor of a voice bank containing a few dozen
characters. What the ensemble genuinely buys is **better-calibrated scores** — a stable decision threshold
across domains — and that is a calibration problem, not a compute problem. **Alternates:** WavLM-based
speaker verification front-ends; multi-resolution / Matryoshka embeddings (sappho192/redimnet-mrl) if you
want one embedding usable at several dimensionalities for a tiered voice-bank index.

### 3. API pick

There is **no quality reason to use an API here** — the best open model is a 12M-parameter file. If a
hosted path is required: **pyannoteAI voiceprints / speaker identification (Precision-3 tier)**, the only
major diarization vendor that exposes enrolment-based speaker identity, available on API and on-premise.
**Alternates:** NVIDIA Riva / NIM speaker models (TitaNet-L class); Azure Speaker Recognition. All are
behind a 0.287%-EER open checkpoint on published VoxCeleb numbers.

### 4. Compute sensitivity

**1.** Solved. ReDimNet2's explicit contribution is *pushing the accuracy-per-GMAC Pareto front*, not
scaling: B2 at 3-ish GMACs already reaches 0.57% EER, and B6 at 13 GMACs reaches 0.287%. Going from
12M to 300M parameters moves EER by hundredths of a percent. Every remaining error in voice-bank matching
comes from the surrounding machinery — segment selection, duration, channel mismatch, threshold
calibration, clustering — none of which a larger model fixes.

### 5. Delta vs prior pass

**Changed — a same-family generational upgrade.** The prior pass named **ReDimNet B3–B6**. **ReDimNet2**
(arXiv 2603.11841, March 2026) supersedes it with official implementation and pretrained weights, reaching
**0.287% EER on Vox1-O at 12.3M parameters / 13 GMACs** and improving the cost-accuracy Pareto front over
ReDimNet across the whole B0–B6 range. A third-party paper measured a 15% relative gain over ReDimNet-B2 on
a hard (whispered-speech) condition, confirming the family is still the reference point. **ERes2NetV2 keeps
its role** as the short-clip and Mandarin/multi-device specialist, so the prior pass's "decide on
multi-domain rather than VoxCeleb numbers" instruction still holds — just run it against ReDimNet**2**.

### Sources
1. ReDimNet2, arXiv 2603.11841 — https://paperswithcode.co/paper/2603.11841
2. ReDimNet2 official repo + weights — https://github.com/PalabraAI/redimnet2
3. MIT-licensed ReDimNet2-B6 checkpoint — https://huggingface.co/sarvam/redimnet2-b6-vb2vox2-lm-mit
4. ReDimNet (v1), arXiv 2407.18223 — https://arxiv.org/html/2407.18223v1
5. Whispered-speech SV comparison, arXiv 2604.20229 — https://arxiv.org/abs/2604.20229
6. 3D-Speaker / ERes2NetV2 — https://github.com/modelscope/3D-Speaker
7. pyannoteAI models (voiceprints / speaker ID) — https://www.pyannote.ai/md/models

---

## A8 Paralinguistic and delivery tagging

### 1. DGX Spark pick

**Primary — two tiers, and the cheap tier comes first.**

*Tier one (not a model):* measured prosody computed directly off the A5 alignments — speaking rate,
relative loudness, F0 median and range, pause profile. Exact, free, language-independent, and it feeds D3,
E1 and E4 directly. Nothing below beats it on cost-effectiveness.

*Tier two:* **Qwen3-Omni-30B-A3B-Instruct (Alibaba, Apache-2.0)** as the delivery-note writer. MoE, 30B
total / **3B active**, so BF16 ≈ **62 GB** resident (fits the 128 GB unified pool with room for the rest of
the pipeline) or **~16–32 GB at FP8/NVFP4**. Because only 3B parameters are active per token, the 273 GB/s
bandwidth limit hurts far less than it would for a dense 30B — expect meaningfully better than the
22.7–23.7 tok/s measured on this box for a dense-ish 120B-A12B NVFP4 model. It is the **largest
open-weights omni-modal model available** — Qwen3.5-Omni (April 2026) is API-only. Spark gotchas: this is
the one Phase A component that hits the real sm_121 problems — **flash-attn has no official sm_121 build**
(use the community wheel index or fall back to SDPA), **vLLM needs the `cu130-nightly` image**, and
FlashInfer's sm_121 FP4/MXFP4 paths are incomplete. Budget a day for the serving stack.

**Alternates.** (a) **emotion2vec+ large (MIT)** as a cheap categorical prior to condition the LLM prompt.
(b) **Audio Flamingo Next-Think (8B, NVIDIA/UMD, April 2026)** — stronger on audio reasoning
(MMAU-v05.15.25 75.01%, MMAU-Pro 58.7 vs Gemini-2.5-Pro's 57.4) and small enough to run at BF16 in ~16 GB,
**but it ships under the NVIDIA OneWay Noncommercial licence (plus Qwen Research + OpenAI ToS components)
— it cannot ship inside an open-source project.** Research comparison only.

### 2. Unconstrained pick

**Primary — Qwen3.5-Omni-Plus (hundreds of billions of parameters, 256K context, April 2026)**; the largest
*open-weights* option remains **Qwen3-Omni-30B-A3B**, i.e. the Spark already runs the best open model in
this category. **Margin over the Spark pick: this is where it is largest in Phase A, but the evidence is
subtler than "bigger is better."** VoxEmo (arXiv 2603.08936) found zero-shot speech LLMs *trail* supervised
specialists on hard-label Macro-F1 (CREMA-D: Qwen2-Audio 61.9% vs 70.6–71% supervised) while capturing
human annotation *distributions* far better — and that **fine-tuning** flipped it, beating EmoBox baselines
on 10 of 30 datasets. It also found the foundation model mattered more than scale: two ~7B models differed
substantially. So the compute play is **fine-tuning plus best-of-N self-consistency over delivery notes**,
not raw parameters. **Alternates:** Audio Flamingo Next-Think for reasoning-heavy tagging (non-commercial);
an ensemble of emotion2vec+ / audEERING wav2vec2-MSP-dim as a numeric prior feeding a large LLM.

### 3. API pick

**Primary — Qwen3.5-Omni-Plus (Alibaba, API-only, technical report 21 April 2026).** It reports
state-of-the-art across 215 benchmarks and claims to surpass **Gemini-3.1 Pro in key audio tasks** (e.g.
LibriSpeech-clean 1.11% vs 3.36% WER), with 113 input languages. **Alternates:** **Gemini 3.8 Live Extended
Thinking** (model card published **15 September 2026**; audio + image + video + text in, 128K context,
64K output, built on Gemini 3 Pro) — the high-reasoning audio option, with **Gemini 3.1 Pro** for batch
work; **OpenAI GPT Live 1 / `gpt-audio-1.5`** (GPT Live 1 released **17 September 2026**), which brings
GPT-5-class reasoning to audio.

### 4. Compute sensitivity

**4.** This is the Phase A category where more compute buys the most. The best open model a Spark can hold
is a 30B-A3B MoE; the API frontier is "hundreds of billions" and claims a 3× WER advantage over Gemini-3.1
Pro on clean speech. Test-time compute compounds: best-of-N with self-consistency over free-text delivery
notes is cheap and directly improves consistency. The counterweight, and the reason this is a 4 and not a
5: **tier one — measured rate, loudness, F0 and pause profile from A5 — is compute-free and already
captures most of the value the downstream TTS can actually act on**, and VoxEmo shows supervised
specialists still beat zero-shot LLMs on hard labels.

### 5. Delta vs prior pass

**Primary stands** — "measured prosody first, then an audio LLM as delivery-note writer" is still right,
and the tier-one build order is still the correct first move. Updates to the model slate: **Qwen3.5-Omni
(April 2026) is API-only**, so **Qwen3-Omni-30B-A3B (Apache-2.0) remains the self-hosted/air-gapped pick**
— the prior pass's hedge was correct. **Audio Flamingo Next (April 2026) supersedes Audio Flamingo 3** and
beats Gemini-2.5-Pro on MMAU-Pro, but its **non-commercial licence rules it out of OpenDub** — a constraint
the prior pass listed it without. On the API side the prior pass's "Gemini 2.5/3 Pro, GPT-4o Audio" is now
stale: **Gemini 3.8 Audio (15 Sept 2026)** and **GPT Live 1 (17 Sept 2026)** both landed inside the last
fortnight, and Qwen3.5-Omni-Plus has the strongest published audio claims.

### Sources
1. Qwen3-Omni-30B-A3B-Instruct — https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
2. Qwen3.5-Omni Technical Report, arXiv 2604.15804 — https://arxiv.org/html/2604.15804v2
3. Audio Flamingo Next — https://huggingface.co/nvidia/audio-flamingo-next-hf
4. Audio Flamingo Next announcement — https://gamma.umd.edu/media/af_next/
5. VoxEmo benchmark, arXiv 2603.08936 — https://arxiv.org/html/2603.08936
6. emotion2vec / emotion2vec+ — https://github.com/ddlBoJack/emotion2vec
7. Gemini 3.8 Audio model card (15 Sep 2026) — https://deepmind.google/models/model-cards/gemini-3-8-audio/
8. September 2026 release roundup (GPT Live 1, StepAudio 3, MAI-Transcribe-2) — https://thursdai.news/releases/2026-09
9. vLLM on DGX Spark (measured tok/s, cu130-nightly) — https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
10. flash-attention sm_121 support issue — https://github.com/Dao-AILab/flash-attention/issues/1969
11. Prebuilt sm_121 wheels (flash-attn etc.) — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels
12. FlashInfer DGX Spark (sm_121) audit — https://github.com/flashinfer-ai/flashinfer/issues/3170
13. vLLM sm_121 tracking issue — https://github.com/vllm-project/vllm/issues/31128
14. CTranslate2 aarch64/CUDA 13 binaries — https://github.com/assix/ctranslate2-aarch64-cuda13-binaries
15. NVIDIA DGX Spark playbooks — https://github.com/NVIDIA/dgx-spark-playbooks

---

# Phase B · Source analysis · picture

## B1 Active speaker detection and face tracking

### 1. DGX Spark pick

**Primary: LoCoNet trained with TalkNCE loss** (Indiana Univ./SJTU + KAIST MM Lab), MIT licence,
checkpoint published at `kaistmm/TalkNCE` (Google Drive), **95.5% mAP on AVA-ActiveSpeaker val** —
still the best ASD number with a downloadable weight file [1,2,3]. Pipeline model, ~tens of MB of
weights; audio branch plus a ResNet-class visual branch over 112×112 face crops. Run BF16;
resident footprint with detector and tracker loaded is **~3 GB** of the 128 GB, i.e. free. It is a
plain PyTorch model: the community GB10 wheel index (torch 2.13.0+cu13.3, flash-attn 2.8.3.post1,
all `cp312`/`linux_aarch64`/`sm_121`) covers everything it needs, and it does not require
flash-attn at all [10]. Throughput is dominated by face cropping and decode, not the network —
expect well above realtime for 1080p with batched crops.

Face detection/tracking: **SCRFD (InsightFace)** for detection plus a ByteTrack-class associator,
shot-cut aware (feed it B2's cut list so tracks never bridge a cut). Gotcha: InsightFace ships
ONNX, so this needs `onnxruntime-gpu` for aarch64 — there is **no official PyPI aarch64+CUDA 13
wheel**; use the prebuilt GB10 index (`onnxruntime-gpu 1.22.0+cu13.3`) or build with
`CMAKE_CUDA_ARCHITECTURES=121` (20–40 min) [10,11]. CPU ORT on 20 Arm cores is also viable here.

**Alternates:** (a) **LR-ASD** (Sichuan Univ., MIT, IJCV 2025) — 94.45% AVA mAP, ~1M-parameter
class, 96.4% F1 on Columbia after TalkSet fine-tuning; only ~1 point behind LoCoNet at a fraction
of the cost, and the safer default if you want one pass over a whole feature in minutes [4].
(b) **SAM 3.1** (Meta, 27 March 2026) for track propagation through occlusion — adds Object
Multiplex, ~7× faster at 128 tracked objects on one H100; heavier than a box tracker but far more
robust on crowd scenes [5].

### 2. Unconstrained pick

**GateFusion** (WACV 2026, arXiv 2512.15707, submitted 17 Dec 2025): pretrained unimodal encoders
plus a Hierarchical Gated Fusion decoder, **77.8% mAP Ego4D-ASD (+9.4)**, **86.1% UniTalk (+2.9)**,
**96.1% WASD (+0.5)** over LoCoNet [6]. Those are the only 2025–26 gains large enough to matter.
Caveat that decides it: **no public code or weights as of 18 Sep 2026** — I could not find a
repository, so it is an unconstrained pick only in the sense of "when weights appear".
Margin over the Spark pick on OpenDub's actual footage (framed film/TV faces, AVA-like) is small,
roughly +1 mAP; the +9.4 is an egocentric-video result that does not transfer to dubbing sources.
**Alternates:** (a) C³ASD (ECCV 2026, arXiv 2607.03018, 3 Jul 2026) — multi-level consistency
training for corruption robustness (noise, occlusion, joint degradation); the right pick for
archival or noisy sources, no benchmark table verifiable from the abstract page [7].
(b) LoCoNet+TalkNCE ensembled over multiple crops/scales with SAM 3.1 tracks — buys maybe +0.5 mAP
for 5× compute; this is what "unlimited compute" actually buys in ASD.

### 3. API pick

There is **no hosted active-speaker-detection API** in September 2026. Closest:
**Primary: Gemini 3.1 Pro** (Google DeepMind, 19 Feb 2026) — native video input, so you can ask
"which visible face is speaking in this 2-second window" directly; works, but is not frame-accurate
and is far worse than a 1M-parameter specialist at the actual task [8].
**Alternates:** (a) Claude Opus 5 (Anthropic, 24 Jul 2026), text/image/audio/video input, ranked
#4 on llm-stats' aggregate vision index vs Gemini 3.1 Pro's #27 — same caveat [9].
(b) AWS Rekognition / Azure face-and-person tracking for the *tracking* half only; you still need
LoCoNet for the speaking decision.

### 4. Compute sensitivity

**2/5.** A 1M-parameter model (LR-ASD, 94.45 mAP) is within ~1 point of the best downloadable model
(LoCoNet+TalkNCE, 95.5 mAP), and the largest 2026 gains come from better fusion and pretraining on
hard domains (Ego4D +9.4) rather than from scale on film-like footage [3,4,6]. What buys quality
here is face-crop resolution, correct track association across cuts, and domain-matched
fine-tuning — none of which the Spark limits.

### 5. Delta vs prior pass

**Unchanged.** LoCoNet+TalkNCE remains the Spark pick; GateFusion still has no usable code, so the
3 September "switch the moment code appears" verdict stands verbatim. Two genuine additions:
**SAM 3.1** (27 Mar 2026) supersedes the SAM 2 track-propagation idea the prior pass filed as
"niche", and **C³ASD** (3 Jul 2026, ECCV) is a new robustness option the prior pass did not have.
Neither displaces the primary.

### Sources
1. LoCoNet: Long-Short Context Network for ASD — https://arxiv.org/abs/2301.08237 (CVPR 2024)
2. LoCoNet code + AVA weights — https://github.com/SJTUwxz/LoCoNet_ASD
3. TalkNCE (LoCoNet+TalkNCE, 95.5 mAP AVA val, MIT) — https://github.com/kaistmm/TalkNCE ; paper https://arxiv.org/abs/2309.12306
4. LR-ASD (MIT, IJCV 2025, 94.45 mAP AVA) — https://github.com/Junhua-Liao/LR-ASD
5. SAM 3.1 Object Multiplex + video tracking (27 Mar 2026) — https://ai.meta.com/blog/segment-anything-model-3/
6. GateFusion (WACV 2026) — https://arxiv.org/abs/2512.15707
7. C³ASD (ECCV 2026, 3 Jul 2026) — https://arxiv.org/abs/2607.03018
8. Gemini model version history incl. Gemini 3.1 Pro, 19 Feb 2026 — https://en.wikipedia.org/wiki/Gemini_(language_model)
9. Claude Opus 5 (24 Jul 2026) spec/vision index — https://llm-stats.com/models/compare/claude-opus-5-vs-gemini-3.1-pro-preview
10. Prebuilt GB10/sm_121 aarch64 wheel index (torch 2.13, flash-attn, onnxruntime-gpu 1.22) — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels
11. ONNX Runtime GPU on DGX Spark, build guide — https://forums.developer.nvidia.com/t/onnx-runtime-gpu-inference-on-dgx-spark-gx10-build-guide-and-prebuilt-binaries/366157

---

## B2 Shot boundary detection

### 1. DGX Spark pick

**Primary: TransNetV2** (Charles University Prague), **MIT**, F1 **96.2 BBC Planet Earth /
93.9 RAI / 77.9 ClipShots**, PyTorch and TensorFlow inference in-repo, no training required [1].
Tiny — under 10M parameters, ~41 GMACs per window [2] — so on a GB10 it is **~1.5 GB resident**
and runs at many times realtime; one batched pass over a feature is minutes, and it is the cheapest
thing in the whole OpenDub graph. No aarch64 gotchas at all: pure PyTorch conv stack, no
flash-attn, no custom kernels, covered by the stock aarch64 CUDA 13 torch wheel [8].

**Alternates:** (a) **AutoShot** (CVPR-NAS 2023 workshop) — +4.2 F1 over TransNetV2 on the SHOT
dataset and +1.1/+0.9/+1.2 on ClipShots/BBC/RAI, at *lower* cost (37 vs 41 GMACs); worth running as
a second channel because it is nearly free [2]. (b) **PySceneDetect** (BSD-3) threshold/fade
detectors for black frames, fades and act breaks — a different failure mode from the learned
detectors, and the cheapest possible insurance.

### 2. Unconstrained pick

**Primary: OmniShotCut** (arXiv 2604.24762v2, 27 Apr / rev 21 May 2026) — shot-query Transformer,
ResNet18 + 3 encoder / 6 decoder layers, 24 shot queries, **34.49M parameters**. **0.881 F1 on
OmniShotCutBench vs AutoShot 0.815 and TransNetV2 0.814**; on legacy BBC, **0.971 vs 0.967**
TransNetV2 [3]. It also predicts *what kind* of transition (dissolve/wipe/fade, hard cut, sudden
jump), which is directly useful for deciding where lip-sync may cross a boundary. Licence on the
preprint is **CC BY-NC-ND 4.0** and code release is promised but not confirmed.
**Margin over the Spark pick: +0.4 F1 on BBC.** That is the honest number — on broadcast-like
material TransNetV2 is already at the ceiling, and OmniShotCut would also run fine on a Spark
(34M params). Compute is not what separates them; annotation quality and transition taxonomy are.
**Alternates:** (a) **PERSIST** (BMVC 2026, arXiv 2608.29287, 29 Aug 2026) — reframes SBD as
"persistent state discrimination", **roughly halves TransNetV2's false positives** on a 2,727-video
diagnostic set and cuts ClipShots false positives ~25% at equal recall, specifically targeting
flashes, text overlays and archival damage [4]. This is OpenDub's real failure mode (a title card
or camera flash inventing a cut mid-line), so it is the most interesting new paper in this
category even though its headline F1 is only at parity. (b) **TransVLM** (ECCV 2026, arXiv
2604.27975v2, rev 4 Aug 2026) — VLM + optical-flow prior, detects transition *spans* rather than
cut points; CC BY-NC-SA, no F1 table verifiable from the abstract page [5].

### 3. API pick

**Primary: Google Cloud Video Intelligence API — `SHOT_CHANGE_DETECTION`.** Mature, returns shot
ranges over a whole file, and is the only shot-detection API a production pipeline can lean on.
**Alternates:** (a) AWS Rekognition Video Segment Detection (shots + technical cues: black frames,
colour bars, end credits — the credits/bars detection is genuinely useful for dubbing and neither
open model does it). (b) Gemini 3.1 Pro over video, as a semantic fallback for act breaks.
Note: a cloud call here is almost never justified — TransNetV2 is free and better than the APIs on
published benchmarks.

### 4. Compute sensitivity

**1/5.** The lowest score in phase B. A sub-10M-parameter 3D CNN from 2020 scores 0.967 F1 on BBC;
a 34.5M Transformer from 2026 scores 0.971 — **+0.4 F1 for 7× the parameters** [1,3]. The remaining
errors are precision errors on flashes and overlays, which PERSIST fixes with *better framing of
the problem*, not more FLOPs (it trains on real ClipShots transitions only, while the baselines use
corpora that are ~85% synthetic) [4]. The Spark is not remotely a bottleneck.

### 5. Delta vs prior pass

**Unchanged primary, one flag.** TransNetV2 + PySceneDetect still stands exactly as the
3 September pass concluded. Two 2026 papers that pass did not surface: **OmniShotCut** (Apr 2026)
and **PERSIST** (29 Aug 2026 — five days before the prior pass, so genuinely new to it). Neither
should displace TransNetV2 today: OmniShotCut is NC-ND with no confirmed code, PERSIST's code
release is asserted in the paper but I could not verify a repository. **Recommendation: keep
TransNetV2, and track PERSIST** — halving false positives on flash/text-overlay frames is worth
more to OpenDub than +0.4 F1.

### Sources
1. TransNetV2 (MIT; F1 96.2 BBC / 93.9 RAI / 77.9 ClipShots) — https://github.com/soCzech/TransNetV2
2. AutoShot (CVPR-NAS 2023 W; +4.2 F1 on SHOT, 37 vs 41 GMACs) — https://arxiv.org/abs/2304.06116
3. OmniShotCut (27 Apr 2026, rev 21 May 2026; 34.49M params; 0.881/0.971 F1) — https://arxiv.org/html/2604.24762v2
4. PERSIST (BMVC 2026, 29 Aug 2026) — https://arxiv.org/abs/2608.29287
5. TransVLM (ECCV 2026, rev 4 Aug 2026) — https://arxiv.org/abs/2604.27975
6. PySceneDetect — https://github.com/Breakthrough/PySceneDetect
7. Google Cloud Video Intelligence shot change detection — https://cloud.google.com/video-intelligence/docs/analyze-shots
8. Prebuilt GB10/sm_121 aarch64 wheels — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels

---

## B3 On-screen text reading

### 1. DGX Spark pick

**Primary: PP-OCRv6_medium** (PaddlePaddle/Baidu, released 11–22 June 2026). **34.5M parameters
total** (15.5M detection + 19M recognition) on the PPLCNetV4 backbone; **86.2% detection Hmean and
83.2% recognition accuracy** on PaddleOCR's multi-scenario benchmark, +4.6/+5.1 over PP-OCRv5_server
— and **+47.9 Hmean / +8.3 accuracy over Qwen3-VL-235B-A22B**, +39.4 / +11.8 over Gemini-3.1-Pro,
at ~6,800× fewer parameters [1,2]. One model covers **50 languages** (zh-Hans/zh-Hant/en/ja + 46
Latin-script). Weights CC BY 4.0, code Apache-2.0. Footprint **~1 GB**; 0.29 s/image end-to-end on
an A100, so a GB10 will do a full feature's sampled frames in minutes.

**The Spark gotcha is the framework, not the model.** PaddlePaddle publishes **no aarch64 GPU
wheel** — official ARM64 builds are CPU-only, and on Jetson/DGX Spark Paddle silently falls back to
CPU even with CUDA/cuDNN/TensorRT present [3,4]. Two working paths, in order:
(a) **Run the ONNX variants.** PaddleOCR 3.x ships ONNX exports and accepts
`engine="onnxruntime"` for PP-OCRv6 directly, so you never build Paddle; pair with the prebuilt
`onnxruntime-gpu 1.22.0+cu13.3` aarch64/sm_121 wheel, or build ORT with
`CMAKE_CUDA_ARCHITECTURES=121` (20–40 min) [1,5,9,10]. **This is the recommendation.**
(b) Build `paddlepaddle-gpu` from source for CUDA 13.0 / SM 12.1 — documented working on a DGX
Spark on 6 Apr 2026 (~40 min build, Eigen vectorisation flags required, produced
`paddlepaddle_gpu-3.4.0.dev20260405-cp312-cp312-linux_aarch64.whl`) [4]. Works, but it is a
from-source dependency in your install path.

**Alternates:** (a) **PaddleOCR-VL-1.6** (0.9B ERNIE-based VLM, June 2026) — **96.33% on
OmniDocBench v1.6**, SOTA, beating Qwen3-VL-235B and Gemini-3 Pro; ~2 GB BF16, plain HF
`transformers`, so **no Paddle build needed**. Use it as the escalation tier for dense/stylised
frames [6,7]. (b) **GoMatching++** for video text *spotting* — freezes an image spotter and adds a
light tracker; SOTA on ICDAR15-video/BOVText/DSText (+13.55 MOTA over TransDETR on ICDAR15-video),
70% fewer trainable parameters. This is what gives you stable text *tracks* across a shot, which
per-frame OCR cannot [8].

### 2. Unconstrained pick

**Primary: PP-OCRv6_medium + PaddleOCR-VL-1.6 consensus, with GoMatching++ track-level voting.**
This is the honest answer and it costs almost nothing. **Adding a frontier open VLM makes OCR
worse, not better**: Qwen3-VL-235B-A22B scores 38.3% detection Hmean and 74.9% recognition against
PP-OCRv6_medium's 86.2 / 83.2 [2]. Margin over the Spark pick: essentially zero on recognition;
the gain is in *reading order, layout and semantics* (is this a subtitle, a sign, a credit roll?),
which is a different task.
**Alternates:** (a) **GLM-5.3-Flash** (Z.ai, 26 Aug 2026) — 320B total / 18B active MoE, natively
multimodal (text/image/video/file), 1,048,576-token context, **MIT licence**; the largest open
natively-multimodal model as of today and the right tool for "explain what this frame's text
means in context". At 4-bit it is ~160–170 GB, so it **does not fit one 128 GB Spark** — it needs
two paired Sparks at best, realistically a B200 node [11]. (b) Qwen3-VL-235B-A22B (Apache-2.0) for
the same semantic role.

### 3. API pick

**Primary: Gemini 3.1 Pro** (Google DeepMind, **19 February 2026**) — native text/image/audio/video
input and 1M context, so you can hand it a shot and ask for every legible on-screen string with
timing and role [12]. Use it for semantics and hard frames, not as the primary recogniser (46.8%
Hmean on PaddleOCR's detection benchmark) [2].
**Alternates:** (a) **Gemini 3.8 Flash** (**2 September 2026**, newest Flash in the line) for
high-volume per-frame passes [12]. (b) **Claude Opus 5** (Anthropic, **24 July 2026**) — video and
image input, ranked #4 on llm-stats' aggregate vision index vs Gemini 3.1 Pro at #27 [13].
(c) Google Cloud Video Intelligence `TEXT_DETECTION`, which returns per-frame boxes *with tracks*
— the only API that gives the geometry F4 (text replacement) needs.

### 4. Compute sensitivity

**2/5.** Compute buys little and can buy negative. The definitive 2026 number: a **34.5M**
specialist beats a **235B** frontier VLM by **+8.3 points recognition and +47.9 points detection
Hmean** [2]; and the best document model on OmniDocBench v1.6 is **0.9B** (PaddleOCR-VL-1.6,
96.33%), ahead of Qwen3-VL-235B and Gemini-3 Pro [6,7]. The genuine wins are temporal — tracking a
string across frames and voting (GoMatching++, +13.55 MOTA) — and resolution/crop strategy, not
parameters. Score 2 rather than 1 only because a VLM tier does measurably help on stylised titles,
low-contrast credits and reading-order decisions.

### 5. Delta vs prior pass

**Changed — the deployment path, not the model.** The 3 September verdict (PP-OCRv6 as the
per-frame engine, VLM as escalation, Qwen3-VL-8B/32B as the local substitute) is still the right
shape, but it assumed PP-OCRv6 just runs. **It does not run on GPU on a Spark out of the box:
PaddlePaddle has no aarch64 CUDA wheel** [3,4]. Ship the **onnxruntime backend** instead. Two
further corrections: the local escalation model should be **PaddleOCR-VL-1.6 (0.9B, 96.33%
OmniDocBench v1.6)** rather than Qwen3-VL-8B/32B — it is smaller, needs no Paddle build, and beats
the 235B model [6,7]; and the cloud tier name should be pinned to **Gemini 3.1 Pro (19 Feb 2026)**
with **Gemini 3.8 Flash (2 Sep 2026)** as the volume tier, not the vague "Gemini 3.x Flash" [12].

### Sources
1. PP-OCRv6 on Hugging Face (tiers 1.5M/7.7M/34.5M, 50 languages, ONNX variants, 22 Jun 2026) — https://huggingface.co/blog/PaddlePaddle/pp-ocrv6
2. PP-OCRv6 paper: 86.2 Hmean / 83.2 acc; Qwen3-VL-235B 38.3/74.9; Gemini-3.1-Pro 46.8/71.4 — https://arxiv.org/html/2606.13108v1
3. PaddleOCR ARM/aarch64 support discussion (no GPU wheel) — https://github.com/PaddlePaddle/PaddleOCR/discussions/17328
4. PaddlePaddle with GPU on DGX Spark — source build, CUDA 13.0, CC 12.1 (6 Apr 2026) — https://news.metaparadigma.de/dgx-spark-installing-paddlepaddle-ocr-on-nvidia-dgx-spark-5348/
5. PaddleOCR Paddle2ONNX / onnxruntime deployment — https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html
6. PaddleOCR-VL-1.6 (0.9B, 96.33% OmniDocBench v1.6) — https://arxiv.org/html/2606.03264v1
7. PaddleOCR-VL-1.6 model card — https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
8. GoMatching++ video text spotting — https://arxiv.org/abs/2505.22228
9. Prebuilt onnxruntime-gpu aarch64 CUDA 13 sm_121 wheel — https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121
10. ONNX Runtime GPU on DGX Spark build guide — https://forums.developer.nvidia.com/t/onnx-runtime-gpu-inference-on-dgx-spark-gx10-build-guide-and-prebuilt-binaries/366157
11. GLM-5.3-Flash (320B-A18B, MIT, 26 Aug 2026) — https://huggingface.co/zai-org/GLM-5.3-Flash
12. Gemini version history (3.1 Pro 19 Feb 2026; 3.8 Flash 2 Sep 2026) — https://en.wikipedia.org/wiki/Gemini_(language_model)
13. Claude Opus 5 (24 Jul 2026), video input, vision index — https://llm-stats.com/models/compare/claude-opus-5-vs-gemini-3.1-pro-preview

---

## B4 Content type classification (live action / 2D animation / 3D CG / hybrid)

### 1. DGX Spark pick

**Primary: SigLIP 2 (so400m) frozen embeddings + a calibrated linear head trained on OpenDub
frames.** SigLIP 2 (Google DeepMind, Feb 2025) is **still the current generation — there is no
SigLIP 3 as of 18 September 2026** [1,2]. The so400m encoder is ~400M parameters; BF16 resident
footprint with the head is **~2 GB**, and embedding a few hundred sampled frames per title is
seconds on a GB10. Pure `transformers`/PyTorch, so the stock aarch64 CUDA 13 wheels cover it; no
flash-attn, no Paddle, no vLLM — **this category has zero Spark compatibility risk** [7].
Train the head on 300–800 frames from OpenDub's own catalogue, because the boundary that matters
(cel-shaded 3D that must be routed to the 2D mouth-retiming path; rotoscope; stop-motion; hybrid
live-action-plus-CG) is defined by *your* downstream routing, not by a public taxonomy. Wrap it in
**split conformal prediction / temperature scaling** so the classifier can abstain and defer rather
than misroute a title into the wrong lip-sync branch.

**Alternates:** (a) **deepghs/anime_classification** (MIT, CPU-runnable, ~100 MB) — 4-way
3D / Bangumi / Comic / Illustration; day-one zero-training gate, and the sibling
**deepghs/anime_real_cls** is a direct anime-vs-photoreal head [3]. Note the community SigLIP 2
fine-tune `prithivMLmods/Anime-Classification-v1.0` reports **0.8648 overall accuracy** on the same
four classes (3D 0.82 F1, Bangumi 0.87, Comic 0.95, Illustration 0.82) — a useful reality check on
the higher numbers quoted for this family [4]. (b) **Qwen3-VL-8B** (Apache-2.0, ~16 GB BF16) as a
local tiebreaker on abstained titles.

### 2. Unconstrained pick

**Primary: GLM-5.3-Flash** (Z.ai, 26 Aug 2026) — 320B total / 18B active, natively multimodal over
text/image/**video**, MIT licence, 1M context; sample 16–32 frames per title, ask for a structured
verdict with a confidence and a reason, and majority-vote [5]. At 4-bit it is ~160–170 GB, so it
**does not fit one Spark** (128 GB) — that is the only category in phase B where the Spark is a
hard wall, and it is a wall you have no reason to hit. **Margin over the Spark pick: negligible on
the common classes, meaningful only on the ~2–5% genuinely ambiguous tail** (rotoscoped animation,
photoreal CG, stop-motion, mixed live-action/animation titles), where a VLM can *explain* its call
and a linear probe cannot.
**Alternates:** (a) Qwen3-VL-235B-A22B (Apache-2.0) for the same multi-frame vote.
(b) An ensemble: SigLIP 2 probe + deepghs head + VLM, with disagreement routed to human review —
in practice this beats any single model because the cost of a wrong route (2D retiming applied to
live action) is high and asymmetric.

### 3. API pick

**Primary: Gemini 3.1 Pro** (19 Feb 2026) — native video input means you classify from the actual
clip rather than stills, which is what separates cel-shaded 3D from 2D (motion smoothness,
frame-rate stepping on 2s/3s) [6]. **Alternates:** (a) **Gemini 3.8 Flash** (2 Sep 2026) for the
per-title high-volume pass [6]. (b) **Claude Opus 5** (24 Jul 2026) as a second opinion — video
input, #4 aggregate vision index [8]. All three should be used *with an explicit abstain option*
in the prompt; a confident wrong answer here is worse than a deferral.

### 4. Compute sensitivity

**2/5.** This is a coarse 3–5-way visual-style decision that a frozen encoder plus a linear head
solves; published numbers on the closest public task sit in the 0.86–0.97 accuracy band for models
of ~100M–400M parameters [3,4]. More compute does not move the easy 95%; it only helps on the
ambiguous tail, and even there the honest gain comes from **multi-frame temporal evidence and
calibrated abstention** rather than from parameter count. Concretely: going from a 400M SigLIP 2
probe to a 320B VLM changes the routing decision on a small minority of titles, and those are
exactly the titles a human should see anyway.

### 5. Delta vs prior pass

**Unchanged.** SigLIP 2 + calibrated linear head remains the Spark primary, with
deepghs/anime_classification as the zero-training day-one gate — exactly the 3 September verdict.
No newer image encoder has appeared (**no SigLIP 3**), and no dedicated live-action-vs-animation
video benchmark or model surfaced in 2026 [1,2]. Two refinements: the escalation tier should now be
named **Gemini 3.1 Pro / Gemini 3.8 Flash (2 Sep 2026)** rather than "Gemini 3.x Flash class" [6],
and **GLM-5.3-Flash (MIT, 26 Aug 2026)** is a new open-weights option for the unconstrained tier
that did not exist when the prior pass was written [5]. Also worth recording honestly: the 97.23%
accuracy figure the prior pass attributed to this model family is not reproducible from the model
cards I could verify — the comparable published number is **0.8648** on the same four classes [4].

### Sources
1. SigLIP 2 (Feb 2025; still current, no SigLIP 3 as of Sep 2026) — https://huggingface.co/blog/siglip2
2. SigLIP 2 paper — https://arxiv.org/abs/2502.14786
3. deepghs/anime_classification (3D / Bangumi / Comic / Illustration) — https://huggingface.co/deepghs/anime_classification ; https://huggingface.co/deepghs/anime_real_cls
4. prithivMLmods/Anime-Classification-v1.0 (SigLIP 2 fine-tune; 0.8648 accuracy) — https://huggingface.co/prithivMLmods/Anime-Classification-v1.0
5. GLM-5.3-Flash (320B-A18B, MIT, native multimodal incl. video, 26 Aug 2026) — https://huggingface.co/zai-org/GLM-5.3-Flash
6. Gemini version history (3.1 Pro 19 Feb 2026; 3.8 Flash 2 Sep 2026) — https://en.wikipedia.org/wiki/Gemini_(language_model)
7. Prebuilt GB10/sm_121 aarch64 wheels (torch 2.13.0+cu13.3) — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels
8. Claude Opus 5 (24 Jul 2026) — https://llm-stats.com/models/compare/claude-opus-5-vs-gemini-3.1-pro-preview

---

# Phase C · Text transformation

## C1 Dub-line segmentation

### 1. DGX Spark pick

**Primary — SaT `sat-12l-sm` (wtpsplit) with the `ted` ASR-style LoRA, driving a pause/duration split search over faster-whisper word timestamps.** 0.3 B parameters (12-layer XLM-R-derived encoder), BF16, ~0.7 GB resident, MIT licence, 85 languages [1]. Throughput is thousands of sentences/s on the GB10; a 45-minute episode segments in well under a second of GPU time. No aarch64 risk at all: it is plain `transformers` + PyTorch with no custom CUDA kernels, no flash-attn, no bitsandbytes — the exact class of model that is trivially safe on SM 12.1. Run it with the ONNX path if you want it off the GPU entirely and leave the memory for C2.

The model is only half the stage. The decisive component is the split *search*: score every candidate boundary as a weighted sum of (a) SaT boundary probability, (b) inter-word pause length from faster-whisper word timestamps, (c) syntactic-constituent cost (the HW-TSC parsing-informed pause selector), and (d) predicted target-side duration fit. That search is CPU-side and free.

**Alternates.** (a) **Qwen3.8-27B** (Apache 2.0, 27 B, 262 K native context) re-segmenting a timed word list at scene level — it is already resident for C2, so the marginal cost is one extra pass; verified to run on a single Spark under vLLM 0.27.1 with NVFP4 [7]. (b) **SHAS** (audio-side, wav2vec2-based, ~300 M) when the transcript is unreliable and you want boundaries from the acoustics rather than the text.

### 2. Unconstrained pick

**Primary — a frontier LLM doing joint segmentation + translation + length budgeting in one scene-level pass, with best-of-N sampling reranked by a QE model (C4).** Segmentation quality is bounded by *information the segmenter does not have* — who is on screen, whether the next line is an interruption, whether the target language can carry the clause — not by parameter count. A frontier model with the full scene, the character list and the timing grid in context is the ceiling. Largest open-weights equivalent: **GLM-5.3** (744 B total / ~40 B active, GLM-5.3 bespoke licence, open weights since end of August 2026) or **Qwen3.8-2.4T-A95B** (12 Aug 2026) [5][6]. Margin over the Spark pick is real but small and mostly indirect: SaT gives you boundaries, the LLM gives you boundaries that survive translation. Expect the gain to show up as fewer C2 retries and lower SubER downstream, not as a sentence-F1 delta.

### 3. API pick

**Primary — Claude Opus 5** (`claude-opus-5`, 1 M context, adaptive thinking, current lineup as of Sept 2026 [8]). The whole episode's timed word list fits in context, which is what this task actually needs. **Alternates:** Gemini 3.1 Pro (19 Feb 2026 [9]) — the model with the strongest published MT evidence, so segmenting and translating with the same model is attractive; GPT-6 Astra (`gpt-6-astra`), OpenAI's current flagship per its model docs, September 2026 [10].

### 4. Compute sensitivity

**2.** The prior pass's own framing is right: this is a search problem with a cheap scoring model. `sat-12l` over `sat-3l` is a ~1–2 point sentence-F1 difference on well-formatted text and the paper's headline claim is a 3× *speed* gain over the previous state of the art, not a quality jump [1][2]. What moves the needle is the pause/parse/duration cost function and word-level timestamps, both of which are free. The one place compute helps is co-deciding segmentation with translation (see 2), and that budget is better booked against C2.

### 5. Delta vs prior pass

**No change.** The SaT checkpoints are unchanged since June 2024 [1] (the wtpsplit toolkit itself is maintained — 2.2.1, 11 Apr 2026) — no 2026 successor exists, and nothing has appeared since 3 September 2026. One new thing worth tracking: **WMT26 adds a Video Subtitle Translation shared task organised by Tencent** (ZH→EN/TH/ID/MS/ZH-TW), which explicitly scores "concise enough to fit on screen while matching the timing of the original subtitles" and requires participants to open-source their models; results land at EMNLP 2026, Budapest, 28–29 Oct 2026 [3]. That is the first public benchmark that measures exactly what C1+C6 do together, and the open-sourcing requirement means usable checkpoints will exist six weeks from now.

### Sources
1. segment-any-text/sat-12l-sm model card (0.3 B, MIT, 85 languages; updated Jun 2024) — https://huggingface.co/segment-any-text/sat-12l-sm
2. Minixhofer et al., *Segment Any Text* (arXiv 2406.16678) — https://arxiv.org/abs/2406.16678
3. WMT26 Shared Task: Video Subtitle Translation — https://www2.statmt.org/wmt26/video-subtitle-translation.html
4. WMT26 General MT task (timeline, human-only evaluation) — https://www2.statmt.org/wmt26/translation-task.html
5. Qwen (Wikipedia) — Qwen3.8-2.4T-A95B, 12 Aug 2026 — https://en.wikipedia.org/wiki/Qwen
6. GLM-5.3 specs (744 B / ~40 B active, open weights end Aug 2026) — https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/
7. Qwen3.8-27B NVFP4 on a single DGX Spark / GB10 (vLLM 0.27.1) — https://github.com/gitcommit90/qwen38-27b-dgx-spark
8. Claude models overview (Fable 5.1 / Opus 5 / Sonnet 5 / Haiku 4.5) — https://platform.claude.com/docs/en/about-claude/models/overview
9. Gemini (language model), Wikipedia — Gemini 3.1 Pro, 19 Feb 2026 — https://en.wikipedia.org/wiki/Gemini_(language_model)
10. OpenAI model docs (GPT-6 Astra, GPT-5.6 Sol), September 2026 — https://developers.openai.com/api/docs/models

---

## C2 Translation and dubbing adaptation

### 1. DGX Spark pick

**Primary — Hy-MT2-30B-A3B in FP8 (`tencent/Hy-MT2-30B-A3B-FP8`), served by vLLM.** 30 B total / ~3 B active MoE, FP8, **~32 GB resident** (weights ~30 GB + KV), Apache 2.0, released **21 May 2026**, 33 languages [1][2]. Because only ~3 B parameters are touched per token, the Spark's 273 GB/s wall barely bites: the closest measured analogue is Qwen3-Coder-30B-A3B at **44.3 tok/s decode / 1,654 tok/s prefill** on llama.cpp on a Spark [6], and Hy-MT2 at FP8 should land in the same 40–60 tok/s band. That is fast enough to generate 8–16 length-variant candidates per dub line and let C4 rerank them — which is the whole point of having 128 GB.

Quality: on Tencent's own FLORES-200 table (XCOMET-XXL / CometKiwi / GEMBA ×100) Hy-MT2-30B-A3B scores **89.83 / 79.03 / 90.26** against **Gemini 3.1 Pro 90.30 / 78.96 / 92.14** and **GPT-5.5 89.60 / 78.96 / 91.65**, and beats DeepSeek-V4-Pro (88.60 / 78.11 / 90.15), Kimi K2.6 (87.68), HY-MT1.5-7B (86.90) and Tower-Plus-72B (79.69) [3]. It natively supports terminology intervention, contextual translation and format/delimiter preservation — i.e. the glossary, show-bible and timing-marker behaviour a dubbing stage actually needs.

**Gotchas.** (a) **8,192-token context** [2] — you cannot put a whole scene plus a show bible in the prompt; budget a rolling 2–3 line context window plus a compact glossary. (b) Vendor self-report; WMT25's human evaluation found the Hunyuan-family system placed first on *automatic* rankings for all but one language pair while human evaluation put it "considerably lower than the top-rated systems" [4] — treat the ~0.5-point gap to Gemini as optimistic. (c) On Spark, **prefer vLLM with FP8/AWQ over TensorRT-LLM with NVFP4**: NVFP4 kernels are still immature on GB10 and users measured AWQ beating NVFP4; NVIDIA's own container failed for several people [7][8].

**Alternates.** (a) **Qwen3.8-27B** (Apache 2.0, Aug 2026, 27 B, **262 K native context** extensible to 1 M, native vision) at NVFP4 — there is a published single-Spark vLLM 0.27.1 recipe measuring **44.46 tok/s with DFlash-2 speculative decoding (64.4% draft acceptance)** at 0.85 GPU-memory utilisation [9][10]. This is the one to use for the *adaptation* half of the job — scene-level rewriting to a syllable budget, character register, callbacks — where 8 K context is disqualifying. (b) **gpt-oss-120b** MXFP4, ~63 GB, **60.6 tok/s decode / 1,956 tok/s prefill measured on a Spark**, Apache 2.0 [6] — the largest model that runs *fast* on the box, but English-centric with no published multilingual MT evidence, so use it as a reranker/critic rather than the translator.

Nothing bigger is viable: Qwen3.5-397B-A17B at 4-bit is ~200 GB and GLM-5.3 (744 B) ships a 756 GB FP8 checkpoint [11][12]; Qwen3-235B-A22B only fits at Q2_K (~80 GB) and measures **20–25 tok/s across *two* linked Sparks** [8].

### 2. Unconstrained pick

**Primary — Gemini 3.1 Pro (19 Feb 2026)**, which tops the FLORES-200 / WMT25 / Mandarin↔minority tables in the Hy-MT2 report [3] and whose predecessor Gemini 2.5 Pro was the **best system overall in the WMT25 human evaluation, in the top cluster for 14 of 15 language pairs** [4]. **Alternates:** GPT-5.5/5.6 (within 0.5–0.7 XCOMET-XXL of Gemini on FLORES-200 [3][13]); **largest open-weights: Kimi K3** (2.8 T, the largest open-weight release to date, July 2026) or **GLM-5.3** (744 B-A40B, top open-weight entry on the Artificial Analysis Intelligence Index as of 8 Sept 2026) [11][14].

**Margin over the Spark pick: real but modest on adequacy, large on adaptation.** On FLORES-200 the gap to Hy-MT2-30B-A3B is **+0.47 XCOMET-XXL and +1.88 GEMBA** [3] — a fraction of a point per line. The gap that matters is elsewhere: WMT25 found the **speech domain the hardest of all domains** (ASR-noised, disfluent, colloquial — exactly OpenDub's input) and that SOTA systems still fail on non-standard input, terminology and gender agreement [4]. Long context, reasoning and instruction-following under a hard syllable budget are where the frontier models pull away, and none of that shows up in FLORES-200.

### 3. API pick

**Primary — Gemini 3.1 Pro** (`gemini-3.1-pro`, released 19 Feb 2026 [13]); it is the only frontier model with *current, independent, translation-specific* evidence in both the WMT25 human evaluation lineage [4] and a third-party 2026 report [3]. **Alternates:** (a) **Claude Opus 5** (`claude-opus-5`, current lineup Sept 2026, 1 M context, adaptive thinking; **Claude Fable 5.1** `claude-fable-5-1` when a line needs maximum reasoning) [15] — no published WMT-grade MT ranking exists for the Claude 5 generation, so pick it for register, dialogue naturalness and instruction-following under constraints rather than on a benchmark number, and A/B it with C4 before committing. (b) **GPT-6 Astra** (`gpt-6-astra`) or **GPT-5.6 Sol**, OpenAI's current lineup per its model docs, September 2026 [16]; GPT-5.5 is the version with published MT numbers [3].

In practice run all three behind the same interface and let the C4 QE stage pick per line — the per-line quality differences are smaller than the per-line variance.

### 4. Compute sensitivity

**5.** This is the most compute-sensitive category in the phase, but not in the naive way. Model scale alone buys **+0.5 to +1.9 points** (Hy-MT2-30B-A3B → Gemini 3.1 Pro on FLORES-200 [3]); within the specialist family, 1.8 B → 7 B → 30 B is **84.26 → 89.45 → 89.83 XCOMET-XXL** [3], i.e. steeply diminishing above 7 B. The real scaling lever is **best-of-N with QE reranking**: 128 GB lets you sample 16–32 length-constrained variants per line and pick with MetricX + a duration model, and WMT25 participants using MBR/QE reranking (Algharb, In2x, Wenyiil, Kaze-MT) clustered at the top of the constrained track [4]. Add a length/isochrony constraint and a second tightening pass and the compute is spent on *search*, which is where it pays.

### 5. Delta vs prior pass

**Changed — on both halves.** (a) The 3 September local pick "Hunyuan-MT-7B / HY-MT1.5-7B" is superseded by **Hy-MT2** (21 May 2026, Apache 2.0, 1.8 B / 7 B / 30 B-A3B), which lifts FLORES-200 XCOMET-XXL from 86.90 (HY-MT1.5-7B) to 89.83 (30B-A3B) and adds instruction-following translation plus the IFMTBench benchmark [1][3]. The MoE 30B-A3B variant is also strictly the better *Spark* choice than a 7 B dense model: same memory class, ~3 B active, so you get the quality of a 30 B at the decode speed of a small model. (b) The 3 September API pick "Gemini 2.5 Pro" is superseded by **Gemini 3.1 Pro** (19 Feb 2026) [13]. (c) A new caution, not in the prior pass: WMT25's headline finding is that **automatic metrics are biased in favour of exactly this family of fine-tuned specialists** [4], so do not let a COMET number alone decide between Hy-MT2 and a frontier model — gate on human or LLM-judge evaluation of *speech-domain* lines. (d) Also new since the prior pass: **TranslateGemma** (Google, 28 Jan 2026, 4 B / 12 B / 27 B open weights, 55 languages, trained with MetricX-QE and AutoMQM in the RL loop; the 12 B beats the 27 B Gemma 3 baseline on WMT24++) [17] — a clean, permissively licensed, Spark-sized fallback if Hy-MT2's 33-language list misses a target.

### Sources
1. Tencent-Hunyuan/Hy-MT2 (GitHub) — sizes, 33 languages, 8,192 ctx, features — https://github.com/Tencent-Hunyuan/Hy-MT2
2. tencent/Hy-MT2-30B-A3B-FP8 model card (Apache 2.0, 21 May 2026) — https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8
3. Hy-MT2 technical report, Table 2 (FLORES-200 / WMT25 / Mand.↔Min., XCOMET-XXL / CometKiwi / GEMBA) — https://arxiv.org/abs/2605.22064
4. Kocmi et al., *Findings of the WMT25 General MT Shared Task: Time to Stop Evaluating on Easy Test Sets* — https://aclanthology.org/2025.wmt-1.22/
5. HY-MT1.5 technical report (WMT25 champion lineage) — https://arxiv.org/abs/2512.24092
6. llama.cpp performance on NVIDIA DGX Spark (gpt-oss-120b MXFP4 60.6 tok/s; Qwen3-Coder-30B Q8_0 44.3 tok/s) — https://github.com/ggml-org/llama.cpp/discussions/16578
7. PSA: state of FP4/NVFP4 support for DGX Spark in vLLM — https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069
8. Qwen3 235B A22B on 2× DGX Spark (15 → 20–25 tok/s; vLLM+AWQ preferred over TensorRT-LLM+NVFP4) — https://forums.developer.nvidia.com/t/question-on-inference-performance-results-of-qwen3-235b-a22b-on-2x-dgx-spark/355053
9. Qwen3.8-27B NVFP4 on a single DGX Spark / GB10, vLLM 0.27.1, 44.46 tok/s with DFlash-2 — https://github.com/gitcommit90/qwen38-27b-dgx-spark
10. Qwen/Qwen3.8-27B model card (Apache 2.0, 262 K ctx, Aug 2026) — https://huggingface.co/Qwen/Qwen3.8-27B
11. GLM-5.3: 744 B / ~40 B active, 756 GB FP8 checkpoint, bespoke licence — https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/
12. Qwen (Wikipedia) — Qwen3.5 397B-A17B, Qwen3.8-2.4T-A95B — https://en.wikipedia.org/wiki/Qwen
13. Gemini (language model), Wikipedia — Gemini 3.1 Pro, 19 Feb 2026 — https://en.wikipedia.org/wiki/Gemini_(language_model)
14. Best open-weight LLMs 2026 (Kimi K3 2.8 T; AA Intelligence Index 8 Sept 2026) — https://kingy.ai/news/best-open-weight-ai-models-in-2026-glm-5-2-vs-deepseek-v4-vs-kimi-k2-6-vs-qwen-vs-mistral/
15. Claude models overview — https://platform.claude.com/docs/en/about-claude/models/overview
16. OpenAI model docs (GPT-6 Astra, GPT-5.6 Sol), September 2026 — https://developers.openai.com/api/docs/models
17. Google introduces TranslateGemma open models (28 Jan 2026, 4/12/27 B, 55 languages) — https://www.infoq.com/news/2026/01/google-translategemma-models/

---

## C3 Viseme-aware phonetic adaptation

### 1. DGX Spark pick

**Primary — PS-Comet-style N-best reranking: LLM paraphrase candidates, scored by a vowel-level DTW cost over G2P output plus a semantic term.** This is the prior pass's verdict, and it is now a fully specified, published method (Hong et al., **ICPR 2026**, arXiv 2604.09111 v4, 2 May 2026) [1] rather than a pattern to be reconstructed. Concrete recipe on one Spark:

- **Candidate generator — `gpt-oss-120b`, MXFP4**, 117 B total / 5.1 B active, Apache 2.0, **~63 GB resident**, measured on a GB10 at **60.6 tok/s decode / 1,956 tok/s prefill** (llama.cpp) [2][3]. Generating 16–32 paraphrases per line is cheap at that rate. Alternate generator: `Qwen3-32B` (Apache 2.0) at Q8_0 ≈ 33 GB, but as a *dense* 32 B it is bandwidth-bound at roughly 273 ÷ 33 ≈ **8 tok/s**, and BF16 (66 GB) halves that again — on this box, prefer MoE or 4-bit for anything dense above ~14 B.
- **Scorer — essentially free.** IPA from `phonemizer`/espeak-ng or `g2pk` (what PS-Comet used), or CharsiuG2P; vowel-centroid Euclidean distances → DTW (`tslearn`); combined as `α·DTW_norm + β·COMET` with **α = 1.6, β = 0.4** [1]. Isochrony gate: accept when predicted duration is within **±26 frames (~350 ms)** and LaBSE cosine ≥ **0.75**, otherwise regenerate, up to 60 iterations [1].
- **Runtime:** PS-Comet measures **1 min 34 s per 10 s clip on a V100** [1]; a GB10 is in that class or better. This is not the pipeline bottleneck.

**Gotchas.** aarch64 + **sm_121**: use NGC containers (PyTorch / vLLM / TensorRT-LLM) rather than PyPI wheels, many of which still have no sm_121 aarch64 build. On the Spark's unified memory, `cudaMemGetInfo` under-reports allocatable memory and `nvidia-smi` prints "Memory-Usage: Not Supported", so do not size your KV cache from either [4]. Note the licence split: espeak-ng is GPLv3 and `phonemizer` is GPLv3+ — shell out or use CharsiuG2P (MIT) if that matters for redistribution.

**Alternates.** (a) **SyncNet LSE-D / LSE-C reranking on rendered candidates** — accurate but requires rendering each candidate through TTS and lip-sync first, so it costs orders of magnitude more per line than the phonetic proxy; use it as an offline validator of the cost function's weights, not in the loop. (b) **Do nothing**, gated off by default — still defensible given the numbers in §4.

### 2. Unconstrained pick

**Primary — the same method with a frontier LLM in the generator slot; there is no meaningful unconstrained win.** PS-Comet's own ablations bound the headroom: on lip-reading test sets, LSE-D improves **12.671 → 12.175** (7.3 % relative) and LSE-C **1.128 → 1.404**, with UTMOS 2.453 → 2.614 [1]. On voice-actor clips (K2E), PS-Comet at **LSE-D 10.561 / LSE-C 1.457** *beat professional human voice actors* (11.118 / 1.260) [1] — i.e. the ceiling for text-side adaptation is already near.

**Margin over the Spark pick: negligible for text-side methods; large only if you change the problem.** The only large lip-sync gain comes from **modifying the video**: a commercial deepfake pipeline reaches **LSE-D 8.965 / LSE-C 2.528**, but VMAF collapses **98.229 → 86.208** and runtime goes from 1 m 34 s to **19 m 29 s (12.4×)** [1]. That trade belongs to OpenDub's lip-sync stage (phase F), not here.

### 3. API pick

Candidate generation only — the scoring stays local, because it is a deterministic function of G2P output and costs nothing. PS-TTS used ChatGPT-4o for paraphrases [1]. Current equivalents: **primary — Claude Opus 5** (`claude-opus-5`, 1 M context, adaptive thinking) [5], which can be given the character's register, the scene and the phonetic constraint in one prompt and asked for a diverse candidate set. **Alternates:** **GPT-6 Astra** (`gpt-6-astra`, 1.05 M context) or GPT-5.6 Sol [6]; **Gemini 3.8 Flash** (`gemini-3.8-flash`, docs updated 17 Sept 2026) [7] — Flash-tier is the right economics for generating 32 paraphrases per line.

### 4. Compute sensitivity

**2.** Quality here is set by the vowel-distance/DTW cost function and the duration gate, not by generator size: the *entire* method moves LSE-D by **0.50** (12.671 → 12.175) while modifying the video moves it by **3.2** (to 8.965) [1]. A bigger paraphrase generator gives you a more diverse candidate pool, which helps the search, but the marginal return is small compared with getting the cost weights (α = 1.6, β = 0.4) and the duration gate right. Implement it as a low-weight reranking term, ship it off by default with an operator weight, and spend the compute on C2 instead.

### 5. Delta vs prior pass

**No change — the prior primary stands, now with hard numbers behind it.** Two corrections to the prior pass's framing, though: (a) **Saboo and Baumann (WMT 2019)** integrates synchrony constraints *into encoder-decoder NMT training*; it is not n-best viseme reranking, so cite it as motivation rather than as the pattern being implemented [8]. (b) **IWSLT 2026 had no dubbing or isochrony track** — the new speech-generation track is Cross-Lingual Voice Cloning [9], so there is no 2026 shared-task evidence for this category. Nothing new on viseme-aware wording selection has appeared since 3 September 2026; a sweep of 2025–26 arXiv "viseme" work is all lip-reading and talking-head generation. The one adjacent update is **HOMURA v3, dated 3 September 2026** [10], which is about duration/syllable budgets rather than visemes and belongs to C2/C6.

### Sources
1. Hong, Song, Park, Bang, Ku, Lee, Kim — PS-TTS / PS-Comet, ICPR 2026 (arXiv 2604.09111 v4, 2 May 2026) — https://arxiv.org/abs/2604.09111
2. openai/gpt-oss-120b model card (Apache 2.0) — https://huggingface.co/openai/gpt-oss-120b
3. llama.cpp performance on NVIDIA DGX Spark (gpt-oss-120b MXFP4: 60.6 tok/s tg32, 1,956 tok/s pp2048) — https://github.com/ggml-org/llama.cpp/discussions/16578
4. NVIDIA DGX Spark known issues (unified memory reporting, `nvidia-smi` Memory-Usage not supported) — https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
5. Claude models overview — https://platform.claude.com/docs/en/about-claude/models/overview
6. OpenAI model docs (GPT-6 Astra, GPT-5.6 Sol), September 2026 — https://developers.openai.com/api/docs/models
7. Gemini API models (updated 17 Sept 2026) — https://ai.google.dev/gemini-api/docs/models
8. Saboo and Baumann, *Integration of Dubbing Constraints into Machine Translation*, WMT 2019 — https://aclanthology.org/W19-5210/
9. Findings of the IWSLT 2026 Evaluation Campaign — https://aclanthology.org/2026.iwslt-1.39/
10. HOMURA / Sand-Glass (Bilibili), arXiv 2601.10187 v3, 3 Sept 2026 — https://arxiv.org/abs/2601.10187
11. Qwen/Qwen3-32B model card (Apache 2.0) — https://huggingface.co/Qwen/Qwen3-32B
12. CharsiuG2P (MIT) — https://github.com/lingjzhu/CharsiuG2P

---

## C4 Reference-free translation quality estimation

### 1. DGX Spark pick

**Primary — MetricX-24-Hybrid-XXL (bfloat16) run in QE (reference-free) mode, as stage 1 over every line.** ~13 B parameters (mT5-XXL encoder-decoder), bfloat16, **~26 GB** resident, **Apache 2.0**, checkpoints `google/metricx-24-hybrid-xxl-v2p6` and `-bfloat16` [1]. The "hybrid" checkpoints score with *or* without a reference from the same weights, which is exactly what OpenDub needs (no reference exists for a dub line). At 26 GB it coexists with Hy-MT2-30B-A3B-FP8 (32 GB) in the 128 GB pool, so C2 generation and C4 scoring can be co-resident and you can rerank best-of-N without swapping models. Pure `transformers`/PyTorch, no custom kernels → no aarch64 or SM 12.1 risk; drop to `metricx-24-hybrid-xl-v2p6` (~3.7 B) if you want the memory back.

**The important correction: MetricX-25 has no public checkpoints.** The google-research/metricx repository releases MetricX-23 and MetricX-24 only, with no MetricX-25 and no 2026 update [1]. MetricX-25 exists as a Google-internal WMT25 submission built on **Gemma 3 12B converted to an encoder-only architecture** [2]; you cannot run it.

**Alternates.** (a) **xCOMET-XXL** (10.7 B, XLM-R-XXL base, ~21 GB BF16, 94 languages) for *error spans* rather than a scalar — it remains the strongest span baseline, best en-de char-F1 in the WMT25 comparison [2][3]. **Licence hazard: CC-BY-NC-SA-4.0, non-commercial**; Unbabel require you to contact them for a commercial service [3], which matters for an open-source project people will deploy. (b) A **GEMBA-style error-span judge run locally on Qwen3.8-27B or gpt-oss-120b** over the worst-scoring tail — the un-fine-tuned Gemma 3 27B baseline already reaches 18.18 avg char-F1 and *beats* everything else on ja-zh (28.42) [2], so a bigger local instruct model prompted for JSON error spans is a credible stage 2. (c) Keep the rule checks (language ID, length-ratio, glossary and named-entity match) in front of both — they cost nothing and catch the failures metrics miss.

### 2. Unconstrained pick

**Primary — an ensemble: MetricX-25-class scalar QE + a GemSpanEval-class fine-tuned span model + a frontier LLM GEMBA-ESA judge, combined AUTORANK-style.** WMT25's own official ranking method combined two reference-based metrics, one QE model and two LLM judges, rescaled and averaged [4] — that is the best-evidenced recipe available, and with unlimited compute you simply run all of it. GemSpanEval is **Gemma 3 27B** fine-tuned for error spans and reaches **19.54 avg char-F1 in QE mode vs xCOMET-XXL-QE's 16.23** [2]. Largest open-weights option for the judge slot: **GLM-5.3 (744 B-A40B)** or **Kimi K3 (2.8 T)** [5][6].

**Margin over the Spark pick:** meaningful. MetricX-25-QE improves segment-level tie-calibrated pairwise accuracy over MetricX-24-Hybrid-QE by **+2.37 (en-de, 54.97 vs 52.60), +0.92 (en-es, 69.42 vs 68.50) and +4.21 (ja-zh, 57.21 vs 53.00)** [2] — and those come from a *better base model at similar size* (Gemma 3 12B vs mT5-XXL 13B), not from more parameters. So the Spark is not the bottleneck; the unavailability of the newer checkpoint is.

### 3. API pick

**Primary — Gemini 3.1 Pro as a GEMBA-ESA error-span judge** (19 Feb 2026) [7]. GEMBA-ESA with frontier LLMs is the protocol WMT25 itself used for its reference QE pass (GPT-4.1 and Command A) [4], and Gemini leads the current MT evidence [8]. **Alternates:** (a) **Claude Opus 5** / **Claude Fable 5.1** [9] — 1 M context lets you judge a whole scene with the source video's character list and glossary in the prompt, which single-segment metrics structurally cannot do; (b) **GPT-6 Astra** (`gpt-6-astra`) [10]. Use two judges with different vendors and disagree-flag: WMT25 documented a systematic single-judge bias (Command A scored an Icelandic human reference **15.04 ESA points** below GPT-4.1 on a language it was not optimised for) [4].

### 4. Compute sensitivity

**4.** Quality here tracks the *base model* and the *ensemble size* more than raw parameter count. Concrete numbers: swapping the 13 B mT5-XXL backbone for a 12 B Gemma 3 backbone buys **+2.4 to +4.2 points** of segment-level pairwise accuracy at constant size [2]; a fine-tuned 27 B span model beats a 10.7 B encoder by **+3.3 char-F1** in QE mode (19.54 vs 16.23) [2]; and WMT25's official ranking needed **five systems ensembled** to be trusted over any single metric [4]. Scaling is real but sub-linear, and the single biggest win — a judge that sees the scene — is a context-window property, not a FLOPs property.

### 5. Delta vs prior pass

**Changed.** The 3 September primary **MetricX-25-QE is not runnable**: no checkpoints have been released, and the google-research/metricx repo still tops out at MetricX-24 as of September 2026 [1]. Substitute **MetricX-24-Hybrid-XXL-bfloat16** (Apache 2.0, ~26 GB) as the always-on stage-1 scorer and keep the rest of the prior verdict — GEMBA-v2-style error-span prompting on a frontier judge over the worst tail, rule checks in front. Two further updates: (a) **xCOMET-XXL's CC-BY-NC-SA licence** should be flagged in OpenDub's provider registry, since the prior pass did not note it and it blocks commercial redistribution [3]; (b) **WMT26's Automated Translation Quality Evaluation task adds Task 3, "Detection of Error-Free Segments"** — binary, scored by Matthews correlation, explicitly framed as "identifying publishable translations without human intervention" [11]. That is precisely OpenDub's gating decision, and it is exactly the metric to adopt once results publish at EMNLP 2026 (28–29 Oct 2026). Submissions closed 2 August 2026, so no results exist yet.

### Sources
1. google-research/metricx — MetricX-23 and MetricX-24 checkpoints only, Apache 2.0, no MetricX-25 — https://github.com/google-research/metricx
2. Juraska et al., *MetricX-25 and GemSpanEval: Google Translate Submissions to the WMT25 Evaluation Shared Task* (Gemma 3 12B / 27B; Tables 2 and 3) — https://aclanthology.org/2025.wmt-1.70/
3. Unbabel/XCOMET-XXL model card (10.7 B, XLM-R-XXL, CC-BY-NC-SA-4.0, 94 languages) — https://huggingface.co/Unbabel/XCOMET-XXL
4. *Findings of the WMT25 General MT Shared Task* (AUTORANK; GEMBA-ESA QE pass; Command A vs GPT-4.1 judge bias) — https://aclanthology.org/2025.wmt-1.22/
5. GLM-5.3 specs — https://kingy.ai/blog/glm-5-3-specs-benchmarks-api-how-to-use/
6. Best open-weight LLMs 2026 (Kimi K3 2.8 T) — https://kingy.ai/news/best-open-weight-ai-models-in-2026-glm-5-2-vs-deepseek-v4-vs-kimi-k2-6-vs-qwen-vs-mistral/
7. Gemini (language model), Wikipedia — https://en.wikipedia.org/wiki/Gemini_(language_model)
8. Hy-MT2 technical report, Table 2 — https://arxiv.org/abs/2605.22064
9. Claude models overview — https://platform.claude.com/docs/en/about-claude/models/overview
10. OpenAI model docs (GPT-6 Astra, GPT-5.6 Sol), September 2026 — https://developers.openai.com/api/docs/models
11. WMT26 Shared Task: Automated Translation Quality Evaluation Systems — https://www2.statmt.org/wmt26/mteval-task.html

---

## C5 Normalization, grapheme-to-phoneme and lexicon

### 1. DGX Spark pick

**Primary — `nemo-text-processing` 1.2.0 (WFST TN/ITN) + `misaki` for en/ja/zh/ko/vi + espeak-ng 1.52 as the explicit OOV fallback + the project PLS lexicon, with a local instruct LLM arbitrating flagged homographs.** All of the deterministic parts are CPU-side and effectively free (<1 GB); the LLM arbiter is any 8–30 B instruct model at BF16/FP8 (16–60 GB of the unified pool) and is invoked only on flagged tokens, so it costs almost nothing per episode. Licences: nemo-text-processing Apache 2.0, misaki Apache 2.0, espeak-ng GPLv3 (shell out, do not link).

nemo-text-processing **1.2.0 (5 June 2026)** is the actionable upgrade: it adds **Hindi and Vietnamese TN/ITN**, Korean TN/ITN, Hebrew ITN, pt-BR and Kinyarwanda TN and ja/fr expansions [1][2] — directly relevant to this branch's Hindi dub.

**aarch64 gotcha, and it is the one that will bite.** nemo-text-processing depends on Pynini, and **PyPI Pynini 2.1.7 ships x86_64 manylinux wheels only — no aarch64 wheels at all** [3]; a naive `pip install` will try to build OpenFst 1.8.4 + Pynini from source on the Spark. Fix: install **`pynini` 2.1.7 from conda-forge, which does publish `linux-aarch64`** [4], then `pip install --no-deps nemo-text-processing` (1.2.0 pins `pynini==2.1.6.post1` and will otherwise force the source rebuild). Second gotcha: misaki's Japanese path needs `pyopenjtalk` 0.4.1, which is **sdist-only with no wheels**, so it compiles from source on aarch64 (needs cmake + Cython) [5]. Third: do **not** take `phonemizer` as a runtime dependency — 3.4.0 (31 July 2026) is **GPLv3+** [6], a licensing hazard for OpenDub.

**Alternates.** (a) **CharsiuG2P ByT5** (MIT, 100 languages, ~300 M for the `byT5_small` checkpoint, ~0.6 GB FP16) for the long tail and for Hindi, where misaki has no coverage — but note the repo's last push was **26 May 2023** and the HF weights were last modified **27 Aug 2022** [7][8]: usable, frozen. (b) **NeMo G2P-Conformer-CTC**, documented as ~20× fewer parameters than the ByT5 G2P with non-autoregressive decoding and sentence-level heteronym handling [9] — the lowest-latency local option. (c) **Epitran** 1.35.2 (MIT-Modern-Variant, 190+ language-script pairs) for transliteration-shaped coverage; its English path needs the Flite `lex_lookup` binary compiled [10].

### 2. Unconstrained pick

**Primary — a frontier LLM in "parse mode" for homograph/polyphone/name decisions, with deterministic WFST TN/ITN kept underneath and dictionary-constrained decoding on top.** The honest finding is that **more compute does not buy much here**: a *small* discriminative CRF scoring paths over a dictionary word lattice, trained on 2 M+ LLM-generated sentences, reaches **99.62% reading accuracy, 0.32% word PER, 0.14% sentence PER** on Joyo-Kanji-Yomi [11] — better than the LLMs it was distilled from. Alternates: the largest open-weights instruct model you can host (GLM-5.3, Kimi K3) in the arbiter slot; or UR-BERT-style romanization if you want to drop G2P entirely (see 5).

**Margin over the Spark pick: small.** Lexicon and dictionary quality dominate model size by a wide margin; the only genuine gain from the frontier is on rare proper nouns and code-switched spans.

### 3. API pick

**Primary — Claude Opus 5** (`claude-opus-5`; **Claude Fable 5.1** for the hardest cases) for contextual G2P, homograph disambiguation and name transliteration [12] — 1 M context means the whole script, the character list and the established pronunciations for the show all fit in one prompt, which is what makes transliteration consistent across an episode. **Alternates — the backend lexicon APIs, which are where the answer is actually delivered:** (a) **Azure AI Speech custom lexicon**, W3C **PLS 1.0**, alphabets `ipa`/`sapi`/`ups`/`x-sampa`, plus `<phoneme>`/`<sub>`/`<say-as>`; limits **100 KB per lexicon file, 15-minute cache TTL, one locale per lexicon** (doc updated 5 June 2026) [13] — richest `say-as` coverage. (b) **ElevenLabs**: `.PLS`/TXT dictionaries; **Eleven v3** takes native IPA via `/…/` across 70+ languages; **Flash v2.5** supports SSML phoneme tags (Arpabet + IPA); **Multilingual v2 does not support phoneme tags — alias substitution only** [14]. (c) **Amazon Polly** PLS 1.0 lexicons, region-scoped, applied per `SynthesizeSpeech` call [15].

### 4. Compute sensitivity

**2.** Going from the best conventional morphological analyser to the best LLM cut Japanese kana CER from **1.03% to below 0.52%** across a >30-model benchmark, with "parse mode" beating direct prompting (Interspeech 2026) [16] — a real but bounded halving. And a tiny CRF-over-dictionary model reaches **0.32% word PER** on the same kind of task [11], i.e. beats the LLMs. Lexicon coverage and TN grammar coverage dominate; the prior pass's verdict that "the project lexicon is worth more than any model choice in this category" is correct and the 2026 evidence strengthens it.

### 5. Delta vs prior pass

**Mostly stands, with three amendments.** (a) **Upgrade nemo-text-processing to 1.2.0 and use the new Hindi TN/ITN grammars** [1][2] — the single most actionable change for the Hindi branch. (b) **misaki is effectively unmaintained** — no release since 0.9.4 (March 2025), no commit since 11 Aug 2025 — **and has no Hindi** [17][18]; keep it for en/ja/zh/ko/vi and route Hindi through espeak-ng or CharsiuG2P plus the project lexicon. (c) **Drop OpenPhonemizer** from the candidate list: 0.1.2, last released 22 March 2024, English-only [19]. Also: **CharsiuG2P is frozen** (2023 repo, 2022 weights) [7][8], and no new flagship multilingual G2P foundation model appeared in 2026 — a HuggingFace sweep returns only community fine-tunes. Worth tracking but not adopting yet: **UR-BERT** (Interspeech 2026) replaces G2P with universal romanization and scales to **495 languages** against ~100 for G2P pipelines [20], and 2026 papers on dictionary-constrained CRF G2P [11] and stress-aware sentence-level ByT5 G2P [21]. Neither removes the need for TN/ITN or a project lexicon. Nothing in this category changed between 3 and 18 September 2026.

### Sources
1. nemo-text-processing on PyPI — 1.2.0, 5 June 2026, Apache 2.0 — https://pypi.org/project/nemo-text-processing/
2. NVIDIA/NeMo-text-processing releases — r1.2.0 adds Hindi/Vietnamese TN-ITN — https://github.com/NVIDIA/NeMo-text-processing/releases
3. Pynini on PyPI — 2.1.7, x86_64 manylinux wheels only — https://pypi.org/project/pynini/
4. conda-forge pynini 2.1.7 for linux-aarch64 — https://anaconda.org/conda-forge/pynini
5. pyopenjtalk on PyPI — 0.4.1, sdist only — https://pypi.org/project/pyopenjtalk/
6. phonemizer on PyPI — 3.4.0, 31 July 2026, GPLv3+ — https://pypi.org/project/phonemizer/
7. lingjzhu/CharsiuG2P — MIT, last push 26 May 2023 — https://github.com/lingjzhu/CharsiuG2P
8. charsiu/g2p_multilingual_byT5_small_100 — weights last modified 27 Aug 2022 — https://huggingface.co/charsiu/g2p_multilingual_byT5_small_100
9. NVIDIA NeMo G2P docs (ByT5 G2P and G2P-Conformer-CTC) — https://docs.nvidia.com/nemo/speech/nightly/tts/g2p.html
10. Epitran on PyPI — 1.35.2, 190+ language-script pairs — https://pypi.org/project/epitran/
11. Dictionary-constrained CRF G2P (99.62% reading accuracy, 0.32% word PER), 17 Sept 2026 — https://arxiv.org/abs/2609.19805
12. Claude models overview — https://platform.claude.com/docs/en/about-claude/models/overview
13. Azure AI Speech SSML pronunciation / custom lexicon (PLS 1.0, 100 KB, 15-min cache) — https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation
14. ElevenLabs pronunciation guide (v3 IPA, Flash v2.5 phoneme tags, ML v2 alias-only) — https://elevenlabs.io/docs/best-practices/prompting/pronunciation
15. Amazon Polly managing lexicons (PLS 1.0) — https://docs.aws.amazon.com/polly/latest/dg/managing-lexicons.html
16. LLM G2P benchmark, 20 June 2026 (kana CER 1.03% → <0.52%) — https://arxiv.org/abs/2606.22009
17. misaki on PyPI — 0.9.4, March 2025, Apache 2.0 — https://pypi.org/project/misaki/
18. hexgrad/misaki commits — last commit 11 Aug 2025 — https://github.com/hexgrad/misaki
19. openphonemizer on PyPI — 0.1.2, 22 March 2024 — https://pypi.org/project/openphonemizer/
20. UR-BERT: universal romanization, 495 languages (Interspeech 2026) — https://arxiv.org/abs/2606.11681
21. Stress-aware sentence-level Filipino ByT5 G2P, 9 Sept 2026 — https://arxiv.org/abs/2609.09974

---

## C6 Subtitle condensation

### 1. DGX Spark pick

**Primary — HW-TSC's IWSLT 2026 two-pass LLM condenser, run on `Qwen3-32B` (Apache 2.0).** This is the only published 2026 result for LLM condensation against explicit per-cue budgets, and it was built on hardware in the Spark's class [1]. The recipe: parse cue timestamps, flag only the cues that violate CPS/CPL, then rewrite them — **pass 1 at temperature 0**, removing redundant auxiliaries and conjunctions only; **pass 2 at temperature 0.3** for deeper compression if the cue is still non-compliant. The prompt pins the character budget, the CPS target and proper-noun retention, and crucially feeds the **English source alongside the target** so the model compresses rather than paraphrases-away.

Measured on dev2026 en→zh, condensation off → on [1]:

| Domain | SubER ↓ | BLEU ↑ | CPS-compliant % ↑ | CPL-compliant % ↑ |
|---|---|---|---|---|
| Asharq | 60.10 → **59.29** | 28.95 → 29.00 | 94.33 → **98.81** | 87.65 → **98.97** |
| ITV | 63.50 → **62.94** | 22.77 → 22.40 | 70.08 → **87.16** | 96.25 → **99.29** |
| YODAS | 54.86 → **54.24** | 29.43 → 29.63 | 75.48 → **87.60** | 93.14 → **99.62** |

Condensation *improved* SubER and (mostly) BLEU — because the references are themselves condensed. That is the single most useful fact in this category: compression is not a quality tax.

**Spark fit.** Qwen3-32B is dense, so at Q8_0 ≈ 33 GB it is bandwidth-bound at roughly **8 tok/s**; that is fine here because only the violating cues are rewritten (3–30 % of a script), but if you want headroom use **gpt-oss-120b** MXFP4 at **60.6 tok/s** or **Qwen3-Coder-30B-A3B** Q8_0 at a measured **44.3 tok/s decode / 1,654 tok/s prefill** on a GB10 [2]. Same sm_121/aarch64 caution as elsewhere: NGC containers, not PyPI wheels; `nvidia-smi` does not report unified memory usage [3].

**Alternates.** (a) **AppTek-style length-class-conditioned MT with iterative re-translation** (below) if you are willing to train; (b) **gpt-oss-120b MXFP4** as the condenser when latency matters more than the extra ~1 BLEU.

### 2. Unconstrained pick

**Primary — AppTek's full stack**, which ranked first on virtually all metrics in the IWSLT 2026 subtitling track [4][5]: Transformer-Big MT conditioned on genre, gender and **length class**, iterative shortening until CPS/CPL/lines-per-block are satisfied, **ILS** (Intelligent Line Segmentation — a neural line-break classifier, proprietary), and a **gpt-4o automatic post-editing pass** over DP-segmented SRT chunks of ~20 sentences each. Results: ITV tst26 es **SubER 60.49**, CPS 99.73 %, CPL 100 %; Asharq tst26 ar **SubER 67.31**, CPS 99.81 % [4].

**Margin over the Spark pick: small, and not in the condensation step.** On Asharq tst26 zh, HW-TSC's Qwen3-32B system **beat AppTek on translation quality** (BLEU **33.64 vs 30.44**, BLEURT **0.5316 vs 0.5129**) while losing SubER by **0.33** (59.72 vs 59.39) [4]. The residual gap is line segmentation and timing, not rewriting — which is where the unconstrained budget should go.

**Alternates.** **HOMURA** (Bilibili): Qwen3-8B fine-tuned with GRPO on 8×H800 for syllable-budget compliance — Zh→En in-bounds rate **89.6 % vs 22.6 %** unconstrained, BLEU-ρ 0.376 vs 0.287, but CometKiwi **0.701 vs 0.748**, i.e. compliance bought at ~4.7 CometKiwi points [6]. Code and the **Sand-Glass** benchmark (1,000 instances, Zh→En/De/Es, five domains) are released research-use-only; **no weights**. Largest open-weights condenser: GLM-5.3 or Kimi K3, but nothing in the literature suggests scale is the binding constraint here.

### 3. API pick

**Primary — Claude Opus 5** (`claude-opus-5`, 1 M context) [7]: the winning IWSLT 2026 system used gpt-4o for automatic post-editing [4], and the current equivalent with a million-token window lets you post-edit a whole episode with the delivery spec, the glossary and the preceding cues in one prompt. **Alternates:** **GPT-6 Astra** (`gpt-6-astra`) or GPT-5.6 Sol [8]; **Gemini 3.8 Flash** (`gemini-3.8-flash`, docs updated 17 Sept 2026) [9] for the per-cue volume. Whichever you pick, keep AppTek's chunking discipline — ~20 sentences per call, and an instruction to "fix only when necessary; do not alter the order, number or length of blocks" [4].

### 4. Compute sensitivity

**2.** A 32 B open model recovered **+17.1 points of CPS compliance** on ITV (70.08 % → 87.16 %) while *improving* SubER [1]; frontier models add little on top because the residual error is line segmentation and timestamp assignment, not the rewriting. The one place compute buys something real is RL for hard budget compliance — HOMURA's **22.6 % → 89.6 %** in-bounds rate [6] — and even that trades ~4.7 CometKiwi points for it, so it is a tuning decision rather than a scaling one.

### 5. Delta vs prior pass

**No change to the primary — the AppTek recipe stands, because it won IWSLT 2026** [4]. But three corrections to the prior pass's specification, all of which affect implementation:

1. **The delivery spec is not "2 × 42 chars, 17–20 CPS" uniformly.** IWSLT 2026 used per-language TED/Netflix profiles: **21 CPS / 42 CPL** for ar, de, es; **9 CPS / 16 CPL** for zh; **4 CPS / 13 CPL** for ja; **max 2 lines** everywhere. AppTek additionally caps the *English* side at 17 CPS before translation [4]. OpenDub's constraint profiles must be per-target-language, not a single global pair of numbers.
2. **Japanese at 4 CPS is a broken profile** — the reference SRTs themselves are only **44.89 %** compliant, rising to **95.74 %** at 6 CPS [4]. Do not gate a Japanese dub on the nominal spec.
3. **IWSLT 2026 removed the separate text-compression track** and added zh/ja to subtitling [1], so "isometric MT" as a standalone task is gone; condensation now lives inside the subtitling pipeline, exactly as the prior verdict assumed.

SubER is confirmed as the primary ranking metric (implementation at `github.com/apptek/SubER`) [4][10]. New since 3 September 2026: **HOMURA v3, dated 3 September 2026** [6]. Also pending: **WMT26 Video Subtitle Translation** (Tencent Hunyuan + Tencent Video) — zh→en/th/id/ms/zh-TW, SRT format, **constrained track only: under 20 B parameters, open weights, non-commercial licence**; test data 1 July 2026, submissions 15 July 2026, but **no results, baselines or system papers are published yet** [11]. That will be the first public benchmark scoring on-screen fit and timing together, and the open-weights requirement means sub-20 B condensers will be downloadable shortly after EMNLP 2026 (28–29 Oct 2026).

### Sources
1. HW-TSC's submission to the IWSLT 2026 subtitling task (two-pass Qwen3-32B condenser; dev2026 SubER/BLEU/CPS/CPL) — https://aclanthology.org/2026.iwslt-1.10/
2. llama.cpp performance on NVIDIA DGX Spark (gpt-oss-120b 60.6 tok/s; Qwen3-Coder-30B 44.3 tok/s) — https://github.com/ggml-org/llama.cpp/discussions/16578
3. NVIDIA DGX Spark known issues — https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
4. Findings of the IWSLT 2026 Evaluation Campaign, Track IV: Subtitling (AppTek and HW-TSC results, CPS/CPL profiles) — https://aclanthology.org/2026.iwslt-1.39/
5. AppTek's submission to the IWSLT 2025 subtitling task — https://aclanthology.org/2025.iwslt-1.21/
6. HOMURA / Sand-Glass (Bilibili), arXiv 2601.10187 v3, 3 Sept 2026 — https://arxiv.org/abs/2601.10187
7. Claude models overview — https://platform.claude.com/docs/en/about-claude/models/overview
8. OpenAI model docs (GPT-6 Astra, GPT-5.6 Sol), September 2026 — https://developers.openai.com/api/docs/models
9. Gemini API models (updated 17 Sept 2026) — https://ai.google.dev/gemini-api/docs/models
10. Wilken, Georgakopoulou and Matusov, *SubER: A Metric for Automatic Evaluation of Subtitle Quality*, IWSLT 2022 — https://aclanthology.org/2022.iwslt-1.1/
11. WMT26 Shared Task: Video Subtitle Translation — https://www2.statmt.org/wmt26/video-subtitle-translation.html
12. Qwen/Qwen3-32B model card (Apache 2.0) — https://huggingface.co/Qwen/Qwen3-32B

---

# Phase D · Speech generation

## D1 Zero-shot cross-lingual voice cloning

### 1. DGX Spark pick

**Primary: Fish Audio S2 (Fish Audio / Hanabi AI), 4.4B (4B slow AR + 400M fast AR), bf16,
≈11 GB resident.** It holds the best published Seed-TTS-eval intelligibility of any model I could
verify — **zh WER 0.54 %, en WER 0.99 %, hard-zh 5.99 %** — explicitly beating Qwen3-TTS
(0.77/1.24), MiniMax Speech-02 (0.99/1.90) and Seed-TTS (1.12/2.25) on the same sets, and on the
24-language MiniMax multilingual testset it takes **best WER in 11 languages and best speaker
similarity in 17**, ahead of both MiniMax and ElevenLabs [S7,S8]. Coverage is ~80 languages from
10 M hours. Speed: RTF 0.195 on an H200 under SGLang; scale to ≈0.5–0.7 on a Spark. Gotcha: the
reference engine is SGLang, and I could **not** verify an aarch64/sm_121 SGLang build — budget for
a plain PyTorch eager fallback (roughly 2× slower) or a vLLM port. Licence not stated in the launch
post; the Fish Speech lineage has shipped CC-BY-NC-SA weights before, so **verify before shipping**.

**Alternates.** (a) **VoxCPM2 (OpenBMB), 2B, Apache-2.0, ~8 GB** — the safe pick: 30 languages plus
9 Chinese dialects, 48 kHz output, Seed-TTS-eval en 1.84/SIM 75.3, zh 0.97/SIM 79.5, hard 8.13,
MiniMax-multilingual mean 1.68, documented RTF 0.30 (0.13 with Nano-vLLM) on a 4090 → ≈0.4–0.9 on
Spark [S9,S10]. (b) **Qwen3-TTS-12Hz-1.7B-Base, Apache-2.0, ~6 GB** — the only D1 candidate with
published GB10 measurements, and it posts the **highest speaker similarity in all 10 of its
languages** against MiniMax-Speech and ElevenLabs, with cross-lingual MER 4.77–8.40 over 12
directed pairs [S5,S11,S12].

### 2. Unconstrained pick

**Primary: Fish Audio S2 at N=32 with cross-family judge re-ranking** (Whisper + a non-Whisper ASR
for WER, WavLM-based SIM for timbre, a naturalness predictor for MOS). The honest finding is that
the unconstrained pick is *the same model with more samples*, because parameter scale has stopped
paying in this category (see §4). **Margin over the Spark pick: small** — the same model runs on
the Spark, and the only thing a rack buys is N=32 instead of N=8 in wall-clock, worth roughly the
−12 % relative WER measured at N=10 [S6].

**Alternates.** (a) **MOSS-TTS-v1.5 (MossTTSDelay) 8B, Apache-2.0** — the largest open-weights
zero-shot TTS I can verify, 31 languages, 48 kHz stereo tokenizer; ~17 GB bf16, so it *also* fits
on a Spark. (b) **Higgs Audio v3 TTS 4B (Boson AI)** — 100+ languages, macro WER/CER 1.11 on
Seed-TTS, 2.74 on MiniMax-Multilingual (32 langs), 3.61 on its own 111-language set [S13,S14].

### 3. API pick

**Primary: Qwen-Audio-3.0-TTS Plus (Alibaba Cloud Model Studio / DashScope)** — cloning from a
short reference, 16 languages plus 20 Chinese dialect regions, 86 inline tags, free-form
natural-language direction; Elo 1259, #2 on the Artificial Analysis Speech Arena as of the
September 2026 snapshot (it was #1 at Elo 1234 on 23 July 2026) [S15,S16].
**Alternates.** (a) **ElevenLabs Eleven v3**, GA 2 February 2026 — 70+ languages with
accent-preserving cross-lingual cloning, the broadest hosted cloning coverage [S17]. (b) **MiniMax
Speech 2.8**, 23 January 2026 — 10 s cloning, explicitly tuned to reduce cross-lingual accent bleed
[S18]. Note Cartesia Sonic 3.6 leads the arena overall (Elo 1277) but that board scores *native*
voices, not cloning [S16,S19].

### 4. Compute sensitivity — **3**

More parameters have stopped buying cloning quality; more *sampling* and more *post-training* still
do. Within MOSS-TTS, the **8B beats its 1.7B sibling on WER but loses on similarity** (Seed-TTS-eval
EN 1.84/SIM 70.86 for 8B vs 1.93/SIM 73.28 for 1.7B; ZH 1.37/76.98 vs 1.44/79.62) [S10]. Within
CosyVoice 3, 0.5B→1.5B moves Seed-TTS-eval en WER the *wrong* way (2.02→2.21) and SIM not at all
(0.780→0.781) [S20]. Against that, DiffRO post-training gives 20–50 % relative WER improvement at
fixed size, and best-of-N re-ranking gives −12 % relative at N=10 [S20,S6]. So: Spark-sized models
are near the ceiling, and the 128 GB box converts directly into the two levers that still work.

### 5. Delta vs prior pass — **changed**

Prior primary was Qwen3-TTS-12Hz-1.7B-Base + VoxCPM2. Both still run well and both are Apache-2.0,
but **Fish Audio S2 (9 March 2026) posts strictly better Seed-TTS-eval WER than Qwen3-TTS on both
languages and best speaker similarity in 17 of 24 languages with ~80-language coverage** — it was
listed as "strong" rather than primary and that under-ranks it. **Higgs Audio v3 (4 June 2026,
100+ languages)** did not appear in the prior list at all. Nothing released 3–18 September 2026
changes this. Keep Qwen3-TTS as the Spark-proven fallback; it is the only one with published GB10
numbers.

### Sources
1. https://github.com/vllm-project/vllm/issues/31128 — Add support of Blackwell SM121 (DGX Spark)
2. https://github.com/vllm-project/vllm/issues/36821 — No sm_121 support on aarch64, DGX Spark
3. https://vllm.ai/blog/2026-06-01-vllm-dgx-spark — vLLM on the DGX Spark (1 Jun 2026)
4. https://forums.developer.nvidia.com/t/running-vllm-omni-for-qwen3-tts-voice-design-voice-clone-on-dgx-spark/361255 — vLLM-Omni + Qwen3-TTS on DGX Spark
5. https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530 — faster-Qwen3-TTS for GB10
6. https://arxiv.org/abs/2607.08256 — Best-of-N TTS Evaluation is Confounded by ASR Family Alignment (9 Jul 2026)
7. https://fish.audio/blog/fish-audio-open-sources-s2/ — Fish Audio open-sources S2 (9 Mar 2026)
8. https://arxiv.org/pdf/2603.08823 — Fish Audio S2 Technical Report
9. https://huggingface.co/openbmb/VoxCPM2 — VoxCPM2 model card
10. https://github.com/OpenBMB/VoxCPM — VoxCPM/VoxCPM2 repo and benchmark tables
11. https://github.com/QwenLM/Qwen3-TTS — Qwen3-TTS repo (22 Jan 2026)
12. https://arxiv.org/html/2601.15621v1 — Qwen3-TTS Technical Report
13. https://www.boson.ai/blog/higgs-audio-v3-tts — Higgs TTS 3 (4 Jun 2026)
14. https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/ — Higgs Audio v3 on SGLang-Omni
15. https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus — Qwen-Audio-3.0-TTS Plus
16. https://artificialanalysis.ai/text-to-speech/leaderboard — AA Speech Arena
17. https://elevenlabs.io/blog/eleven-v3 — Eleven v3
18. https://www.minimax.io/news/minimax-speech-28 — MiniMax Speech 2.8 (23 Jan 2026)
19. https://www.cartesia.ai/blog/sonic-3.6 — Sonic-3.6 (27 Aug 2026)
20. https://arxiv.org/html/2505.17589v2 — CosyVoice 3 (scaling + DiffRO tables)
21. https://github.com/OpenMOSS/MOSS-TTS — MOSS-TTS family

---

## D2 Expressive and style-transfer synthesis

### 1. DGX Spark pick

**Primary: IndexTTS 2.5 (Bilibili), 0.8B, bf16, ≈4 GB.** It is the only candidate whose
architecture matches the dubbing problem: **two independent prompt slots**, so the source segment
drives emotion and prosody while a separate reference supplies timbre — the source actor's voice
never leaks into the dub. Measured: SS/WER of 0.848/1.43 % (zh), 0.855/1.89 % (en), 0.833/9.95 %
(ja), 0.808/5.40 % (es); emotional sets reach MOS 4.11 with emotion-similarity 0.846 (ja) and MOS
3.93 / ES 0.924 (es) [S1]. GRPO RL fine-tuning (ASR-WER reward) takes average WER 6.75→6.00 and
average SIM 73.18→73.63 [S1]. Speed: T2S RTF 0.119 + S2M 0.017 on the reference GPU — **2.28×
faster than IndexTTS2** after halving the codec frame rate 50→25 Hz; expect ≈0.35–0.45 total on a
Spark. No aarch64 gotcha beyond plain PyTorch (GPT backbone + flow-matching + BigVGAN, no FA3).

**Alternates.** (a) **Step-Audio-EditX (StepFun), 3B, Apache-2.0, 12 GB min** — a *second-pass
editor*, not a synthesiser: take any take and iteratively push emotion, style (30+ including
whisper, child, older, news, advertising), or speed; emotion-editing accuracy reaches **81.6 %
(zh) by iteration 3**, and the 29 January 2026 update added exhale, snort, inhale, chuckle,
giggle, clears-throat [S2]. (b) **Higgs Audio v3 TTS 4B**, inline control tokens for emotion,
style, speed, pitch, pauses and sound effects mid-utterance; EmergentTTS win-rate 53.65 % overall
but **68.57 % on paralinguistics** and 61.43 % on questions [S3].

### 2. Unconstrained pick

**Primary: Fish Audio S2 4.4B**, which leads the expressiveness benchmarks I can verify:
**EmergentTTS-Eval 81.88 % win rate**, Fish Instruction Benchmark TAR 93.3 %, Audio Turing Test
posterior mean 0.515, with free-form textual direction (`[whisper in small voice]`,
`[professional broadcast tone]`, `[pitch up]`) rather than a fixed tag vocabulary [S4].
**Margin over the Spark pick: moderate but not decisive** — S2 wins on instruction-following
breadth, IndexTTS 2.5 wins on the disentangled-prompt structure that dubbing actually needs. The
best unconstrained system is the *pair*: IndexTTS 2.5 generates, Step-Audio-EditX or S2 refines,
an emotion-consistency judge picks.
**Alternates.** (a) Higgs Audio v3 4B. (b) MOSS-TTS-v1.5 8B, Apache-2.0, 31 languages, with
`[pause 3.2s]` and IPA/phoneme control [S5].

### 3. API pick

**Primary: ElevenLabs Eleven v3** (GA 2 February 2026) — audio tags plus multi-speaker dialogue
across 70+ languages, still the richest script-level emotional direction on a hosted API [S6].
**Alternates.** (a) **Qwen-Audio-3.0-TTS Plus** — 86 inline tags plus free-style natural-language
direction, Elo 1259 [S7,S8]. (b) **Cartesia Sonic 3.6** (27 August 2026) — infers emotional
subtext and pacing from the transcript with no tags at all; in blind US-English head-to-heads
listeners chose it over Eleven v3 **92 % of the time**, and up to 93 % over Sonic-3 across 15
locales [S9].

### 4. Compute sensitivity — **4**

Expressiveness is the category where extra compute converts most directly into quality, because
the failure mode is *variance* (one take in eight lands the reading) rather than capacity.
Two concrete numbers: Step-Audio-EditX's emotion accuracy climbs to 81.6 % only at **iteration 3**
of iterative editing — that is compute traded for accuracy inside one line [S2] — and IndexTTS
2.5's GRPO pass, itself a sampling-heavy procedure, moves average WER 6.75→6.00 and SIM
73.18→73.63 [S1]. Best-of-N with the G4 emotion-consistency judge is the natural use of the
128 GB box here, and unlike D1 the metric being re-ranked on (emotion similarity) is not the one
the model was trained against, so the headroom is not already consumed.

### 5. Delta vs prior pass — **partially changed**

IndexTTS 2.5 as primary **still stands** and runs comfortably on a Spark. Two corrections: the
prior pass ranked Step-Audio-EditX on its November 2025 release and missed the **29 January 2026
tag expansion**; and it did not have the EmergentTTS numbers that put **Fish Audio S2 (81.88 %
win rate)** and **Higgs Audio v3 (68.57 % on paralinguistics)** ahead of the rest on
instruction-following. CosyVoice 3 instruct-mode should drop from the shortlist — its 0.5B/1.5B
pair shows no expressive gain from scale [S10].

### Sources
1. https://arxiv.org/html/2601.03888v2 — IndexTTS 2.5 Technical Report
2. https://github.com/stepfun-ai/Step-Audio-EditX — Step-Audio-EditX (updated 29 Jan 2026)
3. https://www.boson.ai/blog/higgs-audio-v3-tts — Higgs TTS 3 (4 Jun 2026)
4. https://fish.audio/blog/fish-audio-open-sources-s2/ — Fish Audio S2
5. https://github.com/OpenMOSS/MOSS-TTS — MOSS-TTS family
6. https://elevenlabs.io/blog/eleven-v3 — Eleven v3
7. https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus — Qwen-Audio-3.0-TTS Plus
8. https://artificialanalysis.ai/text-to-speech/leaderboard — AA Speech Arena
9. https://www.cartesia.ai/blog/sonic-3.6 — Sonic-3.6 (27 Aug 2026)
10. https://arxiv.org/html/2505.17589v2 — CosyVoice 3

---

## D3 Duration-controlled synthesis

### 1. DGX Spark pick

**Primary: IndexTTS 2.5, 0.8B, bf16, ≈4 GB.** The IndexTTS2 lineage is built around exactly this
problem: the abstract names video dubbing as the motivating case and offers **two explicit modes —
specify the generated token count to control duration precisely, or free autoregressive generation
that reproduces the prompt's prosody** [S1]. 2.5 keeps that and adds an auxiliary embedding for
fine-grained duration plus new speech-rate control, while running 2.28× faster than IndexTTS2
(T2S RTF 0.119, S2M 0.017) [S2]. On a Spark expect total RTF ≈0.35–0.45 — fast enough that
rejecting and regenerating off-target takes is cheap.

**Alternates.** (a) **MOSS-TTS-Local-Transformer-v1.5, 4B, Apache-2.0, ≈9 GB** (Qwen3-4B backbone,
18 June 2026) — token-level duration via a `tokens` parameter, explicit `[pause 3.2s]` markers, and
pinyin/IPA/phoneme input, which is the only shortlisted path to *phoneme-level* timing control and
therefore the only one that can implement non-isoelastic stretching natively; 31 languages [S3].
(b) **MOSS-TTS-v1.5 8B**, same controls, ~17 GB, also fits.

### 2. Unconstrained pick

**Primary: IndexTTS 2.5 in token-count mode, sampled N=16–32 per line, keeping the take that lands
inside the duration window with the lowest WER and highest SIM.** Duration is a hard constraint, so
rejection sampling is the correct algorithm and it is embarrassingly parallel — this is one of the
few Phase D categories where a rack genuinely beats one Spark, purely on N per unit wall-clock.
**Margin over the Spark pick: low in quality, high in throughput** — same model, same per-take
quality, more takes per second.
**Alternates.** (a) MOSS-TTS-v1.5 8B for phoneme-level control. (b) A duration-aware re-ranker over
any D1 model plus non-isoelastic phoneme-level TSM as the fallback when no take fits.

### 3. API pick

No hosted model I could verify accepts an exact target duration in seconds; this remains the
clearest open-weights-only advantage in Phase D.
**Primary: Google Cloud Text-to-Speech — Chirp 3: HD**, which is the closest: explicit pace control
from 0.25× to 2×, built-in `[pause long]` markup for inserted silence, and IPA/X-SAMPA custom
pronunciation, across 57 locales and 28 voices [S4].
**Alternates.** (a) **Azure AI Speech MAI-Voice-2 / MAI-Voice-2-Flash** with SSML `<prosody rate>`
over 100+ locales (docs current as of 9–10 September 2026) [S5]. (b) **ElevenLabs Dubbing API**,
which performs its own isochrony internally but gives you no per-line duration handle [S6].

### 4. Compute sensitivity — **4**

Not because bigger models fit durations better, but because **rejection sampling against a hard
constraint is the textbook compute-for-quality trade** and it removes the time-stretch artefact
entirely. STEB quantifies the stake: only systems with *explicit* duration control exceeded a
Speech Length Consistency of 0.9 at a 20 % tolerance; everything else misses the window [S7]. On a
Spark, N=16 at RTF 0.4 costs roughly 6–7× realtime per line — acceptable offline, and it turns
"stretch by 12 %" into "pick the take that was already the right length".

### 5. Delta vs prior pass — **changed (version, not strategy)**

The prior verdict — wire `target_duration` through and the tempo stretch goes away — is correct and
unchanged. The version changes: **IndexTTS 2.5 supersedes IndexTTS2** as the Spark pick (2.28×
faster, RL-tuned, adds Japanese and Spanish), and **MOSS-TTS-Local-Transformer-v1.5 at 4B
supersedes the 1.7B checkpoint** the prior pass would have picked (18 June 2026). Note the MOSS
scaling caveat from D1: the 8B sibling is *worse* on speaker similarity, so take the 4B, not the 8B,
unless you need the extra languages.

### Sources
1. https://arxiv.org/abs/2506.21619 — IndexTTS2: emotionally expressive and duration-controlled zero-shot TTS
2. https://arxiv.org/html/2601.03888v2 — IndexTTS 2.5 Technical Report
3. https://github.com/OpenMOSS/MOSS-TTS — MOSS-TTS family (v1.5, 4B Local Transformer, 18 Jun 2026)
4. https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd — Chirp 3: HD
5. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support — Azure AI Speech (MAI-Voice-2)
6. https://elevenlabs.io/blog/eleven-v3 — ElevenLabs model family
7. https://arxiv.org/pdf/2606.25529 — STEB: S2ST Expressiveness Benchmark

---

## D4 Regional and low-resource language voices

### 1. DGX Spark pick

**Primary: VoxCPM2 (OpenBMB), 2B, Apache-2.0, ~8 GB, RTF ≈0.4–0.9 on Spark.** It is the best
breadth-per-licence option: **30 languages** covering Hindi, Arabic, Swahili, Thai, Vietnamese,
Khmer, Lao, Burmese, Malay, Tagalog, Indonesian, Hebrew, Greek, Turkish, Persian-adjacent scripts,
plus 9 Chinese dialects, at 48 kHz, with a MiniMax-multilingual mean of 1.68 % and CV3-eval
per-language numbers (de 4.77, es 3.80, it 4.25, zh 3.65) [S1,S2]. It supports LoRA and full
fine-tuning from 5–10 minutes of reference audio, which is the realistic path for any language
that fails acceptance.

**Alternates.** (a) **Higgs Audio v3 TTS 4B (Boson AI), ~10 GB** — the widest *evaluated* coverage
anywhere: single-digit WER/CER on **100 of 111 languages** on its internal Higgs-Multilingual set
(macro 3.61), 4.41 on CV3's 13 languages, 2.74 on MiniMax-Multilingual's 32 [S3,S4]. (b)
**IndicF5 (AI4Bharat), 0.4B, MIT, ~2 GB** — F5-TTS-based, 11 Indic languages, 1 417 h from Rasa /
IndicTTS / LIMMITS / IndicVoices-R; the Indic specialist and the cleanest licence [S5]. (c)
**OmniVoice (k2-fsa), 0.6B, ~3 GB, RTF 0.025** — **600+ languages**, Qwen3-0.6B base, diffusion-LM
style, coverage of absolute last resort; **weights are CC-BY-NC**, code Apache-2.0, so it is an
evaluation/coverage tool, not a shippable default [S6,S7].

### 2. Unconstrained pick

**Primary: Fish Audio S2 4.4B** — ~80 languages from 10 M hours, and crucially it takes **best
speaker similarity in 17 of 24 languages** on the MiniMax multilingual testset, beating both
MiniMax and ElevenLabs; timbre transfer holding up *outside* English is exactly what a dubbing
pipeline needs [S8]. **Margin over the Spark pick: meaningful on similarity, modest on
intelligibility** — and note S2 also fits on the Spark, so the real unconstrained advantage is
running per-language LoRA fine-tunes of VoxCPM2 for the whole tail in parallel.
**Alternates.** (a) Higgs Audio v3 4B (111-language evaluation). (b) MOSS-TTS-v1.5 8B, Apache-2.0,
31 languages including Swahili, Macedonian, Persian, Hungarian, Romanian [S9].

### 3. API pick

**Primary: Cartesia Sonic 3.6** (27 August 2026) — **61 locales, 11 of them Indic**, with Odia and
Urdu launched above 96 % transcript accuracy, Hindi transcripts accepted in Devanagari, Latin or
mixed script with Hindi↔English code-switching in a single generation, improved Indian name and
place pronunciation, 10 s cloning and sub-90 ms latency; it also tops both Artificial Analysis
speech arenas at Elo 1277 [S10,S11].
**Alternates.** (a) **Sarvam Bulbul V3 / V4** — 11 Indian languages; in an independent Josh Talks
listening study V3 ranked highest at 8 kHz against ElevenLabs and Cartesia Sonic-3, and V4 was
announced at Sarvam's Epoch conference on 30 July 2026 [S12,S13]. (b) **Azure AI Speech
MAI-Voice-2**, 100+ locales, the deepest catalogue [S14].

### 4. Compute sensitivity — **5**

This is the one Phase D category where scale plainly pays, and the numbers are unambiguous. In
CosyVoice 3, going 0.5B→1.5B barely moves Chinese or English (CV3-eval zh 3.89→3.91, en
5.24→4.99) but **collapses the low-resource error: Korean 12.8 % → 5.69 % (−56 %), Japanese
10.4 % → 7.57 % (−27 %)** [S15]. DiffRO post-training on top gives 20–50 % relative WER
improvement overall and **68.7 % relative for Korean** in one configuration [S15]. The tail is
data- and capacity-starved in a way English and Chinese are not, so both parameters and RL compute
keep buying quality here long after they have stopped in D1.

### 5. Delta vs prior pass — **changed**

VoxCPM2 + IndicF5 as the open-weights pair **still stands** and is still the right Apache-2.0 / MIT
combination. Three additions the prior pass did not have: **OmniVoice (31 March 2026, 600+
languages, CC-BY-NC weights)**, **Higgs Audio v3 (4 June 2026, single-digit error on 100 of 111
languages)**, and on the API side **Cartesia Sonic 3.6 (27 August 2026)**, which is a materially
better regional story than the Sonic 3.x the prior pass saw — 11 Indic locales, Odia and Urdu, and
script-agnostic Hindi. **Sarvam Bulbul V4** (30 July 2026) supersedes the V3 the prior pass listed.

### Sources
1. https://huggingface.co/openbmb/VoxCPM2 — VoxCPM2 model card
2. https://github.com/OpenBMB/VoxCPM — VoxCPM2 benchmarks and language list
3. https://www.boson.ai/blog/higgs-audio-v3-tts — Higgs TTS 3 multilingual tables
4. https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/ — Higgs Audio v3 on SGLang-Omni
5. https://huggingface.co/ai4bharat/IndicF5 — IndicF5 model card
6. https://huggingface.co/k2-fsa/OmniVoice — OmniVoice model card (0.6B, CC-BY-NC weights)
7. https://github.com/k2-fsa/OmniVoice/ — OmniVoice repo
8. https://fish.audio/blog/fish-audio-open-sources-s2/ — Fish Audio S2
9. https://github.com/OpenMOSS/MOSS-TTS — MOSS-TTS v1.5 language list
10. https://www.cartesia.ai/blog/sonic-3.6 — Sonic-3.6 (27 Aug 2026)
11. https://artificialanalysis.ai/text-to-speech/leaderboard — AA Speech Arena
12. https://invideo.io/blog/sarvam-bulbul-indian-tts/ — Sarvam Bulbul v3
13. https://explainx.ai/blog/sarvam-bulbul-v4-tts-emotion-voice-july-2026 — Bulbul V4 (30 Jul 2026)
14. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support — Azure MAI-Voice-2
15. https://arxiv.org/html/2505.17589v2 — CosyVoice 3 scaling + DiffRO

---

## D5 Licensed non-cloning voices

### 1. DGX Spark pick

Two readings of this category, and the Spark answers both without strain.

**Primary: VoxCPM2 voice-design mode, 2B, Apache-2.0, ~8 GB.** A *designed* voice has no donor at
all, which is a stronger licensing position than any cleared catalogue. VoxCPM2 posts the best
voice-design numbers I can verify — **InstructTTSEval APS 85.2 (zh) / 84.2 (en)** — generating a
voice from a text description and then reusing it as a stable cast member [S1,S2].

**Alternates.** (a) **Qwen3-TTS-12Hz-1.7B-VoiceDesign and -CustomVoice, Apache-2.0, ~6 GB** — a
natural-language voice designer plus 9 premium preset timbres, and the only Phase D model with
published DGX Spark measurements (bf16, CUDA graphs, ~30 s capture, RTF well under 1) [S3,S4].
(b) **Kokoro-82M v1.0 (hexgrad), Apache-2.0, ~0.5 GB** — still the only candidate with a readable
training-data provenance statement (public-domain, permissively-licensed and synthetic audio from
commercial providers, a few hundred hours), 8 languages, 54 voices; trivially fast on a Spark and
runnable on CPU [S5]. (c) **Chatterbox-Multilingual V3 (Resemble AI), 500M, MIT**, 23+ languages,
driven from a licensed reference library, with 350M Turbo and 110M Nano variants (Nano runs 3×
realtime on 8 CPU cores) [S6].

### 2. Unconstrained pick

**Primary: VoxCPM2 voice-design with best-of-N *casting* — generate 64 candidate voices per role,
score for distinctiveness against the existing cast and for licence-cleanliness, then freeze the
winner.** The compute goes in once per character, not once per line, so it costs almost nothing in
production. **Margin over the Spark pick: negligible** — this is the flattest category in Phase D.
**Alternates.** (a) **MOSS-VoiceGenerator 1.7B, Apache-2.0** — a dedicated voice generator in the
MOSS family [S7]. (b) **Raon-OpenTTS 1B**, open models *and* open data, Seed-TTS-eval WER 1.78 —
worth evaluating precisely because the data is inspectable [S8].

### 3. API pick

**Primary: Azure AI Speech — MAI-Voice-2 / MAI-Voice-2-Flash.** This is the current generation
(superseding Neural HD / HD Omni / HD Flash), with 18+ named emotional styles (angry, confused,
determined, embarrassed, hopeful, jealous, regretful, relieved, shouting, softvoice, whispering…),
100+ locales, 24 kHz and 48 kHz output, and Microsoft's unambiguous commercial-output terms; docs
current 9–10 September 2026 [S9].
**Alternates.** (a) **Cartesia Sonic 3.6** — 500+ preset voices over 61 locales and **Elo 1277,
first on the Artificial Analysis Speech Arena, which scores exactly this case (each provider's own
native voices)** [S10,S11]. (b) **Google Cloud TTS Chirp 3: HD** — 57 locales, 28 named voices
(14 F / 14 M), and **no voice cloning offered at all**, which is the cleanest possible
non-cloning guarantee for a regulated buyer [S12].

### 4. Compute sensitivity — **2**

The binding constraint is provenance and casting depth, not model capacity. Kokoro at **82 M
parameters** is still a credible bundled cast, and the arena leaders (Sonic 3.6, MAI-Voice-2) win
on voice data and tuning rather than size — the AA board that Sonic tops is a *native-voice* board
where model scale is not the differentiator [S11]. Best-of-N applies once per character at casting
time, not per line, so a 128 GB box buys a better audition process and essentially nothing else.
Score 2 rather than 1 only because voice-design quality (InstructTTSEval APS) does still track
model size modestly.

### 5. Delta vs prior pass — **changed**

The Kokoro-82M recommendation **still stands** (v1.0 remains current; I could not verify the
"Kokoro v3, ~25 languages, June 2026" claim that circulates in secondary write-ups — the model card
lists v1.0 as the latest, so treat v3 as unconfirmed). Two real corrections: **Azure's latest
generation is now MAI-Voice-2 / MAI-Voice-2-Flash, not "Neural HD 2.5 / HD V3"** as the prior pass
recorded; and **Cartesia Sonic 3.6 (27 August 2026) supersedes Sonic 3.x** and now leads both
Artificial Analysis speech arenas, which strengthens the case for trialling it against Azure on the
same 200-line script. Add VoxCPM2 voice-design as a first-class non-cloning option — the prior pass
listed VoxCPM2 only under cloning and breadth.

### Sources
1. https://huggingface.co/openbmb/VoxCPM2 — VoxCPM2 model card
2. https://github.com/OpenBMB/VoxCPM — InstructTTSEval voice-design scores
3. https://github.com/QwenLM/Qwen3-TTS — Qwen3-TTS VoiceDesign / CustomVoice
4. https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530 — Qwen3-TTS on GB10
5. https://huggingface.co/hexgrad/Kokoro-82M — Kokoro-82M v1.0 model card and provenance
6. https://github.com/resemble-ai/chatterbox — Chatterbox Multilingual V3 / Turbo / Nano (MIT)
7. https://github.com/OpenMOSS/MOSS-TTS — MOSS-VoiceGenerator
8. https://arxiv.org/pdf/2605.20830 — Raon-OpenTTS: Open Models and Data for Robust TTS
9. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support — Azure AI Speech, MAI-Voice-2
10. https://www.cartesia.ai/blog/sonic-3.6 — Sonic-3.6
11. https://artificialanalysis.ai/text-to-speech/leaderboard — AA Speech Arena (provider-voice board)
12. https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd — Chirp 3: HD

---

## D6 Voice conversion

### 1. DGX Spark pick

**Primary: Vevo2 (CUHK-Shenzhen + ByteDance Seed, in Amphion), 872 M total, bf16, ≈3 GB.**
Released 25 March 2026; the parameter budget splits 509 M autoregressive content-style model +
363 M flow-matching acoustic model + 255 M unified vocoder [S1]. It is the only shortlisted system
that lets you *choose* style-preserved versus style-converted conversion — critical for dubbing,
where you sometimes want the source actor's delivery on the target timbre and sometimes not — and
it unifies speech and singing, so a sung line does not need a separate path. Joint speech+singing
pre-training gives **WER 19.39 → 15.78, N-CMOS −1.15 → −0.66, SS-CMOS −0.65 → −0.17** [S1,S2].
GRPO multi-objective post-training (intelligibility + prosody-similarity rewards) adds stability.
Amphion code is MIT [S3]. No aarch64 gotchas: plain PyTorch, no FA3, ~3 GB — you can hold eight
copies on a Spark.

**Alternates.** (a) **Chatterbox VC (Resemble AI), S3Gen-based, MIT, ~2 GB** — the simplest
drop-in, and it shares weights with Chatterbox-Multilingual V3 so one download serves D4 and D6
[S4]. (b) **kNN-VC / FreeVC, MIT** — non-parametric and near-free; useful as a regression baseline
that cannot hallucinate content.

### 2. Unconstrained pick

**Primary: re-synthesis instead of conversion — drive Fish Audio S2 or VoxCPM2 with the known
transcript and the target speaker reference, and use Vevo2 only where no transcript exists.** In a
dubbing pipeline you *have* the text, so the "conversion" problem collapses into the D1 problem,
where the best systems are far stronger (Seed-TTS-eval WER 0.54/0.99 and best-SIM in 17 of 24
languages for S2) than any VC system's intelligibility [S5]. Keep Vevo2 for non-verbal, sung, or
untranscribable audio. **Margin over the Spark pick: meaningful for transcribable speech,
negligible otherwise** — and both models fit on the Spark anyway.
**Alternates.** (a) Vevo2 at N=16 with a speaker-similarity judge. (b) X-VC (SJTU/Tianjin/Fudan)
as an academic comparator.

### 3. API pick

**Primary: ElevenLabs Voice Changer** (speech-to-speech), the only mature hosted VC with the same
voice library and accent handling as Eleven v3 [S6].
**Alternates.** (a) **Respeecher Marketplace** — a cleared-voice marketplace with contractual
consent, which is the actual differentiator for a commercial dub. (b) **Cartesia Sonic 3.6**
instant cloning used as pseudo-VC via re-synthesis [S7].

### 4. Compute sensitivity — **3**

Vevo2's own ablation says the gains came from data and objectives, not size: adding singing data to
an 872 M model moved WER 19.39→15.78 and SS-CMOS −0.65→−0.17, and GRPO post-training did the rest
[S1]. At under 1 B parameters the model is nowhere near a capacity wall, so scaling up is not the
obvious lever. What the 128 GB box does buy is N=16 conversions re-ranked by a WavLM speaker-
similarity judge (the G2 capability), which is a reliable but modest lift — comparable to the
−12 % relative WER measured for best-of-N in TTS [S8]. Score 3, not 4, because VC errors are
usually systematic (timbre leakage, prosody flattening) rather than sampling noise.

### 5. Delta vs prior pass — **unchanged**

Vevo2 remains the correct primary; it is still the newest voice-conversion model in Amphion as of
18 September 2026, with nothing shipped after 25 March 2026 in that line [S3]. The prior pass's
licence correction (GPL-3.0 is a Seed-VC problem, not a VC problem; Amphion, Chatterbox, kNN-VC and
FreeVC are all MIT) is right and still holds. The one thing I would add: the prior pass treated VC
as a first-class stage, but given D1's numbers, VC should be routed to only when a transcript is
unavailable.

### Sources
1. https://arxiv.org/html/2508.16332v3 — Vevo2: unified controllable speech and singing generation
2. https://www.arxiv.org/pdf/2508.16332 — Vevo2 (PDF)
3. https://github.com/open-mmlab/Amphion — Amphion changelog (Vevo2, 25 Mar 2026; MIT)
4. https://github.com/resemble-ai/chatterbox — Chatterbox / Chatterbox VC (MIT)
5. https://fish.audio/blog/fish-audio-open-sources-s2/ — Fish Audio S2
6. https://elevenlabs.io/blog/eleven-v3 — ElevenLabs model family incl. Voice Changer
7. https://www.cartesia.ai/blog/sonic-3.6 — Sonic-3.6
8. https://arxiv.org/abs/2607.08256 — Best-of-N TTS evaluation

---

## D7 Non-verbal vocalizations

### 1. DGX Spark pick

**Primary: pass-through splicing of the original vocalisation, gated by an audio tagger — not a
synthesiser.** The benchmark evidence has hardened since the prior pass: on STEB, *every* evaluated
system — cascaded, end-to-end and speech-LLM alike — scores at or below **2.3 / 5 on non-verbal
preservation**, the worst-performing dimension in the whole benchmark [S1]. Detect laughs, sighs,
gasps and breaths on the A1 dialogue stem with **CED (Xiaomi) or BEATs (Microsoft)** at a 1 s hop
(both well under 1 GB, milliseconds per clip on a Spark), and splice the source audio through
rather than regenerate it.

**Synthesis fallback, where splicing is impossible: Step-Audio-EditX (StepFun), 3B, Apache-2.0,
12 GB min (16 GB recommended).** Its paralinguistic tag set is the richest open one — breathing,
laughter, surprise-oh/ah/wa, confirmation-en, uhm, sigh, question-ei, dissatisfaction-hnn, and from
the **29 January 2026** update exhale, snort, inhale, chuckle, clears-throat, giggle — and it edits
an existing take rather than regenerating the line [S2].

**Alternates.** (a) **Higgs Audio v3 TTS 4B** — 20+ inline tokens for emotion, style and sound
effects mid-utterance, with an EmergentTTS **paralinguistics win-rate of 68.57 %**, the best
measured figure I found for this specific behaviour [S3]. (b) **MOSS-SoundEffect-v2.0, 1.3B DiT,
Apache-2.0** (26 May 2026) — flow-matching, 48 kHz, up to 30 s, for effects the splice cannot cover
[S4].

### 2. Unconstrained pick

**Primary: Fish Audio S2 4.4B** for tag-driven synthesis — **EmergentTTS-Eval 81.88 % win rate**
and free-form textual direction rather than a closed tag vocabulary, so `[laughs nervously]` and
`[sharp intake of breath]` are expressible without a tag for each [S5].
**Margin over the Spark pick: small and, more importantly, irrelevant** — the splice still beats
every synthesiser on the only benchmark that measures the task, so more compute changes the
fallback quality, not the default action.
**Alternates.** (a) Higgs Audio v3 4B. (b) An NVSpeech-style paralinguistic-aware ASR over
SenseVoice to emit inline event tokens, so the router knows *which* vocalisation to splice.

### 3. API pick

**Primary: Qwen-Audio-3.0-TTS Plus** — **86 inline tags** covering laughter, breathing, whispers
and angry delivery plus free-style natural-language control, the largest hosted tag vocabulary I
could verify; Elo 1259 on the AA Speech Arena [S6,S7].
**Alternates.** (a) **ElevenLabs Eleven v3 audio tags**, GA 2 February 2026, 70+ languages [S8].
(b) **Cartesia Sonic 3.6**, which supports inserted non-verbal expressions such as laughter and
infers them from transcript subtext [S9].

### 4. Compute sensitivity — **2**

The ceiling is set by a capability gap, not by compute. No system on STEB exceeds 2.3/5 on
non-verbal preservation regardless of size — 8B speech LLMs fail here just as 0.5B TTS models do
[S1] — and the winning strategy (splice the original) costs essentially nothing. Best-of-N helps
little because the failure is *categorical*: the model omits the vocalisation or renders it as a
word, and re-sampling mostly produces more of the same omission rather than a better take. The
128 GB box is better spent on the *detector* side (running CED and BEATs and a paralinguistic ASR
in ensemble to decide what to splice) than on the synthesiser.

### 5. Delta vs prior pass — **unchanged in strategy, updated in evidence**

Splicing-first still stands and is now better supported: STEB gives a hard number (≤2.3/5 for every
system) where the prior pass had only a qualitative claim [S1]. Two updates: **Step-Audio-EditX
expanded its paralinguistic tags on 29 January 2026** (the prior pass ranked the November 2025
release), and two models with *measured* paralinguistic performance have appeared —
**Higgs Audio v3 (68.57 % EmergentTTS paralinguistics win-rate, 4 June 2026)** and **Fish Audio S2
(81.88 % overall EmergentTTS win-rate, 9 March 2026)** — neither of which was in the prior
shortlist. Drop Orpheus/Dia from the list; both are superseded.

### Sources
1. https://arxiv.org/pdf/2606.25529 — STEB: non-verbal preservation ≤2.3/5 across all systems
2. https://github.com/stepfun-ai/Step-Audio-EditX — paralinguistic tag set, 29 Jan 2026 update
3. https://www.boson.ai/blog/higgs-audio-v3-tts — Higgs TTS 3, EmergentTTS paralinguistics 68.57 %
4. https://github.com/OpenMOSS/MOSS-TTS — MOSS-SoundEffect-v2.0 (26 May 2026)
5. https://fish.audio/blog/fish-audio-open-sources-s2/ — Fish Audio S2, EmergentTTS 81.88 %
6. https://openrouter.ai/qwen/qwen-audio-3.0-tts-plus — Qwen-Audio-3.0-TTS Plus, 86 inline tags
7. https://artificialanalysis.ai/text-to-speech/leaderboard — AA Speech Arena
8. https://elevenlabs.io/blog/eleven-v3 — Eleven v3 audio tags
9. https://www.cartesia.ai/blog/sonic-3.6 — Sonic-3.6 non-verbals

---

## D8 Direct speech-to-speech translation (evaluate only)

The recommendation is unchanged: **do not adopt; evaluate as a comparator.** The three questions
are answered for the evaluation harness, not for production.

### 1. DGX Spark pick

**Primary: UniSS (ICLR 2026) as the comparator system to run locally.** Single-stage, expressive
S2ST that reuses a text LLM backbone with speech semantic and style modelling, so it slots into
existing LLM inference stacks rather than requiring the nested-Transformer, train-from-scratch
architecture Hibiki needs [S1,S2]. It reports beating both end-to-end and cascaded systems on
translation fidelity, voice preservation, duration consistency and speech quality on its own
benchmark, and on STEB it is the **only** system whose explicit duration control pushes Speech
Length Consistency above 0.9 at a 20 % tolerance [S3]. Sized like a small speech LLM, it fits
comfortably in unified memory; the aarch64 caveat is the usual one — no FA3, build FA2 from source
[S4].

**Alternates.** (a) **Hibiki (Kyutai)** — multi-stream simultaneous S2ST, the right comparator for
latency rather than quality, and useful because it is architecturally unlike everything else in the
pipeline [S2]. (b) **Step-Audio 2**, a speech LLM comparator that STEB evaluates directly [S3].

### 2. Unconstrained pick

**Primary: the speech-LLM class — Seed-Live 2.0 (ByteDance) and Step-Audio 2.** These are where
extra compute visibly shows: on STEB, cascaded systems produce "neutral, TTS-like speech" scoring
around **1.7 / 5 on emotion**, while end-to-end and speech-LLM systems reach **3.1–3.8 / 5, best
3.82** [S3]. **Margin over the Spark pick: meaningful on expressiveness, and irrelevant to the
decision** — the same benchmark shows cascades keeping their BLEU/COMET fidelity advantage, and
**no** system clears 2.3/5 on non-verbal preservation, so none of them is ready to replace a
per-line-editable cascade.
**Alternates.** (a) SeamlessExpressive (Meta) as the established end-to-end baseline. (b) A
three-stage cascade with a voice-preserving TTS as the control condition — which is what OpenDub
already is.

### 3. API pick

I could **not verify** a generally-available hosted direct-S2ST API that beats a cascade on
fidelity; every mature product in this space (including ElevenLabs Dubbing) is itself a cascade with
voice preservation bolted on [S5]. For evaluation, the closest hosted comparators are **Seed-Live
2.0** and **Step-Audio 2**, both of which STEB treats as speech LLMs rather than dedicated S2ST
systems [S3]. **Primary: ElevenLabs Dubbing API** as the productised control (translate finished
video with voice preservation and one-click lip-sync chaining), with **Seed-Live 2.0** and
**Step-Audio 2** as the direct-S2ST alternates. State plainly in the evaluation report that none of
these is a drop-in replacement for the pipeline.

### 4. Compute sensitivity — **4**

S2ST is the one Phase D category where model scale clearly moves the needle: the speech-LLM systems
roughly **double** the cascade's emotion score (3.1–3.8 vs ~1.7 out of 5) and that gap tracks
backbone size and multimodal pre-training, not architecture tricks [S3]. But the score is 4 rather
than 5 because more compute does not flip the adopt/don't-adopt decision — it does not close the
fidelity gap, it does not fix non-verbals (≤2.3/5 everywhere), and it does not create the per-line
edit point that a dubbing tool needs. A rack of B200s buys a better comparator, not a better product.

### 5. Delta vs prior pass — **unchanged**

"Evaluate, do not adopt" still stands, and the prior pass's plan to run **UniSS and Hibiki** as the
two comparison systems is still the right one. What I would refine: the prior pass cited "the
cascade beats the best end-to-end system by 13 BLEU on STEB" — I could not reproduce that exact
figure from the paper, which reports that many cascaded *and* end-to-end systems achieve strong
BLEU/COMET fidelity, and locates the real differences in emotion (1.7 vs up to 3.82 / 5), non-verbal
preservation (≤2.3 / 5 for everyone) and duration alignment (only explicit duration control clears
0.9 SLC at 20 % tolerance) [S3]. Re-derive that BLEU claim before quoting it. Also add **Step-Audio
2** and **Seed-Live 2.0** to the comparator set — STEB evaluates them and the prior pass did not
list them.

### Sources
1. https://arxiv.org/abs/2509.21144 — UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice
2. https://proceedings.iclr.cc/paper_files/paper/2026/file/c29fc04c339cd91b89b70dfe4b6d0fc0-Paper-Conference.pdf — UniSS (ICLR 2026), incl. Hibiki comparison
3. https://arxiv.org/pdf/2606.25529 — STEB: S2ST Expressiveness Benchmark (systems, emotion, NV, duration)
4. https://forums.developer.nvidia.com/t/running-vllm-omni-for-qwen3-tts-voice-design-voice-clone-on-dgx-spark/361255 — aarch64 / FA3 gotchas on DGX Spark
5. https://elevenlabs.io/blog/eleven-v3 — ElevenLabs dubbing / model family

---

# Phase E · Audio post-production

## E1 Timing fit and pause mapping

### 1. DGX Spark pick

**Primary: pause-structure prosodic alignment (DP over A4 word timings) driving a non-uniform time
map, with Signalsmith Stretch as the residual stretcher.** Parameters: 0 — this is dynamic
programming plus a phase-locked STFT stretcher. Precision: n/a (float64 DP, float32 audio).
Memory: ~0.1 GB. Speed: thousands of × realtime on the Arm cores alone; no GPU is touched.
Licence: MIT (Signalsmith Stretch, C++11 library with Python/Rust/WASM bindings; a new
cross-platform plugin front end shipped September 2026). Gotchas: none — it is portable C++,
builds natively on aarch64, and there is no CUDA, no wheel, and no kernel to worry about.

This stays the primary because the failure mode that is actually audible in OpenDub output is
*uniformly sped-up delivery* — the source's pause structure gets flattened — and that is a
segmentation problem, not a stretching problem. Every compute-hungry alternative below attacks the
smaller half of the problem.

**Alternate 1: LLM isochronic paraphrase + phonetic synchronisation (the PS-TTS / PS-Comet method,
ICPR 2026, arXiv 2604.09111).** Rewrite the target line so the take is the right length instead of
stretching it: paraphrase with an LLM under a duration constraint, then align target vowels to
source vowels by DTW on a vowel-distance metric; PS-Comet additionally scores semantic similarity
so the paraphrase does not drift. Reported to beat TTS-without-PS on objective metrics and to beat
*human voice actors* on Korean↔English lip-sync. On a Spark this is free: reuse whatever 27–32B
instruction model C2 already has resident (BF16, ~55–65 GB) and sample 16–32 paraphrases per line —
lines are short, so best-of-N costs seconds. This is the one place in E1 where the box helps.

**Alternate 2: Rubber Band Library R3 ("Finer") engine, formant-preserved**, for the residual
stretches beyond about ±15% where Signalsmith starts to smear transients. GPL / commercial dual
licence. Also 0 params, CPU-only, no aarch64 issues.

### 2. Unconstrained pick

**Primary: duration-controlled regeneration instead of stretching — Dub-S2ST (KAIST, EMNLP 2025
Findings, arXiv 2505.20899).** A discrete-diffusion speech-to-unit translation model with *explicit
duration control*, followed by conditional-flow-matching synthesis on the source speaker's identity,
plus a unit-level speed-adaptation mechanism that holds the speaking rate close to the source
without going through text. Margin over the Spark pick: it removes time-stretch artefacts entirely
rather than making the aligner better — a categorical rather than an incremental win — but it is
*not* meaningfully out of reach of a Spark. Dub-S2ST is a normal-sized speech model; the reason to
put it here is that the best version of this strategy is best-of-N (generate 32 takes at different
rates, keep the one needing <3% residual stretch), and unlimited compute makes N bigger. With 128 GB
of unified memory the Spark can already run a respectable N.

**Alternate 1: HoliDubber (arXiv 2606.09098, June 2026)** — patch-based autoregressive diffusion
transformer that jointly synthesises speech *and* sound effects/ambience from one text prompt,
video-conditioned via cross-attention on the speaker's articulation, evaluated on its own
HoliDub-Bench (1,000 clips) against AlignDiT, VoiceCraft-Dub and FunCineForge. Caveat: the project
page says "Code (Coming Soon)" and the paper is listed as under review — **weights are not
available and the licence is unstated**, so this is a direction, not a dependency.

**Alternate 2: DubWise (Sony AI, Interspeech 2024, arXiv 2406.08802)** — video-guided duration
control inside a GPT-based multimodal TTS, using a duration-controller network over video tokens.
Older, but the cleanest published statement of "let the generator hit the duration" for both
non-parallel and cross-lingual cases.

### 3. API pick

**Primary: ElevenLabs Dubbing v2.** Verified in the vendor docs on 18 September 2026: 90+ languages
with dialect variants (en-AU/CA/GB/US, es-AR/CL/ES/MX, fr-CA/FR, pt-BR/PT, ar-EG, zh-TW), Dubbing
Studio gives transcript editing, speaker reassignment and per-clip regeneration — which is the
timing-fit control surface — but **transcript editing and audio regeneration via the API are
Enterprise-plan only**; creating and downloading dubs is on all plans. Free-tier dubs are
watermarked, paid-tier are not. The docs do not expose an explicit per-clip duration/speed
parameter, so timing fit is achieved by regeneration, not by a stretch knob.

**Alternate 1: a frontier hosted instruction LLM doing the isochronic paraphrase** (the PS-TTS step),
scored against a local duration predictor. I did not verify a specific vendor model version in this
pass and will not invent one — treat the choice as inherited from whatever C2 picks.
**Alternate 2: HeyGen Video Translate**, which performs the same regeneration-based fit behind a
closed API; named in the prior pass under F1 and not re-verified here.

### 4. Compute sensitivity — **2**

The aligner and the stretcher are solved at essentially zero compute and are the parts that fix the
audible defect; the only compute-sensitive component is the paraphrase/regeneration escalation, and
its gain is bounded by how much slack the line has. PS-TTS reports its LLM-paraphrase plus phonetic
sync beating both plain TTS and human voice actors on Ko↔En lip-sync metrics, but publishes no
scaling curve against LLM size, and nobody has shown that a 70B paraphraser beats a 30B one at
"say this in 2.4 seconds". Best-of-N re-ranking is the only lever with headroom, and the Spark's
128 GB already supports a large N for short lines.

### 5. Delta vs prior pass

**No change.** The 3 September primary (pause-structure aligner + Signalsmith Stretch as the
MIT-licensed residual stretcher, Rubber Band R3 behind it) still stands as the Spark pick, and
**nothing relevant was published after 3 September 2026** — the newest item found, PS-TTS
(arXiv 2604.09111, submitted 10 April 2026, accepted to ICPR 2026), predates the prior pass and
simply puts a citation under what that pass already called "duration-controlled re-generation".
One refinement: the prior pass listed regeneration as a *strong* alternative; PS-TTS's result that
the paraphrase route beats professional voice actors on lip-sync metrics argues for promoting it to
the standing escalation for any line needing more than ~10% stretch.

### Sources

1. PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing (ICPR 2026, 10 Apr 2026) — https://arxiv.org/abs/2604.09111
2. Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing (EMNLP 2025 Findings) — https://arxiv.org/abs/2505.20899
3. HoliDubber project page (code "coming soon", under review, Jun 2026) — https://holidubber.github.io/
4. HoliDubber (arXiv 2606.09098) — https://arxiv.org/abs/2606.09098
5. DubWise (Sony AI, Interspeech 2024) — https://arxiv.org/abs/2406.08802
6. Signalsmith Stretch (MIT, C++ library) — https://github.com/Signalsmith-Audio/signalsmith-stretch
7. Signalsmith Stretch design notes — https://signalsmith-audio.co.uk/code/stretch/
8. ElevenLabs Dubbing capability docs (checked 18 Sep 2026) — https://elevenlabs.io/docs/capabilities/dubbing
9. Prosodic alignment for off-screen automatic dubbing (Amazon) — https://arxiv.org/pdf/2204.02530

---

## E2 Bandwidth extension and restoration

**Direct answer to the question posed: no, bigger generative restorers do not beat small ones at
bandwidth extension — but generative does beat deterministic, decisively, at every size.** The
June 2026 survey re-evaluated ten representative methods under one protocol on VCTK (RTX A6000,
speaker-disjoint split). At 16 kHz→48 kHz: the best *discriminative* model, AFiLM at 134.72M
params, scores LSD 2.195 / ViSQOL 2.075; AP-BWE, a **29.76M GAN**, scores LSD 0.749 /
LSD-HF 0.902 / **ViSQOL 3.367**; FLowHigh, a 48.84M flow model, scores LSD 0.901 /
**ViSQOL 3.556**. A 30M generative model beats a 135M discriminative one by **+1.29 ViSQOL and
−1.45 LSD** while being 4.5× smaller. The survey states it plainly: *"model size is not a reliable
predictor of performance, as several compact generative models outperform larger discriminative
counterparts."* And more sampling does not help either — FLowHigh's ViSQOL is **flat from NFE 2 to
NFE 50** while its RTF rises from 0.0389 to 0.1596, UniverSR saturates after NFE 10, and NU-Wave2
actually gets *worse* at NFE 100.

Where bigger models do win is *universal* restoration — simultaneous noise, reverb, codec artefacts,
clipping and bandwidth loss with unknown degradation parameters across many languages. There the
545.7M UniPASE beats the predictive BSRNN-FAN baseline on perceptual metrics (DNSMOS 3.26 vs 3.01,
NISQA 4.18 vs 3.41, UTMOS 2.97 vs 2.40) **but loses on intelligibility and identity: CER 12.90% vs
11.08%, SpkSim 0.81 vs 0.85.** For a voice-cloned dub that trade is bad by default: hallucinated
phonemes and drifted timbre are correctness bugs, not stylistic choices. Hence the gating
recommendation below.

### 1. DGX Spark pick

**Primary: AP-BWE, 48 kHz checkpoint (Lu et al., USTC/NELSLIP, IEEE/ACM TASLP 2024).**
Params ~29.76M. Precision FP32 (FP16 fine; nothing here needs BF16). Memory ~0.5 GB resident.
Speed: 292.3× realtime on an RTX 4090 and **18.1× realtime on a single CPU** — on a GB10 expect
comfortably >100× realtime, and the CPU fallback alone is fast enough to ship. Licence **MIT for
both code and weights** (`weights_LICENSE.txt` alongside the Google Drive checkpoints; 16 kHz and
48 kHz targets published). Why: it is the highest-fidelity option at this size, it is
bandwidth-agnostic (trained over cutoffs {4, 6, 8, 12} kHz), it is deterministic enough that it
cannot invent words, and it wins the 16k→48k LSD/LSD-HF columns outright in the survey's unified
re-evaluation. Gotchas: none material — a dual-stream CNN over amplitude and phase spectra in plain
PyTorch, no custom kernels, no flash-attn, no ONNX needed. Requires Python ≥3.9.

**Alternate 1: FLowHigh (arXiv 2501.04926, ICASSP 2025), 48.84M, single-step flow matching.**
Highest ViSQOL of everything tested in the survey at both 8k→16k (4.652) and 16k→48k (3.556), at
RTF 0.0389 when run at NFE 2. Run it at NFE 2 — the ViSQOL curve is flat to NFE 50, so extra steps
are pure waste. Worth A/B-ing against AP-BWE on real dub takes: AP-BWE wins spectral fidelity,
FLowHigh wins perceptual score, and which matters depends on whether the artefact you hear is
missing air or sibilance.

**Alternate 2: UniPASE (Xiaobin Rong et al., accepted IEEE TASLP, arXiv 2604.14606, 16 Apr 2026),
545.7M total** — DeWavLM-Omni 315.44M (WavLM distilled for de-distortion) + Adapter 113.73M +
Vocoder 113.73M + PostNet 2.77M, 79.2 GMACs/s, CC BY 4.0, checkpoints on Hugging Face. At BF16 that
is ~1.1 GB — trivial on a Spark. Backbone of the system that took **1st place in the URGENT 2026
objective evaluation** (combined with TF-GridNet; 29 valid entries from 80+ registrations). Use it
as a *conditional* stage that only fires when DNSMOS on the raw take is below threshold, not as a
default pass — see the CER/SpkSim regression above. Gotchas: WavLM uses standard attention, so
PyTorch SDPA is fine on Blackwell; no flash-attn dependency.

### 2. Unconstrained pick

**Primary: the same models.** This is the honest answer and it is worth stating loudly — there is
no rack-scale restoration model that beats AP-BWE at 24k→48k extension, and the compute-hungry
options in this space target a different problem.

The largest *open-weights* universal restorer is **UniPASE (545.7M, CC BY 4.0)** or **Sidon
(ICASSP 2026, arXiv 2509.17052)**, an explicitly open alternative to Miipher built on a w2v-BERT 2.0
feature predictor (pretrained on 4.5M hours / 143 languages) plus a vocoder; Sidon runs **up to 500×
realtime on one GPU** and reports **better CER and DNSMOS than Miipher-2, comparable SpkSim, and
slightly lower NISQA**. The closed frontier is **Miipher-2 (Google DeepMind, arXiv 2505.04457,
IEEE 2025)** — a frozen 2B USM covering 300+ languages, parallel adapters, WaveFit vocoder,
RTF 0.0078 — which is not available as weights. The community reimplementation
**Open-Miipher-2 (MIT)** substitutes the Gemma-3 USM encoder up to layer 6 (0.6B, 12 Conformer
layers) with a ~19M-parameter parallel adapter, but **publishes no pretrained checkpoints and no
benchmark numbers**, so it is a training harness, not a drop-in.

Margin over the Spark pick: **essentially zero for this pipeline.** Sidon and UniPASE both fit in
around 1–2 GB and run hundreds of × realtime; a B200 rack changes throughput, not quality.

**Alternate: CogSR (arXiv 2512.16304)** — flow matching conditioned on a Large Audio-Language Model
doing chain-of-thought reasoning as a semantic anchor, with explicit acoustic priors to hold speaker
identity. This is the *only* member of the category that genuinely wants a big box, and it is built
for severe degradation (archival, forensic, very low sample rates) where ordinary generative models
hallucinate words by probability rather than meaning. It is the wrong tool for cleaning up a 24 kHz
TTS take, but it is the right answer if OpenDub ever ingests degraded archival source audio.

### 3. API pick

There is **no first-class hosted bandwidth-extension API**; every hosted product in this space is a
denoiser/dereverberator, and all of them risk altering a cloned timbre.

**Primary: Adobe Podcast Enhance Speech.** The March 2026 "Advanced Source Separation" update added
a music control and downloadable isolated speech/background stems, which makes it the most useful of
the hosted options for a dubbing pipeline (you get stems back, not just a mixed result).
**Alternate 1: ElevenLabs Voice Isolator** — available via the ElevenLabs API, documented as *not*
supporting real-time streaming. **Alternate 2: Dolby.io Media Enhance API** — speech isolation,
noise reduction, sibilance/plosive reduction, dynamic EQ and tone shaping; note the platform is
being rebranded/migrated to Dolby OptiView and existing customers are told to contact Dolby for
migration, so verify availability before building on it.

### 4. Compute sensitivity — **2**

Bandwidth extension is close to solved at ~30M parameters and the scaling evidence points the other
way. Concrete: AP-BWE (29.76M) beats AFiLM (134.72M) by **+1.29 ViSQOL / −1.45 LSD** at 16k→48k;
FLowHigh's ViSQOL is **unchanged from NFE 2 to NFE 50** while RTF quadruples; NU-Wave2 degrades at
NFE 100. The one compute-sensitive axis is universal restoration under unknown degradation, and
there the 545.7M UniPASE buys **+0.25 DNSMOS and +0.77 NISQA** over a predictive baseline at a cost
of **+1.8 points of CER and −0.04 SpkSim** — a trade a dubbing pipeline should usually decline.

### 5. Delta vs prior pass

**Prior primary (AP-BWE) still stands** — and it now stands on a hard number rather than a vibe: the
June 2026 unified re-evaluation on VCTK is the first apples-to-apples comparison that puts AP-BWE
at the top of the 16k→48k LSD column against discriminative, GAN, diffusion and flow baselines at
matched training/eval conditions.

Two changes to the *shortlist* the prior pass proposed:
- **UniPASE is now confirmed, not speculative.** It is published (TASLP, arXiv 2604.14606), has
  checkpoints, and was the backbone of the URGENT 2026 objective winner. The prior pass said
  "shortlist UniPASE against Resemble Enhance" — that contest is over; take UniPASE and drop
  Resemble Enhance from the shortlist.
- **Demote AudioSR and FlashSR.** The literature now documents their specific failure modes:
  both "exhibit high variance" and "often produce audio lacking sufficient high-frequency content",
  AudioSR "tends to produce outputs with excessive sibilance", and FlashSR "often fails to generate
  detailed harmonic structures". For dialogue that is exactly wrong. FlashSR is still the right
  pick if you need one-step diffusion speed (~22× faster than AudioSR at 100 NFEs, comparable or
  better metrics), but AP-BWE is faster *and* better here.
- **Add Sidon** as the open Miipher-class option the prior pass was missing; it beats Miipher-2 on
  CER and DNSMOS and is actually open.

**Nothing found published after 3 September 2026** in this category; the newest items (the survey
at 2605.16681 and UniPASE at 2604.14606) are from April–June 2026.

### Sources

1. A Survey of Advancing Audio Super-Resolution and Bandwidth Extension from Discriminative to Generative Models (Jun 2026; Table 6, Figure 8, unified VCTK re-evaluation) — https://arxiv.org/pdf/2605.16681
2. AP-BWE repo (MIT code + weights, 16 kHz and 48 kHz checkpoints) — https://github.com/yxlu-0102/AP-BWE
3. AP-BWE paper (TASLP 2024) — https://arxiv.org/abs/2401.06387
4. UniPASE (IEEE TASLP, arXiv 2604.14606, 16 Apr 2026; 545.7M, CC BY 4.0) — https://arxiv.org/abs/2604.14606
5. UniPASE code/checkpoints — https://github.com/Xiaobin-Rong/unipase
6. FLowHigh: single-step flow matching audio SR (arXiv 2501.04926, ICASSP 2025) — https://arxiv.org/abs/2501.04926
7. Sidon: fast, robust, open-source multilingual speech restoration (ICASSP 2026) — https://arxiv.org/abs/2509.17052
8. Miipher-2 (Google DeepMind, arXiv 2505.04457) — https://arxiv.org/abs/2505.04457
9. Open-Miipher-2 (MIT; no released checkpoints) — https://github.com/yukara-ikemiya/Open-Miipher-2
10. ICASSP 2026 URGENT Speech Enhancement Challenge overview — https://arxiv.org/abs/2601.13531
11. FlashSR: one-step versatile audio super-resolution via diffusion distillation — https://arxiv.org/abs/2501.10807
12. SAGA-SR (documents AudioSR/FlashSR failure modes) — https://arxiv.org/html/2509.24924
13. CogSR: semantic-aware speech SR via chain-of-thought guided flow matching — https://arxiv.org/pdf/2512.16304
14. Dolby.io Media Enhance API guide — https://docs.dolby.io/media-processing/docs/enhance-api-guide

---

## E3 Acoustic scene matching

### 1. DGX Spark pick

**Primary: learned acoustic embeddings + feedback delay network parameter matching (Götz, Dal Santo,
Schlecht, Välimäki, Habets — International Audio Laboratories Erlangen / Fraunhofer IIS + Aalto;
arXiv 2510.23158, CC BY 4.0).** A variational autoencoder extracts an embedding from reverberant
speech, and a *differentiable* FDN is optimised so its embedding matches the target; the FDN is then
rendered against the dry dub take. Params: a small VAE (single-digit millions) plus the FDN's own
coefficient set — call it well under 50M. Precision FP32. Memory ~0.5 GB. Speed: per-scene
optimisation is seconds on GPU; rendering is realtime DSP. Licence: paper CC BY 4.0, reference
implementation via the FLAMO differentiable-audio framework
(https://github.com/gdalsanto/flamo). Gotchas: plain PyTorch autograd over DSP primitives —
no custom kernels, no aarch64 concerns at all.

Why this stays first for OpenDub specifically: the output is a **parameter set** — per-band T60, DRR,
C50, spectral delta, FDN coefficients — which can be surfaced in the editor as a per-scene profile
that an operator inspects and overrides. A black-box neural re-reverberator cannot be argued with at
3 a.m.

**Alternate 1: Rec-RIR (arXiv 2509.15628)** — blind monaural RIR *identification* via reverberant
spectrum reconstruction, using a multi-task DNN that sequentially removes noise and reverberation
while fusing reverberant and clean speech embeddings. Gives you an actual impulse response to
convolve, which is the cleanest A/B against the parametric FDN and catches early reflections the FDN
cannot render.

**Alternate 2: Gencho (Lin, Su, Anand, Jin, Kim, Smaragdis — Adobe Research / UIUC, ICASSP 2026,
arXiv 2602.09233, submitted 9 Feb 2026).** Diffusion-transformer model that predicts complex
spectrogram RIRs from reverberant speech, with a structure-aware encoder that exploits the
separation between early and late reflections, plus **text-conditioned** RIR generation. Explicitly
framed for "acoustic matching" and designed to slot into standard speech pipelines. Reported to
produce "richer generated RIRs than non-generative baselines while maintaining strong performance in
standard RIR metrics." This is the generative escalation that actually uses the box. Caveat:
**only audio demos are published (linjac.github.io/Gencho/); code and licence are not stated** —
verify before depending on it.

### 2. Unconstrained pick

**Primary: Gencho** as the matcher, escalated to many diffusion samples per scene and re-ranked on a
reverberation-metric agreement score. Diffusion is compute-bound rather than memory-bound, so
unlimited compute buys you a genuinely larger N and a genuinely better draw — this is the one place
in Phase E where the Spark is a real (if mild) limiter.

**Alternate 1: ReverbMiipher (Nakata, Koizumi, Karita, Scheibler, Ishikawa, Guevara-Rukoz, Zen,
Bacchiani — Google, arXiv 2505.05077, May 2025).** It solves the *other* half of the problem
properly: a ReverbEncoder extracts a reverb feature vector from the noisy input and conditions the
vocoder, so the system removes noise while *retaining* the original reverberation — and the learned
reverb representation supports "interpolation between features, replacement with features from other
utterances, or sampling from a latent space." That is literally "take shot X's room and put it on
dub take Y", and the paper reports it beating the conventional two-stage approach (dereverb, then
convolve a simulated RIR) in both objective and subjective evaluation. **Not open-sourced; no code
or checkpoints announced**, so it is an argument for the approach, not a dependency.

**Alternate 2: BUDDy** (blind unsupervised dereverberation with an unconditional diffusion prior over
anechoic speech, jointly estimating a parametric subband filter) run with a large sampling budget,
and **RIRFlow** (JASA 2026, training-free Bayesian flow matching for RIR inverse problems with an
analytic Wiener denoiser and a physically interpretable exponentially-decaying-variance prior).

Margin over the Spark pick: real but not dramatic, and concentrated in hard cases — strongly
coloured or non-exponential decays, and early-reflection structure, which an FDN fundamentally
cannot reproduce. For ordinary interior dialogue the parametric matcher is close to the ceiling.

### 3. API pick

**There is no hosted acoustic-matching API.** This is worth stating plainly rather than dressing up:
the best closed products are GUI plugins, and wiring them into OpenDub means an offline render step,
not a service call.

**Primary: Accentize Chameleon (and Chameleon Surround).** Reviewers comparing it against iZotope's
offering conclude that "reverb matching is where Dialogue Match falls short, with Accentize Chameleon
being notably more effective in this area." Accentize takes a modular per-task plugin approach.
**Alternate 1: iZotope RX 12 Advanced Dialogue Match** — RX 12 shipped 29 April 2026 with a rebuilt
Dialogue Isolate (improved realtime and offline de-noise/de-reverb for dialogue) plus Scene
Rebalance and Stems View; Dialogue Match learns reverb, EQ and ambience from a reference and applies
it to other recordings. **Alternate 2: none credible** — no cloud vendor exposes reverb transfer.

### 4. Compute sensitivity — **3**

The parametric route (VAE embedding + differentiable FDN) is effectively free and gets you most of
the way, which caps the score; but unlike E1/E4/E5 there is a documented quality gap that compute
closes — Gencho's diffusion transformer produces "richer generated RIRs than non-generative
baselines", and unsupervised diffusion dereverberation (BUDDy) improves with sampling budget. The
ceiling is set by the task definition rather than by the model: the target is "plausibly the same
room", and once a listener cannot tell, extra fidelity is invisible. Note this is the *opposite*
finding to E2, where extra sampling steps demonstrably bought nothing.

### 5. Delta vs prior pass

**Prior primary still stands.** The learned-embedding + FDN matcher remains the right Spark pick and
the right architecture for OpenDub's editor.

One promotion: **Gencho should move from absent to the standing generative escalation.** The prior
pass listed BUDDy, Rec-RIR, ST-ITO and ReverbMiipher but not Gencho, which is the first
diffusion-transformer RIR generator with acoustic matching *and* text conditioning as explicit
design goals, and it comes from the Adobe Research group that also produced the dialogue-restoration
line. Treat it as "trial after the FDN baseline is working", and confirm its licence first.

**Nothing found published after 3 September 2026** in this category. One correction to note: the
prior pass dated the FDN matching paper to "ICASSP 2026" — the arXiv preprint (2510.23158v1) is
dated **28 October 2025**, which is consistent with an ICASSP 2026 submission but the paper itself
is a 2025 preprint.

### Sources

1. Matching Reverberant Speech Through Learned Acoustic Embeddings and Feedback Delay Networks (arXiv 2510.23158, 28 Oct 2025, CC BY 4.0) — https://arxiv.org/pdf/2510.23158
2. FLAMO differentiable audio framework (reference implementation) — https://github.com/gdalsanto/flamo
3. Gencho: RIR Generation from Reverberant Speech and Text via Diffusion Transformers (ICASSP 2026, arXiv 2602.09233, 9 Feb 2026) — https://arxiv.org/abs/2602.09233
4. Gencho demo page (code/licence not stated) — https://linjac.github.io/Gencho/
5. Rec-RIR: monaural blind RIR identification via reverberant spectrum reconstruction — https://arxiv.org/html/2509.15628
6. ReverbMiipher: generative speech restoration with reverberation controllability (Google, arXiv 2505.05077) — https://arxiv.org/abs/2505.05077
7. Unsupervised blind joint dereverberation and room acoustics estimation with diffusion models (BUDDy) — https://arxiv.org/abs/2408.07472
8. Solving RIR inverse problems using flow matching with an analytic Wiener denoiser (JASA 2026) — https://pubs.aip.org/asa/jasa/article/159/6/5527/3395888/Solving-room-impulse-response-inverse-problems
9. iZotope releases RX 12 (29 Apr 2026) — https://sonicstate.com/news/2026/04/29/izotope-releases-rx-12/
10. Best iZotope RX alternatives 2026 (Chameleon vs Dialogue Match on reverb matching) — https://www.production-expert.com/production-expert-1/best-izotope-rx-alternatives-for-audio-restoration-and-dialogue-cleanup-in-2026

---

## E4 Dialogue levelling and loudness

**There is no model here, and pretending otherwise would be the wrong recommendation.** The
deliverable is conformance to a published measurement algorithm. A correct two-pass implementation
of ITU-R BS.1770 is *exact*; no amount of compute improves exact. This section is therefore mostly
an engineering specification, which is the honest output.

### 1. DGX Spark pick

**Primary: BS.1770 two-pass measurement (libebur128 or pyloudnorm, both MIT) + a static delivery-preset
chain + dialogue-gated measurement driven by OpenDub's own A4 word timings.** Params: 0. Precision:
float64 measurement, float32 audio. Memory: <0.05 GB. Speed: thousands of × realtime on the Arm
cores; **the GPU is never touched, so there is no aarch64, CUDA, SM 12.1, wheel or NIM question at
all.** Licence: MIT (libebur128, Jan Kokemüller; pyloudnorm, Christian Steinmetz).

Standards to target, verified current as of 18 September 2026:
- **ITU-R BS.1770-5 (November 2023)** is still the current revision; **there is no BS.1770-6.**
  400 ms overlapping gating blocks, two-stage absolute-then-relative gate (the relative gate has
  been in the standard since BS.1770-2, March 2011).
- **EBU R 128 v5.0 (November 2023)**, with four supplements the original 2010 document did not
  foresee: **s1** short-form (adverts/promos), **s2** "Loudness in Streaming" (Nov 2023), **s3**
  radio, **s4** cinematic content. Target −23 LUFS / −1 dBTP.
- **ATSC A/85**: −24 LKFS / −2 dBTP.
- **Netflix**: −27 LKFS ±2 LU **dialogue-gated**, −2 dBTP. Note the spec cites BS.1770-1 rather than
  -4 deliberately: the Dolby dialogue gate already gates the audio, so the relative-level gate of
  the later revisions is not wanted on top of it. A dialogue gate needs ≥2 seconds of audio before
  it can start detecting.
- Web/streaming loudness-normalised delivery: −16 LUFS.

Because OpenDub already has word-level timings from A4, it can build a **VAD-free** dialogue gate —
gate on the known dialogue intervals rather than on a speech detector — which is strictly more
reliable than the detector-based implementations and costs nothing.

**Alternate 1: ST-ITO / DeepAFx-ST (Christian Steinmetz, QMUL C4DM with Adobe Research)** — a learned
parameter *search* over the existing effect chain rather than a replacement for it. A few million
parameters, seconds to run, and it keeps every decision inspectable as plugin settings.
**Alternate 2: Dialog+-style dialogue separation plus automatic remix (Fraunhofer IIS)** for the
dialogue-to-background ratio, reusing OpenDub's own A1 stems. Worth copying one specific finding:
perceptual quality of the separated speech improves with a slight constant spectral boost across
1–4 kHz on the dialogue component, **compensated by an equal cut in the background component**, and
a global plus a time-varying background attenuation can be combined.

### 2. Unconstrained pick

**Unchanged: the same DSP chain.** There is no large model that beats a correct BS.1770
implementation at hitting a loudness target, because the target is a specification rather than a
perceptual judgement, and nobody has published one that claims otherwise.

The only thing unlimited compute buys is **search**: enumerate or learn over many candidate chains
(ST-ITO-style) and score each with a perceptual metric, or run the 2025 JAES automatic-mixing
approach (Liu and Reiss) which combines iterative Harmony Search with integer optimisation over
level balancing, EQ, dynamic range compression and spatialisation, and reports competitive results
against professional sound-engineer mixes in objective and subjective listening tests — note that is
**classical optimisation, not a neural network**, so it is CPU work that parallelises trivially.

**Alternate: an audio-LLM as a mix critic** feeding the parameter search. Plausible and untested —
I found no published evidence that it beats the standard, so it is a research idea, not a pick.

### 3. API pick

**Primary: Dolby.io Media Enhance API.** The closest thing to a purpose-built hosted dialogue
leveller: speech leveling (identifies speech sections and applies time-varying gain so speech levels
converge — the exact fix for soft-spoken versus booming speakers), loudness correction, speech
isolation, noise reduction, dynamic EQ, tone shaping, sibilance and plosive reduction, hum and mouth
click reduction. **Caveat: Dolby.io is being rebranded/migrated to the Dolby OptiView platform and
existing customers are directed to contact Dolby for migration guidance** — confirm current status
before building on it.

**Alternate 1: Auphonic** — Adaptive Leveler (corrects level differences between speakers and
between music and speech, plus dynamic range compression to a balanced overall loudness), loudness
targets, noise and reverb reduction, multitrack processing, batch jobs, watch folders, a REST API
and a CLI. The most pipeline-friendly of the hosted options.
**Alternate 2: NUGEN Audio (LM-Correct / AMB)** — broadcast-grade offline loudness correction, but a
plugin/standalone product rather than a REST API.

### 4. Compute sensitivity — **1**

The lowest possible score, and deservedly. The measurement is a fixed algorithm defined in
BS.1770-5; a two-pass implementation is exact and a Spark is four orders of magnitude more compute
than it needs. The only genuinely subjective parameter in the whole category is the
dialogue-to-background ratio, which is one scalar a human sets once per show and then reuses. No
published work shows any model improving on standards conformance, because "improving on" is not a
meaningful operation on a conformance target.

### 5. Delta vs prior pass

**No change whatsoever.** The 3 September verdict — delivery presets, MIT-licensed two-pass
measurement rather than a single-pass filter, dialogue-gated measurement from OpenDub's own word
timings, and honesty that there is almost no model here — stands verbatim and is the correct call.

Nothing new since 3 September 2026, and nothing new since 2023 at the standards layer:
**BS.1770 is still at revision 5** (November 2023) and EBU R 128 is still at v5.0 with supplements
s1–s4. The one thing worth adding that the prior pass did not spell out is the *reason* Netflix
cites BS.1770-1: the dialogue gate replaces, rather than stacks with, the relative-level gate of
later revisions — implement it that way or the numbers will not match Dolby Media Meter.

### Sources

1. Recommendation ITU-R BS.1770-5 (11/2023), current revision — https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
2. EBU R 128 (v5.0, Nov 2023, supplements s1–s4) — https://en.wikipedia.org/wiki/EBU_R_128
3. Netflix dialogue-gated −27 LKFS spec, and why it cites BS.1770-1 — https://www.production-expert.com/home-page/2018/8/23/has-netflix-turned-the-clock-back-10-years-or-is-their-new-loudness-delivery-spec-a-stroke-of-genius
4. LUFS / loudness standards overview (2026) — https://www.production-expert.com/production-expert-1/loudness-everything-you-need-to-know
5. Dialog+ in Broadcasting: first field tests using deep-learning-based dialogue enhancement (Fraunhofer IIS) — https://arxiv.org/pdf/2112.09494
6. Speech loudness in broadcasting and streaming — https://arxiv.org/pdf/2405.17364
7. An Automatic Mixing Speech Enhancement System (Liu and Reiss, JAES 2025; Harmony Search + integer optimisation) — https://joshreiss.github.io/documents/2025/Liu2025JAESAutomaticMixingSpeech.pdf
8. Dolby.io Media Enhance API guide — https://docs.dolby.io/media-processing/docs/enhance-api-guide
9. Dolby.io / Dolby OptiView platform migration note — https://github.com/api-evangelist/dolby-io
10. Auphonic singletrack post-production algorithms (2026 docs) — https://auphonic.com/help/algorithms/singletrack.html

---

## E5 Watermarking and provenance

The user has deferred implementing this. The three questions are answered anyway, and there is one
finding since the prior pass that materially changes the recommendation.

**The finding: as of 27 August 2026, AudioSeal and SilentCipher are both breakable, training-free,
at good audio quality.** "How Fragile Is Your Watermark? Training-Free Structural Removal of Neural
Audio Watermarks" (arXiv 2608.16566) computes cheap structural probes that reveal which domain a
watermark occupies, then applies a single domain-matched attack. Result: **WavMark, SilentCipher and
audiowmark have their payloads erased outright, and AudioSeal's detection flag is removed, all at
PESQ ≥ 3.6** — i.e. the attacked audio still sounds fine. The probes also fingerprint *which* scheme
is in use with 84% accuracy, so an attacker does not even need to guess. The schemes that survived
every training-free attack are all **latent-domain**: **VoiceMark, WMCodec, AlignMark and AWARE**.

The regulatory position also moved: **EU AI Act Article 50 became enforceable on 2 August 2026** and
is live today. The Commission and the AI Board have confirmed the **Code of Practice on Transparency
of AI-Generated Content** as adequate for demonstrating compliance, and the guidance says explicitly
that **no single marking technique currently meets all four Article 50(2) requirements**
(effectiveness, interoperability, robustness, reliability) — providers are directed to layer
metadata, watermarking and provenance mechanisms. That is a regulator endorsing exactly the
belt-and-braces design the prior pass proposed.

### 1. DGX Spark pick

This is a CPU/tiny-model category; the Spark is irrelevant to it, which is itself the answer.

**Primary: C2PA Content Credentials via c2pa-rs / c2patool.** Params: 0 — this is signing, hashing
and manifest embedding. Memory: <0.3 GB. Speed: far faster than realtime; I/O bound. Licence:
**dual MIT and Apache-2.0**. Implements the **C2PA v2.4 specification** (2.4 published 21 April 2026;
2.3 was published January/February 2026) and supports hard bindings plus several common assertions.
Format coverage includes **WAV, MP3 and M4A** alongside MP4/MOV, so both the audio deliverable and
the muxed video can carry credentials. Note c2pa-rs is still a 0.x beta with breaking changes
batched roughly every two months. The **C2PA Conformance Program and Trust List** are now the trust
layer — a manifest signed by a certificate outside the Trust List validates as "signer unknown".
Gotchas: pure Rust, builds natively on aarch64, no CUDA, no wheels, no NIM container. Nothing here
cares that the box is a Spark.

**Alternate 1: AudioSeal (Meta FAIR, ICML 2024, v0.2 released 12 December 2024).** **MIT licence for
both code and weights** — unusually permissive for a watermarker, and the reason it became the
default. Generator plus detector; detection is single-pass and localised at sample granularity
(1/16,000 s), claimed up to two orders of magnitude faster than prior approaches; optional 16-bit
message giving 65,536 distinct identifiers without hurting detection; default 16 kHz but documented
to work at 24, 44.1 and 48 kHz. Tens of millions of parameters, CPU-fast. **Ship it as the soft
binding, but do not claim it is robust** — see the August 2026 result above.

**Alternate 2: a latent-domain mark — AWARE (arXiv 2510.17512, "Audio Watermarking with Adversarial
Resistance to Edits", which avoids reliance on attack-simulation stacks and handcrafted
differentiable distortions) or VoiceMark (embeds into speaker-specific RVQ latents, specifically
hardened against zero-shot voice cloning).** These are the only families that survived every
training-free structural attack. Both are **research code with licences I could not establish** —
verify before shipping. Also worth watching: **Latent-Mark** (arXiv 2603.05310, robust to neural
codec compression) and **LambdaMark** (arXiv 2606.21365, semantic watermarking with radioactivity).

### 2. Unconstrained pick

**Compute buys essentially nothing in this category, and that is the substantive answer.** The
binding variable is *where in the signal the mark lives* — latent-domain marks survived the 2026
structural attacks while magnitude- and carrier-domain marks did not — and not how many parameters
the embedder has. No published work shows watermark robustness scaling with model size, and there is
no scaling curve to cite because nobody has produced one.

What unlimited compute would actually justify is a **panel**: C2PA hard binding + AudioSeal +
a latent-domain mark + a perceptual-hash soft binding, all embedded, and all cross-checked at
verification time — the provcheck pattern the prior pass identified, which is also precisely what
the Commission's Article 50 guidance asks for ("layered transparency solutions"). Running four
detectors instead of one costs milliseconds.

Largest open-weights option: **AudioSeal remains the best-supported open release** (MIT weights,
maintained, widely benchmarked). Nothing larger exists that is both open and credible.

### 3. API pick

**Primary: Google SynthID, now a cross-vendor audio standard.** Over 100 billion items marked by
May 2026 across Google's own outputs (Lyria audio; Lyria 3 carries the SynthID extension), and the
standard has been extended to **OpenAI, ElevenLabs and Kakao**. **On 31 July 2026 OpenAI extended
SynthID watermarking to audio generated with its tools (ChatGPT and the OpenAI API) and shipped
verification API access** so third parties can build provenance checks into their own workflows.
Consumer-facing verification is available via the Gemini app. **Important limitation: Google's own
audio detection portal is still not publicly available**, so your ability to verify *other people's*
marks depends on whichever vendor issued them.

**Alternate 1: Truepic** — a C2PA founding member (with Adobe, Arm, BBC, Intel and Microsoft,
February 2021) and the first to ship C2PA 2.0 support for enterprises; offers customised, supported
C2PA signing implementations, which is the managed alternative to running c2pa-rs yourself.
**Alternate 2: Adobe Content Authenticity** — Adobe's Content Authenticity team maintains c2pa-rs
and c2patool, so this is the same technology with vendor support and hosted signing.

### 4. Compute sensitivity — **1**

Signing is cryptography (zero learned parameters) and every credible watermarker is under ~100M
parameters and runs faster than realtime on a CPU. The decisive 2026 evidence is architectural, not
scalar: latent-domain schemes (VoiceMark, WMCodec, AlignMark, AWARE) resisted every training-free
structural attack while magnitude- and carrier-domain schemes (AudioSeal, WavMark, SilentCipher,
audiowmark) did not, at matched PESQ ≥ 3.6. Robustness is a property of the embedding domain. A
bigger box changes nothing here.

### 5. Delta vs prior pass

**The prior primary is still the right shape — C2PA first, as the deliverable — but the AudioSeal
half now needs a caveat the prior pass did not have, and one of its shortlisted alternatives should
be dropped.**

- **AudioSeal is no longer defensible as a robust soft binding.** arXiv 2608.16566 (27 August 2026)
  removes its detection flag training-free at PESQ ≥ 3.6. Keep it — it is MIT for weights, it is
  cheap, and a mark that survives casual re-encoding is still worth having — but describe it
  internally as *tamper-evident, not tamper-proof*, and do not let any compliance claim rest on it.
- **Drop SilentCipher from the shortlist.** Its advantage over AudioSeal was multi-bit payload
  capacity; the same attack erases that payload entirely, so the advantage is gone and it carries no
  compensating robustness.
- **Add a latent-domain mark alongside AudioSeal** (AWARE / VoiceMark class), not instead of it.
  Licences unverified; that is the blocker to investigate first if this is ever implemented.
- **SynthID is now a realistic interoperability target rather than a Google-only curiosity**, after
  the 31 July 2026 OpenAI extension to audio with verification API access, and ElevenLabs and Kakao
  adoption. If OpenDub ever needs a mark that third parties can actually check, this is the one with
  ecosystem reach.
- **The regulatory clock the prior pass was anticipating has now run out.** Article 50 became
  enforceable 2 August 2026; the Transparency Code of Practice has been confirmed adequate by the
  Commission and the AI Board; the guidance explicitly endorses layering rather than any single
  technique. This strengthens rather than changes the prior verdict — but it means the deferral is
  now a deferral of a live obligation for anyone placing an EU-facing service on the market, not of
  a future one. (Worth noting: the AI Act's open-source exemption does not cover Article 50.)
- C2PA moved from 2.3 (Jan/Feb 2026) to **2.4 (21 April 2026)**; c2pa-rs implements 2.4 today.

### Sources

1. How Fragile Is Your Watermark? Training-Free Structural Removal of Neural Audio Watermarks (arXiv 2608.16566, Aug 2026) — https://arxiv.org/abs/2608.16566
2. VoxWatermark: a large-scale benchmark for audio watermark detection under perturbations (25 languages, 126,513.89 h; 10 methods, 4 neural + 6 traditional) — https://arxiv.org/pdf/2606.15187
3. AudioSeal (Meta FAIR; MIT code and weights; v0.2, 12 Dec 2024; ICML 2024) — https://github.com/facebookresearch/audioseal
4. AWARE: Audio Watermarking with Adversarial Resistance to Edits — https://arxiv.org/abs/2510.17512
5. Latent-Mark: an audio watermark robust to neural codec compression — https://arxiv.org/pdf/2603.05310
6. LambdaMark: semantic audio watermarking for robustness and radioactivity — https://arxiv.org/pdf/2606.21365
7. c2pa-rs (dual MIT/Apache-2.0; C2PA v2.4; WAV/MP3/M4A support) — https://github.com/contentauth/c2pa-rs
8. C2PA Technical Specification 2.3 (2026-01-05) — https://spec.c2pa.org/specifications/specifications/2.3/specs/_attachments/C2PA_Specification.pdf
9. C2PA Technical Specification 2.4 — https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
10. C2PA Conformance Program — https://c2pa.org/conformance/
11. EU AI Act Article 50 transparency obligations (enforceable 2 Aug 2026) — https://artificialintelligenceact.eu/article/50/
12. Commission confirms Transparency Code of Practice as adequate; final guidelines (Jul 2026) — https://www.faegredrinker.com/en/insights/publications/2026/7/eu-ai-act-commission-confirms-transparency-code-of-practice-as-adequate-and-publishes-final-version-of-its-guidelines-on-transparency-obligations
13. EU transparency obligations FAQ (no single marking technique meets all four Art. 50(2) requirements) — https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act
14. SynthID (Google DeepMind) — https://deepmind.google/models/synthid/
15. OpenAI: Advancing content provenance (SynthID audio + verification API, 31 Jul 2026) — https://openai.com/index/advancing-content-provenance/
16. Truepic first with C2PA 2.0 support for enterprises — https://www.truepic.com/blog/truepic-first-with-c2pa-2-0-support-for-enterprises
17. A comprehensive real-world assessment of audio watermarking algorithms (will they survive neural codecs?) — https://arxiv.org/pdf/2505.19663

---

# Phase F · Picture generation

## F1 Live-action lip sync

### 1. DGX Spark pick

**Primary: LatentSync 1.6 — ByteDance, Apache-2.0** [1].
SD1.5 UNet + Whisper audio conditioning + TREPA/SyncNet supervision, operating on a **512×512 face crop**
(1.6 retrained at 512² specifically to fix 1.5's blurriness). ~0.9B-class UNet plus VAE and Whisper;
**18 GB resident for 1.6** (1.5 needs 8 GB) — trivial inside 128 GB. Precision: BF16, no quantisation needed.
Speed: **no published Spark measurement**; extrapolating from the GB10:5090 video-diffusion ratio [8],
expect **~4–8× slower than realtime on the face crop**, i.e. a 10-minute episode with ~40 % speaking-face
coverage lands at **~16–32 min of Spark time**. This is the only F1 option in the practical band.
Quality anchor: LSE-C **7.58 on HDTF**, 6.75 VFHQ, 6.51 CelebV-HQ [2].
*Gotchas:* the repo pins old torch/xformers — rebuild on aarch64 against the `sm_121` wheel index [12];
use FA2 2.8.3 or SageAttention 2.2, **never** FA3 (unsupported) or SageAttention 3 (artefacts) [12][13];
`insightface` needs the `sm_121` `onnxruntime-gpu` wheel. **LatentSync has shipped nothing since
11 June 2025** [1] — it is frozen upstream, which is a real maintenance risk for a default.

**Alternates:** *KeySync* (Imperial College London + University of Wrocław, arXiv 2505.00497) — same cost
class, explicitly engineered against expression leakage and occlusion with a `LipLeak` metric, and higher
native resolution; the right swap when LatentSync leaks the original mouth [14].
*LTX-2.3-22b IC-LoRA LipDub* (Lightricks, LTX-2-community-licence) at NVFP4 (~20 GB [6]) — fits the Spark
and is far better, but at 22B DiT rates it is an overnight, hero-shot-only escalation, not a default [10][11].

### 2. Unconstrained pick

**Primary: LTX-2.3-22b IC-LoRA LipDub — Lightricks, LTX-2-community-licence** [10][11].
The largest open-weights option that is purpose-built for *dubbing existing footage* rather than avatar
generation: given a source video and the new dialogue it regenerates lips **and** facial expression in a
22B joint audio-video DiT, instead of pasting a 512² mouth. Validated EN/FR/ES/DE/RU; beta limits are
single-speaker and no auto-translation, and LTX-2.5 support is still "in development" [10].
**Margin over the Spark pick: large and structural.** Full-frame DiT methods on Wan2.1-14B already post
Sync-C **7.62** / Sync-D **8.14** / FID **37.3** / FVD **382** on HDTF (OmniAvatar) against HunyuanAvatar's
7.31 / 8.33 / 47.3 / 588 [17] — and unlike LatentSync those FID/FVD numbers are for the *whole frame*,
which is where the visible dubbing artefacts live. A rack also buys the three things the Spark cannot
afford: 25→50+ sampling steps, 720p→4K native, and **best-of-N sampling re-ranked by a SyncNet/LSE judge**,
which is the single highest-yield trick in this category.
**Alternates:** *InfiniteTalk* (MeiGen-AI, arXiv 2508.14033) — sparse-frame video dubbing on Wan2.1-14B,
beats MuseTalk and LatentSync on HDTF/CelebV-HQ/EMTD [15]; *Wan2.2-S2V-14B* (Alibaba, Apache-2.0, 16B BF16,
**≥80 GB single-GPU**, 480P/720P @ 24 fps) [16].

### 3. API pick

**Primary: sync-3 — sync. labs** (current default model as of September 2026) [18].
**4K native** output with built-in super-resolution, 95+ languages, whole-shot global context rather than
per-snippet inpainting, native handling of close-ups, extreme angles and occlusions, static-image input.
$0.107–0.133/s at 25 fps. API, ComfyUI node, Premiere plugin and MCP server.
**Alternates:** *lipsync-2-pro* (sync. labs) — generates faces at **512×512**, so it needs an F2 restoration
pass sync-3 does not, but costs roughly half [18][19]; *HeyGen Video Translate* — 175+ languages and the
strongest third-party benchmark score in the translate-and-dub product category (76.8 vs Rask AI's 51.8),
with smoother jawlines on real uploaded footage [20].

### 4. Compute sensitivity

**5.** This is the most compute-sensitive category in the pipeline. Quality scales along four independent
axes at once — model size (0.9B UNet → 14B → 22B DiT), sampling steps, resolution (512² crop → 4K full
frame), and best-of-N re-ranking — and the Spark can pay for none of them. Concretely: the Spark's practical
ceiling is a 512×512 mouth-region UNet at ~0.15–0.25 s video/s compute, while the quality frontier is a
22B full-frame DiT at **0.002–0.003 s video/s compute on the same box** [5][7][8] — a ~100× compute gap
that maps directly onto whole-frame FID/FVD (37.3/382 for a 14B DiT vs a crop-and-paste composite that
is not even measured on those axes [17]).

### 5. Delta vs prior pass

**Partly changed.** LatentSync 1.6 still stands as the DGX Spark pick — nothing cheaper is better and
nothing better is cheap enough. Four corrections:
(a) **New and better open-weights option:** *LTX-2.3-22b IC-LoRA LipDub* (model card 29 Jan 2026 [11])
was absent from the prior list and should replace InfiniteTalk as the unconstrained primary.
(b) **The paid escalation is sync-3, not lipsync-2-pro** — sync-3 is now sync's default and is 4K-native,
where lipsync-2-pro still renders faces at 512×512 [18][19]. The prior pass listed both as co-primaries.
(c) **Attribution fix:** the prior pass credits KeySync to "Imperial College London / Meta AI"; the project
page lists Imperial College London and the **University of Wrocław** [14].
(d) **Staleness flag:** LatentSync's last release is 11 June 2025 [1]; treating it as an actively maintained
default is no longer safe. Market context: Amazon shipped production visual dubbing on Prime Video
(Maxton Hall S1–S2, 9 September 2026) [21] — the category is now a shipped product feature, not research.
The prior verdict's core claim — that most of the win comes from B1/B2/B4 per-shot gating rather than the
model swap — survives unchanged and is reinforced by the compute numbers above.

### Sources
1. bytedance/LatentSync (1.6, 11 Jun 2025, Apache-2.0, 18 GB) — https://github.com/bytedance/LatentSync
2. HighSync, arXiv 2605.16918 (16 May 2026) — LSE-C table incl. LatentSync/MuseTalk — https://arxiv.org/html/2605.16918v1
3. NVIDIA Developer Forums, "Image to Video Generation using the Spark vs. PC" — https://forums.developer.nvidia.com/t/image-to-video-generation-using-the-spark-vs-pc/370077
4. NVIDIA Developer Forums, "Dgx spark comfyUI" — https://forums.developer.nvidia.com/t/dgx-spark-comfyui/368179
5. ai-muninn, "Running MiniMax-H3 on a DGX Spark" (measured 741 s / 362 frames) — https://ai-muninn.com/en/blog/dgx-spark-minimax-h3-span-upscaler
6. NVFP4 on GB10: distilled LTX-2.3 29 GB → 19.5 GB, no speed gain — https://ai-muninn.com/en/blog/series/dgx-spark
7. LTX-2: Efficient Joint Audio-Visual Foundation Model, arXiv 2601.03233 (Wan2.2-14B = 22.30 s/step, 121f 720p, H100; CC-BY-4.0) — https://arxiv.org/pdf/2601.03233
8. proxpc, DGX Spark GB10 vs RTX 5090 (HunyuanVideo 1.5 FP16: 3606 s vs 1310 s) — https://www.proxpc.com/blogs/nvidia-dgx-spark-gb10-performance-test-vs-5090-llm-image-and-video-generation
9. StorageReview, DGX Spark CES 2026 update (2.5× / 8× claims, 5 Jan 2026) — https://www.storagereview.com/news/nvidia-dgx-spark-achieves-2-5x-performance-and-8x-video-speed-in-ces-2026-enterprise-update
10. LTX docs, LipDub (IC-LoRA) Beta — https://docs.ltx.io/open-source-model/advanced-workflows/lip-dub-beta
11. Lightricks/LTX-2.3-22b-IC-LoRA-LipDub model card (29 Jan 2026) — https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt
12. dgx-spark-wheels: prebuilt sm_121 flash-attn 2.8.3 / sageattention 2.2.0 / onnxruntime-gpu 1.22 — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels
13. SageAttention issue #321: SA3 mosaic artefacts, no speedup on DGX Spark — https://github.com/thu-ml/SageAttention/issues/321
14. KeySync (Imperial College London + Univ. of Wrocław), arXiv 2505.00497 — https://antonibigata.github.io/KeySync/
15. InfiniteTalk, arXiv 2508.14033 (MeiGen-AI) — https://github.com/MeiGen-AI/InfiniteTalk
16. Wan-AI/Wan2.2-S2V-14B (16B BF16, Apache-2.0, ≥80 GB, 480P/720P) — https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
17. OmniAvatar, arXiv 2506.18866 (Wan2.1-T2V-14B; HDTF Sync-C 7.62 / FVD 382) — https://arxiv.org/html/2506.18866v1
18. sync. labs, sync-3 model docs (4K native, 95+ languages, $0.107–0.133/s) — https://sync.so/docs/models/sync-3
19. sync. labs, lipsync-2-pro (512×512 face generation) — https://sync.so/lipsync-2-pro
20. HeyGen vs Rask AI vs Synthesia comparison 2026 (76.8 vs 51.8) — https://itsupplychain.com/rask-ai-vs-heygen-vs-synthesia-ai-video-translation-comparison-2026/
21. Slator, "Amazon Launches First AI Lip Sync on Prime Video" (9 Sep 2026) — https://slator.com/prime-video-ai-lip-sync-dubbing/

---

## F2 Mouth-region restoration

### 1. DGX Spark pick

**Primary: DVFace — Shanghai Jiao Tong University + Meituan, code at zhengchen1999/DVFace** [1].
Spatio-temporal dual-prior diffusion for video face restoration on a **Wan 2.1 T2V backbone**
(1.3B-class), and critically a **one-step** diffusion restorer — no 20–50 step sampling loop, which is
exactly what makes it affordable on a bandwidth-limited box. Run BF16; **~10–14 GB resident**.
Speed: no published Spark figure; one-step on a ~1.3B backbone over a cropped mouth annulus should land
**near or above realtime** on the GB10 and is not the pipeline bottleneck. Licence: arXiv preprint terms,
code public (treat as research-licence until the repo states otherwise).
Quality on VFHQ-Test: **PSNR 31.81 / LPIPS 0.0776 / DOVER 0.8703 / NIQE 5.27**, first on every metric
except NIQE, versus KEEP 28.58/0.1815/0.6428/6.20, DicFace 30.17/0.1439/0.7045/6.57,
PGTFormer 29.03/0.1165/0.8535/**4.97**, SVFR 26.73/0.1366/0.7749/5.82 [1].
*Gotchas:* Wan-2.1-derived code paths assume FA2 or SDPA — use the `sm_121` FA2 2.8.3 wheel or
SageAttention 2.2 [F1-12]; face-alignment dependencies (`insightface`/`onnxruntime`) need the `sm_121`
onnxruntime-gpu wheel. As with all video diffusion on GB10, NVFP4 saves memory but buys no speed [F1-6] —
so stay at BF16 and keep the one-step schedule.

**Mandatory non-model layer:** the classical annulus path (grain synthesis, colour/luma transfer,
feathered or Poisson blend over the F1 mask) still runs *first and always*. It costs nothing, carries no
identity risk, and fixes the two artefacts that are actually visible — the seam and the grain mismatch.
DVFace is the resolution-recovery pass on top, not a replacement.

**Alternates:** *PGTFormer* — best NIQE in the field (4.97, better than DVFace's 5.27 [1]) and a cheap
transformer, so it is the right pick when perceptual naturalness matters more than fidelity;
*DicFace* (ICCV 2025 Highlight) — Dirichlet-constrained codebook, strongest temporal coherence of the
codebook family and the closest non-diffusion competitor at PSNR 30.17 [1][2].

### 2. Unconstrained pick

**Primary: DVFace on a larger Wan backbone with multi-step sampling, ensembled with DicFace and
re-ranked by an identity (ArcFace/CSIM) judge.** Largest open-weights option in this lineage is the
Wan 2.1/2.2 14B family; DVFace's published configuration is the 1.3B-class one-step variant [1].
**Margin over the Spark pick: small — this is the least compute-sensitive category in Phase F.**
The restored region is a ~200×200 px mouth annulus that must match surrounding *unrestored* pixels, so
the objective is bounded by the source footage, not by model capacity. DVFace's own gain over the prior
SOTA is +1.64 dB PSNR over DicFace and +3.23 dB over KEEP [1]; going bigger buys fractions of that, and
over-restoring actively hurts because it makes the annulus look *better* than the rest of the face.
**Alternates:** *DicFace* (as ensemble member, decorrelated prior) [2]; *SVFR* unified video face
restoration framework (weaker on VFHQ at 26.73 PSNR but handles colourisation/inpainting jointly) [1].

### 3. API pick

**There is no credible hosted API dedicated to mouth-region video face restoration** — I could not verify
one, so I am not naming a product to fill the slot. The category is instead absorbed by the F1 vendors:
**Primary: sync-3 — sync. labs**, whose **4K-native generation with built-in super-resolution** removes
the need for a separate F2 stage entirely; this is the honest API answer [F1-18].
**Alternates:** *lipsync-2-pro* (sync. labs) — the inverse case: because it renders faces at 512×512 it
*requires* an F2 pass, so pairing it with a self-hosted DVFace is the sensible hosted configuration
[F1-19]; *HeyGen Video Translate* — restoration is internal and not separately exposed or measurable [F1-20].

### 4. Compute sensitivity

**2.** Bounded by the source footage rather than by model capacity: the output must blend invisibly into
unrestored surrounding pixels, so headroom is small and over-capacity is a *liability*. The whole field
spans **28.58 → 31.81 dB PSNR on VFHQ-Test** (KEEP → DVFace) [1], and the current SOTA already achieves
that in **one** diffusion step — the usual compute levers (more steps, bigger backbone) have been
explicitly engineered out. The classical zero-compute annulus path captures much of the practical win.

### 5. Delta vs prior pass

**Unchanged.** DVFace (arXiv 2604.14560, 16 April 2026) remains the correct model primary and it still
runs comfortably on a Spark — the one-step design is precisely what makes it Spark-friendly, which the
prior pass did not have to reason about. Nothing has appeared since 3 September 2026 that beats its VFHQ
numbers. Two refinements rather than reversals: the prior pass's third co-primary (classical annulus
matching) should be recorded as a *mandatory preceding layer*, not an alternative to DVFace; and
PGTFormer is worth promoting from "strong" to a first-class alternate on the strength of its NIQE 4.97,
which is the best in the comparison and the metric that correlates with "looks like film" [1].
The prior pass's second co-primary, RGFVR (TU Munich, reference-guided flow-matching restoration), I could
not re-verify with published VFHQ numbers in this pass and therefore do not rank it.

### Sources
1. DVFace: Spatio-Temporal Dual-Prior Diffusion for Video Face Restoration, arXiv 2604.14560 (16 Apr 2026, SJTU + Meituan; VFHQ-Test table vs KEEP/DicFace/PGTFormer/SVFR) — https://arxiv.org/html/2604.14560
2. DicFace: Dirichlet-Constrained Variational Codebook Learning, arXiv 2506.13355 (ICCV 2025 Highlight) — https://arxiv.org/abs/2506.13355
3. DVFace code release — https://github.com/zhengchen1999/DVFace
4. Towards Real-world Video Face Restoration: A New Benchmark, arXiv 2404.19500 (VFHQ evaluation protocol) — https://arxiv.org/pdf/2404.19500
5. sync. labs sync-3 docs (4K native, built-in super-resolution) — https://sync.so/docs/models/sync-3
6. sync. labs lipsync-2-pro (512×512 faces — requires restoration) — https://sync.so/lipsync-2-pro
7. dgx-spark-wheels, prebuilt sm_121 wheels — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels

---

## F3 2D animation mouth retiming

### 1. DGX Spark pick

**Primary: Rhubarb Lip Sync — Daniel Wolf, open source, CPU-only** [1], used as a **viseme/flap-count
oracle feeding C3**, not as a picture generator.
Parameters: n/a — it is a classical recogniser, not a neural generator. It emits **6 basic mouth shapes
(A–F) plus 3 optional extended shapes (G, H, X)**, using PocketSphinx for English and a
language-independent phonetic recogniser for everything else [1]. Memory: **~0 GB GPU, <1 GB RAM**;
it runs many times faster than realtime on the Spark's 20 Arm cores and never touches the GPU.
Actively maintained (CI green, ~510 commits) as of this pass [1].
*Gotchas:* none on aarch64 — no CUDA, no wheels, no `sm_121` exposure. The only real caveat is that its
English path is materially stronger than its language-independent path, so for Hindi/regional targets you
should drive it from the A4/A5 phoneme alignment rather than from raw audio.

**Alternates:** *Classical flap-state re-sequencing* — classify each existing drawn mouth into a flap
state, then reorder/hold the existing drawings to the new dub timing. This is the theoretically correct
method and preserves the line art exactly; I still find **no published implementation**, so it remains a
build-it-yourself item. *sync-3* behind a per-shot flag for stylised-but-photoreal content (sync. labs
claim their editing architecture covers "live-action, 3D animation, even AI-generated" [2]) — but note
that claim does not extend to hand-drawn 2D.

### 2. Unconstrained pick

**There is no unconstrained pick. A rack of B200s does not help this category**, and that is the finding.
The ceiling is not compute — it is the absence of any model that edits *drawn* frames while preserving the
drawn line. Every strong F1 model, including LTX-2.3-22b LipDub and Wan/OmniAvatar-class DiTs, works by
*regenerating pixels* [F1-10][F1-17]; applied to cel animation this destroys ink-and-paint line weight,
flat colour fills and the deliberately sparse 2s/3s frame cadence, and no published evaluation of any of
them on hand-drawn 2D exists. Products marketed for "anime lip sync" (DomoAI, MkAnime, Media.io) publish
no metrics and no papers, and I could not verify any of their quality claims [3].
**The correct recommendation is unchanged: decline to lip-sync animation, and move the effort to C3
flap-aware script adaptation** — which is how the anime dubbing industry has always solved this, is
verifiably what human localisation does [4], and which the existing OpenDub stack already delivers.
**Alternates (for the record, not recommended):** *LTX-2.3-22b LipDub* on animation — technically runnable,
unevaluated on 2D, and near-certain to break the drawn style [F1-10]; *retraining a DiT on a cel-animation
corpus* — no such public corpus or model exists.

### 3. API pick

**None credible.** The shipped product answer is the F1 vendors' generic lip-sync endpoints applied to
animation, and none of them publish an animation-specific evaluation.
**Primary (weak, flagged): sync-3 — sync. labs**, on the strength of its stated coverage of 3D-animated
and AI-generated sources and its whole-shot context model [2]; use it only on 3D/CG animation, never on
hand-drawn 2D, and gate it behind human review.
**Alternates:** *DomoAI* — explicitly markets optimisation for non-human and stylised characters, but
publishes no metrics, so this is an unverified vendor claim [3]; *Adobe Character Animator* auto-lip-sync
— real, shipping, deterministic viseme-driven retiming, but it only works on rigs *authored in it*, so it
cannot retime delivered footage [3].

### 4. Compute sensitivity

**1.** Solved-at-small-scale in the degenerate sense: the Spark pick *is* the ceiling because the ceiling
is a CPU-only phoneme recogniser, and the frontier is empty. More compute buys strictly nothing — the
binding constraint is that the task (retime existing drawings) is not the task any available model
performs (regenerate pixels). The measurable lever here lives entirely in C3: adapting the translated
script so its syllable and open-vowel pattern matches the drawn flaps, which is a text-model problem at
Phase C, not a picture problem at Phase F.

### 5. Delta vs prior pass

**Unchanged, and now more firmly supported.** The 3 September verdict — nothing credible exists, keep
declining to lip-sync animation, move effort to C3 — still stands, and nothing has appeared since.
One addition the prior pass did not make: the compute-tiered framing *strengthens* the verdict rather
than weakening it. Even with unlimited B200s there is no model to run, so this is the one Phase F category
where the answer is genuinely invariant to the compute tier — worth recording explicitly so the question
is not re-litigated the next time a larger video DiT ships. Rhubarb remains the right small internal
dependency and is verified still maintained in 2026 [1]; it should be wired as a flap-count oracle into
C3's adaptation loop rather than as an F3 renderer.

### Sources
1. Rhubarb Lip Sync (Daniel Wolf) — 6 basic + 3 extended mouth shapes, PocketSphinx + language-independent recogniser, CPU-only, maintained — https://github.com/DanielSWolf/rhubarb-lip-sync
2. sync. labs sync-3 docs — claimed coverage of live-action, 3D animation and AI-generated sources — https://sync.so/docs/models/sync-3
3. 2D-animation lip-sync tool survey 2026 (OpenToonz + Rhubarb, MkAnime, DomoAI, Media.io, Adobe) — https://dreamina.capcut.com/resource/2d-animation-lip-sync
4. Dubbing in Practice: A Large Scale Study of Human Localization, arXiv 2212.12137 (what human dubbing actually optimises) — https://arxiv.org/pdf/2212.12137
5. Real-Time Lip Sync for Live 2D Animation, arXiv 1910.08685 (viseme-sequence formulation for 2D) — https://arxiv.org/pdf/1910.08685
6. LTX docs, LipDub (IC-LoRA) Beta — pixel-regenerating architecture, no 2D evaluation — https://docs.ltx.io/open-source-model/advanced-workflows/lip-dub-beta

---

## F4 On-screen text replacement

The category is two problems — **erase**, then **re-render** — and they have different compute profiles.

### 1. DGX Spark pick

**Primary (erase): SEDiT — Baidu (Zheng Hui, Yunlong Bai), arXiv 2605.14894, 14 May 2026** [1].
**One-step, mask-free** diffusion transformer on an **LTX-Video-2B-0.9.6** backbone with a rank-256 LoRA
adding **381M trainable parameters (19 % of base)**. Run BF16; **~8–10 GB resident**.
Measured speed: **4 s for a 65-frame 1920×1080 clip on an A800** = 0.68 s of video per second of compute;
derating through the GB10:datacentre-GPU ratio [F1-8] gives **~0.15–0.25 s video/s on the Spark
(4–7× slower than realtime)** — comfortably practical for subtitle/lower-third burn-off.
Quality on VSR-Bench-400: **PSNR 31.59 / SSIM 0.8805 / LPIPS 0.0981 / FVD 24.06 / MOS 4.5**, versus
Minimax-Remover 28.31/0.8785/0.1011/39.35/2.5, DiffuEraser 27.51 and ProPainter 26.82 [1]. Handles
single-shot and multi-shot long video, background-vs-overlay text discrimination, multilingual subtitles,
and logo preservation [1]. Licence: arXiv preprint terms; project page live.

**Primary (re-render): Qwen-Image-Edit-2511 — Alibaba, Apache-2.0, 20B** [2].
The best *open-weights* text-in-image editor: direct add/delete/modify of text preserving original font,
size and style, bilingual CN/EN, GEdit-Bench-EN 7.56 for the family [3]. At BF16 that is ~40 GB resident —
fine in 128 GB; NVFP4 brings it to ~11 GB but buys no speed on GB10 [F1-6]. Expect **~2–5 min per edited
keyframe** on the Spark by analogy with the measured FLUX-dev GB10 times (129–237 s/image [F1-8]) — which
is acceptable because you edit *keyframes* and planar-track the result, not every frame.
*Gotchas:* both stages want FA2 2.8.3 or SageAttention 2.2 from the `sm_121` index; never FA3 or
SageAttention 3 [F1-12][F1-13]. The LTX-Video-2B backbone under SEDiT is the same lineage NVIDIA ships
NVFP8-optimised weights for on Spark, so the kernel path is well-trodden [F1-9].

**Alternates:** *CLEAR* (UESTC/UCAS/TUM/SJTU/NUS, arXiv 2603.21901, 23 Mar 2026) — Wan2.1-Fun-1.3B +
rank-64 LoRA at **0.77 % trainable params**, mask-free, **+6.77 dB PSNR and −74.7 % VFID over the best
mask-based baseline** on Chinese hard-subs (26.80/0.894/20.37 vs Minimax-Remover 20.03/0.773/95.39,
DiffuEraser 17.85/0.672/72.51, ProPainter 17.24/0.658/98.46) [4]. Stronger on burned-in CJK subtitles,
but **4.86 s per frame across 8 GPUs** [4] — roughly two orders of magnitude more expensive than SEDiT,
so it is a per-shot escalation on the Spark, not the default.
*FLUX.2 [klein] 9B* (Black Forest Labs, 15 Jan 2026) — sub-second re-render, but FLUX.2-dev
non-commercial licence; the 4B sibling is Apache-2.0 and the honest cheap option [5].

**And the product default stays the product default:** the overlay track plus editing-template export is
deterministic, editable, and what Education/Marketing buyers actually want. Everything above lives behind
a flag for flat overlay text only.

### 2. Unconstrained pick

**Primary: SEDiT (erase) + FLUX.2 [dev] 32B (re-render), with best-of-N re-ranking by an OCR legibility
judge.** FLUX.2 [dev] is a 32B rectified-flow transformer coupled to a Mistral-3 24B VLM, native output to
**4 MP** and multi-reference conditioning from up to 10 images — the largest open-weights editor available
(FLUX.2-dev non-commercial licence, 25 Nov 2025) [5].
**Margin over the Spark pick is asymmetric and worth stating precisely:**
- **Erasure: essentially none.** SEDiT is already one-step and near the ceiling (FVD 24.06 vs the next
  best 39.35 [1]) — a B200 rack cannot meaningfully improve a solved one-step task. The only real gain is
  running SEDiT *and* CLEAR and picking per shot, which costs compute but not capability [1][4].
- **Re-render: meaningful.** Legibility of replacement text in non-Latin, non-Chinese scripts (Devanagari,
  Arabic, Thai) is where open 20–32B editors still fail and where frontier models are visibly ahead —
  32B + best-of-16 with an OCR re-rank is a real quality step over a single 20B sample.
**Alternates:** *CLEAR at full 8-GPU settings* for hard CJK burn-ins [4]; *Qwen-Image-Edit-2511* 20B as the
Apache-2.0 re-render option when the FLUX non-commercial licence is disqualifying [2][5].

### 3. API pick

**Primary: Nano Banana Pro (Gemini 3 Pro Image) — Google DeepMind, released 17 November 2025** [6].
The best verified model for rendering *correct, legible* text into an image, across multiple languages,
at up to 4K — which is exactly the F4 failure mode. Google's own framing is that it is the first model
where "add the text 'Sale' in bold white on the product" reliably yields readable text rather than
decorative gibberish [6].
**Alternates:** *Nano Banana 2 (Gemini 3.1 Flash Image), 26 February 2026* — ~95 % of Pro's visual quality
at 2–5× the speed and half the cost, which matters when you are re-rendering hundreds of title cards [7];
*Vozo Visual Translate* — the only shipped **end-to-end** product that detects, erases, translates and
rebuilds on-screen text while preserving layout, style and animation, i.e. it does the whole F4 stage
rather than one half of it [8]. *Qwen-Image-2.0* (Alibaba, 10 February 2026, 7B, **API-only — no open
weights**) ranked #1 on AI Arena for both text-to-image and image editing at launch and is the strongest
non-Google alternate [9].

### 4. Compute sensitivity

**3.** Split by sub-problem, which is why it is not a 4 or a 2. **Erasure is solved at small scale** — a
2B backbone with a 381M LoRA in one step beats every multi-step mask-based method (FVD 24.06 vs 39.35)
[1], so the Spark is genuinely at the ceiling there. **Re-render is not** — text legibility in complex
scripts still scales with model size and with best-of-N sampling, and the gap between a 20B open editor
and a frontier model is visible to the naked eye on Devanagari and Arabic [6]. Net: more compute buys you
roughly nothing on half the category and a real, user-visible gain on the other half.

### 5. Delta vs prior pass

**Changed.** The prior pass listed "CLEAR or SEDiT" as interchangeable erasure options ("both roughly 9 dB
PSNR ahead of..."). On the Spark they are **not** interchangeable: SEDiT is one-step and costs
**4 s per 65-frame 1080p clip on a single GPU**, while CLEAR costs **4.86 s per frame across 8 GPUs** [1][4]
— roughly two orders of magnitude apart. **SEDiT should be the DGX Spark primary and CLEAR demoted to a
per-shot escalation for burned-in CJK subtitles**, where its +6.77 dB margin is real.
Two further corrections:
(a) **Qwen-Image-2.0 (10 Feb 2026) is API-only with no open weights** [9], so the open re-render pick must
remain **Qwen-Image-Edit-2511** (20B, Apache-2.0) [2] — a prior-pass entry listing "Qwen-Image-Edit" as a
local primary needs this version pin to stay true.
(b) **FLUX.1 Kontext is superseded**: FLUX.2 [dev] 32B (25 Nov 2025) and FLUX.2 [klein] 4B/9B (15 Jan 2026)
replace it, and the 4B klein is Apache-2.0 [5].
The prior verdict's headline — ship the overlay track and editing-template export as the deliverable, not
the fallback — stands unchanged and is the right default.

### Sources
1. SEDiT: Mask-Free Video Subtitle Erasure via One-step Diffusion Transformer, arXiv 2605.14894 (14 May 2026, Baidu; LTX-Video-2B + 381M LoRA; VSR-Bench-400 PSNR 31.59 / FVD 24.06; 4 s / 65 frames @1080p on A800) — https://arxiv.org/html/2605.14894
2. Qwen/Qwen-Image-Edit-2511 model card (20B, Apache-2.0) — https://huggingface.co/Qwen/Qwen-Image-Edit-2511
3. Qwen-Image technical report, arXiv 2508.02324 (GEdit-Bench-EN 7.56) — https://arxiv.org/abs/2508.02324
4. CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal, arXiv 2603.21901 (23 Mar 2026; Wan2.1-Fun-1.3B, 0.77 % trainable; PSNR 26.80 / VFID 20.37; 4.86 s/frame on 8 GPUs) — https://arxiv.org/html/2603.21901v1
5. FLUX.2 [dev] 32B (25 Nov 2025) and FLUX.2 [klein] 4B Apache-2.0 / 9B (15 Jan 2026) — https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
6. Nano Banana Pro / Gemini 3 Pro Image (17 Nov 2025) — best-in-class legible text rendering, 4K — https://deepmind.google/models/gemini-image/pro/
7. Nano Banana 2 / Gemini 3.1 Flash Image (26 Feb 2026) — ~95 % of Pro quality, 2–5× faster — https://blog.google/innovation-and-ai/technology/ai/nano-banana-2/
8. Vozo Visual Translate — detect, erase, translate and rebuild on-screen text preserving layout and animation — https://www.vozo.ai/blogs/visual-translation/ai-tools-localize-on-screen-text
9. Qwen Image 2.0 (10 Feb 2026, 7B, API-only, #1 AI Arena at launch) — https://qwen-ai.com/qwen-image/
10. dgx-spark-wheels, prebuilt sm_121 flash-attn / sageattention wheels — https://github.com/Fulton-Engineering-Services/dgx-spark-wheels

---

# Phase G · Automatic quality judges

## G1 Intelligibility judge (ASR-in-the-loop WER on generated takes)

### 1. DGX Spark pick

**Primary: Parakeet-TDT-0.6B-v3 (NVIDIA, 600M, CC-BY-4.0) as the *verifier*, with
Whisper-large-v3 (OpenAI, 1.55B, MIT) as a lineage-disjoint *evaluator*, and a wav2vec2-large /
MMS CTC head as the third family.** FP16/BF16; ~6 GB resident for all three plus batching
headroom. Parakeet is RTFx 1720 and 4.81% avg WER on the ASR Leaderboard's 5-language
multilingual set — same WER as Whisper-large-v3 at 15× the throughput [1], so best-of-16
verification costs essentially nothing. Spark compatibility is confirmed, not inferred: NVIDIA's
own `nemotron-speech-streaming-en-0.6b` card carries an "Added DGX spark after testing" commit,
there are working ARM64/CUDA-13 Parakeet deployments on the NVIDIA developer forums, and the
Canary-1B-v2 card lists Blackwell as a supported architecture [2,3]. NeMo runs natively on
aarch64; you do not need vLLM for ASR, which sidesteps the sm_121 problem entirely.

Alternates: **Canary-1B-v2** (978M, CC-BY-4.0, 7.15% HF-leaderboard mean, RTFx 749) where
accuracy beats throughput; **Voxtral Small 24B** (Mistral, Apache-2.0, 3.70% multilingual avg WER
— best open model on that set) at FP8 ≈ 24 GB if you want the strongest single open judge.
**Do not pair Canary with Parakeet** as your "two opinions": both are NeMo/FastConformer lineage,
which is precisely the failure mode below.

### 2. Unconstrained pick

**Cross-family rank ensemble over ≥3 disjoint ASR lineages** (Whisper-large-v3 + Voxtral Small 24B
+ Canary-1B-v2 + an MMS/wav2vec2 CTC posterior). Yu & Kang (ICML 2026 ML-for-Audio workshop) show
this is the right object: on LibriSpeech-PC test-clean with F5-TTS, **verifier rankings reverse
across Whisper, wav2vec 2.0 and HuBERT evaluators**, and same-family verifier–evaluator pairs
recover **2–3× more oracle headroom** than cross-family pairs despite linear CKA of 0.978 — i.e.
it is identity/lineage coupling, a speech analogue of LLM-as-a-judge self-bias, not representation
overlap [4]. Their cross-family rank ensemble (w2v2-base 95M + distil-whisper-v3 756M) attains the
lowest mean WER across three independent evaluators: **1.61% at N=10, −12% relative to F5-TTS**,
with SIM-o within ±0.0006 and UTMOS within ±0.005. Margin over the Spark pick: **none** — the
ensemble is two models totalling 851M parameters. This category cannot be bought with compute.

### 3. API pick

**ElevenLabs Scribe v2** — 2.67% avg WER on the ASR Leaderboard multilingual set, the best number
published anywhere, open or closed [1]. Alternates: **AssemblyAI Universal 3 Pro** (3.23%) and
**Gemini 3.5 Transcribe** (Google, 26 Aug 2026, 85+ languages, plus a Transcribe Live WebSocket
variant) [5]. Use a hosted model as the *evaluator* you report against, never as the verifier you
optimise into.

### 4. Compute sensitivity — **2**

Within a family, scale buys a little: distil-whisper-sm (166M) as verifier was flat (−1.0%,
p=0.93) while distil-whisper-v3 (756M) gave −8.7% (2.06% → 1.88%, p=0.030) under the official
evaluator [4]. But the *family pairing* is a much larger lever than the checkpoint, and large
oracle headroom is left unexploited: at N=3 the oracle is 1.42% vs 2.04% single-shot, and the best
BoN recovers only 26% of that gap. More N and more families beat a bigger judge.

### 5. Delta vs prior pass

Prior primary (Canary-1B-v2 primary + Whisper large-v3 as independent second opinion) **largely
stands**, and the "gate on agreement, not on either score alone" instinct is now backed by a
published result rather than intuition. Two changes: (a) swap the *verifier* to
Parakeet-TDT-0.6B-v3 — equal WER to Whisper-large-v3, RTFx 1720, so best-of-16 is free; (b) add a
wav2vec2/HuBERT CTC third family and report WER under at least two disjoint lineages
("cross-evaluator triangulation"), which the 3 September pass did not specify. Nothing newer than
Canary-1B-v2 / Parakeet-v3 has shipped from NVIDIA as of 18 Sept 2026 [2].

### Sources
1. ASR Leaderboard: Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation, arXiv:2510.06961v4 (27 Mar 2026) — https://arxiv.org/html/2510.06961v4
2. NVIDIA Nemotron speech FAQ, production-ready open ASR for European languages (10 Jul 2026) — https://perspectives.nvidia.com/nemotron-speech/
3. nvidia/canary-1b-v2 model card (978M, CC-BY-4.0, Blackwell supported) — https://huggingface.co/nvidia/canary-1b-v2
4. Yu & Kang, "Best-of-N TTS Evaluation is Confounded by ASR Family Alignment", ICML 2026 Workshop on ML for Audio — https://github.com/yu1012/BoN-TTS
5. Gemini API changelog (Gemini 3.5 Transcribe, 26 Aug 2026) — https://ai.google.dev/gemini-api/docs/changelog
6. NVIDIA Developer Forums, Parakeet ASR on DGX Spark (ARM64 / CUDA 13 / GB10) — https://forums.developer.nvidia.com/t/multilingual-speech-to-text-stt-asr-with-nvidia-parakeet-tdt-0-6b-v3-for-the-dgx-spark/365554

---

## G2 Speaker similarity judge

### 1. DGX Spark pick

**Primary: a two-head judge — ReDimNet2-B6 (Palabra AI / IDRnD, 12.3M params, MIT) for the
identity score, plus a VoxSim-fine-tuned WavLM-ECAPA regressor for the *perceptual* score.**
BF16, ~2 GB resident for both, >1000× realtime; no aarch64 gotchas whatsoever (pure PyTorch, no
flash-attn, no custom kernels). ReDimNet2 (arXiv:2603.11841, 12 Mar 2026) ships seven variants
from 1.1M to 12.3M parameters and hits **0.287% EER on VoxCeleb1-O** at 12.3M / 13 GMACs [1] —
the best published verifier, and free to run at the top of its range.

The second head is the important part. On VoxSim (41,578 pairs, 69,409 human similarity ratings),
raw speaker-verification cosine correlates with human same/different judgements at only
**LCC 0.768 / SRCC 0.758 for ECAPA-TDNN** and **LCC 0.752 / SRCC 0.736 for WavLM-ECAPA** —
*even though WavLM-ECAPA's verification EER (0.43%) is more than twice as good as ECAPA's (0.96%)*
[2]. Fine-tuning either backbone on the human ratings lifts it to **LCC 0.829 / 0.835**. So the
better verifier is the *worse* perceptual judge, and the +0.06–0.08 LCC comes from human-rating
supervision, not from the encoder.

Alternates: **WeSpeaker ResNet293-LM** (Apache-2.0) as a third embedding for ensembling;
**ERes2NetV2 / 3D-Speaker** (Alibaba, Apache-2.0) for a fourth lineage.

### 2. Unconstrained pick

**A VoiceMOS-2026-Track-3-style ensemble**: the winning team T04 used eight models over varied
pretrained speech embeddings and loss functions, with listener-embedding conditioning and
system-level calibration, and ranked first on both speaker-similarity and accent-similarity
utterance SRCC [3]. Every component is <100M parameters. **Margin over the Spark pick:
effectively zero** — the entire winning ensemble fits in a few GB of the Spark's 128, so there is
nothing a B200 rack buys here. The scarce resource is labelled human similarity ratings and
per-listener calibration, not FLOPs.

### 3. API pick

There is no hosted speaker-similarity *judge* worth using, and I would not substitute an audio LLM
for one: in the 2026 LALM shortcut audit, open judges were almost totally position-locked on
pairwise audio comparison (Qwen3-Omni-30B-A3B-Thinking 1.00 lock rate, GPT-Audio 0.97 on
cleanness) [4]. The defensible hosted option is a **speaker-embedding endpoint** — NVIDIA Riva /
NIM speaker recognition (TitaNet) is the one with a current product page and is what Hume's Voice
Replication Benchmark (10 Sept 2026) used as its objective similarity measure. Alternates:
**Azure AI Speech Speaker Recognition**. Treat all of these as identity scores, not perceptual
ones.

### 4. Compute sensitivity — **1**

Halving EER (0.96% → 0.43%) moved human correlation the *wrong* way, by −0.016 LCC [2]. The whole
open SOTA range (0.96% → 0.287% EER) is invisible to human listeners, while fine-tuning on 69k
human ratings is worth +0.061 to +0.083 LCC. This is the most compute-insensitive category in the
entire map: the Spark pick is the ceiling, and a 12.3M-parameter model is the ceiling.

### 5. Delta vs prior pass

Prior primary (ReDimNet2-B3 as working judge, WavLM-Large embeddings as perceptual cross-check)
**stands, with two refinements**. (a) Use B6 rather than B3 — 12.3M parameters costs nothing on
this box and buys 0.42% → 0.287% EER. (b) The prior pass listed WavLM-Large as a *cross-check*
using raw cosine; the VoxSim numbers say raw WavLM cosine is the worst of the three options, so
the perceptual head must be a regressor **fine-tuned on human similarity ratings**, otherwise the
cross-check is measurably worse than the thing it is checking. No newer speaker model has appeared
since 3 September 2026.

### Sources
1. ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping, arXiv:2603.11841 (12 Mar 2026) — https://arxiv.org/abs/2603.11841
2. Ahn et al., "VoxSim: A perceptual voice similarity dataset", arXiv:2407.18505 (Table 3, Table 4) — https://arxiv.org/pdf/2407.18505
3. The VoiceMOS Challenge 2026, arXiv:2609.13792 (challenge 20 May–10 Aug 2026, results 31 Aug 2026) — https://arxiv.org/html/2609.13792
4. Auditing Protocol-Level Shortcuts in Large Audio Language Model Judges for Speech Evaluation, arXiv:2607.13477 — https://arxiv.org/html/2607.13477
5. Hume Voice Replication Benchmark (10 Sept 2026), TitaNet cosine as objective similarity measure — https://www.famulor.io/blog/voice-cloning-evaluation-identity-naturalness-quality

---

## G3 Naturalness prediction (MOS predictors)

### 1. DGX Spark pick

**Primary: UTMOSv2 (UTokyo-SaruLab, MIT) as the per-take ranker, Distill-MOS (Microsoft, 4.3M,
MIT) as the independent second opinion, and TTSDS2 (Edinburgh) as the system-level bench.**
BF16; ~8 GB resident for all three including TTSDS2's wav2vec2/HuBERT/WavLM/Whisper feature
extractors; ~100× realtime, so scoring 32 candidates per line is trivial. No aarch64 problems —
these are plain PyTorch + ONNX, no flash-attn, no bitsandbytes.

UTMOSv2 fuses wav2vec 2.0 SSL features with an EfficientNetV2 image classifier over spectrograms
and took **1st place in 7 of 16 official metrics and 2nd in the remaining 9** in VoiceMOS Challenge
2024 Track 1, the "zoomed-in" track on high-quality synthetic speech — which is exactly the
regime dubbing takes live in [1]. TTSDS2 is the only one of 16 compared metrics to reach Spearman
ρ > 0.50 on **every** dataset × subjective-score combination, with an average ρ of 0.67 [2]; it is
distributional and system-level, so it answers a different question from UTMOSv2 and you need
both.

Alternate: **Meta Audiobox Aesthetics** (WavLM encoder, four axes; Production Quality PCC
comparable to DNSMOS and UTMOSv2) for a separate "does this sound like a finished mix" axis [3].

### 2. Unconstrained pick

**A VoiceMOS-2026-style weighted ensemble with listener-identity conditioning.** Track 1's winner
T09 was a weighted ensemble of 29 scores (27 pretrained SQA models plus fine-tuned Whisper and
HuBERT encoders) reaching **ACR UTT-SRCC 0.779**; Track 2's winner T02 combined listener
embeddings, frozen UTMOS features and fine-tuned WavLM base+large with predictions averaged over
**64 listener identities**, reaching **QMOS UTT-SRCC 0.785** [4]. Note what won: ensembling and
per-listener conditioning, not a larger backbone. A 29-model ensemble of SQA models is a handful
of GB — **margin over the Spark pick is zero on hardware grounds**; the cost is training and
calibration data.

### 3. API pick

There is no dedicated hosted MOS predictor worth ranking. The honest API answer is an audio LLM
used as a MOS judge: **Gemini 3.7 Flash** (Google, 13 Aug 2026) primary, **Gemini 3.1 Pro**
(19 Feb 2026) and **gpt-audio-1.5** (OpenAI) as alternates. Calibrate expectations: the VoiceMOS
2026 organisers used a Gemini LLM-as-judge as their Track 2 baseline and **most participating
teams beat it** with small fine-tuned models [4], and SQ-LLM — a purpose-built speech-quality LLM
— reaches only **PCC ≈ 0.476** with human ratings averaged over eight perceptual dimensions [5].
Use hosted LALMs for explanations, not for the score.

### 4. Compute sensitivity — **2**

Distill-MOS distils a 478M XLS-R-SQA teacher (weighted mean correlation 0.81) into a **4.3M**
student at **0.76** — a 111× parameter reduction for 0.05 correlation [6]. Against that,
VoiceMOS 2026 was won by ensembling and listener conditioning. More compute is worth a couple of
correlation points at most; more *listeners* and more *predictors* is worth more. The one place
the Spark genuinely pays: scoring 32 takes per line with three independent predictors and gating
on their disagreement, which needs memory, not FLOPs.

### 5. Delta vs prior pass

Prior primary (UTMOSv2 per-take ranker + TTSDS2 system-level bench, Distill-MOS as cheap second
opinion, disagreement as review trigger) **stands** — that architecture is exactly what the 2026
evidence supports. But **new since 3 September**: the VoiceMOS Challenge 2026 results (released
31 Aug 2026, paper arXiv:2609.13792 in September) change the recommended *recipe*. Add
listener-identity conditioning averaged over ~64 synthetic listeners and move from
"two predictors + disagreement gate" to a small calibrated weighted ensemble, because that is what
beat every baseline across all three tracks. No new single model supersedes UTMOSv2.

### Sources
1. UTMOSv2 (sarulab-speech), MIT; VoiceMOS 2024 Track 1 results — https://github.com/sarulab-speech/UTMOSv2
2. Minixhofer et al., "TTSDS2: Robust Objective Evaluation for Human-Quality Synthetic Speech", SSW 2025 / arXiv:2506.19441 — https://arxiv.org/abs/2506.19441
3. Meta Audiobox Aesthetics, arXiv:2502.05139 — https://arxiv.org/abs/2502.05139
4. The VoiceMOS Challenge 2026, arXiv:2609.13792 — https://arxiv.org/html/2609.13792
5. SpeechLLM-as-Judges / SQ-LLM, ACL 2026, arXiv:2510.14664 — https://arxiv.org/abs/2510.14664
6. Distillation and Pruning for Scalable SSL-Based Speech Quality Assessment (Distill-MOS), arXiv:2502.05356 — https://arxiv.org/html/2502.05356v1

---

## G4 Emotion consistency judge

### 1. DGX Spark pick

**Primary: the Odyssey 2024 SER multi-attribute WavLM model (arousal / valence / dominance,
~315M, community `3loi` checkpoint) as the continuous judge, with emotion2vec+ Large
(FunAudioLLM / Alibaba, ~300M) as a categorical second opinion.** BF16, ~3 GB resident for both,
~200× realtime. Share the WavLM checkpoint with A8's source-side delivery tagger so the source
line and the dub take are scored on one scale — the judge is a *consistency* judge, so the scale
matters more than the absolute accuracy. No aarch64 issues.

If you are willing to train a head: the VoiceMOS 2026 Track 2 winner T02 (listener embeddings +
frozen UTMOS features + fine-tuned WavLM base and large, averaged over 64 listener identities)
reached **EMOS UTT-SRCC 0.758** on emotion-similarity prediction across 13 systems and 5 emotion
categories [1]. That is the current public ceiling for exactly this task and it is a ~300M WavLM
ensemble.

### 2. Unconstrained pick

**The T02 recipe at full scale**, optionally with **Audio Flamingo Next** (NVIDIA + UMD, 8B,
NVIDIA OneWay Noncommercial, April 2026, AF-Whisper encoder + Qwen2.5 backbone, 30-minute audio
with timestamp-grounded chain-of-thought) as a *describer* that explains disagreements [2].
AF-Next's predecessor AF-3 was the most accurate audio-only emotion classifier of six judges
audited in 2026 (**0.68** accuracy) [3]. Margin over the Spark pick: small, and AF-Next at 8B
BF16 (~16 GB) runs on the Spark anyway. There is no meaningful unconstrained tier here.

### 3. API pick

**Hume AI Expression Measurement (prosody model)** as the primary hosted option — it is the only
commercial product built specifically for vocal emotion rather than repurposed from a chat model.
Alternates: **Gemini 3.7 Flash** (13 Aug 2026), which the VoiceMOS 2026 organisers used as their
EMOS / valence / arousal / dominance baseline, and **gpt-audio-1.5**. Expect to beat all three
with a local WavLM head.

### 4. Compute sensitivity — **2**

Bigger is *worse* here. Audio-only emotion accuracy across the six audited judges:
**Audio-Flamingo-3 (7B) 0.68 > Gemini-3-Flash 0.47 > Qwen3-Omni-30B-A3B-Instruct 0.35 >
GPT-Audio 0.29 > Voxtral-Small-24B 0.17 > Qwen3-Omni-30B-A3B-Thinking 0.13** [3] — a 24B and a
30B MoE both land near chance while a 300M WavLM head trained on MSP-Podcast does the job. The
only reason this is a 2 and not a 1 is that the ceiling (EMOS SRCC 0.758) is far from solved and
ensembling helps.

### 5. Delta vs prior pass

Prior primary (Odyssey 2024 WavLM multi-attribute AVD model, coarse banded comparison between
source line and dub take, shared checkpoint with A8, emotion2vec+ Large as categorical second
opinion) **stands**. Two things are new since 3 September. (a) VoiceMOS 2026 Track 2
(results 31 Aug 2026) is the first public, task-matched emotion-similarity leaderboard and gives
you a target number (0.758 UTT-SRCC) and a winning recipe (listener conditioning). (b) The LALM
shortcut audit (arXiv:2607.13477) should **delete** the prior pass's "niche: Speech-LLM as
emotion judge" option: five of six judges dropped to ≤0.10 emotion accuracy when handed a wrong
specialist label in the prompt, which also means you must **not** put your own A8 tagger's label
into a LALM reviewer's prompt — it will simply echo it.

### Sources
1. The VoiceMOS Challenge 2026 (Track 2, Emotional TTS), arXiv:2609.13792 — https://arxiv.org/html/2609.13792
2. Audio Flamingo Next, arXiv:2604.10905 (Apr 2026); model card — https://huggingface.co/nvidia/audio-flamingo-next-hf
3. Auditing Protocol-Level Shortcuts in LALM Judges for Speech Evaluation, arXiv:2607.13477 — https://arxiv.org/html/2607.13477
4. emotion2vec+ Large (~300M) model card — https://huggingface.co/emotion2vec/emotion2vec_plus_large
5. Odyssey 2024 Speech Emotion Recognition Challenge: Dataset, Baseline Framework and Results — https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf

---

## G5 Lip-sync scoring

### 1. DGX Spark pick

**Primary: PEAVS (Amazon, ECCV 2024, CC-BY-4.0) for the human-calibrated per-shot score, plus
Synchformer (MIT) for the millisecond offset, with SyncNet LSE-C / LSE-D reported alongside purely
for comparability with published numbers.** ~4 GB resident for all three; throughput is bounded by
video decode, not the models. PEAVS is the only AV-sync metric trained on human opinion scores:
100+ hours of human annotation covering nine synchronization-error types, reaching **Pearson 0.79
at set level and 0.54 at clip level** against human labels and a **50% relative gain over a
Fréchet-based AV-sync baseline** [1]. Synchformer scores 0.64-second segments and reports
**86.6 / 99.6 Acc@1 / Acc@1±1-class on LRS3** for offset classification [2].

Spark gotchas: the models are small and safe, but the *video* path is where aarch64 bites —
`decord` has no aarch64 wheel and torchvision's video backend is unreliable on DGX OS. Use PyAV or
NVIDIA DALI's arm64 build with NVDEC, and pin OpenCV to a source build. No flash-attn dependency.

Alternate: **AV-HuBERT sync expert** (AVSu / AVSm / AVSv, Yaman et al., CVPR 2024 NTIRE workshop)
as the confidence head. It is measurably more trustworthy than SyncNet, which is **not shift
invariant** — LSE-C/LSE-D swing dramatically under small horizontal face translations and
fluctuate even on ground-truth clips, while the AV-HuBERT metrics stay flat [3].

### 2. Unconstrained pick

**PEAVS + Synchformer + AV-HuBERT-Large as a three-metric panel, with a frontier video LLM doing
shot-level triage on the disagreements.** The margin over the Spark pick is small and bounded by
the metric, not the hardware: 0.54 clip-level Pearson is the best published human agreement for
*any* automatic AV-sync metric, and nothing larger improves on it. The real lever is structural,
as the prior pass correctly said: cut on shot boundaries, score per face track, and emit offsets in
milliseconds rather than a single clip-level number.

### 3. API pick

No vendor sells a calibrated lip-sync metric. The best hosted proxy is **Gemini 3.7 Flash**
(Google, 13 Aug 2026), which received Google's agentic video-understanding mode on 1 Sept 2026 —
notably that mode covers Gemini 3.7 Flash, 3.6 Flash and 3.5 Flash-Lite but **not** Gemini 3.1 Pro
[4,5]. Alternates: **Gemini 3.1 Pro** (19 Feb 2026, native video+audio input, 1M context) for long
single-pass review. Use these for "is the mouth obviously wrong in this shot", not for a score.

### 4. Compute sensitivity — **2**

The ceiling is the metric's agreement with viewers, and the best number in the literature is
PEAVS's **0.54 clip-level Pearson / 0.79 set-level** [1]; no larger model has beaten it, and the
documented failure mode of the incumbent (SyncNet shift-variance) is an architecture bug, not a
capacity limit [3]. What moves the number is shot segmentation and per-face-track scoring.

### 5. Delta vs prior pass — **CHANGED**

The 3 September pass named Synchformer + the AV-HuBERT sync expert as joint primaries and did not
mention PEAVS. **PEAVS should be promoted to co-primary**: it is the only metric in this category
grounded in viewers' opinion scores, which is exactly what a *judge* needs, whereas Synchformer and
AV-HuBERT give you a well-behaved offset and a confidence but no calibration to human perception.
This is a correction of an omission rather than a new release — nothing has shipped in AV-sync
scoring since 3 September 2026, and Synchformer's last checkpoint is still January 2024.

### Sources
1. PEAVS: Perceptual Evaluation of Audio-Visual Synchrony Grounded in Viewers' Opinion Scores, ECCV 2024, arXiv:2404.07336 — https://arxiv.org/abs/2404.07336
2. Synchformer (Iashin et al.), MIT licence, LRS3 / VGGSound-Sparse results — https://github.com/v-iashin/Synchformer
3. Yaman et al., "Audio-Visual Speech Representation Expert for Enhanced Talking Face Video Generation and Evaluation", CVPR 2024 NTIRE W, arXiv:2405.04327 — https://arxiv.org/html/2405.04327
4. Gemini API changelog — https://ai.google.dev/gemini-api/docs/changelog
5. Gemini 3.1 Pro model card (19 Feb 2026; text, image, audio, video input; 1M context) — https://deepmind.google/models/model-cards/gemini-3-1-pro/

---

## G6 Multimodal review

### 1. DGX Spark pick

**Primary: Qwen3-Omni-30B-A3B-*Instruct* (Alibaba, 30B total / ~3B active MoE, Apache-2.0),
NVFP4 or FP8, ~40 GB resident including vision/audio encoders and KV for a one-minute clip.**
This is still the newest *open-weights* omni model as of 18 Sept 2026 — the Qwen3-Omni HuggingFace
collection has had nothing added since September 2025, and **Qwen3.5-Omni (30 Mar 2026) is API-only:
Alibaba published neither weights nor a licence for Plus or Flash** [1,2]. Expect roughly 40–60
tok/s decode single-stream by analogy with a 27B-class NVFP4 checkpoint measured at **50.7 tok/s
under SGLang on a DGX Spark** [3]; with only ~3B active parameters the MoE should sit at or above
that, and prefill over video frames is the real cost.

**Spark gotchas, verified.** vLLM issue #36821 (opened 11 Mar 2026) is still open: stock wheels
ship sm_120 kernels only and vLLM crashes at start-up on GB10. sm_120 and sm_121 are binary
compatible, so an sm_120 build *does* run — use NVIDIA's arm64 NGC containers and the official
DGX Spark vLLM/SGLang playbooks, or the community `dgx-spark-vllm` package, rather than pip-install
vLLM [4,5]. For MoE you must pass `--moe-backend flashinfer_b12x` or lose ~2.5× throughput [3].
`flash-attn` has no upstream aarch64 sm_121 wheel (community wheels exist); `bitsandbytes` is
unavailable on aarch64, so quantise with NVIDIA ModelOpt (NVFP4), not bnb; and GB10's
101,376-byte shared-memory ceiling constrains Triton kernel configs [5].

Alternate: **Audio Flamingo Next, 8B** (NVIDIA OneWay Noncommercial — *non-commercial only*,
Apr 2026), ~16 GB BF16, as the audio-only describer; 30-minute context with timestamp-grounded
chain-of-thought, and beats Gemini 2.5 Pro on long-audio benchmarks [6]. **Explicitly avoid
Qwen3-Omni-30B-A3B-Thinking**: same base model, but it was **100% position-locked** on pairwise
audio judging and scored 0.13 on audio-only emotion accuracy versus 0.35 for Instruct [7].

### 2. Unconstrained pick

**The same Qwen3-Omni-30B-A3B-Instruct at BF16 with AudioJudge-style decomposition and 16–64-way
self-consistency, ensembled with AF-Next.** This is an unsatisfying answer and it is the true one:
there is **no larger open-weights video+audio model** than Qwen3-Omni-30B-A3B, so the unconstrained
open tier is the Spark tier with more samples, and the **margin over the Spark pick is small**.
The decomposition is what earns the gain, not the hardware: AudioJudge (EACL 2026) reaches
**up to 0.91 Spearman with human preferences at system level** by splitting the judgement into
separate lexical-content, speech-quality and paralinguistic judges and using audio concatenation
plus in-context learning [8]. The genuine quality gap is to the API tier, not to a bigger box.

### 3. API pick

**Primary: Gemini 3.7 Flash (Google, 13 Aug 2026).** It scores 56 on the Artificial Analysis
Intelligence Index at high reasoning against Gemini 3.1 Pro's upper-40s, 85.2% average on Roboflow
Vision Evals (#3 of 53) against 83.3% (#6 of 53), and it — not the Pro tier — received Google's
agentic video-understanding mode on 1 Sept 2026 [9,10]. Alternates: **Gemini 3.1 Pro**
(19 Feb 2026; native text/image/audio/video input, 1M-token context, 64K output) for long
single-pass review of a whole reel [11]; **gpt-audio-1.5** (OpenAI) for audio-only second opinion.
**Claude is not an option for this category** — all current Claude models are text+image in, with
no video input, so it can only review a rendered transcript-plus-frames artefact.

**Judge-vs-human agreement, the numbers that matter.** A 2026 reliability study of Gemini-family
LALM judges over 209 sessions found the LALM–human Spearman ρ departs from **pairwise human–human
ρ by at most 0.07 on five of eight dimensions**, ≥60% agreement within one Likert point on six of
eight dimensions (>89% on three), and it matched or exceeded human defect sensitivity on 45 of 48
defect-dimension cells — at roughly **two orders of magnitude lower cost** than human raters. The
critical caveat: cross-model replication on Gemini 3.5 Flash and Gemini 3.1 Pro Preview showed
**calibration drift despite comparable ranking ability**, so a model swap must be re-validated on
calibration, not assumed from rank correlation [12]. For absolute scoring the state of the art is
still weak — SQ-LLM reaches PCC ≈ 0.476 with humans across eight perceptual dimensions [13].

### 4. Compute sensitivity — **4**

This is the one judge where the frontier tier is measurably better. On pairwise judging,
Gemini-3-Flash showed a position-lock rate of **0.05–0.20** (i.e. it actually listened) while
GPT-Audio hit 0.97 and Qwen3-Omni-Thinking 1.00 [7]; and human-agreement calibration within 0.07 ρ
of human–human is demonstrated only for the Gemini family [12]. It is a 4 rather than a 5 because
the audit is explicit that **post-training, not parameter count, drives robustness** — two
variants of the *same* 30B base differ by 0.22 in emotion accuracy and 0.65 in position-lock rate —
and because AudioJudge's 0.91 system-level Spearman comes from decomposition and in-context
learning, both of which are free.

### 5. Delta vs prior pass — **CHANGED, both halves**

The 3 September pass recommended **Gemini 3.5 Flash** as primary reviewer with **Qwen3.5-Omni**
self-hosted for regulated/on-premises customers. Both need updating.
(a) **Gemini 3.7 Flash (13 Aug 2026) supersedes 3.5 Flash** and is the tier that got agentic video
understanding on 1 Sept 2026; 3.5 Flash is also one of the checkpoints where the reliability study
observed calibration drift, so if you keep a 3.5-era prompt you must re-run the calibration set.
(b) **Qwen3.5-Omni cannot be the on-premises answer — it has no published weights and no licence.**
The correct self-hosted pick is **Qwen3-Omni-30B-A3B-Instruct** (Apache-2.0), and specifically not
the Thinking variant the position-lock data rules out. The prior pass's transcript-plus-non-verbal-
token text review remains a good cheap third opinion and is unaffected.

### Sources
1. Qwen3-Omni collection (30B-A3B Instruct / Thinking / Captioner, Apache-2.0, Sept 2025; nothing newer) — https://huggingface.co/collections/Qwen/qwen3-omni
2. Qwen3.5-Omni Technical Report, arXiv:2604.15804 (17 Apr 2026) — API-only, no weights published — https://arxiv.org/abs/2604.15804
3. Running Qwen3.8-27B on DGX Spark (NVFP4 under SGLang, 50.7 tok/s; `--moe-backend flashinfer_b12x`) — https://blog.kubesimplify.com/qwen3-8-27b-on-dgx-spark
4. vLLM issue #36821, "No sm_121 (Blackwell) support on aarch64 — NVIDIA DGX Spark" (opened 11 Mar 2026, open) — https://github.com/vllm-project/vllm/issues/36821
5. awesome-dgx-spark: engines, aarch64 gotchas (flash-attn, bitsandbytes, 101,376-byte smem ceiling) — https://github.com/bidual/awesome-dgx-spark
6. Audio Flamingo Next, arXiv:2604.10905 / model card (8B, NVIDIA OneWay Noncommercial) — https://huggingface.co/nvidia/audio-flamingo-next-hf
7. Auditing Protocol-Level Shortcuts in LALM Judges for Speech Evaluation, arXiv:2607.13477 — https://arxiv.org/html/2607.13477
8. AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation, EACL 2026, arXiv:2507.12705 — https://aclanthology.org/2026.eacl-long.168/
9. Gemini 3.1 Pro vs Gemini 3.7 Flash comparison (Artificial Analysis index, Roboflow Vision Evals) — https://benchlm.ai/compare/gemini-3-1-pro-vs-gemini-3-7-flash
10. Gemini API changelog (Gemini 3.8 Flash 2 Sept 2026, Gemini 3.8 Live 15 Sept 2026) — https://ai.google.dev/gemini-api/docs/changelog
11. Gemini 3.1 Pro model card (19 Feb 2026) — https://deepmind.google/models/model-cards/gemini-3-1-pro/
12. A Reliability Assessment of LALM Audio Judges for Full-Duplex Voice Agents, arXiv:2607.07985 — https://arxiv.org/html/2607.07985
13. SpeechLLM-as-Judges / SQ-LLM, ACL 2026, arXiv:2510.14664 — https://arxiv.org/abs/2510.14664
