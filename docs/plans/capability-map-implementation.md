# Plan: implement the 41-capability inference map

Source: "OpenDub Model Research" artifact (7 phases A–G, 41 categories, verdict + bench plan each).
Goal: every category exists in code as a configurable **capability**; a **Simple mode** runs only
the bare-minimum set with zero decisions; **Advanced mode** exposes all 41 for enable/disable,
provider choice and tuning.

## 1. Core design

### 1.1 Capability registry (new `server/app/capabilities.py`) — the single source of truth

Today the pipeline is 6 hard-coded provider kinds (`PipelineConfig.separation/asr/...`). That
cannot stretch to 41. Replace it with a declarative registry:

```python
class Capability(BaseModel):
    id: str            # "A1" … "G6"
    kind: str          # provider kind slug: "separation", "enhance", "song_detect", …
    phase: Literal["A","B","C","D","E","F","G"]
    name: str; summary: str
    stage: StageKey    # macro stage the step runs inside
    order: int         # position within the stage
    tier: Literal["core","recommended","advanced","experimental"]
    needs: list[str]   # hard deps  (F1 needs B1,B2,B4; C6 needs A4 words; A5 needs subtitle file…)
    uses: list[str]    # soft deps  (E1 uses B2 if present; A7 uses A2 cleanliness…)
    default_provider: str        # always a zero-dep builtin, so the server boots with no ML deps
    params: list[ConfigField]    # capability-level knobs (thresholds, N candidates, presets)
```

- The 6 existing kinds keep their slugs and provider ids (`separation`=A1, `asr`=A4,
  `diarization`=A6, `translation`=C2, `tts`=D1, `lipsync`=F1) → no provider file is renamed.
- `ProviderKind` becomes `str` validated against the registry; `providers/base.py` gains one ABC
  per new kind (small, typed `run()` signatures) and `load_all()` walks every sub-package.
- **Every capability ships a builtin provider** (`<kind>.builtin` or `<kind>.off`) that is pure
  Python/ffmpeg. Tests and Simple mode never need an ML dependency — same contract as today.

### 1.2 Config model (`models.py`, mirrored in `web/src/types.ts`)

```python
class CapabilityChoice(BaseModel):
    enabled: bool; provider_id: str; options: dict; params: dict

class PipelineConfig(BaseModel):
    mode: Literal["simple","advanced"] = "simple"
    preset: Literal["minimal","balanced","max","custom"] = "minimal"
    runtime: Literal["builtin","local","cloud"] = "builtin"   # simple-mode provider resolver input
    lipsync: bool = False; subtitles: bool = False            # the only two simple-mode feature toggles
    capabilities: dict[str, CapabilityChoice] = {}            # sparse overrides, keyed "A1".."G6"
```

- `resolve(config) -> dict[cap_id, ResolvedChoice]`: preset → runtime provider table → user
  overrides → **dependency closure** (enabling F1 pulls in B1/B2/B4; disabling A4 words disables
  C1/E1 pause-mapping with a visible reason). One function, used by orchestrator, API and UI.
- Migration: a `model_validator(mode="before")` lifts old manifests' six fields into
  `capabilities`. `configs/settings.yaml` `defaults:` keeps working.

### 1.3 Presets

| Preset | Enabled | Intent |
|---|---|---|
| **minimal** (Simple default) | A1 A4 A6 · C1 C2 · D1(+D2 intrinsic) · E1 E4 | Today's pipeline + the free wins (word timings, rule segmentation). No extra models. |
| **balanced** | minimal + A2 A3 A7 · C4 C5 · D3 D7 · E2 · G1 G2 (+ B1 B2 B4 F1 F2 G5 if lip sync on; C6 if subtitles on) | Everything that raises quality without a human in the loop. |
| **max** | all except E5 (deferred), D8 (bench-only) and F3/F4-inpaint (experimental) | Best quality; slow; may need cloud keys. |
| **custom** | whatever Advanced mode sets | Auto-selected when any override diverges from a preset. |

Simple mode asks only: languages · runtime (Built-in / My GPU / Cloud keys) · preset · lip sync? ·
subtitles?. A `RUNTIME_DEFAULTS[runtime][cap_id]` table picks the provider (falls back down
cloud→local→builtin when `available()` is false, and says so in the stage detail).

### 1.4 Execution: macro stages stay, capabilities become steps

Stage order becomes `ingest → separate → analyze* → transcribe → translate → synthesize → mix →
lipsync → review* → render` (*new). Each stage runner becomes a small step loop:

```python
for cap in registry.steps(stage): 
    if resolved[cap.id].enabled: await STEP_IMPL[cap.id](ctx, provider, progress)
```

- Steps write **artifacts** (`analysis/regions.json`, `analysis/shots.json`,
  `analysis/face_tracks.json`, `analysis/words.json`, `subtitles/*.srt|vtt`, `render/manifest.c2pa.json`…)
  and mutate the manifest. Per-step status (`done/skipped/error + detail`) recorded under
  `StageState.steps[cap_id]` so RunView can show 41 dots under 10 stage chips.
- A failing **non-core** step degrades (warn + continue); a failing core step fails the stage.
- `ModelManager` (new, `providers/_runtime.py`): one-GPU-model-at-a-time load/unload with LRU —
  required on the 16 GB RTX 5060 Ti once >3 local models participate.

### 1.5 Domain model additions

`Word{start,end,text,speaker,conf}` on Segment · `Region{start,end,type: song|walla|laugh|tv|nonverbal, action: skip|passthrough|dub}` ·
`Shot`, `FaceTrack` · `Segment.delivery{rate,loudness,f0,arousal,note}` · `Segment.on_screen`,
`Segment.subtitle` · `Take.scores{wer,similarity,mos,emotion_delta,duration_error}` + `Take.rejected_reason` ·
`Segment.flags[]` (QC issues from C4/G1–G6, drives a review queue) · `Project.glossary[]`, `Project.lexicon[]` ·
global `data/voicebank/` (A7) · `Project.content_type` (B4) · `Project.delivery_preset` (E4/C6).

## 2. The 41 capabilities

Builtin = zero-dep implementation shipped first. "First real" = provider(s) from the research verdicts.

| ID | Kind slug | Stage | Tier | Builtin (ships in M-x) | First real providers |
|---|---|---|---|---|---|
| A1 | separation | separate | core | passthrough; **bed = mix − dialogue** rule; supplied-M&E / 5.1-centre skip | Bandit v2 (`multi`), Demucs (existing), AudioShake, LALAL.AI (existing) |
| A2 | enhance | transcribe | recommended | off; quality-scored window selection (SNR/clipping heuristics) | Resemble Enhance (+MelBand-RoFormer dereverb), DeepFilterNet3; identity gate via A7 |
| A3 | region_detect | analyze | recommended | cross-stem energy heuristic; operator-toggleable regions | CED-base, inaSpeechSegmenter, audio-LLM for ambiguous band |
| A4 | asr | transcribe | core | mock (existing) + **word timestamps on** in faster-whisper | WhisperX-style alignment, Qwen3-ASR + ForcedAligner, Scribe v2 |
| A5 | align | transcribe | recommended | proportional word interpolation; **subtitle import** (SRT/ASS) + NW anchoring + offset/rate repair + non-verbatim flag | MFA 3.x, torchaudio MMS_FA, Qwen3-ForcedAligner |
| A6 | diarization | transcribe | core | single speaker; **word-level attribution + overlap mask** | pyannote community-1 (swap for 3.1), Precision-2 (existing API) |
| A7 | speaker_embed | transcribe | recommended | MFCC-stat embedding (test-grade); composite reference selector; cross-project voice bank w/ abstain | ReDimNet-B3 (WeSpeaker/ONNX), ERes2NetV2 |
| A8 | delivery | transcribe | advanced | tier-one DSP: rate, rel. loudness, F0, pause profile → fills `emotion` | Gemini / Qwen3-Omni delivery notes, emotion2vec+ |
| B1 | active_speaker | analyze | recommended* | off (⇒ lip sync whole-frame as today) | LR-ASD (MIT), LoCoNet; SCRFD front-end, min-face-height gate |
| B2 | shots | analyze | recommended* | ffmpeg `scdet` | TransNetV2, PySceneDetect |
| B3 | ocr | analyze | advanced | off | PP-OCR, VLM escalation (Gemini Flash / Qwen3-VL) → glossary seeding |
| B4 | content_type | analyze | recommended* | operator picks (today) | deepghs/anime_classification, SigLIP2 probe, VLM third vote; **blocks F1 on animation** |
| C1 | segmentation | transcribe | core | pause + punctuation + length-budget splitter/merger over words | SaT/wtpsplit + pause selector, LLM repair for unsplittable lines |
| C2 | translation | translate | core | mock (existing) | existing LLM providers upgraded: scene context, show bible, glossary, N candidates, iterative tightening, **per-language syllable budget table** |
| C3 | viseme_rerank | translate | experimental | off; cheap viseme-agreement scorer over C2 candidates, operator weight | PS-Comet-style DTW |
| C4 | qe | translate | recommended | rule checks (lang-ID, length ratio, glossary, numbers) | MetricX-QE, LLM error-span judge (must differ from C2 model) |
| C5 | pronunciation | translate | recommended | project lexicon + respelling + number/unit normalisation | misaki, nemo-text-processing, CharsiuG2P, LLM for flagged names |
| C6 | subtitles | translate | recommended† | CPS/CPL line-breaker over translation, SRT/VTT export, delivery profiles | LLM condensation loop (AppTek recipe), B2 shot snapping |
| D1 | tts | synthesize | core | mock (existing) | F5 (existing), ElevenLabs (existing), **Qwen3-TTS, VoxCPM2** (Apache-2.0); N-candidate + accept test |
| D2 | style | synthesize | core | segment-as-style-ref (existing), made an explicit toggle/threshold | IndexTTS2 dual-prompt; engine instruction tags |
| D3 | duration_control | synthesize | recommended | pass `target_duration` through; free-run → safe band → escalate to C2 rewrite | IndexTTS2, MOSS-TTS, ElevenLabs speed |
| D4 | lang_router | synthesize | advanced | per-language engine routing table + acceptance status | VoxCPM2, IndicF5, Chatterbox Multilingual |
| D5 | stock_voice | synthesize | advanced | espeak/beep (existing mock) as "no-clone" voice; per-speaker `voice_mode: clone|stock` | Kokoro-82M, Azure Speech, Cartesia |
| D6 | voice_convert | synthesize | advanced | off; upload-a-human-take path in Inspector | Chatterbox VC, Vevo2, ElevenLabs Voice Changer |
| D7 | nonverbal | synthesize | recommended | regions tagged nonverbal ⇒ **pass-through splice** of original; inline tags kept out of C2/D1 text | CED tagger, laughter verifier; Step-Audio-EditX / ElevenLabs v3 tags |
| D8 | s2st | — (bench only) | experimental | none; `bench` route only | UniSS, Hibiki-Zero scorecard |
| E1 | timing_fit | mix | core | atempo (existing) → **pause-mapped non-uniform time map**, no hard trim, spill into silence, >8 % ⇒ regeneration event | Signalsmith Stretch, Rubber Band R3 (opt-in, GPL) |
| E2 | bandwidth | mix | recommended | off (plain resample, today) | AP-BWE; conditional restoration gated by G1/G2 |
| E3 | scene_match | mix | advanced | per-line energy match + LTAS EQ match | parametric T60/DRR estimate + FDN reverb; effect chains (phone/radio) |
| E4 | loudness | mix | core | static chain (existing) → two-pass measure, **delivery presets** (R128/A85/Netflix/web), sidechain duck to target DBR, stem export | pyloudnorm/libebur128; Auphonic (bench) |
| E5 | provenance | render | deferred | signed JSON provenance manifest + MP4 metadata tag | c2pa-python manifest; AudioSeal watermark (after E1) + detector CLI |
| F1 | lipsync | lipsync | recommended* | none (existing) | existing Wav2Lip/LatentSync/Replicate wrapped in a **per-shot, per-face-track router**; sync API escalation; MuseTalk preview |
| F2 | mouth_restore | lipsync | advanced | feathered composite + grain/colour match | DVFace w/ ArcFace identity veto |
| F3 | anim_mouth | lipsync | experimental | **decline** (B4-gated) + Rhubarb-style viseme track export | — (research watch) |
| F4 | text_replace | render | advanced | overlay subtitle-style track + editing-template export from B3 | flagged inpaint path (Qwen-Image-Edit) |
| G1 | judge_intelligibility | synthesize | recommended | duration/peak/truncation check (existing guard, generalised) | independent ASR round-trip (Parakeet/Canary vs Whisper), hallucination flags |
| G2 | judge_similarity | synthesize | recommended | shares A7 builtin embedding | ReDimNet (shared with A7), WavLM cross-check; **argmax among passing takes** replaces newest-wins |
| G3 | judge_naturalness | synthesize | advanced | off | UTMOSv2 + Distill-MOS disagreement flag; TTSDS2 in bench |
| G4 | judge_emotion | synthesize | advanced | A8 DSP deltas (advisory only) | WavLM A/V/D banded compare, emotion2vec+ |
| G5 | judge_sync | review | recommended* | off / abstain | Synchformer per-shot offset; SyncNet reported for comparability |
| G6 | reviewer | review | advanced | off | Gemini (video+audio) single-clip rubric; Qwen-Omni local; flagged + 5–10 % audit |

\* only when lip sync is on. † only when subtitles are on.

## 3. Milestones (each is shippable, tests green with builtins only)

**M0 — Foundation (serial, blocks everything; done by the main session, not delegated)**
Registry, `PipelineConfig` v2 + migration, `resolve()` with dependency closure, step loop in
`stages.py`/`orchestrator.py`, `analyze` + `review` stages, per-step state, `ModelManager`,
API (`GET /api/capabilities`, `PATCH project.pipeline`, `GET /api/presets`), `types.ts`/`client.ts`/`store.ts`.
Web: Settings splits into **Simple** (5 controls) and **Advanced → Capability map** (7 phase
columns × cards: toggle, provider select, auto-rendered `ConfigField` form, dependency/“why
disabled” notes, tier badge); RunView shows steps under stages. Update ARCHITECTURE.md/README.

**M1 — Timing spine:** A4 words · C1 · A6 word-level + overlap mask · A5 + subtitle import · E1 pause map · D3 plumbing.
**M2 — Judges + take selection:** G1 · G2/A7 shared embedder · A2 selection+enhance · N-candidate synth loop · G3 · `Segment.flags` + review-queue filter in TranscriptList.
**M3 — Text:** C2 multi-pass + glossary/bible/syllable table · C4 · C5 lexicon UI · C6 + subtitle export · B3.
**M4 — Audio analysis/post:** A3 regions (timeline lane) · D7 splice · A8/G4 · E2 · E3 · E4 presets+stems.
**M5 — Picture:** B2 · B4 gate · B1 · F1 per-shot router · F2 · G5 · F4 overlay · F3 decline/export · C3.
**M7 — Provenance (deferred):** E5 C2PA manifest · AudioSeal watermark · detector CLI.
**M6 — Voice breadth + review:** Qwen3-TTS/VoxCPM2/IndexTTS2 providers · D4 router · D5 Kokoro/Azure · D6 · G6 · D8 bench route.

Every capability also lands a `server/bench/metrics/` entry implementing the artifact's bench
metric for it (null-with-reason when GT/deps are absent — existing convention), so "is the real
provider better than builtin" is measurable.

## 4. Delegation (Opus subagents)

M0 defines the contracts; after it merges, M1–M6 are mostly *one-file-per-provider + one step
function + one test*, which parallelises cleanly. Per milestone: 3–4 Opus agents in isolated
worktrees, split by phase letter so files don't collide (e.g. M4: agent-A `A3+D7`, agent-B
`A8+G4`, agent-C `E2+E3`, agent-D `E4+E5`), each given the capability's verdict text from the
artifact, the ABC, and the acceptance test. Main session integrates, runs `pytest` + `vitest` +
ruff/tsc, and runs `/code-review` before each milestone commit. Web UI work stays in one agent
per milestone (shared `types.ts`/`store.ts`).

## 5. Risks / decisions baked in

- **Licences:** NC-licensed models (F5-TTS, Wav2Lip, CodeFormer, AV-HuBERT, Sortformer) are
  never a preset default; `ProviderMeta` gains `license` + `commercial_ok`, shown as a badge.
- **Model availability:** several verdict picks (GateFusion, UniPASE, DVFace weights, sync-3 API)
  may not be installable; each provider is verified at implementation time and swapped for the
  verdict's named fallback (LR-ASD, Resemble Enhance, classical composite…) if not.
- **VRAM (16 GB):** `max` preset locally is sequential-load only; Simple/local never loads more
  than one model at a time.
- **Scope honesty:** D8 is bench-only and F3 is "decline + export" by the research's own
  verdict — both still appear in the map as configurable entries, marked experimental.
- **E5 deferred (user decision, 2026-09-17):** Article 50 marking / provenance / watermarking is
  done later, after M6. E5 is registered in the map from M0 (so the map stays at 41) with an
  `off` builtin, tier `deferred`, in no preset; its implementation is milestone **M7**.
