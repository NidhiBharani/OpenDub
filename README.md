# OpenDub

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Self-hosted, voice-preserving video dubbing studio.**

OpenDub takes a video (anime first), transcribes it, translates it, and re-voices every line in
English — keeping each character's **own voice and emotional delivery**, with optional lip sync.
A video-editor-style web UI lets you scrub the timeline, check every translated line, edit it,
and re-voice a single segment in one click.

Every AI stage is pluggable: run open-source models on your own GPU, plug in cloud API keys, or
use the built-in zero-dependency fallbacks.

```
 upload → ingest → separate → transcribe → translate → synthesize → mix → lip sync → render
            │         │           │            │            │         │       │         │
         ffmpeg    Demucs    Faster-Whisper  Ollama/vLLM  F5-TTS   fit+mix Wav2Lip   dubbed.mp4
                 (or none)   (or Whisper    (or Claude /  (or Eleven-      (or Latent-
                              API, mock)     GPT / DeepL)  Labs, mock)      Sync, none)
```

## How it keeps voice and emotion

- **Voice** — each speaker gets a reference clip built from their own clearest lines. The TTS
  stage zero-shot-clones that voice: F5-TTS locally, or ElevenLabs in the cloud.
- **Emotion** — every line is synthesized with its *own source audio* as the style reference, so
  the original delivery (shouting, whispering, sobbing) carries into the dub. A free-text emotion
  hint also steers the translation.
- **Timing** — translations are written to a time budget (~15 characters per second) so each dub
  line fits its slot. Any leftover mismatch is gently time-stretched and shown in the UI as a
  ×rate badge.

## Quick start

Requirements: Python 3.11+ ([uv](https://docs.astral.sh/uv/)), Node 18+, ffmpeg. Optional: an
NVIDIA GPU for local models, `espeak-ng` for a nicer placeholder voice.

```bash
make setup        # server venv + web deps
make dev          # API on :8000, UI on :5173
```

Open http://localhost:5173, drop in a video, and press **Run pipeline**.

Out of the box every stage uses a zero-dependency fallback (silence-based segmenter, mock
translator, placeholder voice), so you can try the full workflow — timeline, editing, re-voicing,
render — without installing any model. Switch stages to real providers in **Settings** when
you're ready.

Production-ish serving: `make serve` (everything from FastAPI on :8000) or
`docker compose up --build`.

## Providers

Each pipeline stage delegates to a provider you pick per project in Settings. Included:

| Stage | Self-hosted (OSS) | Cloud API | Zero-dep fallback |
|---|---|---|---|
| Separation | Demucs (htdemucs) | LALAL.AI | passthrough |
| Transcription | Faster-Whisper | OpenAI Whisper | silence segmenter |
| Diarization | pyannote 3.1 | pyannoteAI | single speaker |
| Translation | any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio…) | Claude, OpenAI, DeepL | mock |
| Voice (TTS + cloning) | F5-TTS | ElevenLabs | beep/espeak |
| Lip sync | Wav2Lip, LatentSync | Replicate¹ | none |

¹ Cloud lip-sync uses human-face models — not for 2D anime (see `lipsync.replicate`). Every other
stage has a fully cloud path, so OpenDub can run **without a local GPU** on API keys alone.

Install the local model stack with `make gpu-extras` (Faster-Whisper + Demucs + F5-TTS).

Settings live in `configs/settings.yaml` — written by the UI, documented in
`configs/providers.example.yaml`. Environment variables override the file, e.g.
`OPENDUB_TRANSLATION_ANTHROPIC_API_KEY=sk-ant-…`. Secrets are write-only through the API and
masked everywhere else.

**Adding a provider is one file**: subclass the stage ABC in `server/app/providers/<kind>/`,
declare `meta` (the Settings UI renders its config form automatically), implement one method,
decorate with `@register`. Details in `ARCHITECTURE.md`.

## The editor

- **Timeline** — waveforms, one lane per speaker, dub lane. Click to seek, wheel to pan,
  Ctrl+wheel to zoom, drag segment edges to retime.
- **Player** — toggle original ↔ dubbed audio (`1`/`2`), Space to play, ↑/↓ to walk segments.
- **Inspector** — edit the transcript or translation (with a fits-the-slot meter), set an
  emotion hint, audition takes, **Re-translate** / **Re-voice** just that line.
- **Transcript** — the whole episode as a list, synced to playback. The fastest way to review.

Edits mark exactly the right downstream work dirty (edit a translation → that line re-voices →
mix and render re-run), and the pipeline only redoes dirty work.

## Project layout

```
server/   FastAPI backend — pipeline, providers, jobs, media
web/      React + TypeScript editor UI
configs/  provider settings
data/     per-project media + manifest.json
```

`ARCHITECTURE.md` has the full contracts: domain model, provider system, HTTP API.

## Tests

- `make test` — backend end-to-end pipeline over a synthesized video (zero-dep providers: no
  GPU, no network, no downloads) + frontend component tests (Vitest, headless jsdom).
- `make e2e` — drives the real app in headless Chromium (Playwright), library → editor →
  settings, and captures screenshots.

## Benchmarking & quality assessment

The pipeline now records backend quality reports, retries failed speech and duration mismatches,
analyzes source emotion/delivery, preserves overlapping voices, and validates encoded output.
See [backend quality](docs/backend-quality.md) for configuration, limits, and regression gates.

`server/bench/` measures how each pipeline stage actually performs. `make bench` runs the real
pipeline over every case in `data/benchmarks/cases.yaml` and writes a scored report.

Each stage gets **reference-free** metrics (always) plus **ground-truth** metrics when you supply
reference files:

| stage | reference-free | ground truth (optional files) |
|---|---|---|
| separation | vocal-vs-background loudness gap | SI-SDR (`ref_vocals.wav`) |
| transcribe | speech coverage, segments/min | WER / CER (`ref_transcript.jsonl`) |
| translate | time-budget compliance, LLM judge (adequacy + fluency, via your Ollama) | chrF (`ref_translation.jsonl`) |
| tts | degenerate-take rate, duration-clamp rate | round-trip WER/CER and speaker similarity with configured local models |
| mix | integrated LUFS vs target, lead-in noise floor, dialogue-vs-bed margin | — |
| lipsync | *(SyncNet slot — implemented once a live-action clip exists)* | — |

Plus per-stage wall time, peak VRAM, and realtime factor on every run. Install the ground-truth
extras with `uv pip install -e '.[bench]'` (jiwer, sacrebleu, soundfile); without them those
metrics simply report "not computed" with a reason.

```bash
make bench                      # all cases
cd server && .venv/bin/python -m bench run moshi
.venv/bin/python -m bench compare results/<old>.json results/<new>.json
```

Add a case by dropping `data/benchmarks/<name>/source.mp4` and listing it in `cases.yaml`; add any
of the ground-truth files next to it to unlock the GT metrics for that case.

## License

OpenDub is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

That covers OpenDub's own source. The AI models and libraries it drives are installed separately
and each carries its **own** license; some model weights (e.g. F5-TTS) are **non-commercial**. If
you plan to use OpenDub commercially, check the upstream license of every provider you enable —
see [NOTICE](NOTICE) for the list.

## Roadmap

- More target languages (the pipeline is language-parameterized; EN is the curated default)
- Per-speaker manual reference audio upload & voice profiles
- Segment splitting/merging in the timeline
- Sidechain ducking + music re-balance in the mix stage
- More providers (Seed-VC voice conversion, CosyVoice, Azure/Google cloud stacks)
