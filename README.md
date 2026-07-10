# OpenDub

**Self-hosted, voice-preserving video dubbing studio.** OpenDub chains pluggable AI models — run
open-source models on your own GPU, or bring cloud API keys — to dub video content (anime first)
into English while keeping each character's **own voice and emotional delivery**, with optional
lip sync. A video-editor-style web GUI lets you scrub the timeline, verify every translated line,
edit it, and re-voice a single segment in one click.

```
 upload → ingest → separate → transcribe → translate → synthesize → mix → lip sync → render
            │         │           │            │            │         │       │         │
         ffmpeg    Demucs    Faster-Whisper  Ollama/vLLM  XTTS-v2  atempo  Wav2Lip   dubbed.mp4
                 (or none)   (or Whisper    (or Claude /  (or F5 / fit+mix (or Latent-
                              API, mock)     GPT / DeepL)  ElevenLabs)      Sync, none)
```

## How voice & emotion are preserved

- **Voice**: each speaker gets a reference built from their own clearest lines; the TTS stage
  zero-shot-clones that voice (XTTS-v2 locally — "clone on the edge" — or ElevenLabs instant
  voice cloning in the cloud).
- **Emotion**: every line is synthesized with the *segment's own source audio* as the style
  reference, so the original delivery (shouting, whispering, sobbing) conditions the dub, plus a
  free-text emotion hint that also steers the translation. Translations are duration-aware
  (~15 chars/sec budget) so dubs fit their slots; residual mismatch is time-stretched within
  gentle limits and shown in the UI as a ×rate badge.

## Quick start

Requirements: Python 3.11+ ([uv](https://docs.astral.sh/uv/)), Node 18+, ffmpeg. Optional:
NVIDIA GPU for local models, `espeak-ng` for a nicer mock voice.

```bash
make setup        # server venv + web deps
make dev          # API on :8000, UI on :5173 (proxied)
```

Open http://localhost:5173, drop in a video, and press **Run pipeline**. Out of the box every
stage uses always-available providers (silence-based segmenter, mock translator, beep voice) so
you can exercise the entire workflow — timeline, editing, per-segment re-voicing, final render —
without installing any model. Then switch stages to real providers in **Settings**.

Production-ish: `make serve` (builds the UI, serves everything from FastAPI on :8000), or
`docker compose up --build`.

## Plugging in models

Every stage is a **provider** you pick per project in Settings (with per-provider config forms,
availability probes, and a Test button). Ships with:

| stage | self-hosted OSS | cloud API | zero-dep fallback |
|---|---|---|---|
| Separation | Demucs (htdemucs) | — | passthrough |
| Transcription | Faster-Whisper | OpenAI Whisper | silence segmenter |
| Diarization | pyannote 3.1 | — | single speaker |
| Translation | **any OpenAI-compatible endpoint** (Ollama, vLLM, LM Studio…) | Anthropic Claude, OpenAI, DeepL | mock |
| Voice (TTS + clone) | Coqui XTTS-v2, F5-TTS | ElevenLabs | beep/espeak |
| Lip sync | Wav2Lip, LatentSync (point at your checkout + checkpoint) | — | none |

Install local model extras: `make gpu-extras` (Faster-Whisper + Demucs + XTTS).

Keys/settings live in `configs/settings.yaml` (written by the UI; see
`configs/providers.example.yaml`) or environment variables that override it:
`OPENDUB_TRANSLATION_ANTHROPIC_API_KEY=sk-ant-…`. Secrets are write-only through the API and
masked everywhere else.

Adding a provider is one file: subclass the stage ABC in `server/app/providers/<kind>/`, declare
`meta` (id, name, config fields — the Settings UI renders them automatically), implement one
method, decorate with `@register`. See `ARCHITECTURE.md`.

## The editor

- **Timeline** (canvas): ruler, source waveform, one lane per speaker with segment blocks, dubbed
  waveform. Click to seek, scrub the playhead, wheel to pan, Ctrl+wheel to zoom, drag segment
  edges to retime. Dirty segments show an amber dot.
- **Player**: original ↔ dubbed audio toggle (`1`/`2`), Space to play, ←/→ to step, ↑/↓ to walk
  segments.
- **Inspector**: edit source text / translation (with a chars-per-second budget meter), emotion
  hint, audition takes, pick the active take, **Re-translate** / **Re-voice** just that segment.
- **Transcript**: full list synced to playback — the fastest way to review a whole episode.

Editing anything marks exactly the right things dirty (edit a translation → that line needs
re-voicing → mix/render need re-running); the pipeline only redoes dirty work.

## Project layout

`server/` FastAPI + pipeline + providers · `web/` React/TS/Vite editor · `configs/` settings ·
`data/projects/<id>/` all media + `manifest.json` per project · `ARCHITECTURE.md` the deep dive.

Tests: `make test` runs an end-to-end pipeline over a synthesized video using only zero-dep
providers — no GPU, no network, no model downloads.

## Roadmap

- More target languages (the pipeline is language-parameterized; EN is the curated default)
- Per-speaker manual reference audio upload & voice profiles
- Segment splitting/merging in the timeline
- Sidechain ducking + music re-balance in the mix stage
- More providers (Seed-VC voice conversion, CosyVoice, Azure/Google cloud stacks)
