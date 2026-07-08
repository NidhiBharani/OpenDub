# OpenDub — Architecture

OpenDub is a self-hostable web application that dubs video content (anime first) into English while
preserving each speaker's voice and emotional delivery. It chains pluggable models — self-hosted OSS
or cloud APIs — through a fixed pipeline, and exposes a video-editor-style web GUI for reviewing and
correcting every step (transcript, translation, voice takes) with per-segment regeneration.

```
anime-dub-v2/
├── server/            # Python 3.12 FastAPI backend (uv-managed)
│   ├── app/
│   │   ├── main.py            # app factory, route mounting, serves built web UI
│   │   ├── config.py          # paths + settings persistence (configs/settings.yaml)
│   │   ├── models.py          # pydantic domain models  ← CONTRACT
│   │   ├── store.py           # project persistence: data/projects/<id>/manifest.json
│   │   ├── jobs.py            # async job engine + event bus (SSE)
│   │   ├── api/               # REST + SSE routes (projects, segments, pipeline, providers, media, events)
│   │   ├── pipeline/          # orchestrator.py (stage DAG), audio.py (fit/mix)
│   │   ├── providers/         # base.py (ABCs + registry) ← CONTRACT ; one pkg per kind
│   │   └── media/             # ffmpeg.py (probe/extract/mux), waveform.py (peaks JSON)
│   └── tests/
├── web/               # React 18 + TS + Vite 5 + zustand. No UI framework — custom minimal design system.
│   └── src/
│       ├── types.ts           # mirrors models.py exactly (snake_case)  ← CONTRACT
│       ├── api/client.ts      # typed REST client + SSE subscription   ← CONTRACT
│       ├── state/store.ts     # zustand app store                      ← CONTRACT
│       ├── theme.css          # design tokens
│       ├── views/             # Library, Editor, Settings
│       ├── editor/            # Player, Timeline (canvas), Inspector, TranscriptList, PipelineBar
│       └── components/        # primitives (Button, Select, Field, Modal, Toast…)
├── configs/           # providers.example.yaml ; settings.yaml is runtime-generated (gitignored)
├── data/projects/     # per-project media + manifest (gitignored)
└── specs/             # work-package specs for implementation agents
```

## Pipeline

Fixed stage sequence; each stage delegates to the provider chosen in the project's `pipeline` config:

| stage        | kind         | consumes                          | produces (paths relative to project dir)          |
|--------------|--------------|-----------------------------------|---------------------------------------------------|
| `ingest`     | —            | uploaded file                     | `source.<ext>`, `playback.mp4` (h264/aac), `audio/original.wav` (48k stereo), `media` info, `waveforms/original.json` |
| `separate`   | `separation` | `audio/original.wav`              | `audio/vocals.wav`, `audio/background.wav`, `waveforms/vocals.json` |
| `transcribe` | `asr` + `diarization` | `audio/vocals.wav`       | `segments[]` (start/end/source_text), `speakers[]`, per-speaker `speakers/<id>/reference.wav` (concat of that speaker's clearest source segments, ≤30 s) |
| `translate`  | `translation`| segments' `source_text` + context | segments' `translated_text` (duration-aware; clears `translate_dirty`) |
| `synthesize` | `tts`        | `translated_text` + speaker reference + **the segment's own source audio** as style/emotion reference | `audio/segments/<seg>/take_<n>.wav`, appended to `takes[]`, `active_take_id` set |
| `mix`        | —            | active takes + `background.wav`   | takes time-fitted (ffmpeg atempo, clamped 0.6–1.6, `rate_factor` recorded), placed on timeline → `audio/dub_vocals.wav`; + background → `audio/dub_mix.wav`, `waveforms/dub_mix.json` |
| `lipsync`    | `lipsync`    | `playback.mp4` + `audio/dub_mix.wav` | `render/lipsync.mp4` (provider `lipsync.none` ⇒ status `skipped`) |
| `render`     | —            | lipsync output or `playback.mp4` + `audio/dub_mix.wav` | `render/dubbed.mp4` |

**Emotion preservation strategy:** each TTS call receives both the speaker-level reference (identity)
and the segment's own source-audio clip (emotion/prosody style reference), plus a free-text `emotion`
hint that also conditions the translation prompt. Zero-shot cloning per segment carries the line's
emotional delivery into the dub.

**Dirty tracking / invalidation:** editing `source_text` ⇒ `translate_dirty=true, synth_dirty=true`;
editing `translated_text`/`speaker_id`/timing ⇒ `synth_dirty=true`; any segment dirty ⇒ downstream
stages (`mix`, `lipsync`, `render`) get status `dirty`. Stage runs process only dirty/missing work
(translate: only `translate_dirty` or empty; synthesize: only `synth_dirty` or take-less).
Per-segment regeneration runs translate and/or synthesize for that one segment, then marks `mix`+ dirty
(it does NOT auto-run mix).

## Domain model (contract — mirrored in web/src/types.ts)

See `server/app/models.py`. Key types: `Project`, `Segment` (start/end seconds, `speaker_id`,
`source_text`, `translated_text`, `emotion`, `takes[]`, `active_take_id`, dirty flags), `Speaker`
(id, name, color, `reference_path`), `PipelineConfig` (a `ProviderChoice {provider_id, options}` per
kind), `StageState`, `Job`, `MediaInfo`. IDs are short random hex strings (`uuid4().hex[:8]`,
segments prefixed `seg_`, speakers `spk_`, takes `take_`, jobs `job_`).

## Provider system (contract)

See `server/app/providers/base.py`. Six kinds: `separation`, `asr`, `diarization`, `translation`,
`tts`, `lipsync`. Every provider: subclasses the kind ABC, declares `meta: ProviderMeta`
(`id` = `"<kind>.<slug>"`, display name, `runtime: local|cloud`, `fields: [ConfigField]` describing
its config UI), registers via `@register`, and implements `available() -> (bool, reason)` using
**lazy imports** — the server must boot with zero optional ML deps installed. Heavy work runs via
`asyncio.to_thread` / subprocess; every long call takes a `progress(fraction, message)` callback.

Providers to ship:

| kind        | local (OSS)                                             | cloud (API key)                     | always-available |
|-------------|---------------------------------------------------------|-------------------------------------|------------------|
| separation  | `separation.demucs` (htdemucs)                          | —                                   | `separation.passthrough` (vocals=original, silent bg) |
| asr         | `asr.faster_whisper`                                    | `asr.openai_whisper`                | `asr.mock`       |
| diarization | `diarization.pyannote` (HF token)                       | —                                   | `diarization.single_speaker` |
| translation | `translation.openai_compatible` (Ollama/vLLM/LM Studio: base_url+model) | `translation.anthropic`, `translation.openai`, `translation.deepl` | `translation.mock` |
| tts         | `tts.xtts` (Coqui XTTS-v2, zero-shot clone), `tts.f5_tts` | `tts.elevenlabs` (IVC clone)        | `tts.mock` (espeak-ng if present, else shaped sine beeps via ffmpeg) |
| lipsync     | `lipsync.wav2lip` (configurable repo/checkpoint dir), `lipsync.latentsync` | —                | `lipsync.none`   |

Settings persistence: `configs/settings.yaml` → `{providers: {<provider_id>: {<field_key>: value}}, defaults: {<kind>: provider_id}}`.
Env override wins: `OPENDUB_<PROVIDER_ID upper, dots→underscores>_<FIELD upper>` (e.g.
`OPENDUB_TRANSLATION_ANTHROPIC_API_KEY`). Secret fields are write-only via API (returned masked as `"•••"`;
sending the mask back means "unchanged").

## HTTP API (contract — prefix `/api`)

- `GET  /api/health` → `{ok, version}`
- `GET  /api/projects` → `ProjectSummary[]`
- `POST /api/projects` multipart `{file, name?, source_lang?, target_lang?}` → `Project` (runs `ingest` synchronously-ish as a job; returns immediately with project, ingest job auto-started)
- `GET/DELETE /api/projects/{pid}` → `Project` / `{ok}`
- `PATCH /api/projects/{pid}` body: partial `{name?, source_lang?, target_lang?, pipeline?}` → `Project`
- `PATCH /api/projects/{pid}/segments/{sid}` partial `{source_text?, translated_text?, speaker_id?, start?, end?, emotion?, active_take_id?}` → `Segment` (applies dirty rules)
- `POST /api/projects/{pid}/segments/{sid}/regenerate` `{stages: ("translate"|"synthesize")[]}` → `Job`
- `PATCH /api/projects/{pid}/speakers/{spid}` `{name?, color?}` → `Speaker`
- `POST /api/projects/{pid}/pipeline/run` `{stages?: StageKey[]}` → `Job` (default: everything not `done`, in order)
- `POST /api/jobs/{job_id}/cancel` → `Job`
- `GET  /api/jobs?project_id=` → `Job[]` (active + recent)
- `GET  /api/projects/{pid}/events` → SSE: events `job` (Job JSON), `project` (Project JSON, sent after stage completions / segment mutations by jobs), `ping`
- `GET  /api/media/{pid}/{path...}` → file with Range support (e.g. `playback.mp4`, `audio/dub_mix.wav`, `audio/segments/<sid>/take_1.wav`)
- `GET  /api/projects/{pid}/waveform/{asset}` → `{version, sample_rate: peaks_per_second, peaks: number[]}` (asset ∈ `original|vocals|dub_mix`, min/max pairs flattened, 50 pairs/s)
- `GET  /api/providers` → `ProviderInfo[]` (`{meta, available, reason, configured_options (secrets masked)}`)
- `PUT  /api/settings/providers/{provider_id}` `{options}` → `ProviderInfo`
- `POST /api/providers/{provider_id}/check` → `{available, reason}` (live check incl. API ping where cheap)

Errors: JSON `{detail}` with proper status codes. CORS: allow all in dev (vite proxy handles it anyway).

## Web GUI

Single-page app, three views switched in the zustand store (no router): **Library** (project cards +
drag-drop upload), **Editor**, **Settings** (per-stage provider picker + dynamic config forms generated
from `ConfigField[]`, availability badges, "Test" button per provider).

Editor layout (the heart — must feel like a video editor):

```
┌────────────────────────────────────────────────┬──────────────┐
│  Player (video, orig/dub audio toggle,         │  Inspector   │
│  transport: ⏯ time  ⧗)                        │  (selected   │
├────────────────────────────────────────────────┤   segment:   │
│  PipelineBar: stage chips w/ status + Run      │   src text,  │
├────────────────────────────────────────────────┤   editable   │
│  Timeline (canvas): ruler ▸ waveform lane ▸    │   translation│
│  one lane per speaker with segment blocks ▸    │   emotion,   │
│  dub lane. Playhead, click-seek, wheel-zoom,   │   takes,     │
│  drag-pan, segment click→select               │   ⟳ regen)   │
└────────────────────────────────────────────────┴──────────────┘
```

- Zustand store owns: `playhead` (written each rAF by Player), `seekRequest {t, nonce}` (written by
  Timeline/TranscriptList clicks; Player consumes), `playing`, `audioTrack: original|dub`, `zoom`
  (px/s, 2–500), `scrollX` (seconds at left edge), `selection` (segment id), plus data + actions.
- Dub preview: hidden `<audio src=dub_mix.wav>` kept in sync with the muted video element.
- Keyboard: Space play/pause, ←/→ ±1 s (Shift ±5 s), ↑/↓ prev/next segment, +/- zoom, `1`/`2` audio track.
- Design: dark-first minimal. Tokens in `theme.css` — bg `#0d0f13`, panels `#14171d`, 1px borders
  `#232833`, text `#e8eaf0`/`#8b93a7`, accent `#e8604c` (coral), speaker palette 8 hues. Inter/system
  font, `font-variant-numeric: tabular-nums` for timecodes. No gradients, no shadows heavier than
  `0 1px 2px rgb(0 0 0 / .4)`. Everything keyboard-reachable, 150 ms ease-out transitions.

## Conventions

- Python: ruff-clean, type-hinted, pydantic v2, `pathlib`, async route handlers; blocking/CPU work in
  `asyncio.to_thread` or subprocesses. No global mutable state outside `jobs.py`/`store.py` singletons.
- All media paths stored **relative to the project dir**; only `store.py` knows absolute paths.
- TS: strict mode, no `any` unless quarantined, named exports, function components + hooks.
- ffmpeg is invoked with `-y -hide_banner -loglevel error` via helpers in `media/ffmpeg.py` only.
- Tests: `server/tests/` pytest + httpx `ASGITransport`; mock providers make the full pipeline runnable
  without any ML dependency.
