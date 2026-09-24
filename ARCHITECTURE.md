# OpenDub — Architecture

OpenDub is a self-hosted web app that dubs video (anime first) into English while preserving each
speaker's voice and emotional delivery. A fixed pipeline chains pluggable models — local OSS or
cloud APIs — and a video-editor-style web UI lets you review and correct every step (transcript,
translation, voice takes) with per-segment regeneration.

```
anime-dub-v2/
├── server/            # Python 3.12 FastAPI backend (uv-managed)
│   ├── app/
│   │   ├── main.py            # app factory, route mounting, serves built web UI
│   │   ├── config.py          # paths + settings persistence (configs/settings.yaml)
│   │   ├── models.py          # pydantic domain models  ← CONTRACT
│   │   ├── store.py           # project persistence: data/projects/<id>/manifest.json
│   │   ├── jobs.py            # async job engine + event bus (SSE)
│   │   ├── api/               # REST + SSE routes
│   │   ├── pipeline/          # orchestrator.py (stage sequence), audio.py (fit/mix)
│   │   ├── providers/         # base.py (ABCs + registry) ← CONTRACT ; one pkg per kind
│   │   └── media/             # ffmpeg.py (probe/extract/mux), waveform.py (peaks JSON)
│   └── tests/                 # pytest end-to-end over mock providers
├── web/               # React 18 + TS + Vite + zustand. Custom minimal design system.
│   ├── src/
│   │   ├── types.ts           # mirrors models.py exactly (snake_case)  ← CONTRACT
│   │   ├── api/client.ts      # typed REST client + SSE subscription   ← CONTRACT
│   │   ├── state/store.ts     # zustand app store                      ← CONTRACT
│   │   ├── theme.css          # design tokens
│   │   ├── views/             # Library, Editor, Settings
│   │   ├── editor/            # Player, Timeline (canvas), Inspector, TranscriptList, PipelineBar
│   │   ├── components/        # primitives (Button, Select, Field, Modal, Toast…)
│   │   └── **/*.test.ts(x)    # Vitest + Testing Library (headless jsdom)
│   └── e2e/                   # Playwright real-browser tests (headless Chromium)
├── configs/           # providers.example.yaml ; settings.yaml is runtime-generated (gitignored)
└── data/projects/     # per-project media + manifest (gitignored)
```

## Pipeline

Fixed stage sequence (`ingest → separate → analyze → transcribe → translate → synthesize → mix →
lipsync → review → render`; `analyze` and `review` only run capability steps — see *Capability
map*). Each stage delegates to the provider chosen in the project's `pipeline`
config:

| stage        | kind         | consumes                          | produces (paths relative to project dir)          |
|--------------|--------------|-----------------------------------|---------------------------------------------------|
| `ingest`     | —            | uploaded file                     | `source.<ext>`, `playback.mp4` (h264/aac), `audio/original.wav` (48k stereo), `media` info, `waveforms/original.json` |
| `separate`   | `separation` | `audio/original.wav`              | `audio/vocals.wav`, `audio/background.wav`, `waveforms/vocals.json` |
| `transcribe` | `asr` + `diarization` | `audio/vocals.wav`       | `segments[]` (start/end/source_text), `speakers[]`, per-speaker `speakers/<id>/reference.wav` (that speaker's clearest lines, ≤30 s) |
| `translate`  | `translation`| segments' `source_text` + context | segments' `translated_text` (duration-aware; clears `translate_dirty`) |
| `synthesize` | `tts`        | `translated_text` + speaker reference + **the segment's own source audio** as style reference | `audio/segments/<seg>/take_<n>.wav`, appended to `takes[]`, `active_take_id` set |
| `mix`        | —            | active takes + `background.wav`   | takes time-fitted (ffmpeg atempo, clamped 0.6–1.6, `rate_factor` recorded), placed on the timeline → `audio/dub_vocals.wav`; + background → `audio/dub_mix.wav`, `waveforms/dub_mix.json`, `playback_dub.mp4` (preview: `playback.mp4` picture + dub audio) |
| `lipsync`    | `lipsync`    | `playback.mp4` + `audio/dub_mix.wav` | `render/lipsync.mp4` (provider `lipsync.none` ⇒ status `skipped`) |
| `render`     | —            | lipsync output or `playback.mp4` + `audio/dub_mix.wav` | `render/dubbed.mp4` |

**Emotion preservation.** Every TTS call gets two reference clips: the speaker-level reference
(voice identity) and the segment's own source audio (emotion/prosody style). A free-text
`emotion` hint additionally conditions the translation prompt. Zero-shot cloning per segment
carries each line's delivery into the dub.

**Dirty tracking.** Editing `source_text` ⇒ `translate_dirty=true, synth_dirty=true`. Editing
`translated_text`/`speaker_id`/timing ⇒ `synth_dirty=true`. Any dirty segment ⇒ downstream stages
(`mix`, `lipsync`, `render`) become `dirty`. Stage runs process only dirty/missing work.
Per-segment regeneration runs translate and/or synthesize for that one segment, then marks
`mix`+ dirty (it does not auto-run mix). Structural transcript edits (split, merge, insert, delete,
re-transcribe one line) set both flags on the affected lines and mark `synthesize`+ dirty; they
are refused with 409 while a job runs (the job's checkpoint merge treats its segment list as
authoritative). `_stage_needs_run` and the stage bodies share one predicate
(`stages.translate_targets` / `synth_targets`), so a line without output is always picked up.

**Skip ranges (keep the original).** `Project.skip_ranges: TimeRange[]` are spans excluded from
dubbing. `pipeline/regions.py` normalizes them and derives `Segment.skipped` (midpoint inside a
range, or ≥ 50 % overlap); skipped lines are never translated or voiced, the mix splices
`audio/original.wav` back in over those spans with 50 ms crossfades (`audio.splice_ranges`, before
mastering so levels match), and lip sync overlays the original picture there
(`ffmpeg.overlay_ranges`). Editing ranges marks `mix`+ dirty.

**Anchors.** `GET /projects/{pid}/anchors` merges shot changes (`ffmpeg` scene score, cached in
`analysis/scenes.json` at ingest), dialogue silences (`silencedetect` over `vocals.wav`, cached in
`analysis/silences.json` at separate) and segment edges into one sorted list the timeline draws
under its ruler; `?refresh=1` re-runs the detectors. `GET /projects/{pid}/filmstrip` builds a
contact sheet (`analysis/filmstrip.jpg`) for the timeline's video row.

**Versions.** Every pipeline run that completes `mix` snapshots the on-disk state to
`versions/<vid>/` (`meta.json` = `Version`, `state.json` = `VersionState`, copies of
`dub_mix.wav` / `playback_dub.mp4` / `dubbed.mp4` / the waveform, hardlinks of take files and
speaker references — take files are immutable but `run_transcribe` deletes the directory).
Manual snapshots and restores go through `/versions`; restore refuses while a job runs, first
snapshots the working state as a `pre_restore` version when it is not already a version, then
copies outputs back and relinks takes. `Project.active_version_id` names the version the working
state equals; every mutation route clears it. A project has one working `target_lang`; changing it
snapshots the current dub, flags every line for re-translation and re-voicing, and versions carry
their own `target_lang`, so dubs in several languages coexist as lineages (`Take.lang` records
the language each take was voiced in).

## Capability map

`server/app/capabilities.py` is the declarative registry of the **41 capabilities** in the
inference map (phases A–G: audio in, picture in, text, voice, audio out, picture out, judges).
Each `Capability` names its provider `kind`, the macro stage it runs in, a `slot`
(`pre`/`post` = run by the stage's step loop around the stage body; `inline` = the stage body
consults it), a `tier`, hard `needs` / soft `uses` dependencies, an optional feature gate
(`lipsync` / `subtitles`), an always-available builtin provider and its own `params`.

- The six original kinds are capabilities too (A1 separation, A4 asr, A6 diarization,
  C2 translation, D1 tts, F1 lipsync); their provider choice stays in the `PipelineConfig` field
  of the same name. Every other capability is configured in `PipelineConfig.capabilities[<id>]`
  (`CapabilityChoice {enabled, provider_id, options, params}`).
- **Simple mode** = `apply_preset(cfg, preset, runtime, lipsync, subtitles)`: writes concrete
  choices for all 41 (presets `minimal` | `balanced` | `max`; runtimes `builtin` | `local` |
  `cloud`, falling back cloud → local → builtin when no provider is `available()`, with notes).
  New projects default to `minimal` + `builtin`. **Advanced mode** edits one capability at a time
  and flips `preset` to `custom`.
- `resolve(cfg)` is the single answer to "what will run": explicit choice → preset membership →
  dependency closure (`needs` are switched on) → feature gates → core capabilities are always
  on → a `<kind>.off` provider means off. Each result carries a human `reason`.
- Stages `analyze` (after `separate`) and `review` (before `render`) have no body: they run
  their enabled steps and are `skipped` when none are. A failing step is recorded in
  `StageState.steps[<capability id>]` and never fails the stage.
- Adding a capability provider: subclass `StepProvider` in `server/app/providers/steps/`,
  `meta.id = "<kind>.<slug>"`, implement `run(ctx: StepContext, progress) -> str`, `@register`.
- E5 (watermark + provenance) is registered but tier `deferred`: in no preset, not implemented.

Roadmap and per-capability builtins/providers: `docs/plans/capability-map-implementation.md`.
Model picks per capability under three compute budgets (DGX Spark / unconstrained / API):
`docs/model-candidates-by-compute.md`.

## Domain model (contract — mirrored in web/src/types.ts)

See `server/app/models.py`. Key types: `Project` (… `skip_ranges[]`, `active_version_id`),
`Segment` (start/end seconds, `speaker_id`, `source_text`, `translated_text`, `emotion`,
`words[]` with per-word timing + confidence, `takes[]`, `active_take_id`, dirty flags, `skipped`),
`Take` (… `lang`), `TimeRange`, `Speaker` (id, name, color, `reference_path`), `PipelineConfig`
(a `ProviderChoice {provider_id, options}` per kind), `StageState`, `Job`, `MediaInfo`,
`Version` / `VersionState`. IDs are short random hex (`uuid4().hex[:8]`, prefixed `seg_` /
`spk_` / `take_` / `job_` / `rng_` / `ver_`).

## Provider system (contract)

See `server/app/providers/base.py`. Six kinds: `separation`, `asr`, `diarization`, `translation`,
`tts`, `lipsync`. Every provider subclasses its kind's ABC, declares `meta: ProviderMeta`
(`id` = `"<kind>.<slug>"`, display name, `runtime: local|cloud`, `fields: [ConfigField]` that the
Settings UI renders automatically), registers with `@register`, and implements
`available() -> (bool, reason)` using **lazy imports** — the server must boot with zero optional
ML deps installed. Heavy work runs via `asyncio.to_thread` or subprocesses; long calls report
through a `progress(fraction, message)` callback.

Included providers:

| kind        | local (OSS)                                             | cloud (API key)                     | always-available |
|-------------|---------------------------------------------------------|-------------------------------------|------------------|
| separation  | `separation.demucs` (htdemucs)                          | `separation.lalalai` (LALAL.AI)     | `separation.passthrough` |
| asr         | `asr.faster_whisper`                                    | `asr.openai_whisper`                | `asr.mock` (silence segmenter) |
| diarization | `diarization.pyannote` (HF token)                       | `diarization.pyannote_api` (pyannoteAI) | `diarization.single_speaker` |
| translation | `translation.openai_compatible` (Ollama, vLLM, LM Studio…: base_url+model) | `translation.anthropic`, `translation.openai`, `translation.deepl` | `translation.mock` |
| tts         | `tts.f5_tts` (F5-TTS, zero-shot clone)                  | `tts.elevenlabs` (IVC clone)        | `tts.mock` (espeak-ng if present, else shaped beeps via ffmpeg) |
| lipsync     | `lipsync.wav2lip`, `lipsync.latentsync` (point at your checkout + checkpoint) | `lipsync.replicate` (human faces only) | `lipsync.none`   |

Every stage has a cloud path (key-only, no local GPU). Cloud providers use `httpx` (a core dep),
gate `available()` on their API key, and share upload/poll/download helpers in
`providers/_http.py`; the job-based ones (LALAL.AI, pyannoteAI, Replicate) upload the media,
poll the remote job, then download results.

Settings persistence: `configs/settings.yaml` →
`{providers: {<provider_id>: {<field_key>: value}}, defaults: {<kind>: provider_id}}`.
Env override wins: `OPENDUB_<PROVIDER_ID upper, dots→underscores>_<FIELD upper>`
(e.g. `OPENDUB_TRANSLATION_ANTHROPIC_API_KEY`). Secret fields are write-only via the API
(returned masked as `"•••"`; sending the mask back means "unchanged").

## HTTP API (contract — prefix `/api`)

- `GET  /api/health` → `{ok, version}`
- `GET  /api/projects` → `ProjectSummary[]`
- `POST /api/projects` multipart `{file, name?, source_lang?, target_lang?}` → `Project` (ingest job auto-started)
- `GET/DELETE /api/projects/{pid}` → `Project` / `{ok}`
- `PATCH /api/projects/{pid}` partial `{name?, source_lang?, target_lang?, pipeline?, mode?, capabilities?}` → `Project` (`pipeline` = legacy kinds only; `capabilities` keyed by capability id)
- `POST /api/projects/{pid}/pipeline/preset` `{preset, runtime, lipsync?, subtitles?}` → `{project, notes[]}` (Simple mode)
- `GET  /api/capabilities` → `{phases, capabilities[41], presets[]}` · `GET /api/projects/{pid}/capabilities` → resolved map
- `PATCH /api/projects/{pid}/segments/{sid}` partial `{source_text?, translated_text?, speaker_id?, start?, end?, emotion?, active_take_id?}` → `Segment` (applies dirty rules)
- `POST /api/projects/{pid}/segments/{sid}/regenerate` `{stages: ("translate"|"synthesize")[]}` → `Job` (409 for a skipped line)
- `POST /api/projects/{pid}/segments/{sid}/split` `{at} | {word_index}` → `Project` · `POST …/{sid}/merge-next` → `Project` · `POST /api/projects/{pid}/segments` `{start, end, speaker_id?, source_text?}` → `Project` · `DELETE …/{sid}` → `Project` (all 409 while a job runs)
- `POST /api/projects/{pid}/segments/{sid}/transcribe` → `Job` (re-run ASR over one line)
- `PUT  /api/projects/{pid}/skip-ranges` `{ranges: [{id?, start, end, label?}]}` → `Project`
- `GET  /api/projects/{pid}/anchors?refresh=` → `{version, anchors[{t, kind, confidence, start, end}]}` · `GET /api/projects/{pid}/filmstrip` → `{interval, cols, rows, tile_w, tile_h, count, url}`
- `GET  /api/projects/{pid}/versions?lang=` → `Version[]` · `POST …/versions` `{label?}` → `Version` · `PATCH …/versions/{vid}` `{label}` · `DELETE …/versions/{vid}` · `POST …/versions/{vid}/restore` `{restore_point?}` → `Project` (version outputs are served by the media route at their `outputs` paths)
- `PATCH /api/projects/{pid}/speakers/{spid}` `{name?, color?}` → `Speaker`
- `POST /api/projects/{pid}/pipeline/run` `{stages?: StageKey[]}` → `Job` (default: everything not `done`, in order)
- `POST /api/jobs/{job_id}/cancel` → `Job`
- `GET  /api/jobs?project_id=` → `Job[]` (active + recent)
- `GET  /api/projects/{pid}/events` → SSE: `job` (Job JSON), `project` (Project JSON), `ping`
- `GET  /api/media/{pid}/{path...}` → file with Range support
- `GET  /api/projects/{pid}/waveform/{asset}` → `{version, sample_rate, peaks}` (asset ∈ `original|vocals|dub_mix`; flattened min/max pairs, 50 pairs/s)
- `GET  /api/providers` → `ProviderInfo[]` (`{meta, available, reason, configured_options}`, secrets masked)
- `PUT  /api/settings/providers/{provider_id}` `{options}` → `ProviderInfo`
- `POST /api/providers/{provider_id}/check` → `{available, reason}` (live check; pings the API where cheap)

Errors: JSON `{detail}` with proper status codes.

## Web GUI

Single-page app, three views switched in the zustand store (no router): **Library** (project
browser + drag-drop upload), **Editor**, **Settings** (Simple/Advanced pipeline configuration
with forms generated from `ConfigField[]`). Design spec and audit: `docs/plans/editor-redesign.md`.

Editor layout (an NLE shell):

```
┌ Title bar: project · JA → EN · duration │ Run / progress │ Export · Settings ──────────────┐
├ Status strip: 10 stages as dots + labels (click → popover: detail, run only / run from) ───┤
├ Sidebar (Script | Versions | Cast) ┬ Viewer (black, transport bar) ┬ Inspector (Line|Speaker|Info)┤
├ Timeline: toolbar (tools, snapping, zoom, fit, exclude range) · headers │ ruler + scene strip ┤
│   V filmstrip ▸ ORIGINAL waveform ▸ one lane per speaker ▸ DUB waveform; range selection,   │
│   skip regions (hatched "keep original"), skimmer + playhead                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- The store owns: `playhead` (written each rAF by Player), `seekRequest {t, nonce}` (Player
  consumes), `playing`, `rate`, `audioTrack: original|dub`, `zoom` (px/s, 2–500), `scrollX`
  (seconds at left edge), `selection` / `skipSelection`, `range {start,end}`, `editing`
  (script row + field), `anchors`, `filmstrip`, `versions`, `previewVersionId`, plus data + actions.
- Dub preview: the `<video>` source swaps between `playback.mp4`, `playback_dub.mp4` and a
  version's `playback_dub.mp4` (same picture, different audio) so one media clock drives both;
  position is restored across the swap.
- Script panel: sentence-by-sentence inline correction (Enter commits + next row, Tab
  source→translation, Esc reverts); word spans seek on click and underline low-confidence ASR
  words; per-row actions (play, re-translate, re-voice, split, merge, keep original, delete).
- Keyboard map: `docs/plans/editor-redesign.md` §7.
- Design: neutral dark NLE chrome, one accent (`#3d7eff`), tokens in `theme.css`; system sans at
  12 px, tabular mono timecodes, 24–28 px rows, 4 px radii, no gradients or serif. Colour is
  reserved for content (speaker clips, status dots).

## Conventions

- Python: ruff-clean, type-hinted, pydantic v2, `pathlib`, async route handlers; blocking/CPU
  work in `asyncio.to_thread` or subprocesses. No global mutable state outside the
  `jobs.py`/`store.py` singletons.
- All media paths are stored **relative to the project dir**; only `store.py` knows absolute paths.
- ffmpeg is invoked with `-y -hide_banner -loglevel error` via `media/ffmpeg.py` only.
- TS: strict mode, no `any` unless quarantined, named exports, function components + hooks.
- Tests: `server/tests/` (pytest + httpx `ASGITransport`; mock providers make the full pipeline
  runnable with no ML deps), `web/src/**/*.test.*` (Vitest/jsdom), `web/e2e/` (Playwright).
