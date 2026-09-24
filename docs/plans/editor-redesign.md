# Editor redesign — audit and design spec

Date: 2026-09-24. Inputs: screenshots of the shipped "Neon Dusk" UI, a web survey of Final Cut Pro,
DaVinci Resolve, Premiere, ElevenLabs Dubbing Studio, Rask, HeyGen, Dubverse, Descript and Figma
version history, and a read of the server pipeline. Companion doc for the backend:
`docs/plans/editor-redesign-backend.md`.

## 1. Audit: why the current UI does not read as a professional editor

| # | Finding | Evidence |
|---|---------|----------|
| 1 | The chrome is coloured. Dusk-indigo panels, hot-pink (`#ff4fa3`) and cyan (`#2fe6d6`) accents, a pink original waveform and a teal dub waveform. Pro NLEs use neutral dark grey chrome and one accent; colour is reserved for content (clips, speaker labels, status). | `web/src/theme.css`, `timelineDraw.ts` `drawWaveLane(..., theme.accent)` |
| 2 | Editorial typography. Instrument Serif italic wordmark and serif headings ("Script", "Line 04", "It's a wrap.", "Every line, their voice."). Tools use one sans at 11–13 px plus tabular mono for timecode. | `TopBar.tsx`, `TranscriptList.tsx`, `Inspector.tsx`, `Library.tsx`, `RunView.tsx` |
| 3 | The Library is a landing page (60 px hero, tagline, dashed upload card inside the grid), not a project browser. | `Library.tsx` |
| 4 | The pipeline stepper is ten 12 px labels with 8 px dots in the title bar; skipped stages are struck through; each label runs that stage on click with no confirmation. Illegible and dangerous. | `PipelineBar.tsx` |
| 5 | The viewer is a 10 px-rounded card with a striped placeholder, pill overlays over the picture, and 44/52 px round transport buttons. NLE viewers are edge-to-edge black with a slim icon transport bar. | `Player.tsx` |
| 6 | Timeline: no track-header column (lane pills float over the waveform), 38 px speaker lanes with ellipsised clip labels, no zoom slider or fit, no overview for a 15-minute file, no markers, no range selection, no snapping feedback. | `timelineDraw.ts`, `Timeline.tsx` |
| 7 | Script rows are read-only; edits happen only in Inspector textareas. No inline correction, no word confidence, no split/merge, no keyboard flow, no filters. | `TranscriptList.tsx` |
| 8 | Inspector is four stacked inputs and a fat two-button footer instead of compact label/value rows. | `Inspector.tsx` |
| 9 | Inconsistent scale: button heights 36/38/44/52 px, radii 5/7/8/10/14/20/999 px, three type families. | throughout |
| 10 | Skip regions, scene anchors, inline transcript correction and run versions do not exist in the model, API or UI. | `models.py`, `api/*`, `web/src/*` |

## 2. Design principles (rules the implementation follows)

1. Neutral chrome, one accent. Tokens below. No gradients, glows, serif, or coloured panels.
2. One sans (system stack + Inter fallback) at 12 px base; 11 px labels; timecode in tabular mono.
3. Density: list rows 24–28 px, two-line script rows 44 px, toolbars 32–36 px, icon buttons 24 px in a 28 px hit area, spacing scale 2/4/6/8/12/16/24, radii 3 px (clips) / 4 px (controls) / 6 px (popovers).
4. Panels are separated by 1 px borders, never by gaps or shadows. The timeline is the widest, lowest element and is never nested.
5. State is a dot or a hatch, never a coloured button. Edits mark things stale immediately; regeneration is explicit (ElevenLabs model). No "Apply changes" batching.
6. Keyboard first: every action in the keyboard map (section 7) has a tooltip that shows its key.

### Tokens (dark, default)

| Token | Value | Use |
|---|---|---|
| `--bg` | `#1c1c1e` | window background, gutters |
| `--bg-panel` | `#242426` | panel bodies |
| `--bg-raised` | `#2c2c2e` | toolbars, headers, hover rows |
| `--bg-sunk` | `#19191b` | timeline canvas, viewer surround |
| `--bg-field` | `#1e1e20` | inputs |
| `--border` | `#3a3a3c` | panel separators |
| `--border-subtle` | `#2f2f31` | row dividers |
| `--text` | `#e5e5e7` | primary |
| `--text-dim` | `#9a9aa0` | secondary |
| `--text-faint` | `#6a6a70` | placeholder / disabled |
| `--accent` | `#3d7eff` | selection, focus, primary action |
| `--playhead` | `#f2f2f2` | playhead |
| `--skimmer` | `#ff3b30` | hover-scrub line |
| `--range` | `#ffd60a` | range selection border |
| `--ok` | `#30d158` | done |
| `--warn` | `#ff9f0a` | stale / timing overrun |
| `--err` | `#ff453a` | failed |
| `--spk-1..8` | muted hues (`#5b8def #a06fd8 #4fb286 #d9a441 #e0705e #4db3c9 #c46aa0 #8f9d5c`) | speaker clips and chips only |

Light theme keeps the same structure with grey values inverted; it is secondary.

## 3. Shell

```
┌ Title bar 38 px: [mark] Project name ▾   JA → EN · 15:30 · 536×360   │ run status / Run ▸ │ Export  Versions  Settings ┐
├ Left 340 px ─────────┬ Viewer (flex) ────────────────────┬ Inspector 300 px ┤
│ tabs: Script Versions│ black letterboxed video           │ property rows     │
│ Cast                 │ transport bar 32 px               │                   │
├ 5 px resize handle ──┴───────────────────────────────────┴───────────────────┤
│ Timeline toolbar 32 px: tools · snapping · zoom slider · fit · exclude range · markers │
│ Track headers 120 px │ ruler 22 px + scene strip 8 px + lanes                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

- The pipeline stepper leaves the title bar. Stage state lives in a **status strip** under the title bar (10 compact steps with 6 px dots, 11 px labels; running step underlined in accent). Clicking a step opens a popover (status, detail, "Run from here", "Run only this"); nothing runs on a bare click.
- Run button: primary accent. While running it becomes a progress pill with cancel. Export is an icon button enabled when render is done.

## 4. Script panel (sentence-by-sentence correction)

- Row (44 px, expands when edited): line 1 = speaker chip (speaker colour, 10 px) · start timecode · status glyph (amber dot = stale, hatched = excluded, red = failed take). Line 2 = translated text 12 px. Source text shows as an italic dim line above the translation; both are inline-editable `contenteditable`-free textareas that grow.
- Keyboard: `Enter` commits and moves to the next row's same field, `Shift+Enter` newline, `Tab` moves source → translation → next row, `Esc` reverts. `↑/↓` move selection when not editing.
- Words: with word timings present, the source line renders word spans; click seeks to the word; confidence < 0.6 gets a dotted amber underline and a tooltip. A view-options menu toggles the indicator.
- Row hover toolbar (24 px icons, right): Play from here · Re-translate · Re-voice · Split at playhead · More (Merge with next · Exclude · Delete).
- Header: search, filter segmented control All / Stale / Warnings / Excluded, and counts.
- Timing: if the active take's fitted duration exceeds the slot by >10 %, an amber `+0.8s` badge; >30 % red.

## 5. Timeline

- Track header column (120 px): `V` video filmstrip row (thumbnails from `playback.mp4`, generated by the server per zoom bucket; falls back to a flat dark row), `ORIGINAL` dialogue waveform in muted grey, one lane per speaker with chip + mute, `DUB` mix waveform. Headers carry mute/solo icons that drive the preview track selector where meaningful (Original / Dub).
- Ruler 22 px: major/minor ticks, labels 10 px mono; playhead 1 px `--playhead` with a downward triangle; skimmer 1 px `--skimmer` follows the pointer and does not move the playhead until click.
- **Scene strip** 8 px under the ruler: anchor ticks from `GET /projects/{pid}/anchors` (scene cuts, silence gaps, segment boundaries). Height/opacity = confidence; a threshold slider in the toolbar hides weak ones. Click between two anchors selects that span as the range; `Shift+click` extends. `Alt+[` / `Alt+]` jump to prev/next anchor.
- **Range selection**: `I` / `O` set in/out at the skimmer or playhead, `X` selects the current segment span, `Alt+X` clears. Drawn as an 8 % white fill with a 1 px `--range` border and edge handles across all lanes. Range actions in the toolbar: **Exclude from dub** (`Cmd+E`), Loop, Regenerate lines in range.
- **Skip regions** (`Project.skip_ranges`): drawn as a 30 % black overlay with 8 px diagonal hatch and a "KEEP ORIGINAL" label; segments inside render outline-only and are not synthesised; the mix keeps the original audio there. Click a region to select it (Inspector shows label + delete); drag its edges to trim.
- Clips: radius 3 px, body = speaker colour at 35 % on dark, 1 px inner stroke, label 11 px; selected = 2 px accent; stale = amber corner dot; overrun tail hatched amber.
- Toolbar: segmented Select/Range tools, snapping toggle, zoom slider + `Fit`, clip-height segmented (S/M/L), exclude-range button, marker button, anchor threshold slider.

## 6. Versions

- Left-panel tab **Versions** grouped by target language (tab row: `EN 3`, `HI 1`). Items newest first: label (auto "Run · 24 Sep 14:03" or user-named), timestamp, preset/runtime/provider summary, stage summary, `active` badge for the version the current state came from.
- Item actions: **Preview** (viewer plays that version's `playback_dub.mp4`; banner "Previewing v3 · Restore · Close"), **Restore** (non-destructive: snapshots the current state as "Before restore" first, then applies), **Rename**, **Delete**.
- Auto-snapshot after every pipeline run that completes `mix` (and again if it completes `render`); manual **Snapshot now** (`Cmd+Shift+D`).
- Per-line take history stays in the Inspector (ElevenLabs clip-history model): list of takes with duration, rate factor, provider, play, radio to make active.
- Multi-language: the project's `target_lang` is editable in the title bar; changing it marks translate/synthesize dirty and starts a new lineage. Versions carry their own `target_lang`, so old dubs remain browsable and restorable (restore also restores `target_lang`).

## 7. Keyboard map

`Space` play/pause · `J/K/L` shuttle · `←/→` 1 s (`Shift` 5 s) · `↑/↓` prev/next line · `Enter` edit · `I/O` range · `Alt+X` clear range · `X` select line span · `Cmd+E` exclude range · `Cmd+B` split at playhead · `M` marker · `Alt+[ / ]` prev/next anchor · `1/2` original/dub audio · `+/-` zoom · `Shift+Z` fit · `Cmd+Shift+D` snapshot · `Esc` deselect.

## 8. Library and Settings

- Library: project browser with a toolbar (New project, search, sort, grid/list). Cards: 16:9 thumbnail, name, `JA → EN`, duration, status strip, modified date. No hero copy.
- Settings: dense preferences sheet. Section headers 11 px uppercase; the Simple/Advanced control stays; copy loses the conversational headline.
- Run view: keep the live feed but restyle to the tokens (no serif headline).
