# Editor redesign — backend design (agent report, 2026-09-24)

Companion to `editor-redesign.md`. Line numbers refer to the tree before implementation.



# OpenDub backend design: skip regions + anchors, dub versions, transcript correction

All paths below are absolute under `/home/nidhi/code/OpenDub/server/` unless noted. Line numbers refer to the current working tree (branch `neon-dusk-ui-and-hindi-dub`).

## 0. Ground truth that shapes all three features

These are the existing mechanisms the features must plug into (or fight). Read this section first.

### 0.1 How stages pick work and how "Run pipeline" selects stages

- `app/pipeline/stages.py:424-431` `run_translate` targets: `[s for s in project.segments if s.translate_dirty or not s.translated_text.strip()]`; empty-source segments get `translate_dirty=False` (L427-428).
- `app/pipeline/stages.py:544-549` `run_synthesize` targets: `[s for s in project.segments if s.synth_dirty or not s.takes]`.
- `app/pipeline/stages.py:598-655` `run_mix`: `entries` = every segment with an `active_take()` (L603-607); each take is time-fitted into `audio/segments/<seg>/fitted.wav` via `audio_utils.fit_to_duration` (L618-620, sets `take.rate_factor` in place); `assemble_track(placements, media.duration, audio/dub_vocals.wav)` (L626-628, pure-Python PCM overwrite, later placement wins); `mix_tracks(dub_vocals, background, dub_mix, reference_vocals=audio/vocals.wav, target_lufs=E4)` (L630-638); `ffmpeg.mux(playback.mp4, dub_mix.wav) -> playback_dub.mp4` (L644-645); waveform (L647-648); `mark_downstream_dirty("mix")` (L650).
- `app/pipeline/audio.py:116-166` `mix_tracks`: measure dub vocals (ebur128), static gain to match `reference_vocals` loudness (+ `-1.5 dBTP` limiter), `ffmpeg.amix(leveled, background, tmp_premix)` (L157), then master static gain to `target_lufs` (L159-163). The premix at L157 is "source level" (vocals matched to the original dialogue, bed at unity), which is where original audio can be spliced in without level mismatch.
- `app/pipeline/audio.py:57-113` `assemble_track` / `_assemble_track_sync`: preallocates a 48k s16 stereo buffer and byte-copies clips at `round(start*48000)` frames. Reuse this pattern (and the `wave` module) for the splice.
- `app/pipeline/stages.py:661-674` `run_lipsync`: whole-file `provider.sync(playback.mp4, dub_mix.wav, render/lipsync.mp4)`; no range awareness in any of the four providers (`providers/lipsync/{wav2lip,latentsync,replicate,none}.py`).
- `app/pipeline/stages.py:680-689` `run_render`: `mux(lipsync.mp4 if lipsync done else playback.mp4, dub_mix.wav) -> render/dubbed.mp4`.
- `app/pipeline/orchestrator.py:230-240` `_stage_needs_run`: a `done` stage is re-selected only via `translate_dirty` / `synth_dirty` flags. It does NOT look at `not translated_text` / `not takes`, so a segment that lacks output but carries no dirty flag (e.g. a freshly inserted one) will never get the stage auto-selected even though the stage body would process it. Fix: unify the predicates (see 3.6).
- `app/pipeline/orchestrator.py:97-167` `_pipeline_runner`: reloads the project with `track=True` per stage (L105), checkpoints before and after each stage (L116, L140), and the outer `except BaseException` restores statuses. There is no hook after the last stage; that is where the version snapshot goes (section 2.4).
- `app/pipeline/orchestrator.py:170-216` `_segment_runner`: only `translate`/`synthesize` (`_SEGMENT_STAGES` L24); always `mark_downstream_dirty("synthesize")` at L212 (wrong for a future per-segment transcribe, which must dirty from `translate`).
- `app/models.py:204-210` `Project.mark_downstream_dirty(after)`: flips only `done`/`error` stages after `after` to `dirty`. It never touches the stage itself; `api/projects.py:202-207 _mark_config_changed` handles that case.

### 0.2 The checkpoint/merge logic (`store.save_merged`) — the biggest thing that fights B and C

`app/store.py:79-143`:
- L99-102: project-level fields re-applied from disk are exactly `name, source_lang, target_lang, pipeline`. Any new project-level field (`skip_ranges`, `versions`, `current_version_id`) edited by a route while a job runs will be silently reverted by the job's next `_checkpoint` unless added here.
- L104-109: segment merge is keyed by id and `if b is None or f is None: continue` — the job's in-memory segment list is authoritative for *which segments exist*. A route that deletes/inserts/splits/merges segments while a job holds a tracked copy will have deletions resurrected and insertions dropped at the next checkpoint. Consequence: all structural transcript ops (section 3) and version restore (2.6) must refuse with 409 while a job is active for the project.
- L66-76 `_USER_SEGMENT_FIELDS`: must gain `words` and `skipped` (new field) or a job checkpoint reverts word edits / skip status.
- L130-143 stage merge: the job wins for stages it transitioned. A `PUT skip-ranges` during a running mix would mark mix dirty on disk, then the job's "mix: done" wins. Hence 409-while-busy for skip-range edits too (or accept and re-dirty after; 409 is simpler).

### 0.3 Files rewritten in place vs immutable files (matters for versions)

- Immutable, never rewritten: `audio/segments/<seg>/take_<n>.wav` (`_next_take_number` L487-494 always picks a fresh name), `speakers/<id>/reference.wav` (only rebuilt in `run_transcribe`). Safe to hardlink into a version.
- Rewritten in place by ffmpeg `-y` (open+truncate on the same inode) or `wave.open(path,"wb")`: `audio/dub_vocals.wav`, `audio/dub_mix.wav`, `playback_dub.mp4`, `render/lipsync.mp4`, `render/dubbed.mp4`, `waveforms/dub_mix.json`, `audio/segments/<seg>/source.wav` + `fitted.wav`. A hardlink to these would be clobbered by the next mix. Copy them (or change every writer to tmp+`Path.replace`; not worth it).
- `run_transcribe` L349-356 `shutil.rmtree` of `speakers/` and `audio/segments/`. This deletes take files a version references. Hardlinks in the version dir survive the rmtree (the inode lives on), so hardlinking at snapshot time is the fix; do not rely on the working-dir paths.
- `store.save` (L40-50) is tmp+replace: safe.

### 0.4 Other things worth knowing

- `api/media.py:26-35` `get_media` serves any file under the project dir via `store.resolve` (traversal-checked). `versions/<vid>/playback_dub.mp4` is servable with zero changes. `WAVEFORM_ASSETS` (L23) is a closed set; it needs a `version` query param.
- `api/projects.py:154-155`: `PATCH target_lang` only assigns the field. Translations/takes are not dirtied — a re-run silently keeps the old-language dub. Must be fixed for multi-language (2.7).
- `jobs.py`: no "is this project busy" helper; `api/projects.py:138-140` already does `for job in engine.list(pid): if job.status in ("queued","running")`. Add `JobEngine.active(project_id) -> Job | None`.
- `jobs.py:83-85` `publish_project` sends the full project JSON on every checkpoint, including after every synthesized segment (`stages.py:561`). `Segment.words` is already in that payload (~60 B/word; 500 lines × 30 words ≈ 1 MB per event). Versions metadata adds little; words are the concern (3.7).
- `web/src/types.ts:58-71` `Segment` has no `words` field; the frontend currently ignores the word data the backend already persists.
- Word confidence: `providers/asr/faster_whisper.py:119` fills `Word.confidence=w.probability`; `openai_whisper.py` and `asr/mock.py` produce no words at all.
- `providers/asr/mock.py:25-69` `_detect_silences` already shells out to ffmpeg `silencedetect` (violating the "only `media/ffmpeg.py` invokes ffmpeg" rule). Move it into `media/ffmpeg.py` and reuse it for anchors.
- Capability overlap (`app/capabilities.py`): A3 `region_detect` (L92-96, songs → "passthrough"), D7 `nonverbal` (L186-188), B2 `shots` (L125-127, planned builtin `scdet` per `docs/plans/capability-map-implementation.md:111`), E1 `timing_fit` `uses=["B2"]`. `providers/steps/` is empty, so all three are `.off` today. Design skip ranges and anchors as the data these capabilities will write into later (a `source` field on ranges; `analysis/scenes.json` as B2's output file).
- `Take.rate_factor` is mutated at mix time (L620). A snapshot taken after mix captures it; one taken before would not.

---

## 1. Feature A: skip regions ("keep original") + smart anchors

### 1.1 Models (`app/models.py`)

Add after `Word`:

```python
class TimeRange(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rng"))
    start: float
    end: float
    label: str = ""
    source: Literal["user", "A3", "D7"] = "user"   # who created it; user-editable regardless
```

`Segment`: add `skipped: bool = False` — derived from `Project.skip_ranges` by `regions.apply_skip_ranges` (1.3); persisted so stages, `save_merged`, and the UI agree. It is the "status" the UI shows (badge "kept original"). A per-line manual skip is expressed as a range equal to the segment's bounds (`label=seg.id`), so there is one source of truth.

`Project`: add `skip_ranges: list[TimeRange] = Field(default_factory=list)`.

`Take`: add `lang: str = ""` (see 2.7).

`web/src/types.ts`: mirror `TimeRange`, `Segment.skipped`, `Project.skip_ranges`, `Take.lang`.

`store._USER_SEGMENT_FIELDS` (L66-76): add `"skipped"`. `store._reapply_user_edits` (L99-102): add `ours.skip_ranges = fresh.skip_ranges`.

### 1.2 Normalization rules for ranges

New module `app/pipeline/regions.py`:

```python
def normalize_ranges(ranges: list[TimeRange], duration: float) -> list[TimeRange]:
    # clamp to [0, duration], drop end-start < 0.1, sort by start, merge overlapping/touching
    # ranges (keep the earlier id/label; union the labels with " / " when both non-empty)

def is_skipped(seg: Segment, ranges: list[TimeRange]) -> bool:
    # midpoint inside a range, OR overlap >= 50 % of seg.duration
    mid = (seg.start + seg.end) / 2
    for r in ranges:
        if r.start <= mid < r.end: return True
        ov = min(seg.end, r.end) - max(seg.start, r.start)
        if ov > 0 and ov >= 0.5 * seg.duration: return True
    return False

def apply_skip_ranges(project: Project) -> tuple[set[str], set[str]]:
    """Recompute Segment.skipped for all segments. Returns (newly_skipped_ids, newly_unskipped_ids)."""
```

`apply_skip_ranges` must also be called in `run_transcribe` after `project.segments = segments` (L355) and after every structural segment op in section 3, so `skipped` never goes stale.

### 1.3 Routes (`app/api/projects.py` or new `app/api/regions.py`, mounted in `main.py:41-44`)

- `GET  /api/projects/{pid}/skip-ranges` → `list[TimeRange]` (also present on the Project JSON; the GET exists for symmetry).
- `PUT  /api/projects/{pid}/skip-ranges` body `{"ranges": [{id?, start, end, label?}]}` → `Project`. Replaces the whole list. Under `store.lock(pid)`:
  1. `409` if `engine.active(pid)` (see 0.2 for why).
  2. `normalize_ranges`; ranges with no id get one; `source` preserved from the existing range with the same id, else `"user"`.
  3. `newly_skipped, newly_unskipped = apply_skip_ranges(project)`.
  4. Dirty rules:
     - if ranges changed at all → `project.mark_downstream_dirty("synthesize")` (mix/lipsync/render → dirty; translate/synthesize untouched).
     - for each newly **un-skipped** segment: `translate_dirty = not translated_text.strip()`, `synth_dirty = not takes` (so `_stage_needs_run` selects translate/synthesize for it; with the predicate unification in 3.6 this becomes redundant but harmless).
     - newly skipped segments: flags cleared (`translate_dirty = synth_dirty = False`) so they never keep translate/synthesize "owing work".
  5. `store.save`, `engine.bus.publish_project`.

### 1.4 Stage changes for skipped segments

- `run_translate` (`stages.py:426`): `targets = [s for s in project.segments if not s.skipped and (s.translate_dirty or not s.translated_text.strip())]`. Also at L427-428, extend the flag-clear loop to skipped ones: `for s in project.segments: if s.skipped: s.translate_dirty = False`.
- `run_synthesize` (L547): `targets = [s for s in project.segments if not s.skipped and (s.synth_dirty or not s.takes)]`; clear `synth_dirty` on skipped segments the same way. `_translation_request` context (L419-420) may keep skipped neighbours as context — that is desirable.
- `translate_one` / `synthesize_one` and `orchestrator.start_segment_job` (L71-91): raise `ValueError("segment is inside a skip range")` → `api/segments.py:regenerate_segment` maps to 409 (add an explicit check there).
- `_stage_needs_run` (L236-239): `any(s.translate_dirty for s in project.segments if not s.skipped)`; same for synth.
- `run_mix` (L603-607): add `if not seg.skipped` to `entries`. Note segments that only *partially* overlap a range (under 50 %) still get placed; the splice below overrides their audio inside the range, which is the intended "original wins inside the range" behavior.

### 1.5 Mix: splicing the original audio back in

Change `audio.mix_tracks` signature:

```python
async def mix_tracks(vocals, background, out_wav, reference_vocals=None, target_lufs=-16.0,
                     background_gain_db=0.0,
                     keep_original: list[tuple[float, float]] | None = None,
                     original: Path | None = None, crossfade: float = 0.05) -> None:
```

Insert between the `amix` (L157) and the master measurement (L159):

```python
if keep_original and original is not None and original.exists():
    tmp_spliced = out_wav.parent / f".{out_wav.stem}.splice.tmp.wav"
    await splice_ranges(tmp_premix, original, keep_original, tmp_spliced, fade=crossfade)
    premix = tmp_spliced   # and unlink in finally
else:
    premix = tmp_premix
```

Why here: the premix is at "source level" (dub vocals matched to the separated original dialogue, bed at unity), so `audio/original.wav` (48k s16 stereo, produced by `ffmpeg.extract_audio` at ingest) drops in at the same loudness, and the single static master gain (L159-163) then moves the whole program, spliced sections included, to the E4 delivery target. Splicing after mastering would require a second gain measurement and would mismatch by the master gain.

New `audio.splice_ranges(base, overlay, ranges, out_wav, fade=0.05)` — pure Python like `_assemble_track_sync`, run in `asyncio.to_thread`:

1. Validate both wavs are 48k/2ch/s16 (same check as L82-90); read `base` fully into a `bytearray`; open `overlay` and for each range `[s, e]` read only frames `[s-fade, e+fade]` (`wf.setpos`, `wf.readframes`), clamped to both files' lengths.
2. Inner part `[s, e]`: byte-slice copy (overlay replaces base), exactly like L103-105.
3. Fade-in `[s-fade, s]` and fade-out `[e, e+fade]`: per-sample linear crossfade using `array('h')` on both slices: `out = base*(1-w) + overlay*w`, `w` ramping 0→1 (in) and 1→0 (out), clip to int16. At 48 kHz stereo that is 4,800 samples per edge — negligible in pure Python. Equal-power (`sin/cos`) is unnecessary for a 50 ms cross on correlated material; use linear.
4. Write with `wave.open(out_wav, "wb")` as in L107-113.

`run_mix` (L630-638) passes:

```python
keep_original=[(r.start, r.end) for r in project.skip_ranges],
original=store.resolve(project, "audio/original.wav"),
```

Detail string (L652-654): append `f"; {len(project.skip_ranges)} original region(s) kept"`.

`dub_vocals.wav` is left as-is (skipped segments are simply absent). `playback_dub.mp4` and `waveforms/dub_mix.json` derive from the spliced `dub_mix.wav`, so the preview and timeline reflect the kept audio automatically.

### 1.6 Lipsync and render

Lipsync providers process the whole file. Keep that, and restore the original picture afterwards in `run_lipsync` (`stages.py:661-674`):

```python
raw = _out(project, "render/lipsync_raw.mp4") if project.skip_ranges else out_video
await provider.sync(playback, dub_mix, raw, _sub(progress, 0.0, 0.9 if project.skip_ranges else 1.0))
if project.skip_ranges:
    await ffmpeg.overlay_ranges(raw, playback, [(r.start, r.end) for r in project.skip_ranges], out_video)
    raw.unlink(missing_ok=True)
```

New `ffmpeg.overlay_ranges(base_video, overlay_video, ranges, out_path)` in `app/media/ffmpeg.py`:

```
ffmpeg -y -hide_banner -loglevel error -i <base> -i <overlay> \
  -filter_complex "[1:v][0:v]scale2ref[ov][base];[base][ov]overlay=eof_action=pass:enable='between(t,12.000,15.500)+between(t,40.000,42.250)'[v]" \
  -map "[v]" -an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart <out>
```

`scale2ref` guards a provider that returns a different resolution; `enable=` takes a `+`-joined sum of `between()` terms (one per range; for hundreds of ranges use `-filter_complex_script` with a temp file). `-an` because render muxes the audio (which already has the original spliced). One extra libx264 pass over the episode is the price; acceptable relative to lipsync inference. `run_render` needs no change.

Stage status: lipsync stays `done`. A future optimisation is to pass only non-skipped shots to the provider (B2 shot list), not needed now.

### 1.7 Smart anchor points

Anchors are derived data, not manifest state. Cache the two expensive sources on disk and merge with segment boundaries at request time.

Files: `analysis/scenes.json` `{"version":1,"source_mtime":<playback.mp4 mtime>,"threshold":0.4,"cuts":[{"t":12.288,"score":0.51},...]}` and `analysis/silences.json` `{"version":1,"source_mtime":<vocals.wav mtime>,"noise_db":-35,"min_dur":0.3,"gaps":[{"start":3.10,"end":3.62},...]}`.

Producers (both cheap; do them unconditionally so `minimal` preset users get anchors):
- `run_ingest` (`stages.py:224-266`), after the waveform at L262: `await analysis.detect_scenes(project)` writing `analysis/scenes.json`. When a real B2 provider lands it overwrites the same file.
- `run_separate` (L272-288), after the vocals waveform at L284: `await analysis.detect_silences(project)` writing `analysis/silences.json` from `audio/vocals.wav`.
- `GET …/anchors?refresh=1` recomputes both synchronously (also the path for projects ingested before this feature); if `analysis/silences.json` is missing and `audio/vocals.wav` is absent, fall back to `audio/original.wav`.

New `app/media/ffmpeg.py` helpers:

```python
async def detect_scene_changes(video: Path, threshold: float = 0.4) -> list[tuple[float, float]]:
    """(pts_time, scene_score) per detected cut."""
```
Command (downscale first; the `scene` score is a frame-difference metric so 320 px is plenty and ~10× faster):
```
ffmpeg -y -hide_banner -loglevel error -i playback.mp4 -an \
  -vf "scale=320:-2,select='gt(scene,0.4)',metadata=print:file=<tmpfile>" -f null -
```
`metadata=print:file=` writes, per selected frame, `frame:N pts:… pts_time:12.288` followed by `lavfi.scene_score=0.512345`; parse with `pts_time:(\S+)` and `lavfi.scene_score=(\S+)`. Using a temp file keeps `-loglevel error` and `_run`'s error handling intact. Alternative: `scdet=threshold=10` (ffmpeg ≥ 4.4) logs `lavfi.scd.score`/`lavfi.scd.time` to stderr at info level, which would need the `measure_loudness`-style stderr capture (L358-386); `select+metadata` is the simpler parse.

```python
async def detect_silences(wav: Path, noise_db: float = -35.0, min_dur: float = 0.3) -> list[tuple[float, float | None]]:
```
Move `providers/asr/mock.py:25-69` here verbatim (it already handles the "silence still open at EOF" case) and have the mock import it. Callers substitute the total duration for `None`.

New `app/pipeline/analysis.py`:

```python
class Anchor(BaseModel):
    t: float
    kinds: list[Literal["scene", "silence", "segment"]]
    score: float = 0.0        # scene score; silence gap length; 1.0 for segment edges
    start: float | None = None  # silence gaps carry their extent
    end: float | None = None

class AnchorSet(BaseModel):
    version: int = 1
    duration: float
    anchors: list[Anchor]
    sources: dict[str, bool]   # {"scene": True, "silence": False, ...} — what was available

async def anchors(project: Project, *, refresh: bool = False, scene_threshold=0.4, min_silence=0.3) -> AnchorSet
```
Merge: silence gaps yield an anchor at the gap midpoint (with `start/end`); segment edges yield anchors at every `seg.start` and `seg.end`; sort all by `t`; coalesce anchors within 0.1 s into one (keep `t` of the highest-priority kind: scene > silence > segment; union `kinds`). Clamp to `[0, media.duration]`.

Route: `GET /api/projects/{pid}/anchors?refresh=0&scene_threshold=0.4&min_silence=0.3` → `AnchorSet`. With `refresh=0` it only reads cached files + segments (fast); with `refresh=1` it runs the detectors (seconds to tens of seconds for an episode) — the frontend should call it with `refresh=1` only from an explicit "Re-analyse" action. Frontend snapping (range handles snap to the nearest anchor within N px) is UI work outside this doc.

### 1.8 Dirty-tracking summary for A

| event | segment flags | stages |
|---|---|---|
| PUT skip-ranges (any change) | newly un-skipped: `translate_dirty` if no translation, `synth_dirty` if no takes; newly skipped: both cleared | `mark_downstream_dirty("synthesize")` → mix, lipsync, render dirty |
| translate/synthesize run | skipped segments excluded; their flags cleared | unchanged |
| transcribe re-run | `apply_skip_ranges` after new segments | unchanged (already dirties downstream) |
| anchors | none (derived) | none |

---

## 2. Feature B: versions of dubbing attempts

### 2.1 Directory layout

```
data/projects/<pid>/versions/<vid>/
  version.json               full Version (below) incl. segments, speakers, skip_ranges
  dub_mix.wav                copy   (rewritten in place by the next mix — see 0.3)
  dub_mix.json               copy   (waveform peaks)
  playback_dub.mp4           copy
  dubbed.mp4                 copy, only if render was done
  lipsync.mp4                NOT stored (large; dubbed.mp4 already contains the picture)
  takes/<seg_id>/take_<n>.wav      hardlink (os.link; copy on EXDEV) — every take listed in segments[].takes
  speakers/<spk_id>/reference.wav  hardlink
```

Rationale: takes and speaker references are immutable and are the files `run_transcribe` rmtree's; hardlinks make the version survive that. The mix/render outputs are rewritten in place so they are copied (`shutil.copy2` in `asyncio.to_thread`; ~10 MB/min for the wav, on local NVMe a second or two per version). `dub_vocals.wav`, `fitted.wav`, `source.wav` are derivable and not stored.

Size guard: `Version.bytes` is recorded; `GET /versions` returns it so the UI can show it; deletion frees copies (hardlinked takes only go when the last link goes).

### 2.2 Models (`app/models.py`)

```python
class VersionMeta(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ver"))
    label: str = ""
    created_at: datetime = Field(default_factory=now)
    kind: Literal["auto", "manual", "restore-point"] = "auto"
    job_id: str | None = None
    source_lang: str
    target_lang: str
    preset: str; runtime: str                    # copied from pipeline for grouping/sorting
    providers: dict[str, str]                    # {"tts": "tts.f5_tts", ...} the six legacy kinds
    stages: dict[str, StageState]                # per-stage status at snapshot time
    outputs: dict[str, str]                      # {"dub_mix": "versions/<vid>/dub_mix.wav", "playback_dub": ..., "dubbed": ...}
    segment_count: int = 0
    take_count: int = 0
    skipped_count: int = 0
    bytes: int = 0
    parent_id: str | None = None                 # version the working state was restored from when this was taken (lineage)

class Version(VersionMeta):                      # what version.json holds
    pipeline: PipelineConfig
    media: MediaInfo | None
    speakers: list[Speaker]
    segments: list[Segment]                      # Take.path rewritten to versions/<vid>/takes/...
    skip_ranges: list[TimeRange]
```

`Project` gains `versions: list[VersionMeta] = Field(default_factory=list)` and `current_version_id: str | None = None` (set on snapshot/restore; cleared by any mutation route — a tiny `project.touch()` helper called from `PATCH /segments`, `PUT /skip-ranges`, `PATCH /projects` pipeline/capabilities/target_lang, and the section-3 routes). `ProjectSummary` gains `version_count: int = 0` and `languages: list[str]` (distinct `target_lang` over versions + current).

Old manifests: all new fields default; nothing to migrate.

`store._reapply_user_edits` (L99-102) rule for `versions`: merge by id — start from `fresh.versions` (on-disk labels/deletions win), then append any entry present in `ours` but not in `fresh` and not in `base` (the job's auto-snapshot). `current_version_id`: on-disk wins unless the job changed it relative to `base` (same pattern as stages L139-142).

### 2.3 Snapshot implementation (`app/pipeline/versions.py`, new)

```python
async def snapshot(project_id: str, *, kind, label="", job_id=None) -> VersionMeta
```
Must be called under `store.lock(project_id)` with a **fresh** `store.load(project_id)` (not the job's tracked copy — the copy is stale by construction; see 0.2). Steps (blocking parts in `asyncio.to_thread`):
1. `vid = new_id("ver")`; `vdir = versions/<vid>`; `vdir.mkdir`.
2. Copy `audio/dub_mix.wav`, `waveforms/dub_mix.json`, `playback_dub.mp4`, `render/dubbed.mp4` when present → `outputs`.
3. For each segment/take: `os.link(take_path, vdir/takes/<seg>/<name>)` (copy on `OSError` EXDEV/EPERM); rewrite `take.path` in the snapshot copy of the segment. Skip takes whose file is missing (drop from the snapshot's `takes[]`; if it was the active take, set `active_take_id=None`).
4. Same for `speakers[].reference_path`.
5. Write `version.json` via tmp + `Path.replace` (same pattern as `store.save`).
6. Append the `VersionMeta` to `project.versions`, set `project.current_version_id = vid`, `store.save(project)`, publish.

Default label: `f"v{len(project.versions)+1} · {target_lang} · {preset}/{runtime}"`.

### 2.4 Where the automatic snapshot happens

`orchestrator._pipeline_runner` (L97-167): after the `for` loop completes (right before `job.message = f"completed {total} stage(s)"` at L142), add:

```python
if "mix" in job.stages and project.stage("mix").status == "done":
    async with store.lock(project_id):
        meta = await versions.snapshot(project_id, kind="auto", job_id=job.id)
    engine.bus.publish_project(store.load(project_id))
```

One version per run (not per stage), taken after the last stage so `render/dubbed.mp4` and `rate_factor` are included when render was part of the run. Runs that don't include `mix` (translate-only, segment jobs, ingest) don't snapshot. A job that errors after mix (e.g. lipsync fails) doesn't snapshot either; the user can take a manual one.

### 2.5 Routes (`app/api/versions.py`, new; mount in `main.py`)

- `GET    /api/projects/{pid}/versions` → `list[VersionMeta]` (newest first). The UI groups by `target_lang`; no server grouping needed. Optional `?lang=` filter.
- `GET    /api/projects/{pid}/versions/{vid}` → `Version` (reads `version.json`; 404 if dir missing even if the index lists it — then also prune the index).
- `POST   /api/projects/{pid}/versions` body `{label?}` → `VersionMeta` (manual snapshot; `409` if busy; `400` if `audio/dub_mix.wav` is missing — nothing to version).
- `PATCH  /api/projects/{pid}/versions/{vid}` `{label}` → `VersionMeta` (updates both `project.versions[i]` and `version.json`; allowed while busy — it goes through the merge rule in 2.2).
- `DELETE /api/projects/{pid}/versions/{vid}` → `{ok}`; `409` if busy (a running snapshot could race); rmtree the dir; drop from index; clear `current_version_id` if equal.
- `POST   /api/projects/{pid}/versions/{vid}/restore` body `{restore_point?: bool = true}` → `Project` (2.6).
- Media: `GET /api/media/{pid}/versions/{vid}/playback_dub.mp4` (and `dubbed.mp4`, takes) already works via `store.resolve`. The frontend's `api.mediaUrl(pid, meta.outputs.playback_dub)` swaps the `<video>` source exactly as it does for the current `playback_dub.mp4`.
- Waveform: extend `GET /api/projects/{pid}/waveform/{asset}?version=<vid>` (`api/media.py:38-50`): when `version` is given and `asset == "dub_mix"`, resolve `versions/<vid>/dub_mix.json`.

### 2.6 Restore semantics

Under `store.lock(pid)`; `409 {"detail": "a job is running; cancel it first"}` if `engine.active(pid)` (the tracked-copy merge in 0.2 cannot survive a wholesale segment replacement).

1. Load `Version` from `version.json` (404 if missing).
2. If `restore_point` and the working state has a `dub_mix.wav` and `current_version_id is None` (i.e. the working state is not already identical to a version), take a `kind="restore-point"` snapshot first, labelled `"before restore of <label>"`, `parent_id=vid`.
3. Replace: `project.segments`, `project.speakers`, `project.skip_ranges`, `project.pipeline`, `project.source_lang`, `project.target_lang` from the version. For each take: rewrite `versions/<vid>/takes/<seg>/<name>` → `audio/segments/<seg>/<name>` and `os.link` it back if the working file is missing (copy on EXDEV). Same for speaker references. `source.wav`/`fitted.wav` are regenerated by the next synthesize/mix as needed.
4. Copy outputs back: `dub_mix.wav`, `waveforms/dub_mix.json`, `playback_dub.mp4`, `render/dubbed.mp4` (copies, never links — the working files get rewritten in place). Delete `render/lipsync.mp4` (stale; render already contains it).
5. `project.stages` = version's stages, except: `lipsync` → `"dirty"` if it was `done` (its file isn't stored; render still has the picture, so only a re-render would need it); any stage whose output file failed to restore → `"dirty"`.
6. Dirty flags come from the snapshot as-is (they were taken right after mix; normally all false). `current_version_id = vid`.
7. `store.save`, publish.

Segments/takes present in the working dir but not in the version are simply no longer referenced (their files stay; a later "prune orphans" is optional).

### 2.7 Minimal multi-language support

Keep exactly one `Project.target_lang` (the *working* language). Versions carry `target_lang`, so the version list is the per-language history. Minimal changes:

- `Take.lang: str = ""` — set in `_synthesize_segment` (`stages.py:537`): `Take(..., lang=project.target_lang)`. Old takes have `""`; the UI treats `""` as "unknown". Lets the inspector label/filter takes when a segment accumulates takes in two languages.
- `PATCH /projects/{pid}` with a changed `target_lang` (`api/projects.py:154-155`) becomes, under the existing lock:
  1. `409` if busy.
  2. If `audio/dub_mix.wav` exists and `current_version_id is None`: auto-snapshot `kind="auto"`, label `"before switching to <new>"` (so the old-language dub is never lost).
  3. Set `target_lang`; for every non-skipped segment `translate_dirty = True` (and `synth_dirty = True` follows automatically when translate writes a new line, L452); `_mark_config_changed(project, "translate", "translation")`-style: translate stage `done→dirty`, `mark_downstream_dirty("translate")`.
  4. `translated_text` and `active_take_id` are left in place (the UI shows them as stale via `translate_dirty`); the next translate+synthesize run replaces them. If the engineer prefers a clean slate, clear `translated_text` and `active_take_id` here — either is consistent.
- "Switch back to ja→en v3" = `restore` of that version (it restores `target_lang` too).
- `ProjectSummary.languages` (2.2) lets the Library card show "en, hi".

Backward compatibility: no existing manifest field changes meaning; `versions=[]` for old projects.

### 2.8 Job-engine interaction

- Add `JobEngine.active(project_id) -> Job | None` (`jobs.py`): first job with status in `("queued","running")`. Use it in restore, delete-version, manual snapshot, PUT skip-ranges, target_lang change, and all section-3 structural routes. Replace the inline loop in `api/projects.py:138-140` with it.
- Snapshot inside the runner runs on the same per-project worker task, so no second job can interleave; routes are excluded by the 409 rule + `store.lock`.
- Cancellation during snapshot: file copies run in `to_thread` and cannot be interrupted; wrap the index update in `asyncio.shield` like the checkpoints at L160-165 so a half-written version dir is either completed and indexed or left unindexed (a startup/`GET /versions` pass can prune dirs without `version.json`).

---

## 3. Feature C: interactive transcript correction

### 3.1 What exists

- `PATCH /api/projects/{pid}/segments/{sid}` (`api/segments.py:53-131`): `source_text` → `translate_dirty+synth_dirty` + `mark_downstream_dirty("translate")` (L97-100, L117-118); `translated_text/speaker_id/emotion` → `synth_dirty` (L102-105); timing → `synth_dirty` (L107-110); `active_take_id` → downstream dirty from synthesize (L122-125). Validation clamps to media duration and requires `start < end` (L77-87).
- `Segment.words: list[Word]` with `confidence` (`models.py:63-70, 86`), filled by `faster_whisper` (L119) and carried through `segmentation.resegment` (`_line` keeps `words=words`, `segmentation.py:21-23`). Persisted and already included in project JSON, but **not** in `web/src/types.ts` and not touched by any route: editing `source_text` leaves `words` stale.
- Per-segment regenerate (`translate`/`synthesize` only).

### 3.2 Missing: word consistency on text edits

Extend `SegmentPatchBody` with `words: list[Word] | None`. Rule in `_update_segment_locked`:
- if `words` is given: validate each `start<=end` within `[segment.start, segment.end]` (after any timing patch), set `segment.words`, and set `source_text = "".join(w.text for w in words).strip()` unless `source_text` was also given (then the client is responsible for consistency).
- if only `source_text` changes: keep `words` when its joined text equals the old `source_text` and the new text has the same number of whitespace-separated tokens (spaced languages) — remap token texts onto existing timings; otherwise set `words = []` (timings unknown; the UI falls back to segment-level highlight). Re-ASR (3.5) restores timings.
- Add `"words"` to `store._USER_SEGMENT_FIELDS`.

### 3.3 Structural routes (`app/api/segments.py`)

All: `async with store.lock(pid)`, `409` if `engine.active(pid)` (see 0.2), then `apply_skip_ranges(project)` (1.2), `store.save`, publish. Return the affected `Segment`(s).

**Split** `POST /api/projects/{pid}/segments/{sid}/split` body `{"at": float}` or `{"word_index": int}` (split *before* word `word_index`; `at` = that word's `start`). Validation: `segment.start + 0.05 < at < segment.end - 0.05`.
- Left keeps `sid`, `end = at`; right is a new `Segment` (`start = at`, `end = old end`), `speaker_id/emotion/notes` copied.
- Text: with words, left gets words with `end <= at`, right the rest (`"".join(w.text)` per side, `.strip()`). Without words, split `source_text` at the whitespace/punctuation boundary nearest to `len(text) * (at - start) / duration`.
- `translated_text = ""` on both (cannot be split faithfully), `takes` stay on the left (history), `active_take_id = None` on both, `translate_dirty = synth_dirty = True` on both; `project.mark_downstream_dirty("translate")`.
- Insert right after left in `project.segments`; keep the list sorted by `(start, end)`.

**Merge** `POST /api/projects/{pid}/segments/{sid}/merge` body `{"with": "next" | "prev"}`. The two must be adjacent in `(start,end)` order (no other segment starts between them); different speakers are allowed (result takes `sid`'s speaker) but the response should say so via a `warning` field, or require equal speakers with `400` — pick one; I recommend allowing it.
- Survivor = the earlier segment's id; `start = min`, `end = max`; `source_text = a + sep + b` where `sep = " "` if `b.words` is empty or `b.words[0].text` starts with a space, else `""` (Japanese); `words = a.words + b.words`; `emotion` from the survivor if set else the other; `notes` joined with `"\n"`.
- `translated_text = ""`, `active_take_id = None`, `takes` = survivor's takes; both dirty flags True; `mark_downstream_dirty("translate")`; remove the other segment (its take files stay on disk; versions may hardlink them).

**Insert** `POST /api/projects/{pid}/segments` body `{start, end, speaker_id?, source_text?: "", emotion?}` → `Segment` (201). Validation as in PATCH (clamp, `start<end`, speaker exists; default speaker = the speaker of the nearest preceding segment, else the first speaker). Overlap with existing segments is permitted (the timeline already tolerates overlap; `assemble_track` "later wins").
- `translate_dirty = bool(source_text.strip())`, `synth_dirty = False` (nothing to voice yet; translate sets it when it writes a line, `stages.py:452`); if it has text → `mark_downstream_dirty("translate")`, else no stage dirtied (outputs are still valid until text arrives).

**Delete** `DELETE /api/projects/{pid}/segments/{sid}` → `{ok: true}`. If the segment had an active take → `mark_downstream_dirty("synthesize")` (mix must drop it); otherwise nothing. Speakers with no remaining segments are kept.

### 3.4 Re-running ASR on one segment

`POST /api/projects/{pid}/segments/{sid}/transcribe` body `{"pad": 0.15}` → `Job` (`kind="segment"`, `stages=["transcribe"]`).

- `orchestrator._SEGMENT_STAGES` (L24) → `("transcribe", "translate", "synthesize")`; `start_segment_job` keeps its ordering logic. `_segment_runner` (L184-193) gains `elif key == "transcribe": await stage_impl.transcribe_one(project, asr_provider, segment, progress)`, and the downstream marking at L200/L207/L212 becomes `mark_downstream_dirty("translate" if "transcribe" in job.stages else "synthesize")`.
- `stages.transcribe_one(project, provider: ASRProvider, seg, progress, pad=0.15)`:
  1. `vocals = _require(project, "audio/vocals.wav", ...)`; `clip = audio/segments/<sid>/asr_input.wav`; `await ffmpeg.slice_audio(vocals, clip, seg.start, seg.end, pad=pad)` (L196-218; note it clamps `start-pad` to ≥ 0, so compute `offset = max(0, seg.start - pad)`).
  2. `raw = await provider.transcribe(clip, project.source_lang, progress)`; apply the same cleaning as L301-305; if empty → `StageError("ASR found no speech in this range")`.
  3. Join texts (`" ".join` for spaced languages, `"".join` when the first word of each piece carries no leading space); words: `start/end += offset`, clamped to `[seg.start, seg.end]`; confidence carried.
  4. `seg.source_text`, `seg.words` set; `translate_dirty = synth_dirty = True`; `seg.skipped` unchanged.
  5. Unlink `asr_input.wav`.
- Timing is not changed by re-ASR (the user chose the range). A follow-up "tighten to words" is just `PATCH {start: words[0].start, end: words[-1].end}`.
- The whole-range variant for "insert + transcribe": client calls Insert (3.3) then this route.

### 3.5 Per-word confidence for highlighting

Already in the model; expose it by adding to `web/src/types.ts`:

```ts
export interface Word { start: number; end: number; text: string; confidence: number | null }
// Segment: words: Word[]
```

`confidence` is `null` for providers without per-word probabilities (`openai_whisper`, `asr.mock`), so the UI must treat `null` as "no data" rather than "low". Suggested threshold: `< 0.5` = red, `< 0.8` = amber (faster-whisper probabilities are fairly well calibrated for the Whisper family).

Payload size (0.4): keep `words` in `manifest.json` but strip them from the SSE/GET payload by default: give `Project` a `public_dump_json()` (`model_dump_json(exclude={"segments": {"__all__": {"words"}}})`) used by `EventBus.publish_project` (`jobs.py:85`), `events.py:31` and `GET /projects/{pid}`; add `GET /api/projects/{pid}/segments/{sid}/words` → `Word[]` (and bulk `GET /api/projects/{pid}/words` → `{sid: Word[]}`) which the transcript panel loads on demand and re-fetches when a `project` event shows a changed `source_text` for that segment. If you'd rather not deviate from "Project JSON mirrors models.py", accept the ~1 MB/event cost for now and note it.

### 3.6 Predicate unification (needed by A and C)

Extract into `stages.py`:

```python
def translate_targets(project) -> list[Segment]:
    return [s for s in project.segments if not s.skipped and s.source_text.strip()
            and (s.translate_dirty or not s.translated_text.strip())]

def synth_targets(project) -> list[Segment]:
    return [s for s in project.segments if not s.skipped and s.translated_text.strip()
            and (s.synth_dirty or not s.takes)]
```

Use them in `run_translate` (L426-429), `run_synthesize` (L547), and in `orchestrator._stage_needs_run` (L236-239) as `bool(translate_targets(project))`. This closes the existing gap where a segment without output but without a dirty flag is processed by the stage yet never selected by "Run pipeline".

### 3.7 Dirty-flag rules for C (complete table)

| operation | `translate_dirty` | `synth_dirty` | `mark_downstream_dirty(...)` | `current_version_id` |
|---|---|---|---|---|
| PATCH `source_text` (exists) | True | True | `"translate"` | cleared |
| PATCH `words` (new) | True | True | `"translate"` | cleared |
| PATCH `translated_text` / `speaker_id` / `emotion` (exists) | – | True | `"synthesize"` | cleared |
| PATCH `start`/`end` (exists) | – | True | `"synthesize"` | cleared |
| split | True (both) | True (both) | `"translate"` | cleared |
| merge | True | True | `"translate"` | cleared |
| insert with text | True | – | `"translate"` | cleared |
| insert empty | – | – | none | cleared |
| delete | – | – | `"synthesize"` if it had an active take, else none | cleared |
| re-ASR (segment job) | True | True | `"translate"` (job end) | cleared |
| re-ASR job cancelled/failed | unchanged | unchanged | none | – |

---

## 4. Everything in the current code that fights these features (with locations)

1. `app/store.py:97-143 _reapply_user_edits` — (a) project-level whitelist L99-102 reverts new fields (`skip_ranges`, `versions`, `current_version_id`, and `target_lang`-driven flag changes) on the job's next checkpoint; (b) L104-109 makes the job's segment *list* authoritative, so structural edits during a job are lost/resurrected; (c) `_USER_SEGMENT_FIELDS` L66-76 lacks `words`/`skipped`. Mitigations: extend the whitelist and the field tuple; merge `versions` by id; 409-while-busy for structural routes, restore, skip-range edits and target_lang switches.
2. `app/pipeline/stages.py:349-356 run_transcribe` rmtree of `speakers/` and `audio/segments/` destroys files versions reference → hardlink takes/references into the version dir at snapshot time (2.1/2.3).
3. In-place rewrites by `ffmpeg -y` / `wave.open("wb")` of `dub_mix.wav`, `playback_dub.mp4`, `render/*.mp4`, `waveforms/dub_mix.json`, `fitted.wav`, `source.wav` → never hardlink these; copy (2.1).
4. Target predicates in `run_translate` L426 / `run_synthesize` L547 and `_stage_needs_run` L230-240 disagree and ignore skipping → 3.6.
5. `api/segments.py:134-147 regenerate_segment` and `orchestrator.start_segment_job` L71-91 don't know about skipped segments → 409 check (1.4).
6. `orchestrator._segment_runner` L200, L207, L212 always dirty from `"synthesize"` → parametrize for the transcribe stage (3.4).
7. `api/projects.py:154-155` `PATCH target_lang` doesn't dirty anything → 2.7.
8. `jobs.py:83-85 publish_project` full-JSON events on every per-segment checkpoint (`stages.py:561`) → words bloat (3.5); versions metadata is small but grows; keep `VersionMeta` lean and never embed segments in it.
9. `orchestrator._pipeline_runner` L105 per-stage tracked reload → the snapshot must load a fresh project under the lock rather than use the runner's copy (2.3/2.4).
10. No `engine.active(pid)`; `api/projects.py:138-140` re-implements it inline → add to `JobEngine` (2.8).
11. `api/media.py:23 WAVEFORM_ASSETS` closed set → `version` query param (2.5).
12. `providers/asr/mock.py:25-69` shells out to ffmpeg outside `media/ffmpeg.py` (rule at `ARCHITECTURE.md:201`) → move `detect_silences` into `ffmpeg.py` and reuse (1.7).
13. `capabilities.py` A3 (L92-96) / D7 (L186-188) / B2 (L125-127) overlap conceptually with skip ranges and anchors; no step providers exist yet (`providers/steps/` is empty). The `TimeRange.source` field and the `analysis/scenes.json` file are the integration points so those capabilities can later populate the same data instead of inventing parallel structures.
14. `run_mix` L620 mutates `take.rate_factor` in place — snapshot after mix (2.4), and restore brings the mutated value back, which is correct.
15. `Project.mark_downstream_dirty` L204-210 leaves `skipped` stages (e.g. lipsync with `lipsync.none`) alone, which is right; but it also leaves `pending` alone, so after a restore that sets `lipsync="dirty"` the "Run pipeline" default re-selects lipsync — intended.
16. `web/src/types.ts:58-71` lacks `words`; the TS contract must grow `Word`, `TimeRange`, `VersionMeta`, `Segment.skipped`, `Project.skip_ranges/versions/current_version_id`, `Take.lang`, and `api/client.ts` methods for the new routes (`putSkipRanges`, `getAnchors`, `listVersions`, `getVersion`, `snapshotVersion`, `labelVersion`, `deleteVersion`, `restoreVersion`, `splitSegment`, `mergeSegment`, `insertSegment`, `deleteSegment`, `transcribeSegment`, `getWords`).

## 5. Suggested implementation order and tests

1. `engine.active`, predicate unification (3.6), `store` whitelist/field additions — small, unblock everything.
2. Feature A models + `regions.py` + PUT route + stage exclusions + `splice_ranges` + `mix_tracks` hook; test: `tests/test_mix.py` style — tone at 1 kHz as "dub", tone at 3 kHz as "original", one skip range; assert band RMS inside the range is the original's and outside is the dub's, and no click at the edges (peak of the first-difference signal below a bound). Then `overlay_ranges` (ffmpeg smoke test on a synthetic 2-colour video: sample a frame inside/outside the range with `ffmpeg -ss … -frames:v 1 -f rawvideo` and check the colour).
3. Anchors: `detect_scene_changes` on a synthetic video with a hard colour cut (`tests/test_e2e.py:make_test_video` pattern), `detect_silences` on tone+silence; route test for merge/coalesce.
4. Feature B: `versions.py` snapshot/restore with the mock pipeline in an e2e test: run pipeline → 1 version; edit a segment → run → 2 versions; restore v1 → segments/stages equal v1, `playback_dub.mp4` served from the version path; re-run transcribe → v1 take files still readable through `versions/<vid>/takes/`; restore refused (409) while a job is queued.
5. Feature C routes with unit tests per dirty-rule row of 3.7; re-ASR segment job with `asr.mock` (it yields a `[line 1]` placeholder — assert words offset/clamping using a fake ASR provider that returns words).