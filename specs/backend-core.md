# Backend core contract — jobs engine + orchestrator public surface

Two work packages depend on this file agreeing: **B1** (implements `app/jobs.py`,
`app/pipeline/orchestrator.py`, `app/pipeline/stages.py`) and **B7** (implements `app/api/*` routes
that call it). Implement exactly these signatures.

## `app/jobs.py`

```python
class EventBus:
    def subscribe(self, project_id: str) -> tuple[str, asyncio.Queue]     # (sub_id, queue of (event_name, json_str))
    def unsubscribe(self, project_id: str, sub_id: str) -> None
    def publish_job(self, job: Job) -> None                                # event "job"
    def publish_project(self, project: Project) -> None                    # event "project"

class JobEngine:
    bus: EventBus
    def start(self) -> None                       # create worker task(s); called from lifespan
    async def shutdown(self) -> None
    def submit(self, job: Job, runner: Callable[[Job], Awaitable[None]]) -> Job
        # queue job, publish, run runners sequentially PER PROJECT (one at a time per project;
        # different projects may run concurrently). Runner exceptions => job.status="error",
        # job.error=str(e). asyncio.CancelledError => "cancelled". Always publish updates.
    def get(self, job_id: str) -> Job | None
    def list(self, project_id: str | None = None) -> list[Job]            # newest first, keep last ~50 finished
    async def cancel(self, job_id: str) -> Job | None                     # cancel queued or running

engine = JobEngine()   # module singleton
```

Job progress: runners mutate `job.progress/stage/message` and call `engine.bus.publish_job(job)`;
throttle publishes to ≥10 per job but ≤ ~5/sec.

## `app/pipeline/orchestrator.py`

```python
def start_pipeline_job(project_id: str, stages: list[StageKey] | None = None) -> Job
    # stages=None => every stage in STAGE_ORDER whose status != "done" (and lipsync when provider
    # is lipsync.none => run but mark "skipped"). Explicit list => run exactly those, in STAGE_ORDER order.
    # kind = "ingest" if stages == ["ingest"] else "pipeline".

def start_segment_job(project_id: str, segment_id: str, stages: list[str]) -> Job
    # stages ⊆ {"translate", "synthesize"}; kind="segment". Runs those for ONE segment,
    # then marks mix/lipsync/render dirty and publishes the project.
```

Both raise `KeyError` for unknown project/segment, `ValueError` for bad stage names / already-running
job on that project *if you choose to enforce it* (default: allow queueing; engine serializes per project).

Stage execution lives in `app/pipeline/stages.py`: one `async def run_<stage>(project, job, progress)`
per stage. Orchestrator: loads project via `store.load`, instantiates providers from
`project.pipeline.choice(kind).provider_id` + merged options (`config.provider_options(provider_id)`
overlaid with the choice's per-project options), checks `available()` and fails the job with a clear
message when a provider is unusable, saves the project + publishes `project` event after each stage.

Stage behaviors (see ARCHITECTURE.md pipeline table for artifacts):
- **ingest**: probe `source.*` → `project.media`; make `playback.mp4`; extract `audio/original.wav`;
  `waveforms/original.json`. (Upload itself is done by the route before the job starts.)
- **separate**: provider → `audio/vocals.wav` + `audio/background.wav`; `waveforms/vocals.json`.
- **transcribe**: ASR on `vocals.wav` → segments (replace `project.segments`; fresh ids); diarize →
  map labels to `Speaker` entries ("Speaker 1"… with `SPEAKER_PALETTE` colors); build per-speaker
  reference wav: concat that speaker's longest/clearest segments (sliced from `vocals.wav`) up to 30 s
  → `speakers/<id>/reference.wav`. Segment emotion: leave `""`.
- **translate**: for segments where `translate_dirty or not translated_text` build `TranslationRequest`
  (with ±2 lines context, speaker name, slot duration) → provider (batch) → set `translated_text`,
  clear `translate_dirty`.
- **synthesize**: for segments where `synth_dirty or no takes`: slice the segment's own source audio
  from `vocals.wav` → `audio/segments/<sid>/source.wav` (style reference); TTSRequest with speaker
  reference + segment reference + emotion + target_duration; provider writes
  `audio/segments/<sid>/take_<n+1>.wav`; append `Take` (probe duration), set `active_take_id`, clear
  `synth_dirty`. Continue past per-segment provider errors; collect and report count in stage detail;
  stage fails only if ALL segments failed.
- **mix**: for each segment with an active take: `fit_to_duration` → record `rate_factor` on the take;
  `assemble_track` at segment starts over project duration → `audio/dub_vocals.wav`; `mix_tracks` with
  `audio/background.wav` → `audio/dub_mix.wav`; `waveforms/dub_mix.json`.
- **lipsync**: provider `lipsync.none` → status "skipped"; else provider(`playback.mp4`,
  `audio/dub_mix.wav`) → `render/lipsync.mp4`.
- **render**: mux `render/lipsync.mp4` (if exists & lipsync done) else `playback.mp4` with
  `audio/dub_mix.wav` → `render/dubbed.mp4`.

Stage status transitions: set `running` (publish) → `done`/`skipped`/`error` with `detail` + timestamps.
A failing stage stops the run (later stages stay as they were).

## Dirty rules (applied by B7 in PATCH segment route, helper welcome in models.py)

- `source_text` changed → `translate_dirty=True, synth_dirty=True`
- `translated_text` | `speaker_id` | `start` | `end` | `emotion` changed → `synth_dirty=True`
- `active_take_id` changed → (no segment flags)
- any of the above → `project.mark_downstream_dirty("translate" if source_text else "synthesize")`;
  for `active_take_id` only, `mark_downstream_dirty("synthesize")` (mix must rerun).
- Save project + `engine.bus.publish_project(project)` after mutations.
