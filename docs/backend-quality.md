# Backend dubbing quality

Quality checks run inside the existing stages. There are no editor changes. Pipeline completion
means the implemented technical gates passed, not that a human has approved translation,
performance, or separation quality.

## Processing changes

| Stage | Behavior | Measurement / limitation |
|---|---|---|
| Separation | Demucs defaults to two shifts and 50% chunk overlap; output stems must be readable and retain source duration | Audio level/clipping, stem duration; SI-SDR requires reference stems. Stem level ratios are not leakage detection |
| Transcription | Reject invalid timestamps, retain valid word alignment, keep overlapping word runs separate | Coverage, overlaps, invalid word bounds, reference WER/CER in benchmarks |
| References | Choose isolated, unclipped 1–10 second speech with ASR-confidence ranking; save matching transcript | Per-speaker candidate scores and exclusion reasons; no clean candidate is reported explicitly |
| Emotion | Analyze original audio before translation; use confidence-gated emotion and acoustic delivery features | Language-aware optional classifier; pitch variation, intensity variation, pauses/activity; unknown labels stay unknown |
| Translation | Include scene neighbors, emotion, explicit terminology, and measured-duration feedback | Reject empty results; report length budgets; semantic adequacy still requires a judge or bilingual reference |
| TTS | Bounded candidate retries, audio checks, optional independent target-language ASR, emotion mismatch retry | Persist each attempt; only accepted takes become active. Optional identity cosine is a diagnostic |
| Timing | Trim substantial outer silence conservatively, retain short takes' natural pace, cap acceleration at 1.15× | Overflow triggers retranslation/re-synthesis with compatible LLM translators; unresolved overflow fails, never chops words |
| Mixing | Sum overlaps with headroom, edge fades, disk-backed accumulation, float premix, restrained optional leveling/ducking | LUFS, true peak, clipping, estimated same-window stem balance, fitted-speech checks |
| Render | Check encoded streams, duration, start offsets, full decode and encoded true peak | Structural synchronization, not phoneme/mouth alignment |

Emotion is transmitted through the existing request hint and suitable source style reference.
F5 relies on audio conditioning (it has no generic emotion-tag API). Eleven v3 uses its supported
bracketed delivery tags; other Eleven models are not sent invented tags. Confident categorical
source/output mismatches can cause retries, but acoustic comparison scores are descriptive
heuristics, not calibrated emotion accuracy or MOS scores. An explicit per-line emotion override
wins over automatic conditioning.

Diarization quality still depends on the selected provider. `diarization.single_speaker` cannot
preserve distinct identities in a multi-character scene; use a real diarization provider for that
material. Neither loudness nor a source separation model guarantees absence of original speech
in the background. Listen to both stems and include real examples in the regression corpus.

## Configuration without UI changes

Merge the `quality.runtime` mapping in `configs/quality.example.yaml` into `providers` in the
active settings file. Existing provider settings/credentials must be retained. All runtime policy
keys also accept `OPENDUB_QUALITY_RUNTIME_<KEY>` environment overrides.

- `asr_model`: **local Faster-Whisper model directory**. Missing model/packages produce an
  explicit unavailable measurement. Set `require_verification: true` for production acceptance.
- `speaker_model`: local SpeechBrain ECAPA directory including its hyperparameters and weights.
  Optional diagnostic; values need calibration on the actual voices/languages.
- `emotion_model`: `auto` uses cached English SUPERB; it does not apply English labels to Japanese
  or Hindi. A local Transformers classifier or `funasr:/absolute/model-directory` can be configured.
  `emotion_languages` explicitly declares languages validated for that model/material. Do not
  interpret this allowlist as evidence that arbitrary model weights support those languages.
- `require_emotion_match`: default true; reject mismatches only when both predictions are confident
  and from the same model. An unavailable classifier does not manufacture a match.
- `tts_attempts`: default 3, clamped to 1–5. F5 additionally has its own bounded verification retries.
- `max_cer`: default 0.20, provisional. WER is also recorded; CER handles unspaced scripts and
  short lines more consistently. Both normalize punctuation and preserve Unicode combining marks.
- `level_outliers`: default false. When enabled, speech-level outliers over 6 dB from the median
  receive at most 3 dB correction. Keep disabled if intended whispers/shouts are being flattened.
- `ducking_mix`: default 0 (preserve original bed). Nonzero blends selective sidechain compression
  under speech. Avoid double-ducking an already balanced soundtrack.

Optional packages can be installed with the server extras `asr`, `quality-emotion`, and
`quality-speaker`. Model weights must be installed separately. No new classifier automatically
downloads weights during a normal job. F5's own verifier defaults on and needs a cached
`large-v3-turbo` model; it gives an explicit error when unavailable. `tts.f5_tts.verify: false`
disables that provider-specific pass; use the independent runtime verifier in its place.

Provider-specific backend settings for Demucs: `shifts` (1–10, default 2) and `overlap` (0–0.9,
default 0.5). These trade compute time for stabilization; validate on your clips before increasing.

## Model loading and unloading

Every in-process model (faster-whisper, F5-TTS and its verifier, pyannote, the runtime verifier,
emotion and speaker models) is loaded through one process-wide manager
(`server/app/providers/_runtime.py`). A model loads on first use and is shared while in use.
Each pipeline stage keeps the models it loaded warm until the stage ends. At a stage boundary,
idle models the next stage does not use are unloaded. When a pipeline job ends (done, error or
cancelled) every idle model is unloaded. Work that is still running in an uncancellable worker
thread releases its model as soon as it finishes. Segment jobs keep local models warm for the
idle TTL, so regenerating line after line doesn't reload them. They release LLMs as soon as they
finish. Server shutdown unloads everything.

| Setting | Default | Meaning |
|---|---|---|
| `models.idle_ttl_s` | `60` | Unload a model this many seconds after its last use. `0` unloads as soon as the last user (or stage) releases it. |
| `models.gpu_budget_gb` | total VRAM − 1 GB | Before a load that would exceed this (by rough per-model estimates), idle models are evicted, least recently used first. `0` = auto; no limit when VRAM can't be detected. |
| `providers.translation.openai_compatible.unload_after_use` | `true` | When the server is Ollama (port 11434, or it answers `GET /api/tags`), ask it to unload the model (`keep_alive: 0`) when the manager releases it. This happens after the stage, at job end, or after the idle TTL. No effect on other servers. |
| `providers.tts.f5_tts.unload_after_use` | `true` | The same, for F5's reference-transliteration LLM. |

`models.*` also accept `OPENDUB_MODELS_IDLE_TTL_S` / `OPENDUB_MODELS_GPU_BUDGET_GB`.
`GET /api/system/models` lists resident models with their refcount, age, idle time and VRAM
estimate, plus the `nvidia-smi` totals and this process's VRAM. `POST /api/system/models/unload`
with `{"key": "<key>"}` unloads one model, and with no body unloads all of them. A model in use
unloads the moment its current work finishes. Demucs and the lip-sync models run as subprocesses
and free their memory when they exit.

## Reports and regression checks

Reports are under each project directory:

- `quality/<stage>.json`: stage status, safe provider identifiers/settings, measurements and failures.
- `quality/segments/<id>.json`: source delivery, candidate attempts, verification, timing, identity,
  and fitted delivery. Unknown model measurements have a reason.
- `quality/references/<speaker>.json`: candidate scores/exclusions and selected reference.
- `quality/mix_levels.json`: applied gains, target loudness and ducking settings.
- `quality/render_validation.json`: encoded-media validation and true peak.

Reports are copied into version snapshots. A failed stage labels existing media as potentially
stale. Re-synthesis clears old fitted verification; measurements are associated with take IDs.
Reports deliberately exclude API keys and arbitrary provider options. They contain transcripts
and media paths, so they have the same privacy expectations as the project itself.

Run from `server/`:

```sh
.venv/bin/python -m bench run moshi
.venv/bin/python -m bench compare bench/results/OLD.json bench/results/NEW.json
.venv/bin/python -m bench gate bench/results/OLD.json bench/results/NEW.json \
  --require tts.round_trip_cer --require mix.true_peak
```

`gate` returns nonzero for pipeline errors, lost measurements, required metrics that are unknown,
and directional regressions. A single global `--max-regression` is an absolute tolerance in each
metric's unit; use carefully when comparing different metric types. Configure/inspect verification
coverage as well as averages so a few measured lines cannot hide unmeasured failures.

Expand `data/benchmarks/cases.yaml` with real clips covering quiet dialogue, shouting, whispering,
crying, background music, multiple voices, overlap, short lines, and Japanese→Hindi/English.
Supply `ref_transcript.jsonl`, `ref_translation.jsonl`, and clean stems where available. No ground
truth has been invented for existing clips. Pin model revisions and preserve settings; compare
one change at a time with bilingual listening ratings for meaning, identity, emotion and naturalness.

## Initial existing-output audit

A read-only audit of `prj_6504aae1` found 8 of 10 active takes exceeded 1.15× slot duration;
6 needed more than the previous 1.6× cap. The largest was 35.912 seconds for 9.820 seconds.
The saved mix measured −16.0 LUFS and −1.5 dBTP. These measurements identify duration mismatch
as a concrete issue; they are not evidence that new models or emotion conditioning have been
validated by listening on that project. Existing user outputs were not regenerated or overwritten.

## Upstream references

- [Demucs separation CLI](https://github.com/facebookresearch/demucs/blob/main/demucs/separate.py)
- [emotion2vec+ model card](https://huggingface.co/emotion2vec/emotion2vec_plus_base)
- [Eleven v3 audio tags](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-do-audio-tags-work-with-eleven-v3-alpha)
