# Next steps

- [ ] Test the new backend quality changes with the complete regression suite.
- [ ] Run real Japanese→Hindi and Japanese→English clips through the updated pipeline; compare
      original speech, raw synthesis, fitted speech, and final mix by listening.
- [ ] Revisit the existing Hindi project: 8/10 active takes exceeded the new timing limit.
      Check regeneration resolves the mismatch without losing meaning or words.
- [ ] Assess every stage and the models/providers used in it:
  - Separation: dialogue leakage, music/effect preservation, clipping, stem duration, SI-SDR when
    clean references exist; compare Demucs models and shift/overlap settings.
  - Transcription: WER/CER on manually verified transcripts, missed speech, word boundaries,
    overlap handling, source-language model accuracy.
  - Diarization: speaker assignment accuracy and identity consistency; assess real diarization
    models on multi-character scenes instead of the single-speaker fallback.
  - References: clean single-speaker audio, accurate reference transcripts, reference selection
    and resulting voice-cloning consistency.
  - Emotion/delivery: validate classifier language support for Japanese/Hindi/English, confidence,
    emotional intensity, pitch variation and pauses; compare source versus generated delivery.
  - Translation: bilingual assessment of meaning, names/terminology, emotion, natural phrasing,
    and measured-duration rewrites; compare configured translation models.
  - Synthesis: intelligibility, pronunciation, missing/repeated words, speaker similarity,
    emotion match, candidate retries, and naturalness; compare TTS models/checkpoints.
  - Timing: verify no clipped words, natural pace, silence handling, and rejection of overflow.
  - Mixing: overlapping voices, dialogue audibility, preserved whispers/shouts, fades, headroom,
    optional leveling/ducking, LUFS and true peak.
  - Lip sync (when enabled): assess audiovisual alignment and artifacts on the intended content;
    structural render checks do not measure phoneme-to-mouth synchronization.
  - Rendering: encoded audio/video synchronization, stream duration, decoding, and true peak.
- [ ] Build a representative benchmark set: quiet speech, shouting, whispers, crying, music,
      overlap, multiple speakers, and very short lines. Add verified transcripts/translations/stems.
- [ ] Compare models one stage at a time; record model/checkpoint versions, settings, quality,
      latency, memory use, and cost. Select defaults from measured results and listening tests.
- [ ] Validate quality-report freshness and completeness, including unavailable metrics; configure
      local ASR/emotion/speaker models and calibrate acceptance thresholds before strict gating.
- [ ] Save baseline/candidate benchmark reports and run regression gates. Require measurement
      coverage as well as average scores; missing measurements are not successful evaluations.

## Commands

From the repository root:

```sh
cd /home/nidhi/code/OpenDub
export OPENDUB_ROOT="$PWD"

# Backend regression tests (no model downloads or cloud APIs).
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 server/.venv/bin/python -m pytest \
  -p pytest_asyncio.plugin server/tests -q

# Run only the quality/audio/emotion regression tests while iterating.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 server/.venv/bin/python -m pytest \
  -p pytest_asyncio.plugin server/tests/test_audio_quality.py \
  server/tests/test_emotion.py server/tests/test_quality.py \
  server/tests/test_runtime_quality.py -q

# Start backend and frontend for manual testing.
make dev
```

In another terminal, run a configured real-model benchmark:

```sh
cd /home/nidhi/code/OpenDub
export OPENDUB_ROOT="$PWD"
cd server
.venv/bin/python -m bench run moshi --timeout 7200
```

Prerequisites: place a source clip at `data/benchmarks/moshi/source.mp4` (repository root),
set `defaults` in the active `configs/settings.yaml` to the real providers being assessed, and
configure/install their models or credentials. Otherwise new benchmark projects use placeholder
providers. The existing `moshi` case targets English; add a separate case with `target_lang: hi`
for Hindi. Each benchmark creates a new project and may invoke configured cloud providers.

Compare two result paths printed by the benchmark command (replace OLD and NEW):

```sh
.venv/bin/python -m bench compare bench/results/OLD.json bench/results/NEW.json
.venv/bin/python -m bench gate bench/results/OLD.json bench/results/NEW.json \
  --require tts.round_trip_cer --require mix.true_peak
```

Inspect the reports written for a project after a run:

```sh
find data/projects/PROJECT_ID/quality -type f -name '*.json' -print
python -m json.tool data/projects/PROJECT_ID/quality/mix.json
python -m json.tool data/projects/PROJECT_ID/quality/render_validation.json
```

Use `configs/quality.example.yaml` and `docs/backend-quality.md` for local verifier/model setup.
Do not overwrite existing provider credentials when merging the quality configuration.
