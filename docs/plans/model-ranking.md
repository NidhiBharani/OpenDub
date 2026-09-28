# Plan: rank every model in every category, per language, and pick defaults

Goal: for each of the 41 capabilities (A1–G6), measure every candidate model with metrics specific
to that category, separately for each language or language pair, and turn the results into
defaults. Each default is keyed by **(compute profile, language, capability)**.

Compute profiles, in the order they come online:

1. **`thalassa-16g`**, now. RTX 5060 Ti 16 GB (SM 12.0, x86_64), 24 GB RAM, 12 cores.
2. **`api`**, later. Hosted models, with cost recorded.
3. **`spark`**, later. DGX Spark GB10, 128 GB unified memory, aarch64, SM 12.1.

Inputs:
- Candidate lists: the "Forty-one jobs, three budgets" compute-tiers artifact, mirrored in
  `docs/model-candidates-by-compute.md`.
- Critique of those candidates: `docs/model-candidates-assessment-2026-09-18.md`.
- Metric research: two Opus passes, dated 2026-09-27. Their conclusions are folded into §4 and
  the appendix.

Status: decisions taken 2026-09-27 (§9). M0 in progress.

---

## 0a. Decisions 2026-09-28

- **Code only from here.** The user orchestrates every benchmark run. Nothing runs models
  implicitly.
- **Every candidate in both research artifacts** (Model Research ledger, 310 rows; Compute tiers
  picks and alternatives) is registered.
  - Each candidate carries hardware `requires`, so `arena plan` shows what runs on this machine:
    Thalassa 16 GB, DGX Spark (aarch64, ~112 GB unified), or API (keys present).
  - Results from different machines are combined with `arena merge`.
- **Graceful model lifecycle in the app.** Nothing stays resident when idle. That covers the
  idle-TTL model manager and the Ollama `keep_alive: 0` unloads. Arena workers are
  one-process-per-job.
- **IndicConformer is disabled for now.**
- **SPY×FAMILY ep. 1** (official Muse Asia uploads: Japanese, plus the official English and Hindi
  dubs) is the headline ja→en / ja→hi material. It lives in the `SCENE` pack, and the official
  dubs are the human references.
- **OpenDub server stopped** (it was idle but held 3.9 GB of VRAM).

## 0. Implementation status

| Piece | Where | State (2026-09-27) |
|---|---|---|
| Worker protocol (stdlib SDK, per-process VRAM via nvidia-smi, CUDA-12 lib preload) | `server/bench/arena/workers/_sdk.py` | done |
| Candidate registry + isolated worker envs | `bench/arena/registry.py`, `bench/candidates/*.yaml`, `_envs.yaml` | done |
| Eval packs + builders (FLEURS en/hi/ja, 300 items each) | `bench/arena/packs.py`, `builders.py`, `data/eval/A4/*` | done |
| Content-addressed generation cache, kept outputs | `bench/arena/runner.py`, `data/arena/outputs/` | done |
| Results store | `bench/arena/db.py`, `data/arena/arena.sqlite` | done |
| Judges (versioned, re-score without regenerating) | `bench/arena/judges.py` | A4 only |
| Paired group bootstrap, Holm, winner sets | `bench/arena/stats.py` | done |
| Leaderboard + audit viewer (ok/bad/note marks) | `bench/arena/report.py`, `serve.py` | done |
| **Stage ladder** (one real dub, line × stage, round-trip CER per take) | `bench/arena/ladder.py`, `/ladder/<project>` | done |
| CLI | `python -m bench arena pack|run|score|report|ladder|serve` | done |
| Cascade with arena winners per stage | — | next, after C2 and D1 have candidates |
| Hardware detection + `requires` eligibility, `arena hw` / `arena plan` | `bench/arena/hardware.py` | done |
| Env recipes per arch, `arena env list/setup/check` | `bench/arena/envs.py`, `bench/envs/*.yaml` | done |
| Spec registry (one module per capability) | `bench/arena/specs/` | A4 done; others by phase agents |
| Model judges (isolated, cached per output), derived metrics, gates | `bench/arena/judges.py`, `judgelib.py` | done |
| Corpus metrics (SRCC/Kendall/pairwise acc.) for judge meta-evaluation | `judges.CorpusMetric`, `stats.rank_corpus` | done |
| API support: `kind: api`, per-item cost, `--budget-usd` | `workers/_sdk.py`, `runner.py` | done |
| Portable outputs + `arena merge` (Thalassa ↔ Spark) | `db.py` | done |
| Contributor contract | `bench/arena/CONTRIBUTING.md` | done |
| **All 41 capabilities registered**: 772 candidates (479 local, 203 API, 90 methods; 249 disabled with reasons), 217 workers, 116 env recipes, specs + builders + model-free tests (300 tests pass) | `bench/candidates/`, `bench/arena/{specs,workers,builders}/`, `bench/envs/` | code done 2026-09-28 — **nothing but A4 has produced real outputs yet** |
| Model documentation (generated from the registry) | `docs/arena-models.md` via `arena docs` | done |
| Load/unload smoke tests (workers + app model manager) | `bench/arena/smoke.py` via `arena smoke` | app: 4/4 ok; workers: run started 2026-09-28 |
| App model lifecycle (idle TTL, VRAM budget, stage/job-end unload, Ollama keep_alive 0) | `app/providers/_runtime.py`, `/api/system/models` | done |
| SPY×FAMILY ep.1 scene pack (ja + official en/hi dubs, aligned, 4 clips) | `builders/spyfamily.py`, `data/eval/SCENE/ja` | done |
| `select` → `configs/defaults.by_language.yaml` | — | M6 |

**First results (A4, FLEURS test, 300 clips per language, 2026-09-27).** en: Qwen3-ASR-1.7B has
the best WER at 3.2%; fw-large-v3 (4.0%) is also in the winner set; the turbo baseline (4.5%) is
out. hi: Qwen3-ASR has 13.4% WER against 30.2% for turbo, the only winner. ja: fw-large-v3 has
4.9% CER, with turbo and Qwen3 also in the winner set; kotoba-whisper-v2 is 7.5%.
IndicConformer is registered but waits on the gated HF repo (needs HF_TOKEN).

**Known hazards found while running:**
- Ollama keeps gemma3:12b resident (8.9 GB) after translation, so later GPU stages go out of
  memory. Unload it with `keep_alive: 0` between stages.
- CTranslate2 word timestamps segfault on distil-Whisper (kotoba), so that candidate runs with
  `word_timestamps: false`.
- Whisper's English normaliser drops parenthesised text, so FLEURS references are unbracketed
  first (judge `text_errors@3`).

## 1. What exists today, and why the current bench cannot do this

The current bench is `server/bench`.

- **Whole-pipeline only.** One run executes the full pipeline with the global `settings.yaml`
  defaults. There is no per-stage isolation, no provider override and no language axis. Because of
  that, a TTS score also contains ASR, translation and segmentation noise.
- **No test media.** The only registered case, `moshi` (ja→en), has no directory. There is no
  ground truth.
- **Missing metrics.** There is no MOS predictor, COMET or MetricX, DER, or lip-sync score.
- **Judge problems.** The translation judge prompt hardcodes English. It also uses the same model
  family as the translator.
- **Outputs.** Each run writes flat JSON. There is no cache, no results store, no confidence
  intervals and no significance tests.
- **Providers.** 30 of 41 capabilities have no real provider. `ProviderMeta` does not record
  languages, VRAM, licence or version. `RUNTIME_DEFAULTS` is keyed by runtime and kind only, not
  by language.

Things to reuse:
- The `Metric` record and `score(ctx)` structure.
- `quality.text_error_rates`. It is Unicode-aware and already keeps Indic matras.
- The runtime Whisper verifier and the ECAPA and emotion hooks.
- The `compare` and `gate` CLI.
- The subprocess-with-own-venv pattern used by the Wav2Lip and LatentSync providers.

## 2. Architecture: a stage-isolated "model arena"

The arena lives in the new `server/bench/arena/`. The existing whole-pipeline bench stays as the
end-to-end regression gate.

```
 eval packs (frozen inputs + refs)      candidate registry (yaml)
            │                                   │
            ▼                                   ▼
   ┌──────────── generate ────────────┐   one worker per candidate, isolated env,
   │ worker(candidate, pack, lang)    │── target = local | ssh://spark | api
   └──────────────┬───────────────────┘
                  ▼  content-addressed output cache
   ┌──────────── judge ───────────────┐   judges are workers too (G-category models)
   │ judge(output, refs) → metric rows│
   └──────────────┬───────────────────┘
                  ▼
        results.sqlite ──► aggregate (paired bootstrap, Holm, winner sets)
                  ▼
   leaderboard report  +  generated configs/defaults.by_language.yaml
```

The key decisions:

1. **Stage isolation.** Each category is evaluated on *frozen, gold, or fixed upstream inputs*. For
   example, TTS gets a fixed text, a fixed reference clip and a fixed target duration. Every
   candidate is compared on the same items, which makes the comparison paired. Upstream variance
   never leaks into a category's score. Interactions between categories are tested separately in
   the end-to-end stage (§6, M7).

2. **One worker protocol, three uses.** Each candidate runs as an out-of-process worker. It reads a
   JSONL manifest (item id, input paths, params) and writes one output file per item plus a
   `worker.json` record. That record holds the model id, revision hash, env lock hash, peak VRAM
   measured with `torch.cuda.max_memory_allocated` plus NVML inside the worker, wall time, RTF and
   token or cost counters.
   - The same worker becomes the pipeline provider once a candidate wins, so there is no second
     integration.
   - The same protocol runs over `ssh` on the Spark and becomes an HTTP call for APIs.
   - This is also the `ModelManager` and remote-GPU abstraction that the capability-map plan left
     unbuilt.

3. **Environment isolation.** Each worker family gets its own `uv` venv under
   `~/.opendub/envs/<family>`. The torch pin is overridden to cu128 or cu130 so it has sm_120
   kernels; the LatentSync venv already shows this works. NeMo, fairseq, vLLM and llama.cpp run in
   NGC or official containers; Docker and the NVIDIA Container Toolkit are already installed. The
   server venv never imports a candidate model.

4. **Two caches.**
   - Generation outputs are keyed by `(candidate, revision, params, input hash)`.
   - Judge scores are keyed by `(judge, judge revision, output hash, ref hash)`.
   - So adding a candidate never re-runs the others, and adding or changing a judge never
     re-generates anything; it re-scores cached outputs. Either change invalidates only the
     affected rows.

5. **Results store.** `server/bench/results/arena.sqlite` holds these tables: `runs`, `items`,
   `outputs`, and
   `scores(category, lang, candidate, item, metric, judge, value)`. One row per item makes paired
   statistics possible. Aggregates are views, never stored.

6. **Candidate registry.** Each category has a file `server/bench/candidates/<ID>.yaml`. Every entry
   records:
   - model id and pinned revision
   - worker family and compute profiles
   - languages declared by the vendor
   - licence, and whether it may ship (`ship: true | eval-only`)
   - params
   - a `verified | inferred | untested` flag for artifact, licence, languages and VRAM, as the
     assessment doc asks

   The table in §7 is the seed list.

## 3. Languages

Each language, or language pair for the text categories, is a separate leaderboard. Scores are
never pooled across languages. The working set follows the UI list and current use (ja→en, ja→hi,
Hindi dub).

| Tier | Languages | Why |
|---|---|---|
| **L1** (the only tier in scope now) | en, hi, ja | current directions ja→en, ja→hi, en→hi |
| L2 (later) | es, fr, de, pt, zh, ko | major dub markets, good public data |
| L3 (later) | ta, te, bn | Indic low-resource; the hardest data situation, and the D4 case |

Text and translation categories use the directions ja→en, ja→hi, en→hi and hi→en. The headline
test is **ja→en and ja→hi on anime**; see §5 for the sources.

Rule: if a candidate does not declare a language, it is evaluated there only when that is cheap,
and it can never become that language's default until the user has audited its outputs. Undeclared
languages are where silent failure happens (e.g. accent, script and matras bugs).

## 4. How ranking works, across all categories

These rules are shared. The per-category metrics are in the appendix.

1. **Gates, then rank.** A candidate must pass all of these before it is ranked, in each language:
   - licence recorded. Decision 2026-09-27: non-commercial models may be defaults for now. Every
     candidate still carries its `license` and `ship_ok` fields, so `select --commercial-only` can
     re-run the choice later without re-running anything.
   - fits the profile alongside its co-resident models
   - failure or degenerate rate < 1%
   - the category's own catastrophic-error gate, e.g. hallucinated words per hour of music, dialogue
     ghosting in the bed, boundaries off by more than 200 ms, CER_ratio ≤ 1.5

2. **Primary metric, per language.** The *winner set* is every candidate that is not significantly
   worse than the best:
   - paired bootstrap, 10k resamples, over the natural independent unit (file, speaker, video or
     title, never frames)
   - Holm correction across the candidate grid
   - also within a practical threshold (1 WER point, 0.5 dB, 1 MetricX point, 1 DER point, 0.01
     SIM_norm)

3. **Inside the winner set, a secondary composite.** Rescale each metric as the fraction of the gap
   closed between the current OpenDub default and the best observed value, then take a weighted
   mean. Weights come from dubbing impact and are fixed per category *before* results are seen.
   Raw z-scores are never averaged across metrics.

4. **Tie-breakers, in order.**
   1. RTF on the profile's hardware
   2. peak VRAM
   3. licence clarity
   4. language coverage (fewer routing branches)
   5. maintenance status

   A win inside the confidence interval never justifies a model that is 3× slower.

5. **Global versus per-language default.** Pick the model that is in the winner set for the most
   languages, weighted by expected use. Add a per-language override only when another model wins
   significantly *and* clears the practical threshold.

6. **Normalise per language against humans.**
   - `WER_ratio = WER_synth / WER_human`, where WER_human is the same judge on real recordings of
     the same texts.
   - `SIM_norm = (SIM − SIM_diff) / (SIM_same − SIM_diff)`, calibrated per language pair.
   - Absolute thresholds shared across languages are meaningless.

7. **Keep judges and generators apart.**
   - A judge never shares a lineage with what it ranks, or with anything the candidate was trained
     or rewarded against. Examples:
     - a CAM++-family SIM judge favours CosyVoice
     - a Whisper verifier inflates Whisper WER gains
     - SyncNet LSE-C is exploited by SyncNet-supervised lip-sync models
     - MetricX as reranker and MetricX as judge is circular
   - Every speech category is scored by at least two ASR families. Round-trip speech recognition
     always includes a CTC model with no language model, which does not "repair" misreadings.

8. **Text normalisation traps.**
   - ja, zh and ko use CER. zh is folded from Traditional to Simplified with OpenCC. ko drops
     spaces.
   - ja also gets Kana-CER, via pyopenjtalk readings or a kana-output ASR.
   - Indic uses CER as primary, with IndicNLP normalisation. Whisper's `BasicTextNormalizer`
     defaults strip matras and understate Hindi WER by about 11 points; `text_error_rates` already
     avoids this.
   - Numerals are normalised in every language.

9. **No human raters: the user audits (decision 2026-09-27).** Every stage output is kept for
   the user to inspect. Automatic metrics rank; the user audits and can overrule.
   - **The audit trail keeps everything.** Each run writes every candidate's output per item and
     per stage: audio, text, video and JSON. They are saved under
     `data/arena/runs/<run>/<stage>/<candidate>/<item>.*` next to the metric rows, and cache
     eviction never deletes them.
   - **The audit viewer** is a static HTML page per run, served on the LAN IP.
     - For each stage, it shows items as rows and candidates as columns, with players for the
       inputs and outputs, the reference, and the per-item metrics.
     - Sorting by "worst metric first" puts the failures on top.
     - The user can mark an item ok, bad or a note. Those marks are saved to `audit.jsonl` through
       a tiny local endpoint and feed into judge meta-evaluation (M1) and tie-breaks.
   - **Cascade view: how good each stage is once it sits on top of the previous ones.** Each
     stage is scored twice.
     - *Isolated:* on gold or frozen inputs, as in §2.1.
     - *Cascaded:* on the real outputs of the chosen upstream stages.
     - The cascade grows one stage at a time, in pipeline order:
       1. A1
       2. A1→A4
       3. A1→A4→A6
       4. … →C1→C2
       5. … →D1
       6. … →E1→E4
       7. … →F1
     - At each prefix the audit viewer shows that stage's output and its metrics, and the
       **Δ(cascaded − isolated)**. That delta is the quality lost to upstream error.
     - The final prefix is the full dub, so the user can watch a ja→en or ja→hi clip being built
       up stage by stage.
   - **The user's marks win** over automatic metrics wherever they disagree. Each disagreement is
     logged as meta-evaluation data for the G judges.

10. **Sample sizes.** These are starting points; widen the set if the 95% confidence interval is
    wider than the practical threshold.
    - about 300 utterances per language for WER/CER
    - about 200 for SIM and MOS
    - 50–100 per system for TTSDS2
    - 500 segments per direction for MT
    - 20 recordings or 3 h for diarization
    - at least 1,000 word boundaries per language for alignment

## 5. Evaluation data: "eval packs"

An eval pack sits at `data/eval/<ID>/<lang>/`. It holds `manifest.jsonl` (id, inputs, refs, split)
and `LICENSE.md` (the source, the terms, and `eval-only` if applicable).

- **Public sets are downloaded by script, never committed.** Examples: FLEURS, Common Voice, DnR
  v3, VoxConverse, AMI, WMT24++, IN22-Conv, IndicVoices-R, Rasa, Seed-TTS-eval, CV3-Eval, MiniMax
  multilingual, VoiceMOS/BVCC/SOMOS, and the WMT23 QE DA sets. The full list per category is in the
  appendix.
- **In-house "OpenDub film set", built once and reused by many categories.**
  - Public-domain Japanese anime from archive.org:
    - *Kumo to Chūrippu* (1943, 15.5 min)
    - *Momotarō: Umi no Shinpei* (1945, 74 min)
  - **Recent anime clips (ja→en, ja→hi headline test).** These are 1–5 min Japanese-audio clips
    from official rights-holder uploads, such as official PVs or free first episodes (Naruto or
    Boruto if available), plus any CC-licensed Japanese anime shorts.
    - They are stored only under `data/` (gitignored), marked `eval-only, private`, and never
      committed or redistributed.
    - They have no gold transcript, so they drive the **cascade view** and the user's audit, not
      reference-based metrics.
  - Public Indic and English speech comes from FLEURS, IndicVoices-R and Common Voice, until
    public-domain Hindi films are added.
  - 30–60 min per L1 language.
  - Annotation layers, each serving several categories:
    - transcript (A4)
    - word boundaries on a subset of about 2,000 words per language, 10% double-annotated (A5)
    - speaker turns (A6, A7)
    - speaker face per line (B1)
    - shots (B2)
    - on-screen text (B3)
    - content type (B4)
    - emotion and nonverbal events (A8, D2, D7)
    - dub-line cues (C1)
- **Synthetic sets with free ground truth.**
  - DnR-recipe remixes for languages DnR lacks (hi, ko, pt).
  - Degraded reference clips (A2, E2).
  - Offset injection from −200 to +200 ms (G5).
  - Text overlays on clean frames (B3, F4).
  - Convolution with known room impulse responses (E3).
  - Defect injection for G6: wrong speaker, 150 ms off-sync, untranslated line, clipping, wrong
    emotion, missing laugh.
- **Splits.** A `dev` split is used for tuning prompts and thresholds. A held-out `test` split
  decides defaults and is never used for tuning.

## 6. Milestones

### M0: arena infrastructure and hygiene (no GPU needed)

**Code to build:**
- Worker protocol and runner. Local target first, with the ssh target stubbed.
- Environment manager: `uv` venv per family plus container runner.
- The two caches and `arena.sqlite`.
- Statistics module: paired bootstrap, Holm, winner set, rescaled composite, Bradley–Terry.
- Leaderboard report: Markdown and HTML, one table per category × language, plus a Pareto plot of
  quality against RTF and VRAM.
- CLI:
  - `python -m bench arena run <ID> --lang hi --candidates … --profile thalassa-16g`
  - `… score`
  - `… report`
  - `… select`
- Candidate registry YAMLs seeded from §7. A verification pass fills the `verified` flags from
  model cards, covering the †-marked models dated after June 2026: Fish S2, VoxCPM2, Higgs v3,
  MOSS-TTS, IndexTTS 2.5, Hy-MT2, PP-OCRv6 and others.
- Extend `ProviderMeta` with `languages`, `license`, `ship_ok`, `vram_gb` and `revision`.

**Thalassa hygiene:**
- **GPU lock.** A job refuses to start unless the GPU is at least 15 GB free. Right now 11 GB is
  held by a parakeet job from another project and by the OpenDub server.
- **Disk.** Move `HF_HOME` and the envs to a larger volume, or evict per category. Only 98 GB is
  free, against about 150–250 GB of checkpoints.
- **No CPU offload.** RAM is 24 GB and 7 of 8 GB of swap are in use, so candidates that need
  offload are marked `spark`.

### M1: judges first (G1–G4)

Every later milestone depends on these judges, so they are built and meta-evaluated before any
generator is ranked.

- **G1 ASR panel per language.** Table in the appendix. Plus Kana-CER for ja and IndicConformer for
  Indic.
- **G2 similarity.** Two lineages: WavLM-SV and ERes2Net/ReDimNet2, with ECAPA for continuity.
- **G3 naturalness.** UTMOSv2 (comparisons within a language only), Distill-MOS, TTSDS2 at system
  level, Audiobox Aesthetics, and IndicMOS for Indic.
- **G4 emotion.** Dimensional SER (arousal, valence, dominance) plus emotion2vec+ cosine with a
  neutral-take baseline.
- **Meta-evaluation.** Each judge is checked against:
  - public human ratings: BVCC, SOMOS, VoxSim, VMC2026 tracks, LIMMITS
  - the user's ok/bad marks from the audit viewer, which accumulate over time; no labelling quota
  - agreement between judges from disjoint families. A judge that disagrees with both the others
    and the user's marks is not trusted

  The output is a "judge card" per language, stating which judges are trusted where. Where no judge
  is valid, the ranking for that language relies on the user's audit of the top few candidates.

### M2: speech analysis (A4, A5, A6, A7, A1, A2, A3)

- These are cheap, fit easily on 16 GB, and their outputs are the frozen inputs for everything
  later.
- They rank on public data plus the film set.
- A4 is ranked on text and hallucinations. It is ranked on timing only if A4 timestamps drive
  segmentation; otherwise A5 owns timing.

### M3: voice (D1, D4, D3, D2, D5, D6, D7), the largest and most valuable sweep

- Every L1 language (en, hi, ja), same texts and references. Cross-lingual cloning uses Japanese
  anime references speaking English and Hindi.
- About 12 open models × about 4.5 supported languages each, at N=4 takes, comes to about 14 h of
  GPU time. That is overnight, and the cache makes reruns free.
- D3 reports accept-rate@N and wall time per accepted take, so best-of-N cost is visible.
- D8 is run only as a comparator.

### M4: text (C2, C4, C1, C5, C6)

- **C4 is meta-evaluated first,** because it is the C2 judge and reranker.
  - MetricX-24-Hybrid-XL fits on Thalassa. XXL is Spark-only, and its rank agreement with XL is
    measured once there.
  - CometKiwi and xCOMET are non-commercial: eval-only.
- **C2 on Thalassa:** open LLMs at 16 GB or less: Qwen3-14B, Gemma 3 12B, Hunyuan-MT-7B,
  gpt-oss-20b, and the current qwen2.5:14b. They are served by vLLM in a container with batched
  requests, about 10–20 min per model per direction. Candidates are gated on duration compliance
  (predicted TTS duration within 0.9–1.1× of the slot), then ranked by MetricX.
- **LLM judge (GEMBA-ESA).** It must be a family different from every candidate, so it waits for
  the API profile, or uses a local model excluded from the candidates.

### M5: post and picture on Thalassa (E1, E2, E4, B2, B4, B1, B3, F1 baseline, G5)

- F1 is limited to LatentSync 1.5/1.6 and Wav2Lip on a 10-clip set, scored by G5's panel:
  Synchformer, PEAVS, and SyncNet LSE for comparability only.
- The large video models are marked `spark`.

### M6: generate defaults

- `bench arena select` writes `configs/defaults.by_language.yaml`:
  `{profile: {lang: {capability: provider_id + params}}}`.
- `RUNTIME_DEFAULTS` becomes `[profile][lang][kind]`, falling back to `[profile]["*"][kind]`.
  `pick_provider` and `apply_preset` receive `source_lang` and `target_lang`.
- D4's `lang_router` is the runtime consumer.
- Each default carries a link to its leaderboard row, so every choice is traceable.

### M7: end-to-end validation

- A "golden reel" of about 5 scenes per L1 language covers on-screen and off-screen lines, emotion,
  nonverbal events and on-screen text.
- For each finalist, swap exactly **one** category pick against the current defaults. Score with
  the whole-pipeline bench plus the G6 panel. The user audits the full scene in the cascade view.
- A category winner is adopted only if it wins end to end, or ties while being cheaper.
- Categories that do not move end-to-end preference (likely E2, E4 and F2 on most content) get the
  cheapest acceptable option.

### Later phase: API profile (`api`)

- API workers use the same protocol, with an explicit budget cap per run and cost per item stored
  in `outputs`.
- API models join the existing leaderboards on the same frozen packs, so no reruns are needed.
- The API profile unlocks the cross-family LLM judges for C2/C4/G6 and the ASR APIs for G1
  diversity.
- API versions are pinned and dated, because hosted models change monthly. A monthly re-score job
  detects drift.

### Later phase: Spark profile (`spark`)

- The ssh worker target syncs manifests and pulls outputs back into the same caches.
- aarch64 env builds carry over from the Spark memory notes: flash-attn has no official build,
  CTranslate2 PyPI wheels are CPU-only on aarch64, vLLM needs cu130 builds, and pyannote 4 is
  blocked by TorchCodec.
- Unlocks:
  - Qwen3-Omni for A8/G6
  - MetricX-XXL and xCOMET-XXL
  - 27–32B LLMs for C1/C2/C3/C6
  - LTX-2.3, InfiniteTalk and KeySync at full precision for F1
  - large VLMs for B3/B4
  - best-of-N at higher N
- Winner sets are recomputed per profile. `thalassa-16g` defaults stay as they are; the Spark gets
  its own column.

### Deferred or parked

- E5: deferred by user decision.
- E3, F2, F3: no released weights or no benchmarkable candidates. Only the metric and data are
  built now.
- C3: runs only after F1 and G5 exist, and only on close-up frontal lines.

## 7. Seed candidate list on Thalassa (16 GB)

Legend: † = the artifact dates the release after June 2026, so verify it in M0. NC = the weights
are non-commercial, so the candidate is eval-only. `spark` = deferred to that profile. "API" =
deferred to the API profile.

| ID | Candidates on Thalassa (baseline first) | Deferred |
|---|---|---|
| A1 | HTDemucs (baseline), Bandit v2 DnR-v3, Mel-Band RoFormer (MSST), AuK† | AudioShake and LALAL (API) |
| A2 | UniPASE, ClearerVoice MossFormer2-48K, Resemble Enhance, RoFormer dereverb | AnyEnhance (no weights) |
| A3 | CED-base, inaSpeechSegmenter (CPU), Dasheng-1.2B, Qwen3-ASR as adjudicator | Qwen3-Omni (`spark`) |
| A4 | faster-whisper large-v3 / turbo (baseline), Qwen3-ASR-1.7B, VibeVoice-ASR int8, IndicConformer-600M (Indic), Parakeet-TDT-v3 (EU languages only), MOSS-Transcribe-Diarize† | Scribe v2 and others (API) |
| A5 | Whisper word timestamps (baseline), MFA 3.x, MMS_FA, Qwen3-ForcedAligner (no hi) | ElevenLabs FA (API) |
| A6 | pyannote 3.1 (baseline), pyannote community-1, DiariZen-L (NC), Sortformer v2 (max 4 speakers), MOSS† | Precision-3 (API) |
| A7 | ECAPA (baseline), ReDimNet2-B6, ERes2NetV2, WeSpeaker ResNet293 | — |
| A8 | measured prosody (DSP), emotion2vec+ L, hubert-er, AF-Next int8 (NC) | Qwen3-Omni (`spark`) |
| B1 | LR-ASD (MIT), LoCoNet+TalkNCE, ByteTrack; SAM 3.1 (gated) as tracker option; face detector open (SCRFD weights NC) | — |
| B2 | PySceneDetect, TransNetV2 (PyTorch port), OmniShotCut v1.5 (MIT), AutoShot, TransVLM-4B (Apache), PERSIST (mmcv source build — risk) | — |
| B3 | PP-OCRv6 ONNX (zh/en/ja/Latin only) + PP-OCRv5 script models (ko, Devanagari, ta, te; bn missing), PaddleOCR-VL-1.6 (needs transformers ≥5, own venv) | Qwen3-VL-235B (`spark`) |
| B4 | SigLIP 2 + linear head, deepghs classifiers, Qwen3-VL-8B FP8 | — |
| C1 | pause splitter (baseline), SaT-12l, SHAS, 14B LLM | 27B+ LLMs (`spark`) |
| C2 | qwen2.5:14b (baseline), Qwen3-14B, Gemma 3 12B, Hunyuan-MT-7B, gpt-oss-20b | Hy-MT2-30B, Qwen3.8-27B, gpt-oss-120b (`spark`); Claude, GPT and Gemini (API) |
| C3 | small generator + vowel-DTW scorer (experimental) | gpt-oss-120b (`spark`) |
| C4 | MetricX-24 XL / Large, CometKiwi (NC), local GEMBA with a 12–14B model | MetricX-XXL and xCOMET-XXL (`spark`) |
| C5 | nemo-text-processing, misaki, espeak-ng (shell out, GPL), CharsiuG2P | — |
| C6 | Qwen3-14B two-pass condenser | Qwen3-32B (`spark`) |
| D1 | F5 v1 (baseline, NC), F5-Hindi, Qwen3-TTS-1.7B, Chatterbox-ML, VoxCPM2†, Higgs v3†, MOSS-TTS-8B int8†, Fish S2† (VRAM and licence unverified) | Eleven v3 and others (API) |
| D2 | IndexTTS 2.5†, Step-Audio-EditX, plus the D1 set | — |
| D3 | F5 (explicit duration), MOSS-Local-4B†, IndexTTS pace control; best-of-N rejection sampling | — |
| D4 | IndicF5, F5-Hindi, VoxCPM2†, Higgs v3†, OmniVoice (NC) | Bulbul (API) |
| D5 | Kokoro-82M, Chatterbox-ML, Qwen3-TTS VoiceDesign, VoxCPM2 design† | — |
| D6 | Vevo2, Chatterbox VC, kNN-VC, FreeVC | — |
| D7 | splice + CED gating, Step-Audio-EditX, SenseVoice tagging | — |
| D8 | comparator only: Hibiki (fr→en), SeamlessExpressive (NC) | — |
| E1 | atempo (baseline), Signalsmith Stretch, pause DP, Rubber Band R3 (GPL) | — |
| E2 | AP-BWE, FLowHigh, Sidon, UniPASE | — |
| E4 | pyloudnorm / libebur128, with per-deliverable targets (−23 LUFS EBU, −27 LKFS Netflix) instead of −16 | — |
| F1 | Wav2Lip (NC), LatentSync 1.6 (installed locally; 15.8 GB peak measured on a 16 GB card, so it fits tightly; weights openrail++), LatentSync 1.5, KeySync | LTX-2.3 IC-LoRA DubIt, InfiniteTalk, Wan2.2-S2V (`spark`); sync-3 (API) |
| G1 | Whisper-large-v3, Parakeet-v3, IndicConformer, kotoba-whisper (ja), Paraformer-zh, Voxtral Mini, MMS (NC) | — |
| G2 | WavLM-SV, ReDimNet2, ERes2NetV2, WeSpeaker, ECAPA | — |
| G3 | UTMOSv2, Distill-MOS, TTSDS2, Audiobox Aesthetics, XLS-R-SQA, IndicMOS | — |
| G4 | Odyssey WavLM A/V/D, emotion2vec+, hubert-er | — |
| G5 | Synchformer, PEAVS (no faces, so supplementary only), SyncNet LSE (for comparison only) | AV-HuBERT (NC) |
| G6 | decomposed transcript review by a text LLM | Qwen3-Omni (`spark`), Gemini (API) |

E3, E5, F2, F3 and F4 are parked; see §6. When they are unparked:
- F2's open candidates (PGTFormer, DicFace, SVFR) are all non-commercial, and DVFace has no weights.
- F4 has CLEAR (Apache, released), FLUX.2 klein 4B (Apache) and Qwen-Image-Edit-2511 as a Q4 GGUF. SEDiT has no code.

Known environment fixes on Thalassa:
- Use SDPA or FA2, never FA3. Use SageAttention 2.2, not 3.
- CTranslate2 needs the cuBLAS-12 shim and version 4.7 or later for INT8.
- The official onnxruntime-gpu 1.21 wheel works on sm_120, but only after `ort.preload_dlls()` or `import torch`. Otherwise it silently falls back to CPU, so assert the CUDA execution provider in every worker.
- llama.cpp needs `CMAKE_CUDA_ARCHITECTURES=120` and `nvcc`, which is missing, so use a container.
- vLLM goes in its own venv or container.
- Use PyAV instead of decord.
- The server venv has torchaudio 2.11 against torch 2.14. That mismatch must be fixed before MMS_FA
  is used.

## 8. Rough effort and wall-clock time

| Milestone | Build | GPU time on Thalassa |
|---|---|---|
| M0 | largest code milestone: worker protocol, envs, caches, statistics, report, registry | none |
| M1 | judges plus meta-evaluation | about 2–4 h |
| M2 | adapters for about 25 analysis models | about 4–8 h |
| M3 | adapters for about 20 TTS/VC models | about 10–14 h (L1 only) |
| M4 | vLLM container, MetricX | about 3–6 h |
| M5 | assorted adapters; F1 limited to 10 clips | about 3–6 h |
| M6–M7 | defaults generator, golden reel, cascade audit | about 3 h |

## 9. Decisions (user, 2026-09-27)

1. **Languages:** start with en, hi, ja. L2 and L3 come later.
2. **Licences:** non-commercial models are compared, and they may be defaults for now. Licence
   fields are still recorded so the choice can be flipped later.
3. **No human raters.** The user is the only auditor. Every stage's outputs are saved, and the
   cascade view shows how quality holds as each stage is added (§4.9).
4. **Thalassa:** the parakeet job is gone, so the GPU is free apart from the OpenDub server (about
   4 GB). There is no larger volume, so evict the model cache per category, keeping outputs.
5. **Sources:** the archive.org public-domain shorts, plus recent anime clips from official uploads
   (ja→en, ja→hi), kept private and local.

---

## Appendix: per-category metric specification

Columns:
- **Gate:** the catastrophic check that must pass before ranking.
- **Primary:** the metric the category is ranked by.
- **Secondary:** used for the composite and diagnostics.
- **Data:** public sets plus in-house (IH).

All reference-based metrics are computed per language. "Film set" means the in-house set in §5.

Since 2026-09-27 there are no human raters. Wherever a cell below says human, native, pairwise,
MOS, CMOS, MUSHRA or ESA, read it as "the user audits the top candidates in the audit viewer" (§4.9).
IH annotation targets are also optional, and the user's audit marks fill them in over time.

### A: source audio
| ID | Gate | Primary (rank by) | Secondary | Data |
|---|---|---|---|---|
| A1 Separation | — | **Bed ghost-dialogue**: ASR word recall on the estimated M&E bed during speech (↓). SI-SDRi of dialogue and bed on synthetic mixtures (median). | SIR/SAR (museval), stereo ILD/ICC error, SAM Audio Judge (eval-only) | DnR v3 (32 languages, no hi/ko/pt → IH remix), MUSDB18-HQ, 10–15 film excerpts + MUSHRA |
| A2 Ref enhancement | SpeechBERTScore / LPS ≥ threshold (no invented phonemes) | Speaker-similarity preservation vs the clean clip (judge disjoint from A7) | DNSMOS/NISQA/UTMOS on real clips; similarity of a clone built from the enhanced clip; metrics at native sample rate | Simulated from IndicVoices-R + FLEURS, URGENT 2026 val, 30 film windows per language |
| A3 Speech/music/singing | speech recall ≥ 0.97 | Segment macro-F1 at 100 ms (sed_eval) | singing F1, false speech per hour of music | AVASpeech-SMAD, OpenBMAT, IH song-heavy hi/ta/ja |
| A4 ASR | hallucinated words per hour in non-speech < threshold | WER (en, es, fr, de, pt, hi, ta, te, bn); CER (ja, zh, ko) | word start/end MAE ms and % ≤ 50/100/200 ms (only if A4 timings are used); RTFx | FLEURS, Common Voice, MLS, IndicVoices/Vistaar, ReazonSpeech (ja), Zeroth (ko), AISHELL-1; film set 30–60 min per language |
| A5 Alignment | gross error (>200 ms) < 2% | Word-boundary MAE ms + % ≤ 25/50 ms, with both gold text and ASR text | long-form drift (300 s), phone MAE | Buckeye (en, NC); IH Praat annotation of about 2,000 words per language (no public data for hi/es/fr/de/pt/zh/ta/te/bn) |
| A6 Diarization | — | DER, no collar, overlap scored (0.25 s collar also reported) | **line-level speaker-attribution error**, JER, speaker-count error, cpWER for joint models | VoxConverse, AMI, AISHELL-4, AVA-AVD (films), DISPLACE (Indic), IH ja/ko |
| A7 Embeddings | — | EER on 2–4 s separated segments, same-language and cross-language trials | actual DCF at one fixed threshold (calibration across languages), character-bank top-1 accuracy | VoxCeleb1-O/E/H, CN-Celeb, TidyVoiceX, IH character banks |
| A8 Delivery tags | — | Emotion macro-F1/UAR per language; nonverbal event F1 at 200 ms | arousal/valence CCC; downstream G4 consistency of TTS conditioned on the notes; human pairwise on notes | EmoBox (14 languages), NonverbalTTS, NVV-TimeBench, IH ja/ko/hi/ta/te |

### B: source picture
| ID | Gate | Primary | Secondary | Data |
|---|---|---|---|---|
| B1 Active speaker | — | Line-level speaker→face accuracy, with an "off-screen" class | AVA frame mAP, HOTA/IDF1, ID switches across cuts | AVA-ActiveSpeaker, UniTalk (zh/ko/ja, NC), IH animation |
| B2 Shots | — | Transition F1 (cuts ±2 frames, gradual transitions by overlap) | false positives per hour, false cuts inside a dub line | ClipShots, AutoShot SHOT, BBC/RAI, IH anime (flashes, holds, telecine) |
| B3 OCR | — | End-to-end word F at IoU ≥ 0.5 + 1−NED on translation-worthy regions | per-script CER after NFKC, text-track IDF1 | TextOCR, MLT-19, Bharat Scene Text, JaWildText, BanglaWild, synthetic overlays for ta/te |
| B4 Content type | — | Title-level macro-F1 (live action / 2D / 3D / mixed) | selective accuracy at 95% coverage, ECE, misrouting cost | Blender open movies, archive.org, deepghs data; at least 100 titles |

### C: text
| ID | Gate | Primary | Secondary | Data |
|---|---|---|---|---|
| C1 Segmentation | — | % TTS-fittable segments (fits the slot, no split constituent, boundary at a pause ≥ 150 ms) | boundary F1 ±1 word, Sigma; downstream C2 retry rate | MuST-Cinema, Anim-400K (research only), IH Indic |
| C2 Translation | duration compliance: predicted TTS duration within [0.9, 1.1]× the slot (not character counts) | MetricX-24-Hybrid (reference-based where references exist, QE otherwise) | xCOMET (eval-only), GEMBA-ESA from a disjoint family, speech overlap after render; quality at fixed compliance | WMT24++ (en→all), FLORES+, IN22-Conv (Indic↔en), BSD (ja↔en, NC); ESA on 150–200 lines for the top 2 |
| C3 Viseme adaptation | QE drop ≤ 1 MetricX, LaBSE ≥ 0.75 | Δ G5 sync-panel score on close-up frontal lines | vowel-DTW, bilabial-closure match ±80 ms | IH, about 50 close-up lines per direction |
| C4 QE (meta-evaluation) | — | Within-source pairwise accuracy with tie calibration (acc_eq) against human ratings | segment Kendall/Pearson, ROC-AUC for major errors, system SPA | WMT23 QE DA (en→hi/ta/te), WMT24/25 MQM/ESA, IH ESA on 300 candidates per direction including condensed variants |
| C5 TN/G2P | — | TN sentence exact match per semiotic class; G2P word exact match + PER | homograph and polyphone accuracy, ja kana CER + accent nucleus, hi schwa deletion; round-trip CER | PolyNorm, NeMo TN tests, WikiPron, ja G2P benchmark; IH 50 items × 10 classes for hi/ta/te/bn/ko |
| C6 Subtitles | CPS/CPL/lines within IWSLT 2026 limits | SubER | COMET/BLEURT, compression ratio, Sigma | IWSLT 2026 (en→de/es/zh/ja, NC), MuST-Cinema, IH Indic |

### D: voice
| ID | Gate | Primary | Secondary | Data |
|---|---|---|---|---|
| D1 Cross-lingual clone | CER/WER_ratio ≤ 1.5 on a two-family ASR panel | Cross-lingual SIM_norm (source-language reference, target-language output); two disjoint encoders | monolingual SIM, UTMOSv2/Distill-MOS within a language, accent leakage (language-ID posterior), SIM/WER Pareto front | Seed-TTS-eval, CV3-Eval (cross-lingual subset), MiniMax 24-language set, IndicVoices-R, IH bilingual 50 speakers × 6 lines per pair |
| D2 Expressive | D1 gates | Emotion consistency: A/V/D distance + emotion2vec cosine vs source, minus a neutral-take baseline | EmergentTTS-Eval (en, LLM-judged), F0/energy contour correlation; human "matches delivery" pairwise | ESD, EmoV-DB, Expresso (all NC), JVNV (ja), Rasa (Indic), IH film lines |
| D3 Duration control | speech fraction via VAD (no silence padding, no truncation) | In-window rate at ε = 5% and 10% | mean abs duration error, residual stretch, WER/UTMOS curve over 0.8–1.2 compression (AUC), speaking rate as a per-language percentile, accept-rate@N | IH source slots + translated scripts |
| D4 Low-resource voices | CER_ratio (IndicConformer) | **Native-listener CMOS** (automatic MOS is not validated for Indic) | SIM_norm, IndicMOS, code-mixing and number/name reading on a 100-item list | IndicVoices-R, Rasa, IndicTTS, LIMMITS ratings |
| D5 Stock voices | documented licence and provenance | Native pairwise "fits the character" | WER_ratio, UTMOS within a language, cast separability, stability over 50 lines | IH character briefs, 20 lines × language |
| D6 Voice conversion | content CER_ratio vs the source audio | SIM_norm to the target (cross-lingual calibration) | source-leak EER, log-F0 correlation, UTMOS; SIM on denoised copies | VCC2018, IH dub-actor → original-actor pairs |
| D7 Nonverbal | — | Nonverbal presence event F1 (±200 ms), with the source and output tagged by BEATs/CED | PCER (NV-Bench), splice seam flux, level match in LU | NonverbalTTS, NV-Bench (NC), JVNV/JNV, IH film |
| D8 S2ST (comparator) | — | ASR-chrF/COMET against the cascade on the same clips | BLASER 2.0-QE (NC), SIM, STEB expressiveness | STEB (zh↔en), IH |

### E: audio out
| ID | Gate | Primary | Secondary | Data |
|---|---|---|---|---|
| E1 Timing fit | WER/UTMOS no worse after stretch | Pause-alignment F1 (±100 ms, pauses ≥ 150 ms) | speech-mask overlap (human dubs about 0.66), max/mean stretch, onset error, weighted by on-screen status | IH |
| E2 BWE | ΔCER ≤ +0.5, ΔSIM ≥ −0.01 | ViSQOL (simulated, VCTK 48k) + effective bandwidth and Audiobox PQ on real TTS | LSD-HF (diagnostic); MUSHRA | VCTK, D-phase outputs |
| E3 Room match (parked) | — | T60/DRR/C50 error vs the original dialogue stem | room-embedding distance, ABX | synthetic room-impulse-response convolution; the estimator is validated first |
| E4 Loudness | exact conformance | Dialogue-gated integrated loudness error vs the deliverable target, true peak, LRA | per-line dub-vs-original LU delta, dialogue-to-background ratio delta | any output |

### F: picture out
| ID | Gate | Primary | Secondary | Data |
|---|---|---|---|---|
| F1 Lip sync | identity CSIM ≥ threshold | Cross-lingual: disjoint sync panel (Synchformer offset ms, PEAVS, AV-HuBERT if licensed). Self-reenactment: mouth LPIPS/SSIM, LMD | FVD, CSIM, LSE-C/D for comparability only; human MOS on sync, identity and visual quality | HDTF, VoxCeleb2, LRS3 (research), IH about 40 film shots per language |
| F2 / F3 / F4 (parked) | no-regression on the F1 sync panel | human pairwise (F2); flap F1 against audio visemes (F3); OCR 1−NED of the rendered text plus erase PSNR/LPIPS (F4) | — | IH |

### G: judges (meta-evaluated against humans, not ranked by their own outputs)
| ID | Meta-metric | Human data | Rule |
|---|---|---|---|
| G1 Intelligibility | Detection AUC and precision/recall for human-flagged defective takes, per language | IH 200 labelled takes per L1 language; ASR floor on real speech | Two-family panel: en Whisper-v3 + Parakeet; zh Paraformer + Qwen3-ASR; ja Whisper + kotoba-whisper + Kana-ASR; ko Whisper + one other TBD; es/fr/de/pt Whisper + Parakeet; Indic IndicConformer + Whisper/MMS. The verifier used for best-of-N is never the reporter. |
| G2 Similarity | SRCC/LCC against human similarity ratings | VoxSim, VCC2018, VMC2026 T3, IH 300 bilingual pairs | A better verifier is not a better perceptual judge; prefer fine-tuned on VoxSim |
| G3 Naturalness | utterance- and system-level SRCC, pairwise accuracy | BVCC, SOMOS, VMC2024, Blizzard (zh, fr, es), LIMMITS (Indic), TTSDS2 ratings | Compare within a language only. Exclude any judge used as an RL reward by a candidate. |
| G4 Emotion | EMOS SRCC | VMC2026 T2 (ESD/DailyTalk), IH cross-lingual "same feeling" ratings | No specialist labels in LLM prompts; always swap A/B order |
| G5 Lip-sync score | monotonic response to injected offsets; agreement with human sync MOS | PEAVS annotations, synthetic offsets | Never rank SyncNet-supervised generators with SyncNet |
| G6 Reviewer | per-defect recall/precision on the injected-defect set; agreement with blind ratings on 100 clips | IH defect set | Decompose into lexical, quality and paralinguistic sub-judges (AudioJudge); use a vendor different from every candidate |
