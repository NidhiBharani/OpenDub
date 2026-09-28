# Adding capabilities and candidates to the arena

The arena ranks every candidate model of every capability (A1–G6) per language on frozen eval
packs, on whatever hardware the machine has. Plan and rationale: `docs/plans/model-ranking.md`.
The user orchestrates runs; code here must never download weights, install packages or run
models as a side effect of import or of tests.

## What one capability needs

| piece | file (one per capability or model family — never edit another's) |
|---|---|
| spec: judges, primary metric per language, gates, I/O contract | `bench/arena/specs/<id>.py` (lower-case, e.g. `d1.py`) exporting `SPEC` |
| candidates | `bench/candidates/<ID>.yaml` |
| workers (one per model family / API) | `bench/arena/workers/<id>_<family>.py`, e.g. `d1_qwen3_tts.py` |
| worker envs (setup recipe per arch) | `bench/envs/<env>.yaml` |
| pack builders (download + convert public data) | `bench/arena/builders/<source>.py` |
| tests (model-free) | `tests/test_arena_<phase>.py` |

Shared core (`hardware.py`, `registry.py`, `envs.py`, `runner.py`, `judges.py`, `judgelib.py`,
`stats.py`, `report.py`, `db.py`, `packs.py`, `workers/_sdk.py`) is owned by the orchestrator. If
you need a core change, make the smallest possible edit, keep every existing test green, and
list it in your report.

## Candidates (`bench/candidates/<ID>.yaml`)

```yaml
capability: D1
candidates:
  - id: qwen3-tts-1.7b-base            # unique within the capability; stable (it keys the cache)
    worker: d1_qwen3_tts
    env: d1_qwen3_tts                  # bench/envs/<env>.yaml; "server" only for deps it has
    kind: local                        # local | api | method (composes other models)
    params: {model: Qwen/Qwen3-TTS-12Hz-1.7B-Base, revision: <sha if known>, dtype: bfloat16}
    languages: [zh, en, ja, ko, de, fr, ru, pt, es, it]   # vendor-declared; "*" only if true
    license: Apache-2.0
    ship_ok: true                      # false for non-commercial weights (still allowed)
    requires: {gpu: true, vram_gb: 6, disk_gb: 5}          # see "Hardware" below
    sources: [research:D1:1, compute-tiers:D1:spark]        # where the row came from
    verified: {license: false, languages: false, vram: false, runs: false}
    notes: one line of caveats (gated repo, needs HF_TOKEN, eager attention on sm_120 …)
    url: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base   # REQUIRED: weights / repo / API docs
    params_b: 1.7                      # billions of parameters, if known
```

- Cover **every** candidate named for the capability in both research artifacts (the "Model
  Research" ledger and the "Compute tiers" Spark / Unconstrained / API picks and alternatives),
  deduplicated. A model with several sizes is one worker, several candidate entries.
- Models without released weights, closed products without an API, and pure ideas: add them with
  `enabled: false` and a `notes:` line saying why, so the registry is the complete ledger.
- `method` candidates (ensembles, best-of-N, splicing) are workers too, composing other models.
- `baseline: true` for what OpenDub does today (see `docs/plans/capability-map-implementation.md`
  and `app/capabilities.py`); builtin/DSP baselines are workers in the `server` env.
- Languages in scope now: **en, hi, ja**. Declare everything the vendor claims anyway.
- Everything you did not check against a model card or repo stays `false` in `verified`.

## Hardware (`requires`)

The same registry runs on every machine; `arena plan` shows what is runnable where. Fields:
`gpu, vram_gb, ram_gb, disk_gb, arch [x86_64|aarch64], min_compute_capability, api_keys, tools`.

Target machines (write `requires` so each lands where it can actually run):

- **Thalassa**: RTX 5060 Ti 16 GB (SM 12.0), x86_64, 24 GB RAM, driver CUDA 13.2, no CPU offload
  headroom. FA3 unsupported (use SDPA/FA2), SageAttention 2.2 not 3, CTranslate2 needs the
  cuBLAS-12 preload (`_sdk.preload_pip_cuda_libs`), onnxruntime-gpu needs CUDA libs preloaded.
- **DGX Spark**: GB10 (SM 12.1), aarch64, 128 GB unified memory (~112 GB usable as VRAM), CUDA 13.
  No official flash-attn build, CTranslate2 PyPI wheels are CPU-only on aarch64, vLLM needs cu130
  builds, pyannote 4 blocked by TorchCodec, many research repos have x86-only wheels (decord,
  some ONNX runtimes) → provide an `aarch64` recipe or `arch: [x86_64]`.
- **API**: `kind: api`, `requires: {api_keys: [ELEVENLABS_API_KEY]}` (names in
  `hardware.KNOWN_API_KEYS`; any env-var name works).

`vram_gb` = measured or model-card peak for inference at the params given (+ ~10% headroom);
estimate from parameter count × bytes/param + activations when unknown and say so in `notes`.
A model that only fits quantized gets a separate candidate entry with the quantized params.

## Envs (`bench/envs/<env>.yaml`)

One isolated `uv` venv per model family under `~/.opendub/envs/<env>`; recipes per arch; pinned
versions; a `check:` python snippet. Template: `bench/envs/qwen3asr.yaml`. Never add candidate
dependencies to `server/pyproject.toml`. Repos that must be cloned go under
`~/.opendub/src/<name>` in the recipe (`git clone … {home}/.opendub/src/<name>`), pinned to a
commit. Docker-only stacks: a recipe that builds/pulls the image and a `python` that is a small
wrapper script is acceptable; document it.

## Workers (`bench/arena/workers/*.py`)

Stdlib-only at import time (`from _sdk import serve, Unsupported, api_key, …`); heavy imports go
inside `load()`. One process per job: the model loads once, serves every item, and its GPU memory
is freed when the process exits — never cache models across jobs.

```python
def load(params: dict, lang: str) -> state
def run(state, item: dict, out: Path) -> dict   # item = {id, inputs, meta, out}
def describe(state) -> dict                     # optional: model id, revision, device
serve(load, run, describe)
```

- Write produced files next to `out` (`out.with_suffix(".wav")` …) and list them in the payload
  under `"files": {"audio": "<abs path>", "video": …, "stems": {...}}`. Judges read `files`.
- `raise Unsupported("reason")` for items the model cannot do (language, duration, modality).
- API workers: key via `api_key("ELEVENLABS_API_KEY")`; put the item's spend in the payload as
  `"_cost_usd"` using the vendor's list price (pricing + date in the worker docstring); honour
  rate limits with retries; never log keys. Pin the API model/version in `params`.
- Force the language from `lang`; never let a model silently auto-detect.
- 16 kHz vs 24/44.1/48 kHz: keep the model's native rate in the output file; judges resample.

## Payload contracts by family (what judges expect)

| capability family | item.inputs | payload |
|---|---|---|
| A1 separation | `{audio}` | `files: {dialogue, background, music?, effects?}` |
| A2/E2 enhancement, BWE | `{audio}` | `files: {audio}` |
| A3 regions, A8 tags | `{audio}` | `{segments: [{start, end, label, score}]}` |
| A4 ASR | `{audio}` | `{text, segments: [{start, end, text, words: [...]}]}` |
| A5 alignment | `{audio, text}` | `{words: [{start, end, word}]}` |
| A6 diarization | `{audio}` | `{turns: [{start, end, speaker}]}` |
| A7 embeddings | `{audio}` or `{audio, bank: [...]}` | `{embedding_file}` or `{match, scores}` |
| B* picture | `{video}` | `{shots | tracks | texts | label: …}` |
| C* text | `{text, context?, slot_s?, src_lang}` | `{text}` (+ `candidates` for N-best) |
| D1–D5, D7 TTS | `{text, ref_audio?, ref_text?, target_s?, emotion?}` | `files: {audio}` |
| D6 VC | `{audio, ref_audio}` | `files: {audio}` |
| D8 S2ST | `{audio}` | `{text}` + `files: {audio}` |
| E1/E3/E4 post | `{audio, ...}` | `files: {audio}` (+ `time_map`, `params`) |
| F1–F4 picture | `{video, audio?}` | `files: {video}` |
| G* judges | see `judgelib.py` | `{metrics: {...}, weights?: {...}}` |

## Specs (`bench/arena/specs/<id>.py`)

`SPEC = Spec(id, title, judges={...function judges...}, primary={"*": metric, "ja": ...},
higher_is_better={...every metric...}, threshold={...}, secondary=[...],
model_judges=lambda lang: [...judgelib factories...], gates={metric: ("<=", x)},
derived={...}, packs=[builder names], io="contract")`.

Judge-type capabilities (G1–G6, C4) rank by agreement with human ratings: put the human value in
`item.refs` (e.g. `{"mos": 3.4}`) and declare `corpus={"srcc": CorpusMetric(pred="mos_utmosv2"…,
target=lambda it: it.refs.get("mos"), fn="spearman"|"pearson"|"kendall"|"pairwise_acc")}` with
`primary={"*": "srcc"}` — the leaderboard bootstraps it over groups (`stats.rank_corpus`).

Use the metrics in the appendix of `docs/plans/model-ranking.md` (primary, gate, secondary). Model
judges come from `judgelib.py` (ASR round-trip, speaker similarity, MOS, emotion, lip sync, MT
QE); add a new factory there only if no existing one fits, and report it. Function judges are
pure Python over the payload and refs (numpy/soundfile allowed; they are in the server venv).

## Packs (`bench/arena/builders/<source>.py`)

`@builder(name, capabilities=[...], langs=[...], license="...")` over `fn(lang, n=None, **opts)`
that downloads into `data/eval/_sources/<source>/`, writes `data/eval/<ID>/<lang>/manifest.jsonl`
via `packs.write_pack` (with `LICENSE.md`), and returns the pack paths. Relative file paths,
a `group` per independent unit (speaker/file/title), `meta.duration_s` for audio. Builders never
run models. Gated datasets: read `HF_TOKEN`, fail with a clear message.

## Tests

Model-free: judge functions on synthetic payloads, builders on tiny fixtures (monkeypatch the
download), YAML/spec loading (`load_candidates`, `SPECS`, every `worker` file exists, every
`env` file exists, ids unique). `cd server && uv run --frozen pytest tests/test_arena*.py -q` and
`uv run --frozen ruff check bench/arena` must pass.
