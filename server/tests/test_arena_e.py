"""Arena phase E (audio post-production): specs, judges, builders and DSP workers — model-free."""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.arena import paths
from bench.arena.packs import Item, load_pack

E_CAPS = ["E1", "E2", "E3", "E4", "E5"]
SR16 = 16000
HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    return tmp_path


def _worker(name: str):
    if str(paths.WORKERS_DIR) not in sys.path:
        sys.path.insert(0, str(paths.WORKERS_DIR))
    return importlib.import_module(name)


def _tone(sec: float, sr: int = SR16, f0: float = 200.0, amp: float = 0.3):
    t = np.arange(int(sec * sr)) / sr
    return amp * np.sin(2 * np.pi * f0 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))


def _sil(sec: float, sr: int = SR16, seed: int = 0):
    return np.random.default_rng(seed).normal(0, 1e-4, int(sec * sr))


def _utterance(path: Path, pattern=(0.3, 1.0, 0.5, 0.8, 0.4, 0.6, 0.3), sr: int = SR16):
    parts = [(_tone if i % 2 else _sil)(d, sr) for i, d in enumerate(pattern)]
    x = np.concatenate(parts)
    sf.write(path, x, sr, subtype="PCM_16")
    return x


# ------------------------------------------------------------------ registry / specs

def test_e_registry_and_specs_are_consistent():
    from bench.arena import envs
    from bench.arena.registry import load_candidates
    from bench.arena.specs import SPECS

    for cap in E_CAPS:
        spec = SPECS[cap]
        cands = load_candidates(cap)
        assert cands, cap
        metrics = {spec.primary_for(lang) for lang in ("en", "hi", "ja")}
        metrics |= set(spec.secondary) | set(spec.gates) | set(spec.derived)
        missing = metrics - set(spec.higher_is_better)
        assert not missing, (cap, missing)
        for c in cands:
            assert c.sources or c.baseline, (cap, c.id)
            if not c.enabled:
                assert c.notes, (cap, c.id)  # disabled rows say why
                continue
            assert (paths.WORKERS_DIR / f"{c.worker}.py").exists(), (cap, c.id)
            env = envs.load_env(c.env)
            if env.setup and "aarch64" not in env.setup and "any" not in env.setup:
                assert c.requires.arch == ["x86_64"], (cap, c.id, "x86-only env")
        assert sum(c.baseline for c in cands) <= 1
    assert not any(c.enabled for c in load_candidates("E5"))  # deferred by the user


def test_e_workers_import_stdlib_only():
    """Module level of every phase-E worker imports only the stdlib and _sdk."""
    import ast

    allowed = set(sys.stdlib_module_names) | {"_sdk", "__future__"}
    for f in sorted(paths.WORKERS_DIR.glob("e[0-9]_*.py")):
        tree = ast.parse(f.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert set(names) <= allowed, (f.name, names)


# ------------------------------------------------------------------ E4 meter (EBU Tech 3341/3342)

def _sine(db: float, sec: float, sr: int = 48000, f: float = 1000.0):
    t = np.arange(int(sec * sr)) / sr
    return 10 ** (db / 20) * np.sin(2 * np.pi * f * t)


def _st(x):
    return np.stack([x, x], 1)


def test_bs1770_meter_matches_ebu_reference_signals():
    from bench.arena.specs.e4 import (
        dialogue_gated_loudness,
        integrated_loudness,
        loudness_range,
        true_peak_dbtp,
    )

    sr = 48000
    assert integrated_loudness(_st(_sine(-23, 20)), sr) == pytest.approx(-23.0, abs=0.1)
    assert integrated_loudness(_st(_sine(-33, 20)), sr) == pytest.approx(-33.0, abs=0.1)
    case3 = np.concatenate([_sine(-36, 10), _sine(-23, 60), _sine(-36, 10)])
    assert integrated_loudness(_st(case3), sr) == pytest.approx(-23.0, abs=0.1)
    case4 = np.concatenate([_sine(-72, 10), _sine(-36, 10), _sine(-23, 60), _sine(-36, 10),
                            _sine(-72, 10)])
    assert integrated_loudness(_st(case4), sr) == pytest.approx(-23.0, abs=0.1)
    assert integrated_loudness(_sine(0, 10, f=997), sr) == pytest.approx(-3.01, abs=0.05)
    lra = np.concatenate([_sine(-20, 20), _sine(-30, 20)])
    assert loudness_range(_st(lra), sr) == pytest.approx(10.0, abs=0.2)
    assert true_peak_dbtp(_st(_sine(-6, 2)), sr) == pytest.approx(-6.0, abs=0.2)
    # dialogue gating keeps only blocks inside the dialogue: the quiet bed is ignored
    prog = np.concatenate([_sine(-26, 10), _sine(-20, 10)])
    assert dialogue_gated_loudness(_st(prog), sr, [[10.0, 20.0]]) == pytest.approx(-20, abs=0.1)
    assert integrated_loudness(_st(prog), sr) < -20.5


def test_e4_conformance_and_balance_judges(tmp_path):
    from bench.arena.specs.e4 import DELIVERIES, balance, conformance, window_loudness

    sr = 48000
    x = _st(_sine(-27, 12))
    wav = tmp_path / "mix.wav"
    sf.write(wav, x, sr, subtype="FLOAT")
    lines = [[1.0, 3.0, window_loudness(x, sr, 1, 3)], [5.0, 7.0, window_loudness(x, sr, 5, 7)]]
    item = Item(id="p-netflix", lang="en",
                inputs={"delivery": DELIVERIES["netflix"], "speech": [[1, 3], [5, 7]]},
                refs={"speech": [[1, 3], [5, 7]], "lines": lines, "speech_bed_db": 0.0})
    rows = {m: v for m, v, _ in conformance(item, {"files": {"audio": str(wav)}}, None)}
    assert rows["loudness_err"] < 0.1 and rows["conform"] == 1.0
    assert rows["true_peak_over_db"] == 0.0
    item.inputs["delivery"] = DELIVERIES["ebu"]  # −23 target: 4 LU off, fails conformance
    rows = {m: v for m, v, _ in conformance(item, {"files": {"audio": str(wav)}}, None)}
    assert rows["loudness_err"] == pytest.approx(4.0, abs=0.1) and rows["conform"] == 0.0
    bal = {m: v for m, v, _ in balance(item, {"files": {"audio": str(wav)}}, None)}
    assert bal["line_lu_err"] < 0.05


def test_e4_bs1770_true_peak_limiter():
    from bench.arena.specs.e4 import true_peak_dbtp

    w = _worker("e4_bs1770")
    y = np.random.default_rng(0).normal(0, 0.3, (48000 * 2, 2))
    z = w.tp_limit({"np": np}, y, 48000, -2.0, 5.0)
    assert true_peak_dbtp(z, 48000) <= -2.0 + 1e-6
    assert np.sqrt(np.mean(z ** 2)) > 0.5 * np.sqrt(np.mean(y ** 2))  # limits, not mutes


# ------------------------------------------------------------------ E1 judges + worker

def test_e1_pause_judges_and_stretch(tmp_path):
    from bench.arena.specs.e1 import (
        internal_pauses,
        pause_alignment,
        speech_intervals,
        stretch_profile,
        timing,
    )

    src = _utterance(tmp_path / "src.wav")
    speech = speech_intervals(src, SR16)
    pauses = internal_pauses(speech)
    assert len(pauses) == 2 and pauses[0][0] == pytest.approx(1.3, abs=0.03)
    target = len(src) / SR16
    item = Item(id="x", lang="en", inputs={"audio": str(tmp_path / "src.wav"), "target_s": target},
                refs={"pauses": pauses, "speech": speech, "target_s": target, "text": "t"})
    same = {m: v for m, v, _ in pause_alignment(item, {"files": {"audio": str(tmp_path /
                                                                              "src.wav")}}, None)}
    assert same["pause_f1"] == 1.0
    _utterance(tmp_path / "flat.wav", (0.1, 1.0, 0.16, 0.8, 0.16, 0.6, 0.1))
    rows = {m: v for m, v, _ in pause_alignment(item, {"files": {"audio": str(tmp_path /
                                                                              "flat.wav")}}, None)}
    assert rows["pause_f1"] < 0.5
    t = {m: v for m, v, _ in timing(item, {"files": {"audio": str(tmp_path / "flat.wav")}}, None)}
    assert t["overflow"] == 0.0 and t["dur_err_ms"] > 900
    long = np.concatenate([src, _sil(0.5)])
    sf.write(tmp_path / "long.wav", long, SR16)
    t = {m: v for m, v, _ in timing(item, {"files": {"audio": str(tmp_path / "long.wav")}}, None)}
    assert t["overflow"] == 1.0
    assert stretch_profile([[0, 0], [1, 0.8], [2, 1.8]], [[0, 2]], [[0, 1.8]]) == \
        pytest.approx((0.25, 0.125))


def test_on_input_judge_and_delta():
    from bench.arena.judges import ModelJudge
    from bench.arena.specs.e1 import delta_vs_input, on_input

    j = ModelJudge(id="asr_roundtrip.x@1", worker="w",
                   inputs=lambda it, out, p: {"audio": out["files"]["audio"]},
                   to_rows=lambda it, pay, out: [("rt_x_cer", pay["v"], 10.0)])
    wrapped = on_input(j)
    item = Item(id="i", lang="en", inputs={"audio": __file__})
    assert wrapped.id == "input.asr_roundtrip.x@1" and wrapped.key != j.key
    assert wrapped.inputs(item, {"files": {"audio": "/nope.wav"}}, Path("."))["audio"] == __file__
    assert wrapped.to_rows(item, {"v": 0.1}, {}) == [("in_rt_x_cer", 0.1, 10.0)]
    d = delta_vs_input("rt_", "_cer")({"rt_x_cer": (0.12, 10.0), "in_rt_x_cer": (0.10, 10.0)})
    assert d[0] == pytest.approx(0.02)
    assert delta_vs_input("mos_")({"rt_x_cer": (0.1, 1.0)}) is None


def test_e1_pause_dp_plan_moves_pauses_not_rate():
    w = _worker("e1_timefit")
    take = [[0.04, 1.28], [1.45, 2.40], [2.57, 3.29]]         # speech 15 % long, pauses short
    src = [[0.30, 1.30], [1.80, 2.60], [3.00, 3.60]]
    groups = w.align(take, src, 4.0, 0.05, 0.3)
    assert groups == [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)]
    plan = w.plan(take, src, 3.9, {"mode": "pause_dp"})
    starts = [g[2] for g in plan]
    assert starts == pytest.approx([0.30, 1.80, 3.00], abs=0.02)
    assert plan[-1][2] + plan[-1][3] <= 3.9
    uni = w.plan([[take[0][0], take[-1][1]]], [[src[0][0], src[-1][1]]], 3.9, {"mode": "uniform"})
    assert len(uni) == 1


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_e1_timefit_worker_aligns_pauses(tmp_path):
    from bench.arena.specs.e1 import internal_pauses, match_pauses, speech_intervals

    w = _worker("e1_timefit")
    src = _utterance(tmp_path / "src.wav")
    sp = speech_intervals(src, SR16)
    ps = internal_pauses(sp)
    _utterance(tmp_path / "take.wav", (0.08, 1.15, 0.2, 0.92, 0.2, 0.69, 0.1))
    item = {"inputs": {"audio": str(tmp_path / "take.wav"), "target_s": len(src) / SR16,
                       "src_pauses": ps, "src_speech": sp}}
    tps = {}
    for mode in ("uniform", "pause_dp"):
        out = w.run(w.load({"backend": "atempo", "mode": mode}, "en"), item, tmp_path / mode)
        y, _ = sf.read(out["files"]["audio"])
        assert len(y) / SR16 == pytest.approx(len(src) / SR16, abs=0.01)
        tps[mode] = match_pauses(ps, internal_pauses(speech_intervals(y, SR16)))
        assert out["time_map"][0] == [0.0, 0.0]
    assert tps["pause_dp"] == 2 and tps["uniform"] < 2


# ------------------------------------------------------------------ E2 / E3 judges

def test_e2_spectral_metrics(tmp_path):
    from scipy.signal import resample_poly

    from bench.arena.specs.e2 import effective_bandwidth_hz, lsd, spectral

    rng = np.random.default_rng(0)
    full = rng.normal(0, 0.1, 48000 * 2)
    band = resample_poly(resample_poly(full, 1, 3), 3, 1)[: len(full)]
    assert lsd(full, full) == pytest.approx(0.0, abs=1e-9)
    assert lsd(full, band, f_lo=8000) > lsd(full, band) > 0.1
    assert effective_bandwidth_hz(full) > 20000
    assert 7000 < effective_bandwidth_hz(band) < 9500
    sf.write(tmp_path / "clean.wav", full, 48000, subtype="FLOAT")
    sf.write(tmp_path / "out.wav", band, 48000, subtype="FLOAT")
    item = Item(id="i", lang="en", inputs={}, refs={"clean": str(tmp_path / "clean.wav")},
                meta={"input_sr": 16000, "cutoff_hz": 8000})
    rows = {m for m, _, _ in spectral(item, {"files": {"audio": str(tmp_path / "out.wav")}}, None)}
    assert rows == {"eff_bw_khz", "lsd", "lsd_hf"}


def test_e3_deconvolution_recovers_room(tmp_path):
    from scipy.signal import fftconvolve

    from bench.arena.builders.e3_rooms import synth_rir
    from bench.arena.specs.e3 import rir_params, room_match

    dry = np.random.default_rng(1).normal(0, 0.1, SR16 * 3) * np.abs(np.sin(np.arange(SR16 * 3)
                                                                            / SR16 * 12))
    sf.write(tmp_path / "dry.wav", dry, SR16, subtype="FLOAT")
    h = synth_rir(0.6, 4.0, SR16, seed=3)
    truth = rir_params(h, SR16)
    assert truth["t60_s"] == pytest.approx(0.6, rel=0.2)
    sf.write(tmp_path / "wet.wav", 0.5 * fftconvolve(dry, h), SR16, subtype="FLOAT")
    item = Item(id="r", lang="en", inputs={"audio": str(tmp_path / "dry.wav")}, refs=truth)
    good = {m: v for m, v, _ in room_match(item, {"files": {"audio": str(tmp_path / "wet.wav")}},
                                           None)}
    dry_out = {m: v for m, v, _ in room_match(item, {"files": {"audio": str(tmp_path /
                                                                             "dry.wav")}}, None)}
    assert good["room_err_jnd"] < 1.0 and good["t60_rel_err"] < 0.05
    assert dry_out["room_err_jnd"] > 10


def test_e3_blind_parametric_worker(tmp_path):
    from scipy.signal import fftconvolve

    from bench.arena.builders.e3_rooms import synth_rir

    w = _worker("e3_dsp")
    dry = _utterance(tmp_path / "dry.wav")
    ref = fftconvolve(_utterance(tmp_path / "r.wav", (0.2, 0.7, 0.4, 0.9, 0.3)),
                      synth_rir(0.8, 2.0, SR16, seed=5))
    sf.write(tmp_path / "ref.wav", ref / np.max(np.abs(ref)) * 0.5, SR16, subtype="FLOAT")
    st = w.load({"mode": "blind_parametric"}, "en")
    out = w.run(st, {"inputs": {"audio": str(tmp_path / "dry.wav"),
                                "reference": str(tmp_path / "ref.wav")}}, tmp_path / "o")
    assert 0.1 <= out["params"]["t60_s"] <= 3.0
    y, _ = sf.read(out["files"]["audio"])
    assert len(y) >= len(dry)


# ------------------------------------------------------------------ builders (no downloads)

def _fake_utts(tmp_path: Path, n: int):
    utts = []
    for i in range(n):
        p = tmp_path / f"u{i}.wav"
        pattern = (0.2 + 0.05 * (i % 3), 0.9, 0.35 + 0.05 * (i % 4), 0.7, 0.4, 0.5, 0.2)
        _utterance(p, pattern)
        utts.append({"id": f"fleurs-{i}", "path": p, "text": f"line {i}", "group": str(i // 2),
                     "duration_s": 3.0, "gender": "F"})
    return utts


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_e1_fleurs_synth_builder(data, tmp_path, monkeypatch):
    from bench.arena.builders import e1_takes

    utts = _fake_utts(tmp_path, 4)
    monkeypatch.setattr(e1_takes, "fleurs_utterances", lambda lang, n, seed=0: utts)
    e1_takes.e1_fleurs_synth("en", n=3)
    items = load_pack("E1", "en")
    assert len(items) == 3
    it = items[0]
    assert Path(it.inputs["audio"]).exists() and it.refs["pauses"] == it.inputs["src_pauses"]
    assert it.meta["source"] == "fleurs-synth" and 0.85 <= it.meta["factor"] <= 1.25


def test_e1_arena_builder_and_merge(data, tmp_path, monkeypatch):
    from bench.arena.builders import e1_takes

    take = tmp_path / "take.wav"
    _utterance(take)
    src = tmp_path / "srcline.wav"
    _utterance(src, (0.2, 1.1, 0.6, 0.9, 0.2))
    it = Item(id="d1-7", lang="en", inputs={"text": "hello there", "target_s": 3.5},
              group="g7", meta={"src_audio": str(src)})
    monkeypatch.setattr(e1_takes, "arena_outputs",
                        lambda cap, lang, cands=None: [{"candidate": "tts-a", "cand_key": "k",
                                                        "item": it, "audio": str(take),
                                                        "payload": {}}])
    # an existing row from another source survives the merge
    from bench.arena.builders.e_common import merge_pack

    merge_pack("E1", "en", [{"id": "keep", "inputs": {}, "refs": {}}], "# other\n", "other")
    e1_takes.e1_arena_takes("en")
    items = {i.id: i for i in load_pack("E1", "en")}
    assert set(items) == {"keep", "e1a-D1-tts-a-d1-7"}
    row = items["e1a-D1-tts-a-d1-7"]
    assert row.inputs["target_s"] == 3.5 and len(row.refs["pauses"]) == 1
    assert row.refs["text"] == "hello there"
    lic = (paths.eval_dir() / "E1" / "en" / "LICENSE.md").read_text()
    assert "# other" in lic and "arena outputs" in lic


def test_e2_sim_builder_from_local_dir(data, tmp_path):
    from bench.arena.builders.e2_bwe import e2_vctk_sim

    corpus = tmp_path / "corpus"
    for spk in ("s1", "s2"):
        (corpus / spk).mkdir(parents=True)
        for u in range(2):
            sf.write(corpus / spk / f"{spk}_{u}.wav",
                     np.random.default_rng(u).normal(0, 0.1, 48000), 48000)
    (corpus / "transcripts.tsv").write_text("s1_0\thello\ns2_1\tworld\n")
    e2_vctk_sim("hi", n=2, rates=(16000,), local_dir=str(corpus))
    items = load_pack("E2", "hi")
    assert len(items) == 2
    it = items[0]
    assert sf.info(it.inputs["audio"]).samplerate == 16000
    assert sf.info(it.refs["clean"]).samplerate == 48000
    assert Path(it.refs["ref_audio"]).exists() and it.meta["cutoff_hz"] == 8000
    with pytest.raises(ValueError):
        e2_vctk_sim("ja", n=1)


def test_e3_rooms_builder(data, tmp_path, monkeypatch):
    from bench.arena.builders import e3_rooms

    utts = _fake_utts(tmp_path, 4)
    monkeypatch.setattr(e3_rooms, "fleurs_utterances", lambda lang, n, seed=0: utts)
    e3_rooms.e3_synth_rooms("ja", n=2)
    items = load_pack("E3", "ja")
    assert len(items) == 2
    it = items[0]
    assert {"t60_s", "drr_db", "c50_db", "rir", "text"} <= set(it.refs)
    assert Path(it.inputs["reference"]).exists()


def test_e4_programme_builder(data, tmp_path, monkeypatch):
    from bench.arena.builders import e4_programmes
    from bench.arena.specs.e4 import DELIVERIES

    utts = _fake_utts(tmp_path, 40)
    monkeypatch.setattr(e4_programmes, "fleurs_utterances", lambda lang, n, seed=0: utts)
    e4_programmes.e4_fleurs_programmes("en", n=1)
    items = load_pack("E4", "en")
    assert len(items) == len(DELIVERIES) and len({i.group for i in items}) == 1
    it = items[0]
    for key in ("dialogue", "background", "audio", "reference_dialogue"):
        assert Path(it.inputs[key]).exists(), key
    assert len(it.refs["lines"]) == len(it.inputs["speech"]) >= 6
    assert it.refs["speech_bed_db"] is not None
    manifest = (paths.eval_dir() / "E4" / "en" / "manifest.jsonl").read_text().splitlines()
    assert all(json.loads(line)["meta"]["source"] == "fleurs-programmes" for line in manifest)
