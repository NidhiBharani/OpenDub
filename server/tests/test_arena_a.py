"""Phase A (A1–A8) arena pieces, model-free: specs, judges on synthetic payloads, worker helpers,
candidate registries and pack builders on tiny fixtures (downloads monkeypatched away)."""
from __future__ import annotations

import ast
import importlib
import json
import sys

import numpy as np
import pytest
import soundfile as sf

from bench.arena import paths
from bench.arena.packs import Item, load_pack

A_CAPS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
LEGACY_A4 = {"fw-large-v3-turbo", "fw-large-v3", "fw-medium", "kotoba-whisper-v2",
             "qwen3-asr-1.7b", "indicconformer-600m"}
SR = 16000


@pytest.fixture()
def workers_path(monkeypatch):
    monkeypatch.syspath_prepend(str(paths.WORKERS_DIR))


def _wav(path, x, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr)
    return str(path)


def _tone(seconds=1.0, f=220.0, sr=SR, seed=0):
    t = np.arange(int(seconds * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * f * t) + 0.01 * np.random.default_rng(seed).standard_normal(
        len(t))


# ------------------------------------------------------------------ specs & registries

def test_specs_registered_with_directions():
    from bench.arena.specs import SPECS

    for cap in A_CAPS:
        spec = SPECS[cap]
        for lang in ("en", "hi", "ja"):
            assert spec.primary_for(lang) in spec.higher_is_better
        for m in spec.secondary + list(spec.gates):
            assert m in spec.higher_is_better, (cap, m)
        for name in spec.derived:
            assert name in spec.higher_is_better


def test_spec_packs_are_registered_builders():
    from bench.arena.builders import BUILDERS
    from bench.arena.specs import SPECS

    for cap in ["A1", "A2", "A3", "A5", "A6", "A7", "A8"]:
        for name in SPECS[cap].packs:
            assert name in BUILDERS and cap in BUILDERS[name].capabilities, name


def test_candidates_complete_and_consistent():
    from bench.arena.envs import load_env
    from bench.arena.registry import load_candidates

    for cap in A_CAPS:
        cands = load_candidates(cap)  # raises on duplicate ids
        assert cands, cap
        assert sum(c.baseline for c in cands) >= 1, cap
        for c in cands:
            assert (paths.WORKERS_DIR / f"{c.worker}.py").exists(), (cap, c.id, c.worker)
            load_env(c.env)
            if c.id not in LEGACY_A4:
                assert c.url.startswith("http"), (cap, c.id)
                assert c.sources, (cap, c.id)
                assert c.notes or c.enabled, (cap, c.id)  # disabled rows must say why
            if c.kind == "api":
                assert c.requires.api_keys or not c.enabled, (cap, c.id)


def test_a4_legacy_entries_untouched():
    from bench.arena.registry import load_candidates

    ids = [c.id for c in load_candidates("A4")]
    assert ids[:6] == ["fw-large-v3-turbo", "fw-large-v3", "fw-medium", "kotoba-whisper-v2",
                       "qwen3-asr-1.7b", "indicconformer-600m"]
    ic = next(c for c in load_candidates("A4") if c.id == "indicconformer-600m")
    assert ic.enabled is False


STDLIB_OK = set(sys.stdlib_module_names) | {"_sdk", "__future__"}


def test_a_workers_import_stdlib_only_at_module_level():
    files = sorted(p for p in paths.WORKERS_DIR.glob("a[1-8]_*.py"))
    assert len(files) > 40
    for f in files:
        tree = ast.parse(f.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for n in names:
                assert n in STDLIB_OK or n.startswith(("a1_", "a3_", "a4_", "a5_", "a7_", "a8_")), \
                    (f.name, n)


def test_a_envs_have_recipes():
    from bench.arena.envs import all_envs

    envs = {e.name: e for e in all_envs()}
    mine = [n for n in envs if n[:1] == "a" and n[1:2].isdigit()]
    assert len(mine) >= 20
    for n in mine:
        e = envs[n]
        assert "x86_64" in e.setup and e.check, n
        assert e.python.startswith("~/.opendub/envs/"), n


# ------------------------------------------------------------------ A1

def test_a1_si_sdr_and_stem_rows(tmp_path):
    from bench.arena.specs.a1 import _stem_rows, si_sdr

    rng = np.random.default_rng(0)
    speech, bed = _tone(2, 300), 0.2 * rng.standard_normal(2 * SR)
    assert si_sdr(speech, speech) > 60
    assert si_sdr(speech, speech + 0.1 * rng.standard_normal(len(speech))) < 20
    mix = speech + bed
    refs = {"dialogue": _wav(tmp_path / "d.wav", speech), "background": _wav(tmp_path / "b.wav",
                                                                              bed)}
    item = Item(id="x", lang="en", inputs={"audio": _wav(tmp_path / "m.wav", mix)}, refs=refs)
    good = {"files": {"dialogue": refs["dialogue"], "background": refs["background"]}}
    rows = {m: v for m, v, _ in _stem_rows(item, good, None)}
    assert rows["sisdri_dialogue"] > 10 and rows["sisdri_background"] > 10
    assert rows["recon_sdr"] > 30
    passthrough = {"files": {"dialogue": item.inputs["audio"], "background": item.inputs["audio"]}}
    rows = {m: v for m, v, _ in _stem_rows(item, passthrough, None)}
    assert abs(rows["sisdri_dialogue"]) < 1e-6 and abs(rows["sisdri_background"]) < 1e-6
    missing = {m: v for m, v, _ in _stem_rows(item, {"files": {}}, None)}
    assert missing["missing_dialogue"] == 1.0


def test_a1_ghost_recall_and_judges(monkeypatch):
    from bench.arena import judges
    from bench.arena.specs import a1

    monkeypatch.setattr(judges, "english_normalizer", lambda: (lambda s: s))
    assert a1.word_recall("hi", "एक दो तीन", "") == (0.0, 3.0)
    assert a1.word_recall("hi", "एक दो तीन", "तीन और एक") == (2 / 3, 3.0)
    assert a1.word_recall("ja", "今日は", "今日")[0] == pytest.approx(2 / 3)
    mj = a1.ghost_dialogue("ja")
    assert {"ghost_dialogue.whisper@1", "ghost_dialogue.qwen3@1"} <= {j.id for j in mj}
    it = Item(id="x", lang="hi", inputs={}, refs={"text": "एक दो"})
    whisper = next(j for j in mj if j.id == "ghost_dialogue.whisper@1")
    rows = whisper.to_rows(it, {"text": "एक"}, {})
    assert rows == [("ghost_whisper_recall", 0.5, 2.0)]
    derived = a1.SPEC.derived["ghost_recall"]({"ghost_whisper_recall": (0.2, 3),
                                              "ghost_qwen3_recall": (0.4, 3)})
    assert derived[0] == pytest.approx(0.3)


# ------------------------------------------------------------------ A2

def test_a2_signal_rows(tmp_path):
    from bench.arena.specs.a2 import _signal_rows

    clean = _tone(1.5, 200)
    noisy = clean + 0.2 * np.random.default_rng(1).standard_normal(len(clean))
    item = Item(id="x", lang="en", inputs={"audio": _wav(tmp_path / "n.wav", noisy)},
                refs={"clean": _wav(tmp_path / "c.wav", clean)})
    rows = {m: v for m, v, _ in _signal_rows(item, {"files": {"audio": item.refs["clean"]}},
                                             None)}
    assert rows["sisdri"] > 5 and rows["lsd_db"] < 1e-3 and rows["missing_audio"] == 0.0
    assert _signal_rows(item, {"files": {}}, None) == [("missing_audio", 1.0, 1.0)]


# ------------------------------------------------------------------ A3

def test_a3_region_f1_recall_and_false_speech():
    from bench.arena.specs.a3 import canon_label, region_rows

    assert canon_label("Male") == "speech" and canon_label("noEnergy") is None
    ref = [{"start": 0, "end": 10, "label": "speech"}, {"start": 10, "end": 20, "label": "music"}]
    item = Item(id="x", lang="en", inputs={}, refs={"segments": ref},
                meta={"duration_s": 20, "classes": ["speech", "music"]})
    perfect = {m: v for m, v, _ in region_rows(item, {"segments": ref}, None)}
    assert perfect["macro_f1"] == 1.0 and perfect["speech_recall"] == 1.0
    assert perfect["false_speech_per_h_music"] == 0.0
    hyp = [{"start": 0, "end": 20, "label": "female", "score": 0.9}]  # speech everywhere
    rows = {m: (v, w) for m, v, w in region_rows(item, {"segments": hyp}, None)}
    assert rows["f1_speech"][0] == pytest.approx(2 / 3) and rows["f1_speech"][1] == 300
    assert rows["false_speech_per_h_music"][0] == pytest.approx(3600.0)
    assert "f1_music" in rows and rows["f1_music"][0] == 0.0


# ------------------------------------------------------------------ A5

def test_a5_boundaries_across_tokenisations(monkeypatch):
    from bench.arena.specs.a5 import align_rows

    ref = [{"start": 0.0, "end": 0.5, "word": "今日は"}, {"start": 0.5, "end": 1.2, "word": "晴れ"}]
    chars = [{"start": 0.01, "end": 0.2, "word": "今"}, {"start": 0.2, "end": 0.3, "word": "日"},
             {"start": 0.3, "end": 0.51, "word": "は"}, {"start": 0.52, "end": 0.9, "word": "晴"},
             {"start": 0.9, "end": 1.25, "word": "れ"}]
    item = Item(id="x", lang="ja", inputs={}, refs={"words": ref})
    rows = {m: v for m, v, _ in align_rows(item, {"words": chars}, None)}
    assert rows["boundary_mae_ms"] == pytest.approx((10 + 10 + 20 + 50) / 4)
    assert rows["within_25ms"] == 0.75 and rows["gross_err"] == 0.0 and rows["unaligned"] == 0.0
    bad = {m: v for m, v, _ in align_rows(item, {"words": chars[:3]}, None)}
    assert bad["unaligned"] == 0.5 and bad["gross_err"] == 0.5
    pseudo = Item(id="y", lang="ja", inputs={}, refs={"words_pseudo": ref})
    names = {m for m, _, _ in align_rows(pseudo, {"words": chars}, None)}
    assert names and all(n.startswith("pseudo_") for n in names)


def test_a5_worker_helpers(workers_path):
    a5 = importlib.import_module("a5_common")
    assert a5.tokenize("今日は晴れです。明日も", "ja") == ["今日は", "晴れです", "明日も"]
    assert a5.tokenize("Hello, world !", "en") == ["Hello,", "world"]
    units = [{"start": 0.0, "end": 0.4, "word": " hello"}, {"start": 0.5, "end": 0.9,
                                                           "word": "word"}]
    out = a5.project(["Hello,", "big", "world"], units)
    assert [w["word"] for w in out] == ["Hello,", "big", "world"]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 0.4
    assert 0.4 <= out[1]["start"] <= out[1]["end"] <= 0.5  # interpolated in the gap


# ------------------------------------------------------------------ A6

def test_a6_der_mapping_overlap_and_lines(tmp_path):
    from bench.arena.specs.a6 import diarization_rows, read_rttm

    ref = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 5, "end": 10, "speaker": "B"},
           {"start": 8, "end": 10, "speaker": "A"}]  # overlap 8–10
    item = Item(id="x", lang="en", inputs={}, refs={"turns": ref}, meta={"duration_s": 10})
    relabelled = [{"start": t["start"], "end": t["end"], "speaker": {"A": "s1", "B": "s2"}[
        t["speaker"]]} for t in ref]
    rows = {m: v for m, v, _ in diarization_rows(item, {"turns": relabelled}, None)}
    assert rows["der"] == pytest.approx(0.0, abs=1e-3) and rows["line_attr_err"] == 0.0
    assert rows["jer"] == pytest.approx(0.0, abs=1e-3) and rows["spk_count_err"] == 0.0
    one = [{"start": 0, "end": 10, "speaker": "X"}]
    rows = {m: v for m, v, _ in diarization_rows(item, {"turns": one}, None)}
    # ref speech = 12 s; X maps to A: miss 2 s (overlap 8–10), confusion 3 s (B alone 5–8)
    assert rows["der"] == pytest.approx(5 / 12, abs=0.01)
    assert rows["der_miss"] == pytest.approx(2 / 12, abs=0.01)
    assert rows["der_confusion"] == pytest.approx(3 / 12, abs=0.01)
    assert 0 <= rows["der_collar"] <= 1
    assert rows["line_attr_err"] == pytest.approx(1 / 3)
    empty = {m: v for m, v, _ in diarization_rows(item, {"turns": []}, None)}
    assert empty["der"] == pytest.approx(1.0) and empty["line_attr_err"] == 1.0
    rttm = tmp_path / "x.rttm"
    rttm.write_text("SPEAKER x 1 0.50 1.25 <NA> <NA> spk0 <NA> <NA>\n")
    assert read_rttm(rttm) == [{"start": 0.5, "end": 1.75, "speaker": "spk0"}]


# ------------------------------------------------------------------ A7

def test_a7_eer_min_dcf_and_rows():
    from bench.arena.specs.a7 import SPEC, eer, embedding_rows, min_dcf

    assert eer([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 0.0
    assert eer([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) >= 0.5
    assert eer([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0]) == pytest.approx(0.5)
    assert min_dcf([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 0.0
    assert 0 < min_dcf([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0]) <= 1.0
    trial = Item(id="t", lang="en", inputs={}, refs={"label": 1})
    assert embedding_rows(trial, {"scores": [0.7]}, None) == [("trial_score", 0.7, 1.0)]
    bank = Item(id="b", lang="en", inputs={}, refs={"match": -1})
    assert embedding_rows(bank, {"scores": [0.1, 0.2], "match": None}, None) == \
        [("bank_top1", 1.0, 1.0)]
    assert SPEC.corpus["eer"].target(trial) == 1.0


def test_a7_score_item_bank_keys(tmp_path, workers_path):
    a7 = importlib.import_module("a7_common")
    vecs = {"t": [1, 0], "b0": [0, 1], "b1": [1, 0.1]}
    item = {"inputs": {"audio": "t", "bank_00": "b0", "bank_01": "b1"}}
    res = a7.score_item(lambda p: vecs[p], item, tmp_path / "o", {})
    assert res["match"] == 1 and res["scores"][0] == pytest.approx(0.0, abs=1e-6)
    assert np.load(res["embedding_file"]).shape == (2,)
    res = a7.score_item(lambda p: vecs[p], item, tmp_path / "o2", {"abstain": 0.999})
    assert res["match"] is None


# ------------------------------------------------------------------ A8

def test_a8_labels_corpus_metrics_and_events():
    from bench.arena.specs.a8 import (
        EMOTIONS,
        ccc,
        delivery_rows,
        event_counts,
        macro_f1,
        predicted_emotion,
        uar,
    )

    assert predicted_emotion({"segments": [{"label": "生气/angry", "score": 0.9},
                                           {"label": "neu", "score": 0.1}]}) == "angry"
    assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0 and uar([0, 0, 1], [0, 1, 1]) == 0.75
    assert ccc([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == pytest.approx(1.0)
    item = Item(id="x", lang="ja", inputs={}, refs={"emotion": "happiness", "arousal": 0.7,
                                                    "events": [{"start": 1.0, "label": "laugh"}]})
    out = {"emotion": "Happy", "dims": {"arousal": 0.6},
           "segments": [{"start": 1.1, "end": 1.5, "label": "laughter"}]}
    rows = {m: v for m, v, _ in delivery_rows(item, out, None)}
    assert rows["emo_pred"] == EMOTIONS.index("happy") and rows["emo_acc"] == 1.0
    assert rows["pred_arousal"] == 0.6 and rows["nv_event_f1"] == 1.0
    assert event_counts([{"start": 1.0, "label": "cry"}], [{"start": 1.5, "label": "crying"}]) \
        == (0, 1, 1)


def test_a8_llm_json_parsing(workers_path):
    llm = importlib.import_module("a8_llm")
    reply = 'Sure!\n```json\n{"emotion": "sad", "arousal": 0.2, "valence": 0.1, "events": ' \
            '[{"start": 0.5, "end": 0.9, "label": "sigh"}], "note": "quiet"}\n```'
    p = llm.to_payload("delivery", reply, 2.0)
    assert p["emotion"] == "sad" and p["dims"] == {"arousal": 0.2, "valence": 0.1}
    assert {s["label"] for s in p["segments"]} == {"sad", "sigh"}
    r = llm.to_payload("regions", '{"segments": [{"start": 0, "end": 3, "label": "Music"}]}')
    assert r["segments"][0]["label"] == "music"
    assert "Hindi" in llm.prompt("asr", "hi")
    assert "angry" in llm.prompt("delivery", "en", {"classes": ["angry", "calm"]})


def test_shared_worker_helpers(workers_path):
    http = importlib.import_module("a4_http")
    body, ctype = http.multipart({"model": "x", "flag": True, "langs": ["hi", "ja"]}, {})
    assert b'name="flag"\r\n\r\ntrue' in body and body.count(b'name="langs"') == 2
    assert ctype.startswith("multipart/form-data; boundary=")
    turns = http.words_to_turns([{"start": 0, "end": 1, "speaker": "a"},
                                 {"start": 1.2, "end": 2, "speaker": "a"},
                                 {"start": 2.1, "end": 3, "speaker": "b"}])
    assert turns == [{"start": 0.0, "end": 2.0, "speaker": "a"},
                     {"start": 2.1, "end": 3.0, "speaker": "b"}]
    a3 = importlib.import_module("a3_common")
    segs = a3.to_segments([0.25, 0.75, 1.25, 1.75], 0.5, {"speech": [0.9, 0.8, 0.1, 0.9]},
                          threshold=0.5, min_dur=0.2)
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.0 and len(segs) == 2
    assert a3.group_of("Female singing") == "singing" and a3.group_of("Laughter", "events") \
        == "laughter"


# ------------------------------------------------------------------ builders

@pytest.fixture()
def arena_data(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    return tmp_path


def _fake_clips(tmp_path, n=12, secs=1.0):
    out = []
    for i in range(n):
        p = tmp_path / "clips" / f"c{i}.wav"
        _wav(p, _tone(secs, 150 + 20 * i, seed=i))
        out.append({"id": f"c{i}", "path": p, "text": f"word{i} other{i}", "duration_s": secs,
                    "group": f"g{i}", "gender": "FEMALE"})
    return out


def test_aphase_loudness_and_textgrid_roundtrip(tmp_path):
    from bench.arena.builders import _aphase

    x = _tone(3, 440)
    y = _aphase.set_loudness(x, SR, -23.0)
    assert _aphase.loudness(y, SR) == pytest.approx(-23.0, abs=0.1)
    tg = tmp_path / "a.TextGrid"
    _aphase.write_textgrid(tg, 2.0, {"words": [{"start": 0.1, "end": 0.6, "label": 'say "hi"'},
                                               {"start": 0.8, "end": 1.5, "label": "there"}],
                                     "status": [{"start": 0, "end": 2.0, "label": "stub"}]})
    tiers = _aphase.read_textgrid(tg)
    labels = [r["label"] for r in tiers["words"] if r["label"]]
    assert labels == ['say "hi"', "there"] and tiers["status"][0]["label"] == "stub"
    assert tiers["words"][1]["start"] == pytest.approx(0.1)


def test_write_merged_keeps_other_sources(arena_data):
    from bench.arena.builders._aphase import write_merged

    write_merged("A6", "en", [{"id": "a", "inputs": {}}], "# A\n", "voxconverse")
    write_merged("A6", "en", [{"id": "b", "inputs": {}}], "# B\n", "ami")
    write_merged("A6", "en", [{"id": "c", "inputs": {}}], "# A2\n", "voxconverse")
    assert sorted(it.id for it in load_pack("A6", "en")) == ["b", "c"]
    lic = (arena_data / "eval" / "A6" / "en" / "LICENSE.md").read_text()
    assert "# B" in lic and "# A2" in lic and "# A\n" not in lic


def test_builder_a1_remix_synthetic(arena_data, monkeypatch):
    from bench.arena.builders import a1_dnr

    clips = _fake_clips(arena_data)
    monkeypatch.setattr(a1_dnr, "fleurs_clips", lambda lang, n, dest, seed=0: clips)
    a1_dnr.dnr_remix("hi", n=2, bed_source="synthetic", seconds=6.0)
    items = load_pack("A1", "hi")
    assert len(items) == 2 and items[0].refs["text"].startswith("word")
    mix, sr = sf.read(items[0].inputs["audio"])
    parts = sum(sf.read(items[0].refs[k])[0] for k in ("dialogue", "music", "effects"))
    assert sr == 44100 and np.allclose(mix, parts, atol=1e-5)


def test_builder_a2_degrade(arena_data, monkeypatch):
    from bench.arena.builders import a2_degrade

    clips = _fake_clips(arena_data, n=14)
    monkeypatch.setattr(a2_degrade, "fleurs_clips", lambda lang, n, dest, seed=0: clips)
    a2_degrade.a2_degrade("ja", n=4)
    items = load_pack("A2", "ja")
    assert len(items) == 4 and all(it.meta["degradations"] for it in items)
    assert all(it.refs["clean"].endswith(".wav") for it in items)


def test_builder_a3_regions_synth(arena_data, monkeypatch):
    from bench.arena.builders import a3_regions

    clips = _fake_clips(arena_data, n=40)
    monkeypatch.setattr(a3_regions, "fleurs_clips", lambda lang, n, dest, seed=0: clips)
    a3_regions.a3_regions_synth("en", n=2, seconds=12.0)
    items = load_pack("A3", "en")
    assert len(items) == 2 and items[0].meta["classes"] == ["speech", "music"]
    labels = {s["label"] for it in items for s in it.refs["segments"]}
    assert labels <= {"speech", "music"} and "speech" in labels


def test_builder_a5_stubs_then_gold(arena_data, monkeypatch):
    from bench.arena.builders import _aphase, a5_align_gold

    clips = _fake_clips(arena_data, n=3)
    monkeypatch.setattr(a5_align_gold, "fleurs_clips", lambda lang, n, dest, seed=0: clips)
    a5_align_gold.a5_align_gold("hi", n=3)
    items = load_pack("A5", "hi")
    assert all(not it.refs for it in items) and items[0].inputs["text"] == "word0 other0"
    tg = arena_data / "eval" / "_sources" / "a5_gold" / "hi" / "c0.TextGrid"
    tiers = _aphase.read_textgrid(tg)
    words = [r for r in tiers["words"] if r["label"]]
    assert [w["label"] for w in words] == ["word0", "other0"]
    _aphase.write_textgrid(tg, 1.0, {"words": words, "status": [{"start": 0, "end": 1.0,
                                                                 "label": "done"}]})
    a5_align_gold.a5_align_gold("hi", n=3)
    gold = {it.id: it for it in load_pack("A5", "hi")}["align-c0"]
    assert [w["word"] for w in gold.refs["words"]] == ["word0", "other0"]
    assert gold.meta["ref_kind"] == "gold"


def test_builder_a6_rttm_ih(arena_data):
    from bench.arena.builders.a6_diarization import a6_rttm_ih

    src = arena_data / "eval" / "_sources" / "a6_ih" / "ja"
    _wav(src / "ep1.wav", _tone(2))
    (src / "ep1.rttm").write_text("SPEAKER ep1 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n")
    (src / "ep1.lines.json").write_text(json.dumps([{"start": 0, "end": 1, "speaker": "A"}]))
    a6_rttm_ih("ja")
    (it,) = load_pack("A6", "ja")
    assert it.refs["rttm"].endswith("ep1.rttm") and it.refs["lines"][0]["speaker"] == "A"


def test_builder_a7_cv_speakers(arena_data):
    from bench.arena.builders.a7_speakers import a7_cv_speakers

    src = arena_data / "eval" / "_sources" / "commonvoice" / "hi"
    rows = ["client_id\tpath\tsentence\tgender"]
    for s in range(14):
        for k in range(3):
            _wav(src / "clips" / f"s{s}_{k}.wav", _tone(0.5, 100 + s * 10, seed=k))
            rows.append(f"spk{s:02d}\ts{s}_{k}.wav\tx\t{'male' if s % 2 else 'female'}")
    (src / "validated.tsv").write_text("\n".join(rows) + "\n")
    a7_cv_speakers("hi", n=20, bank_items=6, bank_size=4)
    items = load_pack("A7", "hi")
    trials = [it for it in items if it.meta["task"] == "trial"]
    banks = [it for it in items if it.meta["task"] == "bank"]
    assert len(trials) == 20 and {it.refs["label"] for it in trials} == {0, 1}
    assert len(banks) == 6 and all(sum(k.startswith("bank_") for k in it.inputs) == 4
                                   for it in banks)
    assert all(it.inputs["bank_00"].startswith("/") for it in items)  # resolved by the loader


def test_builder_a8_labelled_ih(arena_data):
    from bench.arena.builders.a8_emotion import a8_labelled_ih

    src = arena_data / "eval" / "_sources" / "a8_ih" / "hi"
    _wav(src / "l1.wav", _tone(1))
    _wav(src / "l2.wav", _tone(1, 300))
    (src / "l1.txt").write_text("0.2\t0.5\tlaughter\n")
    (src / "labels.csv").write_text("path,emotion,arousal,speaker\nl1.wav,Happy,0.8,a\n"
                                    "l2.wav,sad,,b\n")
    a8_labelled_ih("hi")
    items = {it.id: it for it in load_pack("A8", "hi")}
    assert items["ih-l1"].refs["events"][0]["label"] == "laughter"
    assert items["ih-l1"].refs["arousal"] == 0.8 and items["ih-l2"].refs["emotion"] == "sad"
    assert items["ih-l1"].meta["classes"] == ["happy", "sad"]
