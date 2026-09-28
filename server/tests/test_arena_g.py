"""G phase (judges): model-free tests of judge workers' pure parts, G specs, builders, registry."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from bench.arena import envs, judgelib, paths, stats
from bench.arena.packs import Item, load_pack
from bench.arena.registry import load_candidates

WORKERS = paths.WORKERS_DIR
G_WORKERS = ["judge_audio", "judge_speaker_sim", "judge_mos", "judge_emotion", "judge_lipsync",
             "g1_ctc", "g1_panel", "g6_rubric", "g6_gemini", "g6_openai", "g6_qwen3_omni",
             "g6_audio_flamingo", "g6_transcript_review"]
HEAVY = ["numpy", "torch", "transformers", "soundfile", "funasr", "speechbrain", "google",
         "openai", "anthropic", "faster_whisper", "ctranslate2"]


@pytest.fixture()
def workers_path(monkeypatch):
    monkeypatch.syspath_prepend(str(WORKERS))
    yield


def _item(**kw) -> Item:
    base = {"id": "x", "lang": "hi", "inputs": {}, "refs": {}, "group": "g", "meta": {}}
    base.update(kw)
    return Item(**base)


# ------------------------------------------------------------------ registry / envs / specs

@pytest.mark.parametrize("cap", ["G1", "G2", "G3", "G4", "G5", "G6"])
def test_g_candidates_load_and_resolve(cap):
    cands = load_candidates(cap)
    assert len(cands) >= 5
    assert len({c.id for c in cands}) == len(cands)
    for c in cands:
        assert c.url.startswith("https://"), (cap, c.id)
        assert c.kind in ("local", "api", "method")
        assert c.sources, (cap, c.id)
        if not c.enabled:
            assert c.notes, (cap, c.id)  # disabled rows say why
            continue
        assert (WORKERS / f"{c.worker}.py").exists(), (cap, c.id, c.worker)
        env = envs.load_env(c.env)
        if c.env != "server":
            assert env.setup.get("x86_64") or env.setup.get("any"), c.env
            if "aarch64" not in env.setup and "any" not in env.setup:
                assert "x86_64" in c.requires.arch, (cap, c.id)
        if c.kind == "api":
            assert c.requires.api_keys, (cap, c.id)


def test_g_envs_have_both_arch_recipes_or_pin_x86():
    for name in ["judge_speaker", "judge_mos", "judge_emotion", "judge_lipsync_synchformer",
                 "judge_lipsync_syncnet", "judge_lipsync_peavs", "g6_api", "g6_qwen3_omni",
                 "g6_audio_flamingo"]:
        env = envs.load_env(name)
        assert {"x86_64", "aarch64"} <= set(env.setup), name
        assert env.check and env.python.startswith("~/.opendub/envs/"), name


def test_g_specs_registered_with_corpus_primaries():
    from bench.arena.specs import SPECS

    for cap in ["G1", "G2", "G3", "G4", "G5", "G6"]:
        spec = SPECS[cap]
        prim = spec.primary_for("hi")
        assert prim in spec.higher_is_better
        for m in spec.secondary:
            assert m in spec.higher_is_better, (cap, m)
        for name, cm in spec.corpus.items():
            assert name in spec.higher_is_better, (cap, name)
            assert cm.pred in spec.higher_is_better or cm.pred.startswith("p_"), (cap, cm.pred)
        from bench.arena.builders import BUILDERS

        for b in spec.packs:
            assert b in BUILDERS and cap in BUILDERS[b].capabilities, (cap, b)


def test_workers_import_stdlib_only():
    code = (f"import sys; sys.path.insert(0, {str(WORKERS)!r})\n"
            + "".join(f"import {w}\n" for w in G_WORKERS)
            + f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
            + "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ judgelib factories

def test_judgelib_factories_match_workers():
    names = {j.id.split("@")[0] for j in judgelib.asr_roundtrip("hi")}
    assert "asr_roundtrip.mms" in names
    mms = next(j for j in judgelib.asr_roundtrip("hi") if j.id.startswith("asr_roundtrip.mms"))
    assert mms.worker == "g1_ctc" and (WORKERS / "g1_ctc.py").exists()
    for enc in judgelib.SIM_ENCODERS:
        j = judgelib.speaker_similarity(enc)
        assert j.worker == "judge_speaker_sim" and j.params == {"encoder": enc}
        rows = j.to_rows(_item(), {"metrics": {f"sim_{enc}": 0.7}, "score": 0.7}, {})
        assert rows == [(f"sim_{enc}", 0.7, 1.0)]
    for model in judgelib.MOS_MODELS:
        assert judgelib.naturalness(model).env == "judge_mos"
    emo = judgelib.emotion_consistency()
    assert emo.params == {"model": "emotion2vec", "avd": True}
    assert judgelib.emotion_consistency("odyssey-avd").params["avd"] is True
    for m in judgelib.LIPSYNC_MODELS:
        j = judgelib.lipsync_score(m)
        assert j.env == f"judge_lipsync_{m}" and (WORKERS / f"{j.worker}.py").exists()
        envs.load_env(j.env)


# ------------------------------------------------------------------ G spec function judges

def test_auc_matches_mann_whitney_with_ties():
    from bench.arena.specs._gmeta import auc

    assert auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(auc([0.1, 0.2], [1, 1]))
    assert math.isnan(auc([0.1, 0.2, 0.3], [0.5, 1, 0]))
    val = stats.corpus_value(auc, np.array([0.3, 0.2, 0.9, 0.1]), np.array([1.0, 0, 1, 0]))
    assert val == 1.0


def test_g1_defect_rows_cer_floor_and_worker_score():
    from bench.arena.specs.g1 import defect_rows

    clean = _item(lang="hi", inputs={"text": "नमस्ते दुनिया"},
                  refs={"text": "नमस्ते दुनिया", "defective": 0})
    rows = {m: v for m, v, _ in defect_rows(clean, {"text": "नमस्ते दुनिया"}, None)}
    assert rows["cer"] == 0.0 and rows["floor_cer"] == 0.0 and rows["defect_score"] == 0.0
    bad = _item(lang="hi", refs={"text": "नमस्ते दुनिया", "defective": 1})
    rows = {m: v for m, v, _ in defect_rows(bad, {"text": "नमस्ते", "defect_score": 3.2}, None)}
    assert "floor_cer" not in rows and rows["defect_score"] == 3.2 and rows["cer"] > 0


def test_score_judge_and_g5_rows():
    from bench.arena.specs._gmeta import score_judge
    from bench.arena.specs.g5 import _detect, sync_rows

    assert score_judge()(_item(), {"score": 0.4}, None) == [("pred", 0.4, 1.0)]
    assert score_judge()(_item(), {"score": None}, None) == []
    it = _item(refs={"offset_ms": -120.0})
    rows = {m: v for m, v, _ in sync_rows(it, {"desync": 100.0, "offset_ms": -100.0}, None)}
    assert rows == {"desync": 100.0, "offset_ms": -100.0, "offset_abs_err_ms": 20.0}
    assert sync_rows(it, {"desync": 1.2, "offset_ms": None}, None) == [("desync", 1.2, 1.0)]
    assert _detect(_item(refs={"offset_ms": 40.0})) is None
    assert _detect(_item(refs={"offset_ms": 0.0})) == 0.0
    assert _detect(_item(refs={"offset_ms": -80.0})) == 1.0


def test_g6_review_rows_balanced_accuracy():
    from bench.arena.specs.g6 import SPEC, review_rows

    clean = _item(refs={"defects": [], "has_video": False, "weight": 0.8})
    out = {"metrics": {"p_wrong_speaker": 0.1, "p_off_sync": 0.9, "p_clipping": 0.2,
                       "p_untranslated": 0.0}}
    rows = {m: (v, w) for m, v, w in review_rows(clean, out, None)}
    assert "p_off_sync" not in rows  # not applicable without video; its 0.9 is ignored
    assert rows["bal_acc"] == (1.0, 0.8) and rows["false_alarm"] == (0.0, 1.0)
    clip = _item(refs={"defects": ["clipping"], "has_video": False, "weight": 1.5})
    rows = {m: (v, w) for m, v, w in review_rows(clip, out, None)}
    assert rows["recall_clipping"] == (0.0, 1.0) and rows["bal_acc"] == (0.0, 1.5)
    assert SPEC.corpus["auc_off_sync"].target(clean) is None
    assert SPEC.corpus["auc_clipping"].target(clip) == 1.0


# ------------------------------------------------------------------ worker pure parts

def test_rubric_plan_combine_and_parse(workers_path):
    import g6_rubric as R

    inputs = {"audio": "/t.wav", "ref_audio": "/r.wav", "text": "hello", "video": "/v.mp4"}
    hol = R.plan("review", "holistic", inputs, "en", {"video": True})
    assert len(hol) == 1 and set(hol[0]["keys"]) == set(R.DEFECTS)
    dec = R.plan("review", "decomposed", inputs, "en", {"video": False})
    keys = [q["key"] for q in dec]
    assert keys.count("wrong_speaker") == 2 and "off_sync" not in keys  # both orders; no video
    assert all("wrong_speaker" not in q["prompt"] for q in dec)  # no label leakage
    answers = [{"p": 0.9, "evidence": "German speech"}, {"p": 0.1, "evidence": ""},
               {"p": 0.8, "evidence": "different timbre"}, {"p": 0.6, "evidence": ""}]
    out = R.combine("review", dec, answers, {"video": False}, inputs)
    m = out["metrics"]
    assert m["p_untranslated"] == 0.9 and m["p_off_sync"] == 0.0 and out["no_video"]
    assert m["p_wrong_speaker"] == pytest.approx((0.8 + 0.3) / 2)  # unevidenced 0.6 halved
    assert R.parse_json('Sure!\n```json\n{"p": 0.2, "x": {"y": 1}}\n```') == {"p": 0.2,
                                                                               "x": {"y": 1}}
    with pytest.raises(ValueError):
        R.parse_json("no json here")
    spk = R.plan("speaker", "", inputs, "en", {})
    res = R.combine("speaker", spk, [{"rating": 5}, {"rating": 3}], {}, inputs)
    assert res["score"] == pytest.approx(0.75)
    assert R.cost_usd({"in": 1_000_000, "out": 2_000}, {"in": 0.75, "out": 3.75}) == \
        pytest.approx(0.7575)


def test_rubric_run_item_retries_unparseable(workers_path):
    import g6_rubric as R

    calls = []

    def ask(state, parts, prompt):
        calls.append(prompt)
        return ("garbage" if len(calls) == 1 else '{"rating": 4, "evidence": "ok"}',
                {"in": 10, "out": 5})

    item = {"inputs": {"audio": "/a.wav"}}
    out = R.run_item(ask, None, item, "ja", task="mos", protocol="", caps={},
                     prices={"in": 1.0, "out": 1.0})
    assert out["metrics"]["mos_llm"] == 4.0 and out["unparsed"] == 0
    assert out["usage"] == {"in": 20, "out": 10} and out["_cost_usd"] == pytest.approx(3e-5)


def test_judge_audio_helpers(workers_path, tmp_path):
    import judge_audio as A

    assert A.cosine([1, 0], [1, 0]) == 1.0 and A.cosine([1, 0], [0, 2]) == 0.0
    with pytest.raises(ValueError):
        A.cosine([1, 0], [1, 0, 0])
    wav = tmp_path / "a.wav"
    sf.write(wav, np.sin(np.linspace(0, 100, 16000)).astype(np.float32), 16000)
    x = A.load_mono(wav, 16000)
    assert x.dtype == np.float32 and len(x) == 16000
    seen = []
    cache = A.RefCache(lambda p: seen.append(p) or [1.0, 2.0])
    cache(str(wav))
    cache(str(wav))
    assert len(seen) == 1
    if shutil.which("ffmpeg"):
        assert abs(len(A.load_mono(wav, 8000)) - 8000) < 50


def test_rubric_concat_single_audio(workers_path, tmp_path):
    import g6_rubric as R

    for name in ("a", "b"):
        sf.write(tmp_path / f"{name}.wav", np.zeros(16000, dtype=np.float32), 16000)
    parts, note = R.concat_for_single_audio(
        [("text", "Recording A:"), ("audio", str(tmp_path / "a.wav")), ("text", "Recording B:"),
         ("audio", str(tmp_path / "b.wav"))], str(tmp_path))
    assert len(parts) == 1 and "Recording B: from 2.0s to 3.0s" in note
    assert len(sf.read(parts[0][1])[0]) == 3 * 16000


def test_g1_panel_cer_and_ctc_targets(workers_path):
    import g1_ctc
    import g1_panel

    assert g1_panel._cer("Hello, world!", "hello world") == 0.0
    assert g1_panel._cer("abcd", "abxd") == 0.25

    vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}

    class Tok:
        pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, 1, None, None

        def __call__(self, text):
            return types.SimpleNamespace(input_ids=[vocab.get(ch, 1)
                                                    for ch in text.replace(" ", "|")])

    state = {"processor": types.SimpleNamespace(tokenizer=Tok()), "vocab": vocab}
    assert g1_ctc._target_ids(state, "AB, a!") == [3, 4, 2, 3]


# ------------------------------------------------------------------ builders (no downloads)

def _wav(path, n=16000 * 3, freq=220.0):
    t = np.arange(n) / 16000
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 16000)
    return path


def test_g1_inject_and_build_items(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    from bench.arena.builders.g1_defects import DEFECTS, build_items, inject

    rng = np.random.default_rng(0)
    x = np.random.default_rng(1).normal(0, 0.1, 16000 * 4).astype(np.float32)
    for d in DEFECTS:
        y = inject(x, d, rng, other=x[:8000])
        assert y.ndim == 1 and len(y) > 0 and not np.array_equal(y, x), d
    src = tmp_path / "src"
    src.mkdir()
    rows = [{"sid": str(i), "file": _wav(src / f"{i}.wav", freq=200 + 30 * i).name,
             "text": f"sentence {i}"} for i in range(8)]
    out = tmp_path / "eval" / "G1" / "hi"
    manifest = build_items(rows, src, out, seed=0)
    assert {m["refs"]["defective"] for m in manifest} == {0, 1}
    for m in manifest:
        assert (out / m["inputs"]["audio"]).exists()
        assert (m["meta"]["defect"] is None) == (m["refs"]["defective"] == 0)


def test_merge_pack_keeps_other_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    from bench.arena.builders._gcommon import merge_pack

    row = {"id": "a", "inputs": {"audio": "x.wav"}, "refs": {"mos": 3.0}}
    merge_pack("G3", "en", [dict(row)], "bvcc", "# BVCC")
    merge_pack("G3", "en", [dict(row, id="b")], "somos", "# SOMOS")
    merge_pack("G3", "en", [dict(row, id="c")], "bvcc", "# BVCC v2")
    ids = sorted(it.id for it in load_pack("G3", "en"))
    assert ids == ["b", "c"]
    lic = (tmp_path / "eval" / "G3" / "en" / "LICENSE.md").read_text()
    assert "BVCC v2" in lic and "SOMOS" in lic and "# BVCC\n" not in lic


def test_rating_parsers():
    from bench.arena.builders.g2_similarity import parse_vcc_similarity, parse_voxsim
    from bench.arena.builders.g3_mos import ttsds_mos

    vox = parse_voxsim(["id1/a/1.wav,id2/b/2.wav,diff,L1,2", "id1/a/1.wav,id2/b/2.wav,diff,L2,4",
                        "clip1,clip2,label,listener,rating"])
    assert vox == {("id1/a/1.wav", "id2/b/2.wav"): 3.0}
    lines = ["121,X,1,32,N06,S00,VCC2SM2,VCC2TF2,HUB,S,,1,,2017",
             "121,X,1,33,T00,D05,VCC2SFX,VCC2TF2,HUB,D,,3,,2017",
             "121,X,1,34,D05,T00,VCC2SFX,VCC2TF2,HUB,D,,1,,2017",
             "121,X,1,0,N14,\\N,VCC2SM1,VCC2TF2,HUB,\\N,1,,,2017"]
    cells = parse_vcc_similarity(lines)
    assert cells == {("D05", "VCC2TF2"): pytest.approx((1 / 3 + 1.0) / 2)}
    recs = [{"audio": "a.wav", "rating_type": "mos", "value": 4, "system": "s", "dataset": "d"},
            {"audio": "a.wav", "rating_type": "mos", "value": 2, "system": "s", "dataset": "d"},
            {"audio": "a.wav", "rating_type": "smos", "value": 1, "system": "s", "dataset": "d"}]
    assert ttsds_mos(recs) == {"a.wav": {"mos": 3.0, "system": "s", "domain": "d"}}


def test_read_mos_list_and_user_ratings(tmp_path):
    from bench.arena.builders.g3_mos import read_mos_list
    from bench.arena.builders.g_user_ratings import read_ratings

    (tmp_path / "test_mos_list.txt").write_text("sys1-u1.wav,3.5\nsys2-u2.wav,4.25\n")
    assert read_mos_list(tmp_path / "test_mos_list.txt") == {"sys1-u1.wav": 3.5,
                                                              "sys2-u2.wav": 4.25}
    csv_path = tmp_path / "t2.csv"
    csv_path.write_text("audio,ref_audio,score,group\nwav/a.wav,ref/r.wav,3.75,sysA\n")
    rows = read_ratings(csv_path, "G4")
    assert rows[0]["refs"] == {"emo": 3.75} and rows[0]["group"] == "sysA"
    assert rows[0]["inputs"]["ref_audio"] == str(tmp_path / "ref" / "r.wav")
    with pytest.raises(ValueError):
        (tmp_path / "bad.csv").write_text("audio,score\na.wav,1\n")
        read_ratings(tmp_path / "bad.csv", "G2")


def test_emotion_pairs_balanced_and_disjoint(tmp_path):
    from bench.arena.builders.g4_emotion import index_corpus, make_pairs

    for spk in ("0011", "0012", "0013", "0001"):
        for k, emo in enumerate(("Angry", "Happy", "Neutral")):
            d = tmp_path / spk / emo
            d.mkdir(parents=True)
            for j in range(3):
                (d / f"{spk}_{k * 350 + j + 1:06d}.wav").write_bytes(b"")
    clips = index_corpus(tmp_path, "esd")
    assert {c["lang"] for c in clips} == {"en", "zh"}
    pairs = make_pairs(clips, "en", 20, seed=0)
    assert len(pairs) == 20 and sum(p["same"] for p in pairs) == 10
    for p in pairs:
        assert p["take"]["spk"] != p["ref"]["spk"] and p["take"]["text_id"] != p["ref"]["text_id"]
        assert (p["take"]["emo"] == p["ref"]["emo"]) == bool(p["same"])
    xl = make_pairs(clips, "en", 6, seed=1, ref_lang="zh")
    assert all(p["ref"]["lang"] == "zh" for p in xl)


def test_signal_edits():
    from bench.arena.builders._gcommon import class_weights, clipped_share, hard_clip, shift_audio

    x = np.arange(1, 17, dtype=np.float32)
    late = shift_audio(x, 1000 * 2 / 16000)       # +2 samples: audio delayed
    early = shift_audio(x, -1000 * 2 / 16000)
    assert list(late[:3]) == [0, 0, 1] and list(early[:2]) == [3, 4] and len(late) == len(x)
    s = np.sin(np.linspace(0, 20, 16000)).astype(np.float32) * 0.5
    assert clipped_share(hard_clip(s)) > 0.3 > clipped_share(s)
    w = class_weights(["clean", "clean", "clean", "clip"])
    assert w["clip"] * 1 == pytest.approx(w["clean"] * 3)


def test_g6_audio_items_and_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    from bench.arena.builders.g6_defects import add_weights, audio_items

    tgt_dir, src_dir = tmp_path / "tgt", tmp_path / "src"
    tgt_dir.mkdir()
    src_dir.mkdir()
    tgt = [{"sid": str(i), "file": _wav(tgt_dir / f"t{i}.wav", freq=150 + 20 * i).name,
            "text": f"line {i}", "gender": "MALE" if i % 2 else "FEMALE"} for i in range(8)]
    src = {str(i): {"sid": str(i), "file": _wav(src_dir / f"s{i}.wav").name, "text": f"源 {i}"}
           for i in range(8)}
    items = add_weights(audio_items(tgt, src, tgt_dir, src_dir, tmp_path / "out", 8))
    kinds = [(it["refs"]["defects"] or ["clean"])[0] for it in items]
    assert set(kinds) == {"clean", "wrong_speaker", "clipping", "untranslated"}
    for it in items:
        inp = it["inputs"]
        if it["refs"]["defects"] == ["untranslated"]:
            assert inp["audio"].startswith(str(src_dir)) and inp["text"].startswith("line")
        own = f"{it['id'].rsplit('-', 1)[-1]}_head.wav"
        if it["refs"]["defects"] == ["wrong_speaker"]:
            assert not inp["ref_audio"].endswith(own)
        elif it["refs"]["defects"] in ([], ["clipping"]):
            assert inp["ref_audio"].endswith(own)
        assert it["refs"]["weight"] > 0
    json.dumps(items)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_lipsync_offset_rows_mux(tmp_path):
    from bench.arena.builders.g5_offsets import offset_rows

    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x64:rate=25", "-t", "1", "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    wav = _wav(tmp_path / "a.wav", n=16000)
    rows = offset_rows("clip000", video, wav, tmp_path / "pack", [-40, 0, 0, 80])
    assert [r["refs"]["offset_ms"] for r in rows] == [-40.0, 0.0, 0.0, 80.0]
    assert len({r["id"] for r in rows}) == 4
    shifted = sf.read(tmp_path / "pack" / rows[-1]["inputs"]["audio"])[0]
    assert np.allclose(shifted[:1280], 0) and len(shifted) == 16000
    assert (tmp_path / "pack" / rows[0]["inputs"]["video"]).exists()
