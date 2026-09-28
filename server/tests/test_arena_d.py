"""Model arena, phase D (voice): registry, envs, specs, judges, workers' pure logic and builders.
Model-free: no weights, no GPU, no network (downloads are pre-seeded fixtures)."""
from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.arena import paths
from bench.arena.envs import load_env
from bench.arena.hardware import Gpu, Hardware, check
from bench.arena.packs import Item, load_pack
from bench.arena.registry import load_candidates

CAPS = [f"D{i}" for i in range(1, 9)]
SCOPE = {"en", "hi", "ja"}
WORKERS = paths.WORKERS_DIR
TOOLS = ["ffmpeg", "git", "uv", "ollama"]
THALASSA = Hardware(host="thalassa", arch="x86_64", ram_gb=24, disk_free_gb=500,
                    gpus=[Gpu("NVIDIA GeForce RTX 5060 Ti", 15.9, "12.0")],
                    api_keys=["HF_TOKEN"], tools=TOOLS)
SPARK = Hardware(host="spark", arch="aarch64", ram_gb=128, disk_free_gb=2000,
                 gpus=[Gpu("NVIDIA GB10", 112.0, "12.1", unified=True)],
                 api_keys=["HF_TOKEN"], tools=TOOLS)


def wmod(name: str):
    """Import a worker module (stdlib-only at import time) from the workers directory."""
    if str(WORKERS) not in sys.path:
        sys.path.insert(0, str(WORKERS))
    import importlib

    return importlib.import_module(name)


def tone(path: Path, seconds: float, sr: int = 16000, amp: float = 0.3) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


# ------------------------------------------------------------------ registry

@pytest.mark.parametrize("cap", CAPS)
def test_candidates_load_and_are_complete(cap):
    cands = load_candidates(cap)
    assert cands, cap
    assert len({c.id for c in cands}) == len(cands)
    for c in cands:
        assert (WORKERS / f"{c.worker}.py").exists(), (cap, c.id, c.worker)
        env = load_env(c.env)
        assert c.kind in ("local", "api", "method"), c.id
        assert c.url, (cap, c.id, "url is required")
        assert c.license and c.ship_ok is not None, c.id
        assert set(c.verified) >= {"license", "languages", "vram", "runs"}, c.id
        assert c.sources, c.id
        if not c.enabled:
            assert c.notes, (c.id, "disabled rows say why")
            continue
        assert c.worker != "d_unwired", c.id
        assert c.languages == "*" or SCOPE & set(c.languages), (c.id, c.languages)
        if c.kind == "api":
            assert c.requires.api_keys, c.id
        elif c.env != "server" or c.requires.gpu:
            assert c.requires.vram_gb > 0 or c.worker == "d5_kokoro", c.id
        # every enabled row lands somewhere: Thalassa, the Spark, or an API machine
        spots = [not check(c.requires, hw) for hw in (THALASSA, SPARK)]
        assert any(spots) or c.kind == "api", (c.id, c.requires)
        # env recipes cover the arches the candidate may run on
        arches = c.requires.arch or ["x86_64", "aarch64"]
        if env.python != "auto":
            for a in arches:
                assert env.commands(a), (c.id, c.env, a)


def test_thalassa_and_spark_placement():
    by_id = {c.id: c for c in load_candidates("D1")}
    assert not check(by_id["qwen3-tts-1.7b-base"].requires, THALASSA)
    assert check(by_id["fish-s2-pro"].requires, THALASSA)  # 24 GB → Spark only
    assert not check(by_id["fish-s2-pro"].requires, SPARK)
    assert check(by_id["step-audio-editx"].requires, SPARK)  # x86_64-only deps
    assert by_id["f5-v1-base"].baseline and by_id["f5-v1-base"].env == "server"
    assert by_id["f5-hindi-small"].languages == ["hi"]


def test_env_recipes_have_both_arches_or_x86_only_users():
    users: dict[str, list] = {}
    for cap in CAPS:
        for c in load_candidates(cap):
            users.setdefault(c.env, []).append(c)
    for env_file in sorted((paths.SERVER_DIR / "bench" / "envs").glob("d*.yaml")):
        env = load_env(env_file.stem)
        assert env.check, env.name
        assert env.commands("x86_64"), env.name
        if not env.commands("aarch64"):
            assert all(c.requires.arch == ["x86_64"] for c in users.get(env.name, [])), env.name


def test_d_workers_import_stdlib_only():
    names = sorted(p.stem for p in WORKERS.glob("d*_*.py"))
    assert "d1_f5_tts" in names and "d3_best_of_n" in names
    code = (f"import sys; sys.path.insert(0, {str(WORKERS)!r})\n" +
            "".join(f"import {n}\n" for n in names) +
            "heavy = {'torch', 'numpy', 'transformers', 'soundfile', 'librosa'} & set(sys.modules)\n"
            "assert not heavy, heavy\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=WORKERS)


# ------------------------------------------------------------------ specs

def test_specs_registered_and_consistent():
    from bench.arena.specs import SPECS

    for cap in CAPS:
        spec = SPECS[cap]
        for lang in SCOPE:
            prim = spec.primary_for(lang)
            assert prim in spec.higher_is_better, (cap, prim)
            for m in spec.secondary + list(spec.gates):
                assert m in spec.higher_is_better, (cap, m)
            ids = spec.judge_ids(lang)
            assert "audio_stats@1" in ids
    d1 = SPECS["D1"].judges_for("hi")
    ids = {j.id for j in d1}
    assert {"speaker_sim.wavlm-sv@1", "speaker_sim.eres2netv2@1", "mos.utmosv2@1"} <= ids
    assert any(i.startswith("asr_roundtrip.") for i in ids)
    assert any(i.startswith("asr_human_floor.") for i in ids)
    assert SPECS["D1"].gates["rt_cer"] == ("<=", 0.2)
    assert SPECS["D3"].primary_for("en") == "in_window_10"
    assert SPECS["D5"].primary_for("hi") == "rt_cer"
    assert "nv_events.ced@1" in {j.id for j in SPECS["D7"].judges_for("ja")}
    assert SPECS["D8"].primary_for("hi") == "asr_chrf"


def test_derived_metrics():
    from bench.arena.specs import _dvoice as dvs

    vals = {"rt_whisper_cer": (0.10, 50.0), "rt_qwen3_cer": (0.06, 50.0),
            "rt_whisper_wer": (0.2, 10.0), "hf_whisper_cer": (0.02, 50.0),
            "hf_qwen3_cer": (0.02, 50.0), "sim_wavlm-sv": (0.8, 1.0),
            "sim_eres2netv2": (0.6, 1.0), "simdiff_wavlm-sv": (0.3, 1.0),
            "simdiff_eres2netv2": (0.1, 1.0), "leak_wavlm-sv": (0.5, 1.0)}
    assert dvs.rt_cer(vals)[0] == pytest.approx(0.08)
    assert dvs.rt_cer_ratio(vals)[0] == pytest.approx((0.08 + 0.02) / (0.02 + 0.02))
    assert dvs.rt_cer_excess(vals)[0] == pytest.approx(0.06)
    assert dvs.sim_mean(vals)[0] == pytest.approx(0.7)
    assert dvs.sim_margin(vals)[0] == pytest.approx(0.5)
    assert dvs.sim_minus_leak(vals)[0] == pytest.approx(0.2)
    assert dvs.rt_cer_ratio({"rt_whisper_cer": (0.1, 1.0)}) is None  # no human floor


def test_function_judges(tmp_path):
    from bench.arena.specs import _dvoice as dvs

    wav = tone(tmp_path / "a.wav", 2.0)
    it = Item(id="x", lang="en", inputs={"text": "hello there friend", "target_s": 2.1},
              refs={"text": "hello there friend"})
    out = {"files": {"audio": str(wav)}}
    stats = {m: v for m, v, _ in dvs.audio_stats(it, out, None)}
    assert stats["dur_s"] == pytest.approx(2.0) and stats["degenerate"] == 0.0
    assert stats["speech_frac"] > 0.9
    fit = {m: v for m, v, _ in dvs.duration_fit(it, out, None)}
    assert fit["in_window_10"] == 1.0 and fit["in_window_5"] == 1.0
    assert fit["dur_err"] == pytest.approx(0.1 / 2.1)
    acc = {m: v for m, v, _ in dvs.accept_at_n(it, out, None)}
    assert acc["accept_rate"] == 1.0
    takes = {**out, "takes": [{"accepted": True}, {"accepted": False}, {"accepted": False},
                              {"accepted": True}]}
    acc = {m: (v, w) for m, v, w in dvs.accept_at_n(it, takes, None)}
    assert acc["accept_rate"] == (0.5, 4.0)
    # padded with silence: speech fraction drops (D3 gate), empty output is degenerate
    x, _ = sf.read(str(wav))
    sf.write(str(tmp_path / "pad.wav"), np.concatenate([x, np.zeros(16000 * 6)]), 16000)
    padded = {m: v for m, v, _ in dvs.audio_stats(it, {"files": {"audio": str(tmp_path /
                                                                              "pad.wav")}}, None)}
    assert padded["speech_frac"] < 0.4
    assert dvs.audio_stats(it, {"files": {}}, None) == [("degenerate", 1.0, 1.0)]


def test_chrf_and_text_judge():
    from bench.arena.specs._dvoice import chrf, text_chrf, untagged_text

    assert chrf("the cat sat", "the cat sat") == pytest.approx(100.0)
    assert chrf("xyz", "abc") == 0.0
    assert 0 < chrf("the cat sat down", "the cat sat") < 100
    it = Item(id="x", lang="en", inputs={"audio": "a.wav"}, refs={"text": "Hello, world."})
    assert text_chrf(it, {"text": "hello world"}, None)[0][1] == pytest.approx(100.0)
    tagged = Item(id="t", lang="ja", inputs={"text": "[laugh] そうなの? <|emotion:joy|>ええ"})
    assert untagged_text(tagged, {}) == "そうなの? ええ"


def test_model_judge_inputs_use_human_audio_and_source(tmp_path):
    from bench.arena.specs._dvoice import human_floor_asr, sim_to_human, source_leak

    human, ref, src, out_wav = (tone(tmp_path / f"{n}.wav", 1.0) for n in ("h", "r", "s", "o"))
    it = Item(id="x", lang="en", inputs={"text": "hi", "ref_audio": str(ref), "audio": str(src)},
              refs={"text": "hi", "human_audio": str(human)})
    out = {"files": {"audio": str(out_wav)}}
    j = human_floor_asr("en")[0]
    assert j.inputs(it, out, tmp_path) == {"audio": str(human)}
    rows = j.to_rows(it, {"text": "hi"}, out)
    assert rows and all(m.startswith("hf_") for m, _, _ in rows)
    s = sim_to_human("wavlm-sv")
    assert s.inputs(it, out, tmp_path) == {"audio": str(human), "ref_audio": str(ref)}
    assert s.to_rows(it, {"metrics": {"sim_wavlm-sv": 0.2}}, out) == [("simdiff_wavlm-sv", 0.2,
                                                                         1.0)]
    lk = source_leak("eres2netv2")
    assert lk.inputs(it, out, tmp_path) == {"audio": str(out_wav), "ref_audio": str(src)}
    assert lk.key != s.key
    no_human = Item(id="y", lang="en", inputs={"text": "hi"}, refs={"text": "hi"})
    assert j.inputs(no_human, out, tmp_path) is None


def test_d7_event_tagger_logic():
    from bench.arena.specs.d7 import event_tagger

    t = wmod("d7_event_tagger")
    assert t.group_of_tag("[Laughter]") == "laugh" and t.group_of_tag("sigh") == "sigh"
    assert t.group_of_tag("[whatever]") is None
    ref = [(0.0, 0.5, "laugh"), (2.0, 2.4, "sigh")]
    m = t.f1_rows(ref, [(0.1, 0.6, "laugh")])
    assert m["nv_precision"] == 1.0 and m["nv_recall"] == 0.5
    assert m["nv_f1"] == pytest.approx(2 / 3) and m["nv_f1_timed"] == pytest.approx(2 / 3)
    late = t.f1_rows(ref[:1], [(1.0, 1.4, "laugh")])
    assert late["nv_f1"] == 1.0 and late["nv_f1_timed"] == 0.0
    assert t.f1_rows([], [])["nv_f1"] == 1.0
    j = event_tagger()
    it = Item(id="x", lang="en", inputs={"text": "[laugh] ok [sigh]"}, refs={})
    assert j.inputs(it, {"files": {}}, Path(".")) is None


# ------------------------------------------------------------------ worker helpers

def test_d_voice_helpers(tmp_path):
    dv = wmod("d_voice")
    assert dv.msgpack_dumps({"a": 1}) == b"\x81\xa1a\x01"
    assert dv.msgpack_dumps([True, None, b"xy"]) == b"\x93\xc3\xc0\xc4\x02xy"
    assert dv.msgpack_dumps(-1) == b"\xff" and dv.msgpack_dumps(300)[:1] == b"\xd3"
    assert dv.cer("Hello, world!", "hello world") == 0.0
    assert dv.cer("abcd", "abxd") == pytest.approx(0.25)
    item = {"id": "i1", "meta": {"gender": "MALE"}}
    voices = {"en": {"female": "F", "male": "M"}, "*": "X"}
    assert dv.pick_voice(voices, "en", item) == "M"
    assert dv.pick_voice(voices, "ja", item) == "X"
    with pytest.raises(dv.Unsupported):
        dv.pick_voice({"en": "E"}, "hi", item)
    assert dv.item_seed({**item, "seed": 7}, {"seed": 3}) == 7
    assert dv.item_seed(item, {"seed": 3}) == 3
    assert dv.item_seed(item, {}) == dv.item_seed(item, {})
    with pytest.raises(dv.Unsupported):
        dv.require_lang("hi", ("en", "ja"))
    body, ctype = dv.multipart({"name": "x"}, {"file": ("a.wav", b"RIFF", "audio/wav")})
    assert ctype.startswith("multipart/form-data; boundary=") and b'name="file"' in body
    wav = dv.pcm16_to_wav(b"\x00\x00" * 2400, 24000, tmp_path / "p.wav")
    assert dv.duration_s(wav) == pytest.approx(0.1)
    assert dv.audio_payload(wav)["sample_rate"] == 24000


def test_api_request_builders():
    az = wmod("d5_azure")
    ssml = az._ssml("hi-IN-Kavya:MAI-Voice-2", "hi-IN", "a < b & c", style="happy", rate=1.2)
    assert "a &lt; b &amp; c" in ssml and 'rate="+20%"' in ssml and 'style="happy"' in ssml
    assert "name='hi-IN-Kavya:MAI-Voice-2'" in ssml
    kokoro = wmod("d5_kokoro")
    assert kokoro.LANG_CODES["hi"] == "h" and kokoro.DEFAULT_VOICES["ja"]["female"] == "jf_alpha"
    sm = wmod("d8_seamless")
    assert {"eng", "hin", "jpn"} <= sm.SPEECH_TGTS and sm.ISO3["hi"] == "hin"


class _FakeInner:
    """Inner worker whose take k lasts 1.0 + 0.5 k seconds."""

    def __init__(self, tmp: Path):
        self.tmp = tmp

    def run(self, state, item, out):
        k = item["seed"] - 100
        wav = tone(Path(str(out) + ".wav"), 1.0 + 0.5 * k, sr=24000)
        return {"files": {"audio": str(wav)}, "sample_rate": 24000}


class _FakeVerifier:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, lang):
        return self.texts.pop(0)


def test_best_of_n_picks_accepted_take_in_window(tmp_path):
    bon = wmod("d3_best_of_n")
    item = {"id": "i", "seed": 100, "inputs": {"text": "good morning", "target_s": 2.0}}
    state = {"mod": _FakeInner(tmp_path), "inner": None, "lang": "en",
             "verifier": _FakeVerifier(["good morning", "good morning", "good mourning",
                                        "good morning"]),
             "params": {"n": 4, "inner": {"params": {}}, "window": 0.10, "accept_cer": 0.05}}
    out = bon.run(state, item, tmp_path / "o")
    # takes last 1.0/1.5/2.0/2.5 s for a 2.0 s slot: only take 2 is in the window, and it is
    # mis-heard (CER 1/11 > 0.05), so nothing is accepted; the lowest CER + duration-error score
    # still picks take 2.
    assert out["n"] == 4 and out["chosen"] == 2
    assert [t["accepted"] for t in out["takes"]] == [False, False, False, False]
    assert out["accept_rate"] == 0.0
    state["verifier"] = _FakeVerifier(["good morning"] * 4)
    out = bon.run(state, item, tmp_path / "o2")
    assert out["chosen"] == 2 and out["accept_rate"] == 0.25
    assert sf.info(out["files"]["audio"]).duration == pytest.approx(2.0)


def test_splice_inserts_source_vocalisation(tmp_path):
    spl = wmod("d7_splice")
    laugh = tone(tmp_path / "laugh.wav", 0.5, sr=16000, amp=0.9)

    class Inner:
        def run(self, state, item, out):
            wav = tone(Path(str(out) + ".wav"), 1.0, sr=24000, amp=0.1)
            return {"files": {"audio": str(wav)}, "sample_rate": 24000}

    state = {"mod": Inner(), "inner": None, "lang": "en",
             "params": {"inner": {"params": {}}, "gap_ms": 0, "fade_ms": 10}}
    item = {"id": "i", "inputs": {"text": "well [laugh] okay [sigh]", "ref_audio": "r.wav",
                                  "events": ["laugh"], "event_audio_0": str(laugh)}}
    out = spl.run(state, item, tmp_path / "o")
    assert out["events_spliced"] == 1
    assert out["segments"] == ["speech", "event", "speech"]  # [sigh] has no clip: kept inline
    assert out["duration_s"] == pytest.approx(2.5, abs=0.01)


# ------------------------------------------------------------------ builders

def _fleurs_fixture(root: Path, ids=("1", "2", "3", "4")) -> None:
    codes = {"en": "en_us", "hi": "hi_in", "ja": "ja_jp"}
    for lang, code in codes.items():
        d = root / "_sources" / "fleurs" / "data" / code
        (d / "audio").mkdir(parents=True, exist_ok=True)
        rows, files = [], {}
        for sid in ids:
            for r in range(2):
                name = f"{lang}{sid}{r}.wav"
                buf = io.BytesIO()
                secs = 4.0 + int(sid) * 0.5
                sf.write(buf, np.zeros(int(16000 * secs), dtype="float32"), 16000,
                         format="WAV")
                files[name] = buf.getvalue()
                rows.append([sid, name, f"{lang} sentence {sid}", f"{lang} sentence {sid}", "",
                             str(int(16000 * secs)), "FEMALE" if r else "MALE"])
        with (d / "test.tsv").open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)
        with tarfile.open(d / "audio" / "test.tar.gz", "w:gz") as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(f"test/{name}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


@pytest.fixture()
def evalroot(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    return tmp_path / "eval"


def test_fleurs_voice_packs(evalroot):
    from bench.arena.builders import BUILDERS

    _fleurs_fixture(evalroot)
    b = BUILDERS["fleurs_voice"].fn
    b("en", n=3, capability="D1", n_control=2)
    items = load_pack("D1", "en")
    assert len(items) == 3 and all(it.meta["pair"] == "ja-en" for it in items)
    it = items[0]
    sid = it.meta["flores_id"]
    assert it.inputs["text"] == f"en sentence {sid}"          # target = same FLoRes id …
    assert Path(it.inputs["ref_audio"]).name.startswith(f"ja{sid}")  # … read in Japanese
    assert Path(it.refs["human_audio"]).exists() and it.group == sid
    ctl = load_pack("D1", "en", split="control")
    assert len(ctl) == 2 and all(c.meta["src_lang"] == "en" for c in ctl)
    assert all(c.meta["flores_id"] not in Path(c.inputs["ref_audio"]).name[2:3] for c in ctl)

    b("hi", n=2, capability="D3", n_control=0, compress=[0.8, 1.0])
    d3 = load_pack("D3", "hi")
    assert {it.meta["pair"] for it in d3} == {"ja-hi", "en-hi"} and len(d3) == 8
    one = next(it for it in d3 if it.meta["compress"] == 0.8)
    assert one.inputs["target_s"] == pytest.approx(one.meta["duration_s"] * 0.8, abs=1e-3)

    b("en", n=2, capability="D8")
    d8 = load_pack("D8", "en")
    assert all(Path(it.inputs["audio"]).name.startswith(f"ja{it.meta['flores_id']}") and
               it.refs["text"].startswith("en sentence") for it in d8)
    b("ja", n=2, capability="D5")
    d5 = load_pack("D5", "ja")
    assert all(set(it.inputs) == {"text"} and it.meta["gender"] for it in d5)
    b("hi", n=2, capability="D6", n_control=0)
    d6 = load_pack("D6", "hi")
    assert all({"audio", "ref_audio", "text"} <= set(it.inputs) for it in d6)
    assert (evalroot / "D6" / "hi" / "LICENSE.md").read_text().startswith("# FLEURS")


def test_seedtts_minimax_parsers(tmp_path):
    from bench.arena.builders.d_minimax_ml import parse_prompts, speakers
    from bench.arena.builders.d_seedtts import parse_meta

    rows = parse_meta("u1|prompt one|en/prompt-wavs/a.wav|target one\nbad line\n")
    assert rows == [{"utt": "u1", "prompt_text": "prompt one", "prompt_wav": "en/prompt-wavs/a.wav",
                     "text": "target one"}]
    base = tmp_path / "speaker" / "hindi"
    (base / "hindi_female").mkdir(parents=True)
    tone(base / "hindi_female" / "common_voice_hi_1.wav", 1.0)
    (base / "prompt_text.txt").write_text("hindi_female|नमस्ते दुनिया\n")
    assert parse_prompts(base / "prompt_text.txt") == {"hindi_female": "नमस्ते दुनिया"}
    spk = speakers(tmp_path, "hindi")
    assert spk["hindi_female"][1] == "नमस्ते दुनिया"


def test_cv3_and_indicvoices_builders(evalroot, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bench.arena.builders import BUILDERS, d_cv3eval, d_indicvoices_r

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
    audio = buf.getvalue()
    cv3 = evalroot / "_sources" / "cv3_eval" / "data"
    cv3.mkdir(parents=True)
    for split in ("cross_lingual_zeroshot_to_ja", "zero_shot_ja"):
        pq.write_table(pa.table({"id": ["a", "b"], "prompt_text": ["p", "q"],
                                 "prompt_audio": [{"bytes": audio, "path": "x.wav"}] * 2,
                                 "target_text": ["こんにちは", "さようなら"]}),
                       cv3 / f"{split}-00000-of-00001.parquet")
    BUILDERS["cv3_eval"].fn("ja", capability="D1")
    assert len(load_pack("D1", "ja")) == 2 and len(load_pack("D1", "ja", split="control")) == 2
    with pytest.raises(ValueError):
        d_cv3eval.cv3_eval("hi", capability="D1")

    ivr = evalroot / "_sources" / "indicvoices_r" / "Hindi"
    ivr.mkdir(parents=True)
    tbl = pa.table({"text": ["एक", "दो", "तीन"], "normalized": ["एक", "दो", "तीन"],
                    "speaker_id": ["s1", "s1", "s2"], "gender": ["Female", "Female", "Male"],
                    "duration": [4.0, 5.0, 4.0], "audio": [{"bytes": audio, "path": "a"}] * 3})
    pq.write_table(tbl, evalroot / "_sources" / "indicvoices_r" / d_indicvoices_r.SHARDS[0])
    pq.write_table(tbl.slice(2, 1), evalroot / "_sources" / "indicvoices_r" /
                   d_indicvoices_r.SHARDS[1])
    BUILDERS["indicvoices_r"].fn("hi", capability="D4")
    items = load_pack("D4", "hi")
    # s1 has two clips in shard 0, s2 one in each shard: every item pairs two distinct clips
    assert len(items) == 4 and {it.group for it in items} == {"s1", "s2"}
    assert all(it.inputs["ref_audio"] != it.refs["human_audio"] for it in items)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_scene_voice_builder(evalroot):
    from bench.arena.builders import BUILDERS

    scene = evalroot / "SCENE" / "ja"
    (scene / "clips").mkdir(parents=True)
    tone(scene / "clips" / "s1.ja.wav", 10.0)
    tone(scene / "clips" / "s1.en.wav", 10.0)
    (scene / "manifest.jsonl").write_text(json.dumps(
        {"id": "s1", "group": "ep1", "inputs": {"audio": "clips/s1.ja.wav"},
         "refs": {"dub_en_audio": "clips/s1.en.wav"}}) + "\n")
    (scene / "lines").mkdir()
    lines = [{"start": 1.0, "end": 3.0, "speaker": "loid", "ja": "おはよう", "en": "Morning.",
              "emotion": "happy", "tagged": {"en": "[laugh] Morning."},
              "nv": [{"tag": "laugh", "start": 1.0, "end": 1.4}]},
             {"start": 4.0, "end": 4.2, "speaker": "anya", "ja": "え", "en": "Eh."},  # too short
             {"start": 5.0, "end": 8.0, "spk": "anya", "text": "わくわく",
              "dub_en": {"start": 5.1, "end": 7.9, "text": "So exciting!"}}]
    (scene / "lines" / "s1.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    BUILDERS["scene_voice"].fn("en", capability="D3")
    d3 = load_pack("D3", "en")
    assert [it.inputs["text"] for it in d3] == ["Morning.", "So exciting!"]
    assert d3[1].inputs["target_s"] == pytest.approx(2.8) and d3[1].group == "anya"
    assert Path(d3[0].refs["human_audio"]).exists()
    BUILDERS["scene_voice"].fn("en", capability="D7")
    d7 = load_pack("D7", "en")
    assert len(d7) == 1 and d7[0].inputs["events"] == ["laugh"]
    assert Path(d7[0].inputs["event_audio_0"]).exists()  # resolved to an absolute path
    BUILDERS["scene_voice"].fn("en", capability="D6")
    d6 = load_pack("D6", "en")
    assert all(Path(it.inputs["audio"]).parent.name == "dub" for it in d6)
