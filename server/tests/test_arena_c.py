"""Model arena, phase C (text): registry, specs, judges, workers' pure logic and builders.

Model-free: no weights, no GPU, no network (downloads are monkeypatched or pre-placed)."""
from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from bench.arena import envs, judges, paths, runner
from bench.arena.packs import Item, load_pack, write_pack
from bench.arena.registry import load_candidates
from bench.arena.report import leaderboard
from bench.arena.specs import SPECS
from bench.arena.specs.c2 import worker_module

CAPS = ["C1", "C2", "C3", "C4", "C5", "C6"]
MY_WORKERS = sorted(p.stem for p in paths.WORKERS_DIR.glob("c*_*.py")) + ["judge_mt_qe"]


@pytest.fixture()
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="module")
def cc():
    return worker_module("c_common")


@pytest.fixture(scope="module")
def rate():
    return worker_module("c_speech_rate")


def item(lang="ja-en", inputs=None, refs=None, meta=None, id="x", group="g"):
    return Item(id=id, lang=lang, inputs=inputs or {}, refs=refs or {}, group=group,
                meta=meta or {})


# ------------------------------------------------------------------ registry and specs

def test_c_candidates_are_complete_and_consistent():
    for cap in CAPS:
        cands = load_candidates(cap)
        assert cands, cap
        assert sum(c.baseline for c in cands) <= 1, cap
        for c in cands:
            assert (paths.WORKERS_DIR / f"{c.worker}.py").exists(), (cap, c.id)
            env = envs.load_env(c.env)
            assert c.sources and isinstance(c.verified, dict) and c.license, (cap, c.id)
            if not c.enabled:
                assert c.notes, (cap, c.id)          # disabled rows say why
                continue
            assert c.url, (cap, c.id)                # where to get it
            if c.kind == "api":
                assert c.requires.api_keys, (cap, c.id)
            if c.env != "server":                   # arch recipes (or an explicit x86-only)
                assert "x86_64" in env.setup, (cap, c.id)
                assert "aarch64" in env.setup or c.requires.arch == ["x86_64"], (cap, c.id)
            langs = c.languages if isinstance(c.languages, list) else []
            for lang in langs:
                if cap in ("C2", "C3", "C6"):
                    assert "-" in lang, (cap, c.id, lang)
                if cap == "C5":
                    assert lang in SPECS["C5"].primary, (cap, c.id, lang)


def test_gemba_judge_family_differs_from_every_c2_candidate():
    from bench.arena.specs.c2 import GEMBA_FAMILY, GEMBA_JUDGE

    for c in load_candidates("C2"):
        blob = json.dumps(c.params).lower() + c.id
        assert GEMBA_FAMILY not in blob and GEMBA_JUDGE["model"] not in blob, c.id


def test_c_specs_declare_every_metric():
    for cap in CAPS:
        spec = SPECS[cap]
        for lang in ("*", *spec.primary):
            assert spec.primary_for(lang) in spec.higher_is_better, (cap, lang)
        from bench.arena.builders import BUILDERS

        for pack in spec.packs:
            assert pack in BUILDERS and cap in BUILDERS[pack].capabilities, (cap, pack)
        for m in [*spec.secondary, *spec.gates, *spec.derived, *spec.corpus]:
            assert m in spec.higher_is_better, (cap, m)
        for mj in spec.judges_for("ja-en"):
            assert (paths.WORKERS_DIR / f"{mj.worker}.py").exists(), (cap, mj.id)
            envs.load_env(mj.env)


def test_workers_import_only_stdlib_at_module_level():
    allowed = set(sys.stdlib_module_names) | {"_sdk", "c_common", "c_speech_rate", "__future__"}
    for name in MY_WORKERS:
        tree = ast.parse((paths.WORKERS_DIR / f"{name}.py").read_text())
        for node in tree.body:
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                assert m.split(".")[0] in allowed, (name, m)


def test_workers_parse_their_job_without_heavy_deps():
    """Every C worker module imports cleanly in a bare interpreter (heavy deps only in load)."""
    code = (f"import sys; sys.path.insert(0, {str(paths.WORKERS_DIR)!r}); sys.argv=['x'];\n"
            f"import importlib\nfor n in {MY_WORKERS!r}: importlib.import_module(n)")
    subprocess.run([sys.executable, "-I", "-c", code], check=True, cwd=paths.WORKERS_DIR)


# ------------------------------------------------------------------ speaking-rate model

def test_spoken_units_per_script(rate):
    assert rate.spoken_units("The quick brown fox jumps over the lazy dog.", "en") == 11
    assert rate.spoken_units("make time", "en") == 2
    assert rate.spoken_units("こんにちは", "ja") == 5
    assert rate.spoken_units("きょう", "ja") == 2              # small ょ is not a mora
    assert rate.spoken_units("学校", "ja") == pytest.approx(2 * rate.KANJI_MORAE)
    assert rate.spoken_units("कमल", "hi") == 2                 # final schwa deleted
    assert rate.spoken_units("मैं घर जा रहा हूँ", "hi") == 6
    assert rate.base_lang("hi.g2p") == rate.base_lang("hi_IN") == "hi"
    # characters are not comparable across scripts: same meaning, very different char counts
    en, ja = "I'll be back by tomorrow.", "明日までに戻る。"
    assert len(en) > 2 * len(ja)
    assert 0.5 < rate.predicted_seconds(ja, "ja") / rate.predicted_seconds(en, "en") < 2.0


def test_calibrate_recovers_rate(rate):
    pairs = [("hello there my friend", rate.spoken_units("hello there my friend", "en") / 5.0)]
    assert rate.calibrate(pairs * 3, "en") == pytest.approx(5.0)


# ------------------------------------------------------------------ C2 judges

def test_c2_duration_gate_uses_target_language_units():
    from bench.arena.specs import c2

    it = item("ja-en", {"text": "明日までに戻る。", "slot_s": 1.5},
              refs={"text": "I'll be back by tomorrow."}, meta={"src_lang": "ja",
                                                                "tgt_lang": "en"})
    fits = c2.duration(it, {"text": "Back by tomorrow."}, None)
    long = c2.duration(it, {"text": "I promise you that I will definitely come back home "
                                    "again before tomorrow evening."}, None)

    def get(rows, m):
        return next(v for k, v, _ in rows if k == m)
    assert get(long, "dur_compliance") == 0.0 and get(long, "dur_ratio") > 1.1
    assert get(fits, "dur_ratio") < get(long, "dur_ratio")
    assert any(k == "ref_dur_compliance" for k, _, _ in fits)
    # no measured slot: the predicted source duration is the slot
    it2 = item("en-hi", {"text": "Where are you going?"})
    assert c2.slot_seconds(it2) == pytest.approx(
        c2.rate.predicted_seconds("Where are you going?", "en"))


def test_chrf_and_text_checks():
    from bench.arena.specs import c2

    assert c2.chrf("the cat sat", "the cat sat") == pytest.approx(100.0)
    assert 0 < c2.chrf("the cat sat down", "the cat sat") < 100
    assert c2.chrf("", "abc") == 0.0
    it = item("en-hi", {"text": "Hello"}, refs={"text": "नमस्ते"})
    rows = {k: v for k, v, _ in c2.text_checks(it, {"text": "Hello"}, None)}
    assert rows["untranslated"] == 1.0 and rows["empty_output"] == 0.0


def test_c2_derived_metricx_prefers_reference_based():
    from bench.arena.specs import c2

    vals = {"qe_metricx-24-hybrid-xl": (5.0, 1.0), "hyb_metricx-24-hybrid-xl": (3.0, 1.0),
            "dur_compliance": (1.0, 1.0)}
    assert c2._metricx(vals) == (3.0, 1.0)
    assert c2._metricx_compliant(vals) == (3.0, 1.0)
    assert c2._metricx_compliant({**vals, "dur_compliance": (0.0, 1.0)}) is None
    assert c2._metricx({"qe_metricx-24-hybrid-xl": (5.0, 1.0)}) == (5.0, 1.0)
    judges_ = c2.model_judges("ja-en")
    it = item("ja-en", {"text": "src"}, refs={"text": "ref"}, meta={"tgt_lang": "en"})
    inp = judges_[0].inputs(it, {"text": "hyp"}, Path("/tmp/x"))
    assert inp["reference"] == "ref" and inp["tgt_lang"] == "en"
    g = judges_[-1].inputs(it, {"text": "hyp"}, Path("/tmp/x"))
    assert g == {"source": "src", "hypothesis": "hyp", "src_lang": "ja", "tgt_lang": "en"}


# ------------------------------------------------------------------ c_common (LLM plumbing)

def test_opendub_prompt_matches_the_app(cc):
    from app.providers.translation import _llm

    for s, t in (("Japanese", "English"), ("", "Hindi"), ("auto", "Japanese")):
        assert cc.opendub_system_prompt(s, t) == _llm.build_system_prompt(s, t)
    for d in (0.0, 0.5, 2.3, 10.0):
        assert cc.opendub_max_chars(d) == _llm.max_chars_for(d)


def test_parse_json_and_cleaning(cc):
    assert cc.parse_json('<think>hmm</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert cc.parse_json('Sure! [{"n": 1, "translation": "x"}] done') == [{"n": 1,
                                                                            "translation": "x"}]
    with pytest.raises(ValueError):
        cc.parse_json("no json here")
    assert cc.clean_line('Translation: "Hello."') == "Hello."
    assert cc.direction("ja-en") == ("ja", "en")
    assert cc.direction("hi.g2p") == ("hi", "hi")
    assert cc.direction("", {"inputs": {"src_lang": "en", "tgt_lang": "ja"}}) == ("en", "ja")


@pytest.mark.parametrize("style", ["opendub", "dub", "plain", "hunyuan_mt", "seed_x", "tower",
                                   "translategemma"])
def test_translate_prompt_styles(cc, style):
    it = {"inputs": {"text": "明日までに戻る。", "slot_s": 1.4, "context_before": ["え？"]}}
    msgs, as_json = cc.translate_messages(style, it, "ja", "en")
    blob = json.dumps(msgs, ensure_ascii=False)
    assert "明日までに戻る" in blob and msgs[-1]["role"] == "user"
    user = msgs[-1]["content"] if isinstance(msgs[-1]["content"], str) else ""
    assert as_json == (style in ("opendub", "dub"))
    if style == "dub":
        assert "syllables" in blob and "characters" in blob    # counts units, not chars
    if style == "opendub":
        assert '"max_chars": 21' in user


def fake_complete(replies):
    calls = []

    def complete(messages, *, temperature=None, json_mode=False, max_tokens=2048):
        calls.append({"messages": messages, "temperature": temperature, "json": json_mode})
        return replies[min(len(calls) - 1, len(replies) - 1)], {"input_tokens": 10,
                                                                "output_tokens": 5}
    complete.calls = calls
    return complete


def test_run_task_translate_with_retry_and_cost(cc):
    comp = fake_complete(["not json", '{"translation": "Back by tomorrow."}'])
    it = {"inputs": {"text": "明日までに戻る。"}, "meta": {"src_lang": "ja", "tgt_lang": "en"}}
    out = cc.finish(cc.run_task(comp, it, "ja-en", {"prompt": "dub"}), (5.0, 25.0))
    assert out["text"] == "Back by tomorrow." and len(comp.calls) == 2
    assert out["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert out["_cost_usd"] == pytest.approx((20 * 5 + 10 * 25) / 1e6)
    nbest = cc.run_task(fake_complete(['{"translation": "a"}']), it, "ja-en",
                        {"prompt": "dub", "n_candidates": 3})
    assert nbest["candidates"] == ["a", "a", "a"]


def test_run_task_segment_gemba_c5_condense(cc):
    words = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "word": w}
             for i, w in enumerate(["I", "see", ".", "we", "go", "now"])]
    seg = cc.run_task(fake_complete(['{"ends": [2, 5]}']), {"inputs": {"words": words}},
                      "en", {"task": "segment"})
    assert [(ln["start_word"], ln["end_word"]) for ln in seg["lines"]] == [(0, 3), (3, 6)]
    g = cc.run_task(fake_complete(['{"errors": [{"span": "x", "severity": "major"}], "score": 62}']),
                    {"inputs": {"source": "s", "hypothesis": "h", "reference": "r",
                                "src_lang": "en", "tgt_lang": "hi"}}, "", {"task": "gemba"})
    assert g["metrics"]["gemba_esa"] == 62.0 and g["metrics"]["gemba_major"] == 1.0
    kana = cc.run_task(fake_complete(['{"kana": "キョウ"}']), {"inputs": {"text": "今日"}},
                       "ja.kana", {"task": "c5"})
    assert kana["text"] == "キョウ"
    cues = [{"start": 0.0, "end": 1.0, "text": "This subtitle is much too long to read in a "
                                                 "single second", "source": "src"},
            {"start": 1.0, "end": 4.0, "text": "Short."}]
    comp = fake_complete(['{"text": "Too long to read."}', '{"text": "Too long."}'])
    out = cc.run_task(comp, {"inputs": {"cues": cues, "limits": {"cps": 12, "cpl": 42,
                                                                 "lines": 2}}},
                      "ja-en", {"task": "condense"})
    assert out["cues"][1]["text"] == "Short."                   # compliant cues untouched
    assert out["cues"][0]["text"] == "Too long." and out["cues"][0]["condensed_pass"] == 2
    assert [c["temperature"] for c in comp.calls] == [0.0, 0.3]


class _FakeOllama(BaseHTTPRequestHandler):
    posts: ClassVar[list] = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5:14b", "model": "qwen2.5:14b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).posts.append((self.path, body))
        if self.path == "/api/chat":
            self._send({"message": {"content": '{"translation": "hi"}'},
                        "prompt_eval_count": 7, "eval_count": 3})
        else:
            self._send({})


def test_ollama_backend_unloads_model_at_exit(cc):
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _FakeOllama.posts.clear()
        b = cc.OpenAICompatBackend({"backend": "ollama", "model": "qwen2.5:14b",
                                    "base_url": f"http://127.0.0.1:{srv.server_port}"})
        text, usage = b.complete([{"role": "user", "content": "x"}], json_mode=True)
        assert text == '{"translation": "hi"}' and usage["input_tokens"] == 7
        chat = _FakeOllama.posts[0][1]
        assert chat["format"] == "json" and chat["stream"] is False
        cc._run_cleanups()                               # what atexit does at worker exit
        assert _FakeOllama.posts[-1] == ("/api/generate", {"model": "qwen2.5:14b",
                                                           "keep_alive": 0})
        with pytest.raises(RuntimeError, match="ollama pull"):
            cc.OpenAICompatBackend({"backend": "ollama", "model": "missing:1b",
                                    "base_url": f"http://127.0.0.1:{srv.server_port}"})
    finally:
        srv.shutdown()


def test_server_commands_and_stop(cc, monkeypatch):
    b = object.__new__(cc.OpenAICompatBackend)
    b.p, b.model, b.name, b.port = {"server_args": ["--max-model-len", "8192"]}, "Qwen/X", "x", 8011
    for kind, health in (("vllm", "/v1/models"), ("vllm_docker", "/v1/models"),
                         ("llamacpp", "/health"), ("llamacpp_docker", "/health")):
        b.kind = kind
        cmd, h = b._server_cmd()
        assert h == health and "--max-model-len" in cmd
        if kind.endswith("docker"):
            assert cmd[:3] == ["docker", "run", "--rm"] and "--gpus" in cmd
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            start_new_session=True)
    cc._stop_process(proc, grace=5)
    assert proc.poll() is not None


# ------------------------------------------------------------------ C1

def _words(spec):
    """spec: list of (word, duration, pause_after)."""
    t, out = 0.0, []
    for w, d, p in spec:
        out.append({"start": round(t, 3), "end": round(t + d, 3), "word": w})
        t += d + p
    return out


def test_c1_fittable_and_coverage():
    from bench.arena.specs import c1

    words = _words([("Hello", .4, .05), ("there.", .4, .6), ("How", .3, .05), ("are", .2, .05),
                    ("you?", .3, .0)])
    it = item("en", {"words": words}, refs={"ends": [1, 4]})
    good = {"lines": [{"start_word": 0, "end_word": 2, "start": 0, "end": .85},
                      {"start_word": 2, "end_word": 5, "start": 1.45, "end": 2.35}]}
    bad = {"lines": [{"start_word": 0, "end_word": 4, "start": 0, "end": 2.0},
                     {"start_word": 4, "end_word": 5, "start": 2.05, "end": 2.35}]}
    g = {k: v for k, v, _ in c1.segmentation(it, good, None)}
    b = {k: v for k, v, _ in c1.segmentation(it, bad, None)}
    assert g["fittable"] == 1.0 and g["coverage"] == 1.0 and g["boundary_f1"] == 1.0
    assert b["fittable"] < 1.0 and b["boundary_f1"] < 1.0
    dropped = {"lines": good["lines"][:1]}
    assert {k: v for k, v, _ in c1.segmentation(it, dropped, None)}["coverage"] == 0.4


def test_c1_sat_split_search_respects_pauses_and_length():
    sat = worker_module("c1_sat")
    words = _words([("a", .3, .05), ("b", .3, .8), ("c", .3, .05), ("d", .3, 1.5),
                    ("e", .3, .05)])
    probs = [0.0, 0.2, 0.0, 0.1, 1.0]
    ends = sat.split_search(words, probs, {"line_cost": 0.4})
    assert 1 in ends and 3 in ends and ends[-1] == 4              # pause and hard pause
    long = _words([("w", .9, .02)] * 20)
    ends = sat.split_search(long, [0.0] * 20, {"max_line_seconds": 5.0})
    spans = [(a + 1 if i else 0, b) for i, (a, b) in enumerate(zip([-1, *ends], ends))]
    assert all(long[b]["end"] - long[a]["start"] <= 5.0 for a, b in spans)
    text, char_ends = sat.word_char_ends([{"word": "ab"}, {"word": "c"}], "en")
    assert text == "ab c" and char_ends == [2, 4]


def test_c1_pause_worker_through_the_runner(arena):
    words = _words([("Hi", .3, .5), ("there", .4, .05), ("friend.", .5, .2), ("Go", .3, .0)])
    write_pack("C1", "en", [{"id": "w1", "group": "g", "inputs": {"words": words},
                             "meta": {"src_lang": "en"}}], "# t\n")
    items = load_pack("C1", "en")
    cand = [c for c in load_candidates("C1") if c.id == "pause-splitter"]
    made = runner.generate("C1", "en", cand, items, check_gpu=False, log=lambda _: None)
    assert made[cand[0].key] == 1
    judges.score("C1", "en", cand, items, run_models=False, check_gpu=False, log=lambda _: None)
    _md, rows = leaderboard("C1", "en", cand, items)
    assert rows and rows[0]["candidate"] == "pause-splitter"


# ------------------------------------------------------------------ C3

def test_c3_rerank_prefers_mouth_shape_match():
    vr = worker_module("c3_viseme_rerank")
    assert vr.mouth_sequence("ま", "ja") == ["B", "A"]
    assert vr.dtw(["A", "B"], ["A", "B"]) == 0.0
    text, scored = vr.rerank("Wait for me here", "ままま", ["mama may", "zzz zip it"], "ja",
                             "en", None, {"beta": 0.0})
    assert text == "mama may" and len(scored) == 3
    text, _ = vr.rerank("Wait for me", "まま", ["mama mama mama mama mama mama"], "ja", "en",
                        1.0, {})
    assert text == "Wait for me"                               # out-of-slot candidate dropped


# ------------------------------------------------------------------ C4

def _brute_acc_eq(p, t_src, h):
    best = 0.0
    pairs = [(i, j) for i in range(len(p)) for j in range(i + 1, len(p)) if t_src[i] == t_src[j]]
    diffs = sorted({abs(p[i] - p[j]) for i, j in pairs} | {-1.0})
    for eps in diffs:
        ok = 0
        for i, j in pairs:
            dh, dp = np.sign(h[i] - h[j]), (0 if abs(p[i] - p[j]) <= eps else np.sign(p[i] - p[j]))
            ok += dh == dp
        best = max(best, ok / len(pairs))
    return best


def test_c4_acc_eq_matches_brute_force_and_is_within_source():
    from bench.arena.specs import c4

    rng = np.random.default_rng(0)
    src = rng.integers(0, 6, 40)
    human = rng.integers(0, 5, 40).astype(float)
    pred = human + rng.normal(0, 1.2, 40)
    pred[:5] = np.round(pred[:5])                           # some metric ties
    targets = [c4.encode_target(item(refs={"human": h, "src_id": int(s)}))
               for s, h in zip(src, human, strict=True)]
    assert c4.acc_eq(list(pred), targets) == pytest.approx(_brute_acc_eq(pred, src, human))
    assert c4.acc_eq(list(human), targets) == pytest.approx(1.0)
    # one translation per source: falls back to all pairs
    single = [c4.encode_target(item(refs={"human": h, "src_id": k}))
              for k, h in enumerate(human[:10])]
    assert c4.acc_eq(list(human[:10]), single) == pytest.approx(1.0)


def test_c4_rules_worker_ranked_by_human_agreement(arena):
    """The rules baseline runs through the real runner and C4 ranks it with the corpus metric."""
    rows = []
    for i in range(12):
        good = i % 3 != 0
        rows.append({"id": f"s{i}", "group": f"d{i // 3}",
                     "inputs": {"source": "Where are you going tonight?",
                                "hypothesis": "आज रात तुम कहाँ जा रहे हो?" if good
                                else "Where are you going tonight?",
                                "src_lang": "en", "tgt_lang": "hi"},
                     "refs": {"human": 80.0 if good else 5.0, "src_id": i // 2},
                     "meta": {"src_lang": "en", "tgt_lang": "hi"}})
    write_pack("C4", "en-hi", rows, "# t\n")
    items = load_pack("C4", "en-hi")
    cand = [c for c in load_candidates("C4") if c.id == "rules"]
    runner.generate("C4", "en-hi", cand, items, check_gpu=False, log=lambda _: None)
    judges.score("C4", "en-hi", cand, items, run_models=False, check_gpu=False,
                 log=lambda _: None)
    _md, ranked = leaderboard("C4", "en-hi", cand, items)
    assert ranked[0]["mean"] == pytest.approx(1.0)


def test_judge_mt_qe_names_metrics_by_mode(monkeypatch):
    qe = worker_module("judge_mt_qe")
    monkeypatch.setattr(qe, "_metricx_scores", lambda exs, *a: [float(len(e["mt"])) for e in exs])
    items = [{"id": "a", "inputs": {"source": "s", "hypothesis": "xx"}},
             {"id": "b", "inputs": {"source": "s", "hypothesis": "yyy", "reference": "r"}}]
    out = qe._score(items, {"model": "metricx-24-hybrid-xl"})
    assert out == {"a": ("qe_metricx-24-hybrid-xl", 2.0), "b": ("hyb_metricx-24-hybrid-xl", 3.0)}
    out = qe._score(items, {"model": "metricx-24-hybrid-xl", "use_reference": False})
    assert out["b"][0] == "qe_metricx-24-hybrid-xl"


def test_c4_rules_scores(rate):
    rules = worker_module("c4_rules")
    good, _ = rules.rule_score("Where are you going?", "तुम कहाँ जा रहे हो?", "en", "hi")
    copy, flags = rules.rule_score("Where are you going?", "Where are you going?", "en", "hi")
    assert good > 90 and copy < 10 and flags == ["copy_of_source"]


# ------------------------------------------------------------------ C5

def test_c5_scoring():
    from bench.arena.specs import c5

    assert c5.ipa_phones("ˈkæt") == ["k", "æ", "t"]
    assert c5.ipa_phones("t͡ʃiː z") == ["tʃ", "iː", "z"]
    assert c5.ipa_phones("kʰa") == ["kʰ", "a"]
    tn = item("en.tn", {"text": "25"}, refs={"text": "twenty-five"},
              meta={"semiotic_class": "cardinal"})
    rows = {k: v for k, v, _ in c5.c5_scores(tn, {"text": "Twenty five"}, None)}
    assert rows["tn_exact"] == 1.0 and rows["tn_exact.cardinal"] == 1.0
    g2p = item("hi.g2p", {"text": "कमल"}, refs={"text": "k ə m ə l"})
    rows = {k: v for k, v, _ in c5.c5_scores(g2p, {"text": "kəməl"}, None)}
    assert rows["per"] == 0.0 and rows["g2p_exact"] == 1.0
    kana = item("ja.kana", {"text": "今日"}, refs={"text": "きょう"})
    rows = {k: v for k, v, _ in c5.c5_scores(kana, {"text": "キョウ。"}, None)}
    assert rows["kana_cer"] == 0.0
    misaki = worker_module("c5_misaki")
    assert misaki.to_ipa("hˈOm") == "h oʊ m"


# ------------------------------------------------------------------ C6

def test_c6_cue_checks_and_wrap():
    from bench.arena.specs import c6

    ok = [{"start": 0, "end": 3, "text": "Short line\nand another"}]
    bad = [{"start": 0, "end": 1, "text": "x" * 50 + "\nb\nc"}]
    assert c6.cue_checks(ok, "en") == {"cps_ok": 1.0, "cpl_ok": 1.0, "lines_ok": 1.0}
    assert c6.cue_checks(bad, "en") == {"cps_ok": 0.0, "cpl_ok": 0.0, "lines_ok": 0.0}
    assert c6.char_len("ABCあ", "ja") == 2.5                   # half-width counts 0.5
    it = item("ja-en", {"cues": [{"start": 0, "end": 3, "text": "Short line and another"}]},
              refs={"cues": ok})
    rows = {k: v for k, v, _ in c6.subtitle_checks(it, {"cues": ok}, None)}
    assert rows["chrf"] > 90 and rows["cue_count_kept"] == 1.0
    wrap = worker_module("c6_rules").wrap
    wrapped = wrap("one two three four five six seven eight nine ten eleven", 42, False)
    assert all(len(ln) <= 42 for ln in wrapped.split("\n"))
    suber = worker_module("c6_suber_judge")
    assert "00:00:01,500 --> 00:00:02,000" in suber.to_srt([{"start": 1.5, "end": 2,
                                                             "text": "hi"}])


# ------------------------------------------------------------------ builders

def test_mt_builders_with_fake_datasets(arena, monkeypatch):
    from bench.arena.builders import c_mt_public as mt

    fake = {
        ("google/wmt24pp", "en-ja_JP"): [
            {"document_id": "d1", "segment_id": k, "source": f"Hello {k}", "target": f"こんにちは{k}",
             "is_bad_source": k == 2, "domain": "speech"} for k in range(4)],
        ("ryo0634/bsd_ja_en", None): [
            {"id": "sc1", "no": k, "ja_sentence": f"文{k}", "en_sentence": f"s{k}",
             "ja_speaker": "田中", "en_speaker": "Tanaka", "original_language": "ja",
             "tag": "t", "title": "T"} for k in range(3)] + [
            {"id": "sc2", "no": 0, "ja_sentence": "訳", "en_sentence": "orig",
             "ja_speaker": "A", "en_speaker": "A", "original_language": "en"}],
        ("openlanguagedata/flores_plus", "jpn_Jpan"): [{"id": 1, "text": "猫", "url": "u"}],
        ("openlanguagedata/flores_plus", "hin_Deva"): [{"id": 1, "text": "बिल्ली", "url": "u"}],
    }
    monkeypatch.setattr(mt, "_rows", lambda repo, cfg, split, gated=False: fake[(repo, cfg)])
    mt.wmt24pp("en-ja", n=None)
    items = load_pack("C2", "en-ja")
    assert len(items) == 3 and items[1].inputs["context_before"] == ["Hello 0"]
    assert items[0].meta["tgt_lang"] == "ja" and items[0].refs["text"] == "こんにちは0"
    mt.bsd("ja-en", n=None)
    items = load_pack("C2", "ja-en")
    assert len(items) == 3 and items[0].inputs["speaker"] == "田中"   # en-original scenario dropped
    mt.flores_plus("ja-hi", n=None)
    assert load_pack("C2", "ja-hi")[0].refs["text"] == "बिल्ली"


def _tgz(path: Path, files: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_mtme_builder(arena):
    from bench.arena.builders import c_qe_human

    base = "mt-metrics-eval-v2/wmt24/"
    _tgz(paths.eval_dir() / "_sources" / "mtme" / "mt-metrics-eval-v2.tgz", {
        base + "sources/en-ja.txt": "one\ntwo\n",
        base + "documents/en-ja.docs": "news\tdocA\nnews\tdocB\n",
        base + "references/en-ja.refA.txt": "一\n二\n",
        base + "system-outputs/en-ja/sysX.txt": "いち\nに\n",
        base + "system-outputs/en-ja/sysY.txt": "壱\n弐\n",
        base + "system-outputs/en-ja/refA.txt": "一\n二\n",
        base + "human-scores/en-ja.esa.seg.score":
            "sysX\t80\nsysX\t70\nsysY\t40\nsysY\tNone\nrefA\t95\nrefA\t90\n",
    })
    c_qe_human.mtme("en-ja", n_sources=None)
    items = load_pack("C4", "en-ja")
    assert {i.meta["system"] for i in items} == {"sysX", "sysY"}         # refA is the reference
    assert len(items) == 3 and items[0].inputs["reference"] == "一"
    assert items[0].refs == {"human": 80.0, "src_id": 0} and items[0].group == "docA"


def test_wmt23_qe_builder(arena):
    from bench.arena.builders import c_qe_human

    tsv = ("index\toriginal\ttranslation\tscores\tmean\tz_scores\tz_mean\n"
           "0\tHello\tनमस्ते\t[90]\t90\t[0.5]\t0.5\n1\tBye\tअलविदा\t[20]\t20\t[-1]\t-1.0\n")
    _tgz(paths.eval_dir() / "_sources" / "wmt23_qe" /
         "WMT-QE-Task__wmt-qe-2023-data-main.tar.gz",
         {"wmt-qe-2023-data-main/train_dev/task_1/en-hi/dev.enhi.df.short.tsv": tsv})
    c_qe_human.wmt23_qe_da("en-hi", n=None)
    items = {i.id: i for i in load_pack("C4", "en-hi")}
    assert items["wmt23qe-dev-1"].refs == {"human": -1.0, "src_id": 1}


def test_c5_builders(arena):
    from bench.arena.builders import c_tn_g2p

    wp = paths.eval_dir() / "_sources" / "wikipron" / "hin_deva_broad.tsv"
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text("कमल\tk ə m ə l\nकमल\tk a m a l\nदो शब्द\tx\n", encoding="utf-8")
    c_tn_g2p.wikipron("hi.g2p", n=None)
    items = load_pack("C5", "hi.g2p")
    assert len(items) == 1 and items[0].refs["text"] == "k ə m ə l"
    _tgz(paths.eval_dir() / "_sources" / "nemo_tn" / "NVIDIA__NeMo-text-processing-v1.2.0.tar.gz",
         {"NeMo-text-processing-1.2.0/tests/nemo_text_processing/hi/data_text_normalization/"
          "test_cases_cardinal.txt": "25~पच्चीस\n3~तीन\n"})
    c_tn_g2p.nemo_tn_tests("hi.tn", n_per_class=None)
    items = load_pack("C5", "hi.tn")
    assert {i.meta["semiotic_class"] for i in items} == {"cardinal"} and len(items) == 2
    tpl = c_tn_g2p.c5_inhouse("ja.kana")
    assert tpl.suffix == ".tsv" and "今日は" in tpl.read_text()
    lines = tpl.read_text().splitlines()
    lines[1] = lines[1].replace("\t\t", "\tキョウワアメデスネ\t", 1)
    tpl.write_text("\n".join(lines) + "\n")
    pack = c_tn_g2p.c5_inhouse("ja.kana")
    assert pack.name == "ja.kana" and load_pack("C5", "ja.kana")[0].meta["task"] == "kana"


def test_scene_builders(arena):
    from bench.arena.builders import c_scene

    root = paths.eval_dir() / "SCENE" / "ja"
    (root / "refs_asr" / "qwen3-asr-1.7b").mkdir(parents=True)
    ja_words = _words([("こんにちは", .8, .1), ("。", .01, .6), ("元気", .5, .05),
                       ("ですか", .5, 2.0), ("はい", .4, .0)])
    json.dump({"text": "", "segments": [{"words": ja_words}]},
              (root / "refs_asr" / "qwen3-asr-1.7b" / "clip1.ja.json").open("w"))
    en = {"text": "Hello. How are you? Yes.", "segments": [{"words": [
        {"start": 0.1, "end": 0.6, "word": "Hello."}, {"start": 1.6, "end": 1.9, "word": "How"},
        {"start": 1.9, "end": 2.1, "word": "are"}, {"start": 2.1, "end": 2.4, "word": "you?"}]}]}
    json.dump(en, (root / "clip1.en.json").open("w"))
    write_pack("SCENE", "ja", [{"id": "clip1", "group": "ep1",
                                "inputs": {"audio": "clip1.ja.wav"},
                                "refs": {"dub_en_asr": "clip1.en.json",
                                         "dub_hi_text": "नमस्ते"}}], "# t\n")
    c_scene.scene_c2("ja-en")
    items = load_pack("C2", "ja-en")
    assert [i.refs.get("text") for i in items] == ["Hello.", "How are you?", None]
    assert items[1].inputs["slot_s"] == pytest.approx(1.05) and items[1].group == "clip1"
    c_scene.scene_c2("ja-hi")
    assert all(not i.refs for i in load_pack("C2", "ja-hi"))       # no timed Hindi: QE only
    c_scene.scene_c1("ja")
    c1_items = load_pack("C1", "ja")
    assert len(c1_items) == 1 and c1_items[0].inputs["words"][0]["word"] == "こんにちは"


def test_srt_builders(arena):
    from bench.arena.builders import c_srt

    d = paths.eval_dir() / "_sources" / "c6_srt" / "ja-en"
    d.mkdir(parents=True)
    (d / "ep.full.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nThis is far too long a "
                                   "line to read\n\n2\n00:00:01,000 --> 00:00:03,000\nOK\n")
    (d / "ep.ref.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nToo long\n\n"
                                  "2\n00:00:01,000 --> 00:00:03,000\nOK\n")
    (d / "ep.src.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n長すぎる\n")
    c_srt.c6_srt_pairs("ja-en")
    it = load_pack("C6", "ja-en")[0]
    assert it.inputs["limits"]["cpl"] == 42 and it.inputs["cues"][0]["source"] == "長すぎる"
    assert len(it.refs["cues"]) == 2
    w = paths.eval_dir() / "_sources" / "c1_srt" / "en"
    w.mkdir(parents=True)
    words = _words([("Hi", .3, .5), ("you", .3, .05), ("there", .3, .0)])
    (w / "talk.words.json").write_text(json.dumps(words))
    (w / "talk.srt").write_text("1\n00:00:00,000 --> 00:00:00,300\nHi\n\n"
                                "2\n00:00:00,700 --> 00:00:01,700\nyou there\n")
    c_srt.c1_srt_words("en")
    assert load_pack("C1", "en")[0].refs["ends"] == [0, 2]
    with pytest.raises(FileNotFoundError, match="IWSLT"):
        c_srt.iwslt_subtitling("en-ja")
