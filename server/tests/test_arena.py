"""Model arena: worker protocol, caching, paired stats, reports — with the model-free echo worker."""
from __future__ import annotations

import json

import numpy as np
import pytest

from bench.arena import db, judges, runner, stats
from bench.arena.packs import load_pack, write_pack
from bench.arena.registry import Candidate
from bench.arena.report import audit_data, audit_html, leaderboard

SENTENCES = [f"the quick brown fox number {i} jumps over the lazy dog" for i in range(40)]
LONG = "a very long reference sentence " * 20  # longer than a file name may be


@pytest.fixture()
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    rows = [{"id": f"s{i:02d}", "group": f"g{i // 2}", "inputs": {"text": s},
             "refs": {"text": s}, "meta": {"duration_s": 2.0}} for i, s in enumerate(SENTENCES)]
    rows.append({"id": "zz-long", "split": "dev", "inputs": {"text": LONG}, "refs": {"text": LONG}})
    write_pack("T0", "de", rows, "# test pack\n")
    return tmp_path


def _cands():
    return [
        Candidate(id="exact", capability="T0", worker="echo", params={"field": "text"},
                  baseline=True),
        Candidate(id="drops", capability="T0", worker="echo",
                  params={"field": "text", "drop_words": 2}),
        Candidate(id="flaky", capability="T0", worker="echo",
                  params={"field": "text", "fail_ids": ["s03"]}, languages=["de"]),
        Candidate(id="hindi-only", capability="T0", worker="echo", languages=["hi"]),
    ]


def test_generate_score_rank_and_cache(arena):
    items = load_pack("T0", "de")
    cands = _cands()
    made = runner.generate("T0", "de", cands, items, check_gpu=False, log=lambda _: None)
    assert made[cands[0].key] == 40 and made[cands[2].key] == 39
    assert cands[3].key not in made  # language not declared: skipped, never generated

    # second run: everything ok is cached; the failed item is retried
    again = runner.generate("T0", "de", cands, items, check_gpu=False, log=lambda _: None)
    assert again[cands[0].key] == 0 and again[cands[1].key] == 0

    assert judges.score("T0", "de", cands, items) > 0
    assert judges.score("T0", "de", cands, items) == 0  # already scored at this judge version

    md, rows = leaderboard("T0", "de", cands, len(items))
    by_id = {r["candidate"]: r for r in rows}
    assert rows[0]["candidate"] in ("exact", "flaky") and by_id["exact"]["mean"] == 0.0
    assert by_id["drops"]["in_winner_set"] is False
    assert by_id["drops"]["mean"] == pytest.approx(2 / 11)
    assert "hindi-only" in md  # listed as not evaluated in de


def test_outputs_are_kept_per_item_for_audit(arena):
    items = load_pack("T0", "de")
    cands = _cands()[:2]
    runner.generate("T0", "de", cands, items, check_gpu=False, log=lambda _: None)
    judges.score("T0", "de", cands, items)
    con = db.connect()
    db.add_audit(con, "T0", "de", cands[1].key, "s00", "bad", "drops the article")
    con.close()

    data = audit_data("T0", "de", cands, items)
    first = data["items"][0]
    assert first["outputs"][cands[0].key]["text"] == SENTENCES[0]
    assert first["outputs"][cands[1].key]["mark"] == {"verdict": "bad",
                                                      "note": "drops the article"}
    page = audit_html(data, "leaderboard")
    assert "</script>" in page and json.dumps("s00") in page
    out = runner.output_prefix(cands[0], items[0])
    assert out.with_suffix(".json").exists() and str(arena) in str(out)


def test_rank_holm_and_practical_threshold():
    groups = [f"g{i}" for i in range(50)]
    rng = np.random.default_rng(1)
    base = rng.uniform(0.05, 0.15, size=50)

    def sums(v):
        return np.stack([v * 10, np.full_like(v, 10.0)], axis=1)

    table = stats.MetricTable(groups=groups, n_items=50, sums={
        "best": sums(base), "tiny-worse": sums(base + 0.002), "much-worse": sums(base + 0.05)})
    ranked = {r.cand_key: r for r in stats.rank(table, higher_is_better=False, threshold=0.01)}
    assert ranked["best"].in_winner_set
    assert ranked["much-worse"].in_winner_set is False and ranked["much-worse"].p_adj < 0.05
    # significantly worse but within the practical threshold stays in the winner set
    assert ranked["tiny-worse"].in_winner_set
    assert ranked["tiny-worse"].diff_vs_best == pytest.approx(0.002)


def test_text_errors_unspaced_and_marks():
    from bench.arena.packs import Item

    ja = Item(id="a", lang="ja", inputs={}, refs={"text": "今日は、晴れです。"})
    rows = {m: v for m, v, _ in judges.text_errors(ja, {"text": "今日は晴れ です"}, None)}
    assert rows["cer"] == 0.0 and "wer" not in rows
    hi = Item(id="b", lang="hi", inputs={}, refs={"text": "किताब"})
    rows = {m: v for m, v, _ in judges.text_errors(hi, {"text": "कताब"}, None)}
    assert rows["wer"] == 1.0  # a dropped matra is an error, not normalised away


def test_long_reference_text_is_not_treated_as_a_path(arena):
    items = {it.id: it for it in load_pack("T0", "de", split=None)}
    assert items["zz-long"].refs["text"] == LONG and len(items) == 41


def test_hardware_requirements_gate_candidates(tmp_path, monkeypatch):
    from bench.arena import hardware

    hw = hardware.Hardware(host="t", arch="x86_64", ram_gb=24, disk_free_gb=90,
                           gpus=[hardware.Gpu("RTX 5060 Ti", 15.9, "12.0")], api_keys=[],
                           tools=["ffmpeg"])
    ok = hardware.Requires.parse({"vram_gb": 8})
    assert hardware.check(ok, hw) == []
    big = hardware.Requires.parse({"vram_gb": 60, "arch": "aarch64", "api_keys": ["X_KEY"],
                                   "tools": ["nvcc"]})
    why = hardware.check(big, hw)
    assert any("VRAM" in w for w in why) and any("arch" in w for w in why)
    assert any("X_KEY" in w for w in why) and any("nvcc" in w for w in why)
    spark = hardware.Hardware(host="s", arch="aarch64", ram_gb=128, disk_free_gb=900,
                              gpus=[hardware.Gpu("NVIDIA GB10", 112, "12.1", unified=True)])
    assert hardware.check(big, spark) == ["missing API key(s): X_KEY", "missing tool(s): nvcc"]
    assert spark.profile == "spark-112g"

    cand = Candidate(id="c", capability="T0", worker="echo", enabled=False,
                     requires={"vram_gb": 60})
    assert cand.blockers(hw)[0] == "disabled in registry" and cand.requires.gpu


def test_model_judges_derived_metrics_gates_and_cache(arena, monkeypatch):
    from bench.arena import judges as jmod
    from bench.arena.judges import ModelJudge, Spec, mean_of

    judge = ModelJudge(id="echo_judge@1", worker="echo_judge",
                       inputs=lambda item, out, prefix: {"text": out["text"]},
                       to_rows=lambda item, p, out: [(k, float(v), 1.0)
                                                     for k, v in p["metrics"].items()])
    spec = Spec(id="T0", title="t", judges={"text_errors@1": judges.text_errors},
                primary={"*": "wer"}, higher_is_better={"wer": False, "echo_len": True},
                model_judges=[judge], derived={"echo_mean": mean_of("echo_len", "")},
                gates={"echo_len": (">=", 10.0)})
    monkeypatch.setattr(jmod, "get_spec", lambda cap: spec)
    import bench.arena.report as rep
    monkeypatch.setattr(rep, "get_spec", lambda cap: spec)

    items = load_pack("T0", "de")
    cands = _cands()[:2]
    runner.generate("T0", "de", cands, items, check_gpu=False, log=lambda _: None)
    assert jmod.score("T0", "de", cands, items, check_gpu=False, log=lambda _: None) > 0
    con = db.connect()
    rows = {(r["cand_key"], r["metric"]): r["value"] for r in db.scores(con, "T0", "de")
            if r["item"] == "s00"}
    con.close()
    assert rows[(cands[0].key, "echo_len")] == 11 and rows[(cands[1].key, "echo_len")] == 9
    assert rows[(cands[0].key, "echo_mean")] == pytest.approx((11 + 22) / 2)
    # judged outputs are cached: a second pass starts no judge job and writes nothing
    assert jmod.score("T0", "de", cands, items, check_gpu=False, log=lambda _: None) == 0

    md, ranked = leaderboard("T0", "de", cands, len(items))
    by_id = {r["candidate"]: r for r in ranked}
    assert by_id["drops"]["in_winner_set"] is False and "echo_len" in md  # 9 words fails ≥ 10


def test_api_budget_caps_spend(arena):
    items = load_pack("T0", "de")
    cand = Candidate(id="paid", capability="T0", worker="echo", kind="api",
                     params={"field": "text", "cost_per_item": 0.25})
    runner.generate("T0", "de", [cand], items, check_gpu=False, budget_usd=1.0,
                    log=lambda _: None)
    con = db.connect()
    rows = con.execute("SELECT status, cost_usd FROM outputs WHERE candidate='paid'").fetchall()
    con.close()
    assert sum(r["status"] == "ok" for r in rows) == 4
    assert sum(r["cost_usd"] or 0 for r in rows) == pytest.approx(1.0)
    assert sum(r["status"] == "skipped" for r in rows) == 36


def test_registry_is_consistent():
    """Every enabled candidate names an existing worker and env; ids are unique; specs load."""
    from bench.arena import envs, paths
    from bench.arena.registry import capabilities, load_candidates
    from bench.arena.specs import SPECS

    assert "A4" in SPECS
    for cap in capabilities():
        cands = load_candidates(cap)  # raises on duplicate ids / unknown fields
        for c in cands:
            assert c.kind in ("local", "api", "method"), (cap, c.id)
            if not c.enabled:
                continue
            assert (paths.WORKERS_DIR / f"{c.worker}.py").exists(), (cap, c.id, c.worker)
            envs.load_env(c.env)  # raises if the env is not defined
            assert c.languages == "*" or isinstance(c.languages, list), (cap, c.id)


def test_corpus_metric_ranks_judges_by_agreement_with_humans(arena, monkeypatch):
    """G-style meta-evaluation: the judge whose scores track the human ratings wins."""
    from bench.arena import judges as jmod
    from bench.arena import report as rep
    from bench.arena.judges import CorpusMetric, Spec

    rows = [{"id": f"u{i:02d}", "group": f"g{i}", "inputs": {"text": "x " * (i + 1)},
             "refs": {"mos": float(i)}} for i in range(30)]
    write_pack("T0", "ja", rows, "# t\n")
    spec = Spec(id="T0", title="t",
                judges={"len@1": lambda it, out, row: [("pred", float(len(out["text"])), 1.0)]},
                primary={"*": "srcc"}, higher_is_better={"srcc": True, "pred": True},
                corpus={"srcc": CorpusMetric("pred", lambda it: it.refs.get("mos"))})
    monkeypatch.setattr(jmod, "get_spec", lambda cap: spec)
    monkeypatch.setattr(rep, "get_spec", lambda cap: spec)
    items = load_pack("T0", "ja")
    good = Candidate(id="tracks-humans", capability="T0", worker="echo", params={"field": "text"})
    bad = Candidate(id="ignores-input", capability="T0", worker="echo", params={"field": "nope"})
    runner.generate("T0", "ja", [good, bad], items, check_gpu=False, log=lambda _: None)
    jmod.score("T0", "ja", [good, bad], items, check_gpu=False, log=lambda _: None)
    _md, ranked = leaderboard("T0", "ja", [good, bad], items)
    assert ranked[0]["candidate"] == "tracks-humans" and ranked[0]["mean"] == pytest.approx(1.0)


def test_item_ids_with_dots_get_distinct_outputs(arena):
    from bench.arena.packs import Item

    cand = _cands()[0]
    a = Item(id="clip.1", lang="de", inputs={"text": "a"})
    b = Item(id="clip.2", lang="de", inputs={"text": "b"})
    pa, pb = runner.output_prefix(cand, a), runner.output_prefix(cand, b)
    assert pa.with_suffix(".json") != pb.with_suffix(".json")
