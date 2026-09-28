"""Phase B (source picture: B1 active speaker, B2 shots, B3 OCR, B4 content type) — model-free.

Judges on synthetic payloads, worker helpers, builders on tiny fixtures (downloads and ffprobe
monkeypatched), registry/spec consistency, and one end-to-end run of the B4 baseline worker.
"""
from __future__ import annotations

import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from bench.arena import paths
from bench.arena.hardware import Gpu, Hardware, check
from bench.arena.packs import Item, load_pack, write_pack
from bench.arena.registry import load_candidates
from bench.arena.specs import SPECS, b1, b2, b3, b4
from bench.arena.specs._bpic import iou, same_face, split_words

CAPS = ["B1", "B2", "B3", "B4"]
TOOLS = ["ffmpeg", "ffprobe", "git", "uv"]
MACHINES = {
    "thalassa": Hardware(host="thalassa", arch="x86_64", ram_gb=24, disk_free_gb=500,
                         gpus=[Gpu("NVIDIA GeForce RTX 5060 Ti", 16.0, "12.0")], tools=TOOLS),
    "spark": Hardware(host="spark", arch="aarch64", ram_gb=128, disk_free_gb=2000,
                      gpus=[Gpu("NVIDIA GB10", 112.0, "12.1", unified=True)], tools=TOOLS),
    "api": Hardware(host="api", arch="x86_64", ram_gb=16, disk_free_gb=100, tools=TOOLS,
                    api_keys=["GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                              "GOOGLE_APPLICATION_CREDENTIALS", "AWS_ACCESS_KEY_ID", "HF_TOKEN"]),
}


def _item(lang="en", **kw) -> Item:
    return Item(id=kw.pop("id", "x"), lang=lang, inputs=kw.pop("inputs", {}),
                refs=kw.pop("refs", {}), meta=kw.pop("meta", {}))


def _rows(rows) -> dict[str, tuple[float, float]]:
    return {m: (v, w) for m, v, w in rows}


def _worker(name: str):
    wdir = str(paths.WORKERS_DIR)
    if wdir not in sys.path:
        sys.path.insert(0, wdir)
    return importlib.import_module(name)


# ------------------------------------------------------------------ registry + specs

@pytest.mark.parametrize("cap", CAPS)
def test_registry_and_spec(cap):
    from bench.arena import envs

    spec = SPECS[cap]
    cands = load_candidates(cap)
    assert cands and len({c.id for c in cands}) == len(cands)
    assert sum(c.baseline for c in cands) == 1, "exactly one baseline per capability"
    for c in cands:
        assert c.url, (cap, c.id)
        assert c.kind in ("local", "api", "method")
        assert c.languages == "*" or {"en", "hi", "ja"} & set(c.languages), (cap, c.id)
        if not c.enabled:
            assert c.notes, f"{cap}/{c.id}: disabled entries say why"
            continue
        assert (paths.WORKERS_DIR / f"{c.worker}.py").exists(), (cap, c.id)
        env = envs.load_env(c.env)
        if env.python != "auto":
            assert env.setup.get("x86_64") and env.setup.get("aarch64"), (cap, c.env)
        runnable = [m for m, hw in MACHINES.items() if not check(c.requires, hw)]
        assert runnable, f"{cap}/{c.id} runs on none of {list(MACHINES)}"
        if c.kind == "api":
            assert c.requires.api_keys and "price_in_per_mtok" in c.params or \
                c.worker in ("b2_google_vi", "b2_rekognition"), (cap, c.id)
    for m in [*spec.primary.values(), *spec.secondary]:
        assert m in spec.higher_is_better, (cap, m)
    assert set(spec.packs) <= set(_builders())


def _builders():
    from bench.arena.builders import BUILDERS

    return BUILDERS


def test_thalassa_and_spark_split():
    """The Spark-only picks do not land on Thalassa; the small models run on both."""
    b3c = {c.id: c for c in load_candidates("B3")}
    assert check(b3c["qwen3-vl-32b"].requires, MACHINES["thalassa"])
    assert not check(b3c["qwen3-vl-32b"].requires, MACHINES["spark"])
    assert not check(b3c["pp-ocrv6-medium"].requires, MACHINES["spark"])
    b1c = {c.id: c for c in load_candidates("B1")}
    assert not check(b1c["lr-asd"].requires, MACHINES["thalassa"])
    assert check(b1c["gemini-3.1-pro"].requires, MACHINES["thalassa"])  # no key there


# ------------------------------------------------------------------ geometry helpers

def test_geometry_helpers():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 1], [0.5, 0, 1, 1]) == 0.0
    # a loose box around a tight one: low IoU but mutual centre containment
    assert same_face([0.4, 0.4, 0.5, 0.5], [0.35, 0.3, 0.55, 0.6])
    assert not same_face([0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.7, 0.7])
    words = split_words("hello big world", [0.0, 0.0, 0.15, 0.02])
    assert [w for w, _ in words] == ["hello", "big", "world"]
    assert words[0][1][2] < words[1][1][0] < words[2][1][0]


# ------------------------------------------------------------------ B1

FACE_A, FACE_B = [0.1, 0.2, 0.3, 0.5], [0.6, 0.2, 0.8, 0.5]


def _b1_item():
    return _item(refs={"lines": [
        {"start": 0.0, "end": 2.0, "t": 1.0, "box": FACE_A},
        {"start": 2.0, "end": 4.0, "t": 3.0, "box": FACE_B},
        {"start": 4.0, "end": 6.0, "t": 5.0, "box": None}]})


def _track(tid, box, t0, t1, p):
    ts = [round(t0 + k * 0.2, 2) for k in range(int((t1 - t0) / 0.2) + 1)]
    return {"id": tid, "boxes": [[t, *box] for t in ts], "speaking": [p(t) for t in ts]}


def test_b1_line_accuracy_from_tracks():
    item = _b1_item()
    good = {"tracks": [_track("a", FACE_A, 0, 6, lambda t: 0.9 if t < 2 else 0.1),
                       _track("b", FACE_B, 0, 6, lambda t: 0.9 if 2 <= t < 4 else 0.1)]}
    r = _rows(b1.speaker_lines(item, good, None))
    assert r["line_acc"] == (1.0, 3.0) and r["offscreen_f1"][0] == 1.0
    # face-only baseline: A "speaks" throughout → wrong on B's line and on the off-screen line
    lazy = {"tracks": [_track("a", FACE_A, 0, 6, lambda t: 1.0),
                       _track("b", FACE_B, 0, 6, lambda t: 0.0)]}
    r = _rows(b1.speaker_lines(item, lazy, None))
    assert r["line_acc"][0] == pytest.approx(1 / 3)
    assert r["offscreen_f1"] == (0.0, 1.0)


def test_b1_line_accuracy_from_payload_lines():
    out = {"lines": [{"start": 0.0, "end": 2.0, "box": [0.12, 0.22, 0.31, 0.52]},
                     {"start": 2.0, "end": 4.0, "offscreen": True},
                     {"start": 4.1, "end": 5.9, "box": None}]}
    r = _rows(b1.speaker_lines(_b1_item(), out, None))
    assert r["line_acc"][0] == pytest.approx(2 / 3)
    assert r["onscreen_acc"] == (0.5, 2.0)


def test_b1_ava_ap():
    faces = [{"t": 1.0, "box": FACE_A, "speaking": True},
             {"t": 1.0, "box": FACE_B, "speaking": False},
             {"t": 3.0, "box": FACE_B, "speaking": True}]
    item = _item(refs={"faces": faces})
    perfect = {"tracks": [_track("a", FACE_A, 0, 4, lambda t: 0.9),
                          _track("b", FACE_B, 0, 4, lambda t: 0.8 if t > 2 else 0.1)]}
    assert _rows(b1.ava_frame_ap(item, perfect, None))["ava_ap"] == (1.0, 3.0)
    assert b1.average_precision([0.1, 0.9], [True, False]) == pytest.approx(0.5)


def test_b1_worker_line_assignment():
    asd = _worker("b1_asd")
    tracks = [_track("a", FACE_A, 0, 4, lambda t: 0.8 if t < 2 else 0.2)]
    lines = asd._assign_lines(tracks, [{"start": 0, "end": 2}, {"start": 2, "end": 4}])
    assert lines[0]["track"] == "a" and lines[1]["offscreen"]


def test_b1_tracker_cuts_at_shot_boundaries():
    pytest.importorskip("scipy")
    asd = _worker("b1_asd")
    box = [100.0, 100.0, 200.0, 220.0, 0.99]
    dets = [[box] for _ in range(40)]
    tracks = asd.track_faces(dets, cuts=[20], p={"min_track": 10})
    assert [(t["frames"][0], t["frames"][-1]) for t in tracks] == [(0, 19), (20, 39)]


# ------------------------------------------------------------------ B2

def test_b2_transition_f1_and_false_cuts():
    fps = 25.0
    item = _item(meta={"fps": fps, "duration_s": 3600.0}, refs={
        "transitions": [{"start": 10.0, "end": 10.0}, {"start": 20.0, "end": 21.0},
                        {"start": 30.0, "end": 30.0}],
        "lines": [{"start": 40.0, "end": 44.0}]})
    out = {"transitions": [{"start": 10.08, "end": 10.08},   # +2 frames: hit
                           {"start": 20.5, "end": 20.5},     # inside the dissolve: hit
                           {"start": 30.2, "end": 30.2},     # +5 frames: miss
                           {"start": 42.0, "end": 42.0}]}    # mid-line false cut
    r = _rows(b2.transition_scores(item, out, None))
    assert r["f1"] == (pytest.approx(2 * 2 / 7), 7.0)
    assert r["recall"][0] == pytest.approx(2 / 3) and r["precision"][0] == pytest.approx(0.5)
    assert r["cut_recall"][0] == 0.5 and r["gradual_recall"][0] == 1.0
    assert r["fp_per_hour"] == (2.0, 1.0)
    assert r["false_cuts_per_line"] == (1.0, 1.0)


def test_b2_shots_to_transitions():
    fps = 10.0
    tr = b2.transitions_of({"shots": [{"start": 0, "end": 5.0}, {"start": 5.0, "end": 8.0},
                                      {"start": 9.0, "end": 12.0}]}, fps)
    assert tr == [(50, 50), (80, 90)]
    assert b2.transitions_of({"cuts": [1.0]}, fps) == [(10, 10)]


def test_b2_worker_helpers():
    tn = _worker("b2_transnetv2")
    assert tn._runs([0.1, 0.9, 0.2, 0.7, 0.8, 0.9, 0.95, 0.1], 0.5) == [(1, 1), (3, 6)]


# ------------------------------------------------------------------ B3

def test_b3_word_level_e2e_with_line_predictions():
    refs = {"texts": [{"text": "Platform", "box": [0.10, 0.10, 0.20, 0.15]},
                      {"text": "3", "box": [0.21, 0.10, 0.23, 0.15]},
                      {"text": "###", "box": [0.5, 0.5, 0.6, 0.6], "care": False}]}
    item = _item(refs=refs)
    out = {"texts": [{"text": "Platform 3", "box": [0.10, 0.10, 0.23, 0.15]},
                     {"text": "junk", "box": [0.5, 0.5, 0.6, 0.6]}]}   # on don't-care: ignored
    r = _rows(b3.ocr_scores(item, out, None))
    assert r["e2e_f"] == (1.0, 4.0)
    assert r["one_minus_ned"][0] == 1.0 and r["text_f"][0] == 1.0
    wrong = {"texts": [{"text": "Platfrm 3", "box": [0.10, 0.10, 0.23, 0.15]}]}
    r = _rows(b3.ocr_scores(item, wrong, None))
    assert r["e2e_f"][0] == 0.5 and r["det_f"][0] == 1.0
    assert 0.5 < r["one_minus_ned"][0] < 1.0


def test_b3_unspaced_lines_and_boxless_outputs():
    refs = {"texts": [{"text": "大阪行きの電車", "box": [0.2, 0.8, 0.8, 0.9]}]}
    item = _item(lang="ja", refs=refs)
    r = _rows(b3.ocr_scores(item, {"texts": [{"text": "大阪行きの電軍",
                                              "box": [0.2, 0.8, 0.8, 0.9]}]}, None))
    assert r["one_minus_ned"][0] == pytest.approx(1 - 1 / 7)
    assert "text_f" not in r and r["char_f"][0] == pytest.approx(12 / 14)
    boxless = _rows(b3.ocr_scores(item, {"texts": [{"text": "大阪行きの電車"}]}, None))
    assert boxless["e2e_f"][0] == 0.0 and boxless["char_f"][0] == 1.0


def test_b3_video_frames_matched_by_time():
    refs = {"texts": [{"text": "EXIT", "box": [0.1, 0.1, 0.2, 0.2], "t": 5.0},
                      {"text": "OPEN", "box": [0.1, 0.1, 0.2, 0.2], "t": 10.0}]}
    out = {"texts": [{"text": "exit", "box": [0.1, 0.1, 0.2, 0.2], "t": 5.1},
                     {"text": "OPEN", "box": [0.1, 0.1, 0.2, 0.2], "t": 5.0}]}  # wrong frame
    r = _rows(b3.ocr_scores(_item(refs=refs), out, None))
    assert r["e2e_f"] == (pytest.approx(2 / 4), 4.0)


def test_b_common_vlm_parsing():
    bc = _worker("b_common")
    assert bc.box_from_1000([100, 200, 300, 400]) == [0.2, 0.1, 0.4, 0.3]
    assert bc.parse_json('Sure:\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert bc.parse_json('noise {"label": "2d"} tail') == {"label": "2d"}
    assert bc.norm_poly([[10, 20], [30, 20], [30, 40], [10, 40]], 100, 100) == [0.1, 0.2,
                                                                                0.3, 0.4]
    assert bc.sample_times(100.0, 3, margin=0.0) == [0.0, 50.0, 100.0]


# ------------------------------------------------------------------ B4

def test_b4_judge_and_corpus_metrics():
    it = _item(refs={"label": "2d"})
    r = _rows(b4.content_type(it, {"label": "live_action", "confidence": 0.9}, None))
    assert r["accuracy"][0] == 0.0 and r["misroute_cost"][0] == 1.0
    assert r["block_recall_2d"][0] == 0.0 and r["conf_signed"][0] < 0
    r = _rows(b4.content_type(it, {"abstain": True}, None))
    assert r["label_idx"][0] == -1.0 and r["misroute_cost"][0] == b4.ABSTAIN_COST
    r = _rows(b4.content_type(it, {"label": "anime", "probs": {"2d": 0.8, "3d": 0.2}}, None))
    assert r["accuracy"][0] == 1.0 and r["brier"][0] == pytest.approx(0.08)
    assert b4.macro_f1([0, 1, 1, 2], [0, 1, 2, 2]) == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
    sel = b4.selective_accuracy(0.5)
    assert sel([0.9, 0.8, -0.1, -0.2], [0, 0, 0, 0]) == 1.0
    assert b4.ece([0.95, -0.95], [0, 0]) == pytest.approx(0.45)


def test_b4_frame_aggregation_and_deepghs_mapping():
    bc = _worker("b_common")
    pure = bc.aggregate_title([{"2d": 0.9, "3d": 0.05, "live_action": 0.05}] * 10)
    assert pure["label"] == "2d" and pure["probs"]["mixed"] == 0.0
    mixed = bc.aggregate_title([{"live_action": 0.9}] * 6 + [{"3d": 0.9}] * 4)
    assert mixed["label"] == "mixed" and mixed["probs"]["mixed"] == pytest.approx(0.8)
    dg = _worker("b4_deepghs")
    p = dg.frame_probs({"3d": 0.1, "bangumi": 0.6, "comic": 0.1, "illustration": 0.1,
                        "not_painting": 0.1}, {"anime": 0.8, "real": 0.2})
    assert p["live_action"] == 0.2 and p["2d"] == pytest.approx(0.8 * 0.8 / 0.9)


def test_b4_end_to_end_with_baseline_worker(tmp_path, monkeypatch):
    from bench.arena import judges, runner
    from bench.arena.registry import Candidate
    from bench.arena.report import leaderboard

    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    rows = [{"id": f"t{i}", "group": f"t{i}", "inputs": {"video": f"v{i}.mp4"},
             "refs": {"label": lab}} for i, lab in enumerate(
                 ["live_action", "live_action", "2d", "3d", "2d", "live_action"])]
    write_pack("B4", "en", rows, "# test\n")
    items = load_pack("B4", "en")
    live = Candidate(id="live", capability="B4", worker="b4_constant",
                     params={"label": "live_action"}, baseline=True)
    drawn = Candidate(id="drawn", capability="B4", worker="b4_constant", params={"label": "2d"})
    runner.generate("B4", "en", [live, drawn], items, check_gpu=False, log=lambda _: None)
    assert judges.score("B4", "en", [live, drawn], items, check_gpu=False,
                        log=lambda _: None) > 0
    _md, ranked = leaderboard("B4", "en", [live, drawn], items)
    by = {r["candidate"]: r for r in ranked}
    assert by["live"]["mean"] == pytest.approx((6 / 9) / 3)   # F1(live) = 2·3/(2·3+3), 3 classes
    assert by["drawn"]["mean"] == pytest.approx((4 / 8) / 3)


# ------------------------------------------------------------------ builders

def test_ava_style_lines_from_rows():
    from bench.arena.builders._b_common import ava_clip_item, offscreen_lines, speaking_runs

    rows = [{"t": 100 + k * 0.04, "box": FACE_A, "entity": "e1",
             "label": "SPEAKING_AUDIBLE" if k < 50 else "NOT_SPEAKING"} for k in range(100)]
    runs = speaking_runs(rows)
    assert len(runs) == 1 and runs[0]["start"] == 100.0 and runs[0]["box"] == FACE_A
    off = offscreen_lines([(103.0, 115.0)], runs)
    assert [round(o["end"] - o["start"], 3) for o in off] == [4.0, 4.0, 4.0]
    item = ava_clip_item("c", "vid", "clips/c.mp4", rows, [(103.0, 115.0)], 99.0, 20.0,
                         {"fps": 25})
    assert item["refs"]["lines"][0]["box"] == FACE_A
    assert sum(ln["box"] is None for ln in item["refs"]["lines"]) == 3
    assert item["inputs"]["lines"][0] == {"start": 1.0, "end": 3.0}


def _fake_media(monkeypatch, fps=25.0, dur=60.0):
    import bench.arena.builders._b_common as bc

    fake = lambda _p: {"duration_s": dur, "fps": fps, "width": 640.0, "height": 360.0}
    for mod in ("bench.arena.builders._b_common", "bench.arena.builders.b2_shots",
                "bench.arena.builders.b4_titles", "bench.arena.builders.b_local"):
        monkeypatch.setattr(importlib.import_module(mod), "probe", fake)
    return bc


def test_clipshots_builder(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path / "data"))
    import bench.arena.builders.b2_shots as shots

    _fake_media(monkeypatch)
    media = tmp_path / "media"
    media.mkdir()
    (media / "a.mp4").write_bytes(b"x")

    def fake_download(url, dst, **kw):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"a.mp4": {"transitions": [[99, 100], [200, 225]],
                                             "frame_num": 300},
                                   "missing.mp4": {"transitions": [], "frame_num": 1}}))
        return dst

    monkeypatch.setattr(shots, "download", fake_download)
    shots.clipshots("en", media_dir=str(media))
    items = load_pack("B2", "en")
    assert len(items) == 1
    assert items[0].refs["transitions"] == [{"start": 4.0, "end": 4.0, "type": "cut"},
                                            {"start": 8.0, "end": 9.0, "type": "gradual"}]
    assert Path(items[0].inputs["video"]).exists()
    assert shots._parse_shot_lines("v1 0 10 11 20\n") == {"v1": [(0, 10), (11, 20)]}


def test_textocr_builder(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    import bench.arena.builders.b3_ocr as ocr

    ann = {"imgs": {"i1": {"file_name": "train/i1.jpg", "width": 100, "height": 50}},
           "anns": {"a1": {"image_id": "i1", "bbox": [10, 5, 20, 10], "utf8_string": "EXIT"},
                    "a2": {"image_id": "i1", "bbox": [50, 5, 20, 10], "utf8_string": "."},
                    "a3": {"image_id": "i1", "bbox": [0, 0, 5, 5], "utf8_string": "A"}},
           "imgToAnns": {"i1": ["a1", "a2", "a3"]}}

    def fake_download(url, dst, **kw):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".json"):
            dst.write_text(json.dumps(ann))
        else:
            with zipfile.ZipFile(dst, "w") as zf:
                zf.writestr("train_images/i1.jpg", b"jpeg")
        return dst

    monkeypatch.setattr(ocr, "download", fake_download)
    ocr.textocr("en", min_words=2)
    (it,) = load_pack("B3", "en")
    assert it.refs["texts"][0] == {"text": "EXIT", "box": [0.1, 0.1, 0.3, 0.3], "care": True}
    assert it.refs["texts"][1]["care"] is False and Path(it.inputs["image"]).exists()


def test_b4_titles_splits_and_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path / "data"))
    import bench.arena.builders.b4_titles as titles

    _fake_media(monkeypatch)
    vids = []
    for k in range(7):
        v = tmp_path / f"v{k}.mp4"
        v.write_bytes(b"x")
        vids.append(v)
    labels = ["2d", "2d", "2d", "live_action", "live_action", "live_action", "mixed"]
    tf = tmp_path / "titles.json"
    tf.write_text(json.dumps([{"id": f"t{k}", "label": lab, "path": str(vids[k])}
                              for k, lab in enumerate(labels)]))
    titles.b4_titles("hi", titles_file=str(tf), replace=True)
    items = load_pack("B4", "hi", split=None)
    split = {it.refs["label"]: [] for it in items}
    for it in items:
        split[it.refs["label"]].append(it.split)
    assert split["2d"].count("dev") == 1 and split["live_action"].count("dev") == 1
    assert split["mixed"] == ["test"]


def test_b_local_template_and_pack(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    import bench.arena.builders.b_local as local

    _fake_media(monkeypatch)
    base = local.local_dir("B3", "ja")
    (base / "media" / "clip.mp4").write_bytes(b"x")
    tpl = local.b_local("ja", capability="B3", template=True)
    row = json.loads(tpl.read_text().splitlines()[0])
    assert row["refs"] == {"texts": [], "granularity": "line"} and len(row["inputs"]["times"]) == 6
    row["refs"]["texts"] = [{"text": "出口", "box": [0.1, 0.1, 0.2, 0.2], "t": 5.0}]
    (base / "annotations.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    local.b_local("ja", capability="B3")
    (it,) = load_pack("B3", "ja")
    assert it.inputs["times"] and Path(it.inputs["video"]).exists()
    assert it.refs["texts"][0]["text"] == "出口"


def test_synth_overlay_builder(tmp_path, monkeypatch):
    PIL = pytest.importorskip("PIL")
    from PIL import ImageFont

    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    import bench.arena.builders.b3_synth_overlay as synth

    monkeypatch.setattr(synth, "_fonts", lambda lang, d: (Path("r.ttf"), Path("b.ttf")))
    import io

    data = ImageFont.load_default(10).font_bytes  # Pillow's embedded font; no download
    real = ImageFont.truetype
    monkeypatch.setattr(ImageFont, "truetype",
                        lambda path, size, layout_engine=None: real(io.BytesIO(data), size))
    synth.b3_synth_overlay("en", n=4, size=(320, 180))
    items = load_pack("B3", "en")
    assert len(items) == 4 and PIL
    for it in items:
        assert it.refs["texts"] and all(0 <= b <= 1 for t in it.refs["texts"] for b in t["box"])
        assert Path(it.inputs["image"]).exists()
