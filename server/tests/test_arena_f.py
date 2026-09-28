"""Phase F (picture out: F1–F4) — model-free tests: registry, specs, function judges, frame
helpers, builders on tiny ffmpeg fixtures and model-free workers end to end."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bench.arena import db, envs, judges, paths, runner
from bench.arena.packs import Item, load_pack, write_pack
from bench.arena.registry import Candidate, load_candidates

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

WORKERS = paths.WORKERS_DIR
F_CAPS = ["F1", "F2", "F3", "F4"]


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def _video(path: Path, seconds: float = 2.0, size: str = "160x120", src: str = "testsrc2",
           audio: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-f", "lavfi", "-i", f"{src}=size={size}:rate=25:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    _ff(*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path))
    return path


def _wav(path: Path, seconds: float = 2.0, bursts: list[tuple[float, float]] | None = None,
         sr: int = 16000) -> Path:
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for s, e in bursts or [(0.0, seconds)]:
        m = (t >= s) & (t < e)
        x[m] = 0.3 * np.sin(2 * np.pi * 220 * t[m])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr)
    return path


def _rows(rows) -> dict[str, float]:
    return {m: v for m, v, _ in rows}


def _font() -> Path | None:
    try:
        import matplotlib

        p = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if p.exists():
            return p
    except ImportError:
        pass
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return p if p.exists() else None


@pytest.fixture()
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture()
def wk(monkeypatch):
    """Import phase-F worker helper modules (they live on the workers path, not a package)."""
    monkeypatch.syspath_prepend(str(WORKERS))
    import f_frames

    return f_frames


# ---------------------------------------------------------------- registry, specs, envs

def test_f_specs_candidates_envs_are_consistent():
    from bench.arena.builders import BUILDERS
    from bench.arena.specs import SPECS

    for cap in F_CAPS:
        spec = SPECS[cap]
        assert spec.primary_for("en") in spec.higher_is_better
        for m in spec.secondary + list(spec.gates):
            assert m in spec.higher_is_better, (cap, m)
        assert all(b in BUILDERS for b in spec.packs), (cap, spec.packs)
        assert spec.judges_for("en"), cap
        cands = load_candidates(cap)
        ids = [c.id for c in cands]
        assert len(ids) == len(set(ids))
        assert sum(c.baseline for c in cands) == 1, cap
        for c in cands:
            assert c.url or not c.enabled, (cap, c.id)  # where to get it
            assert (WORKERS / f"{c.worker}.py").exists(), (cap, c.id)
            env = envs.load_env(c.env)
            if not c.enabled:
                assert c.notes, (cap, c.id)  # disabled rows say why
                continue
            if c.kind == "api":
                assert c.requires.api_keys and c.env == "server", (cap, c.id)
            if env.python != "auto" or env.setup:
                arches = set(env.setup)
                assert "x86_64" in arches or "any" in arches, (cap, c.env)
                # every recipe covers aarch64 or the candidate is pinned to x86_64
                assert "aarch64" in arches or "any" in arches or c.requires.arch == ["x86_64"], \
                    (cap, c.id, c.env)


def test_f_candidates_land_on_the_right_machine():
    from bench.arena import hardware

    thalassa = hardware.Hardware(host="t", arch="x86_64", ram_gb=24, disk_free_gb=500,
                                 gpus=[hardware.Gpu("RTX 5060 Ti", 15.9, "12.0")],
                                 tools=["ffmpeg"])
    spark = hardware.Hardware(host="s", arch="aarch64", ram_gb=128, disk_free_gb=900,
                              gpus=[hardware.Gpu("NVIDIA GB10", 112, "12.1", unified=True)],
                              api_keys=["HF_TOKEN"], tools=["ffmpeg"])
    f1 = {c.id: c for c in load_candidates("F1")}
    for cid in ("wav2lip-gan", "latentsync-1.6", "latentsync-1.5", "musetalk-1.5"):
        assert hardware.check(f1[cid].requires, thalassa) == [], cid
    for cid in ("ltx-2.3-dubit-bf16", "ltx-2.3-dubit-nvfp4", "infinitetalk-480-bf16",
                "infinitetalk-480-fp8", "wan2.2-s2v-14b"):
        assert hardware.check(f1[cid].requires, thalassa), cid          # never on 16 GB
        assert hardware.check(f1[cid].requires, spark) == [], cid       # fits the Spark
    f4 = {c.id: c for c in load_candidates("F4")}
    assert hardware.check(f4["flux2-klein-4b"].requires, thalassa) == []
    assert hardware.check(f4["qwen-image-edit-2511-nunchaku-fp4"].requires, spark)  # x86 only
    assert "SYNC_API_KEY" in f1["sync-3"].requires.api_keys


def test_f_workers_import_only_the_stdlib():
    heavy = ["numpy", "torch", "PIL", "cv2", "diffusers", "insightface", "easyocr", "lpips",
             "soundfile", "yaml", "scipy", "huggingface_hub"]
    mods = sorted(p.stem for p in WORKERS.glob("f*_*.py"))
    assert "f1_latentsync" in mods and "f_judge_face" in mods
    code = (f"import sys; sys.argv=['x']; sys.path.insert(0, {str(WORKERS)!r})\n"
            f"import importlib\nfor m in {mods!r}: importlib.import_module(m)\n"
            f"bad = [h for h in {heavy!r} if h in sys.modules]\nprint(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", f"module-level heavy imports: {out}"


# ---------------------------------------------------------------- function judges

def test_fullframe_vs_gt_psnr_ssim(tmp_path):
    from bench.arena.specs.f1 import duration_match, fullframe_vs_gt

    gt = _video(tmp_path / "gt.mp4")
    noisy = tmp_path / "noisy.mp4"
    _ff("-i", str(gt), "-vf", "noise=alls=40:allf=t", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(noisy))
    item = Item(id="a", lang="en", inputs={"video": str(gt)}, refs={"video": str(gt)},
                meta={"audio_duration_s": 2.0})
    same = _rows(fullframe_vs_gt(item, {"files": {"video": str(gt)}}, None))
    worse = _rows(fullframe_vs_gt(item, {"files": {"video": str(noisy)}},
                                                        None))
    assert same["psnr"] == 99.0 and same["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert worse["psnr"] < 35 and worse["ssim"] < same["ssim"]
    assert same["frames_missing"] == 0.0
    no_gt = Item(id="b", lang="en", inputs={"video": str(gt)})
    assert fullframe_vs_gt(no_gt, {"files": {"video": str(gt)}}, None) == []
    assert duration_match(item, {"duration_s": 1.5}, None) == [("duration_err_s", 0.5, 1.0)]


def test_ssim_matches_reference_properties():
    from bench.arena.specs.f1 import psnr, ssim

    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, (2, 32, 32)).astype(np.float32)
    assert ssim(a, a) == pytest.approx(1.0)
    assert ssim(a, 255 - a) < 0
    assert psnr(a, a + 16) == pytest.approx(10 * np.log10(255 ** 2 / 256), rel=1e-4)


def test_derived_abs_offset_and_sync_panel():
    from bench.arena.specs import SPECS
    from bench.arena.specs.f1 import SYNC_PRIMARY

    spec = SPECS["F1"]
    fn = spec.derived[SYNC_PRIMARY]
    assert fn({"sync_offset_ms_synchformer": (-120.0, 1.0)}) == (120.0, 1.0)
    assert fn({}) is None
    ids = [j.id for j in spec.judges_for("en")]
    assert "lipsync.synchformer@1" in ids and "lipsync.syncnet@1" in ids
    assert "face.arcface@1" in ids and "video.lpips_fvd@1" in ids
    # SyncNet is for comparability only: never the primary
    assert "syncnet" not in spec.primary_for("en") and "lse" not in spec.primary_for("en")


def test_face_and_video_judges_apply_only_where_they_should(tmp_path):
    from bench.arena.specs.f1 import face_judge, video_judge

    v = _video(tmp_path / "v.mp4")
    self_item = Item(id="s", lang="en", inputs={"video": str(v)}, refs={"video": str(v)})
    cross_item = Item(id="c", lang="en", inputs={"video": str(v)})
    out = {"files": {"video": str(v)}}
    assert face_judge().inputs(self_item, out, tmp_path)["gt_video"] == str(v)
    assert "gt_video" not in face_judge().inputs(cross_item, out, tmp_path)
    assert video_judge().inputs(cross_item, out, tmp_path) is None
    assert video_judge().inputs(self_item, out, tmp_path) == {"video": str(v), "gt_video": str(v)}
    assert face_judge().inputs(self_item, {"files": {}}, tmp_path) is None


def test_f3_flap_alignment_and_vad(tmp_path):
    from bench.arena.specs.f3 import energy_vad, flap_alignment, frame_f1, intervals_to_frames

    dub = _wav(tmp_path / "dub.wav", 4.0, bursts=[(0.5, 1.5), (2.5, 3.5)])
    vad = energy_vad(dub)
    assert len(vad) == 2 and vad[0][0] == pytest.approx(0.5, abs=0.08)
    item = Item(id="x", lang="en", inputs={"video": "v.mp4", "audio": str(dub)},
                meta={"duration_s": 4.0})
    perfect = flap_alignment(item, {"duration_s": 4.0, "mouth_open": vad}, None)
    assert _rows(perfect)["flap_f1"] == pytest.approx(1.0)
    shifted = flap_alignment(item, {"duration_s": 4.0, "mouth_open": [[1.0, 2.0]],
                                    "cues": [{"start": 0.5, "end": 1.5, "shape": "D"},
                                             {"start": 2.5, "end": 3.5, "shape": "A"}]}, None)
    rows = _rows(shifted)
    assert rows["flap_f1"] < 0.5 and rows["cue_f1"] == pytest.approx(2 / 3, abs=0.02)
    # the pack's dub line timings take precedence over the VAD
    item.refs["dub_speech"] = [[1.0, 2.0]]
    again = _rows(flap_alignment(item, {"duration_s": 4.0,
                                                             "mouth_open": [[1.0, 2.0]]}, None))
    assert again["flap_f1"] == pytest.approx(1.0)
    assert frame_f1(intervals_to_frames([], 10), intervals_to_frames([], 10)) == 1.0


def test_f4_cer_and_ocr_rows():
    from bench.arena.specs.f4 import cer, ocr_judge

    assert cer("Hello World", "hello world") == 0.0
    assert cer("第一章", "第二章") == pytest.approx(1 / 3)
    item = Item(id="t", lang="en", inputs={"video": "v.mp4", "texts": [
        {"box": [0, 0, 10, 10], "start": 0, "end": 1, "src_text": "第一章", "src_lang": "ja",
         "tgt_text": "Chapter One", "tgt_lang": "en"}]}, meta={"src_lang": "ja"})
    rows = _rows(ocr_judge().to_rows(item, {"texts": ["Chapter One"]}, {}))
    assert rows["ocr_cer"] == 0.0 and rows["ocr_exact"] == 1.0 and rows["src_leak"] == 0.0
    leak = _rows(ocr_judge().to_rows(item, {"texts": ["第一章"]}, {}))
    assert leak["ocr_cer"] == 1.0 and leak["src_leak"] == 1.0


# ---------------------------------------------------------------- frame helpers (numpy)

def test_edit_mask_annulus_and_paste(wk):
    rng = np.random.default_rng(1)
    src = rng.integers(60, 200, (4, 64, 64, 3)).astype(np.uint8)
    edited = src.copy()
    edited[:, 30:44, 24:40] = 128  # the "generated mouth" patch
    mask = wk.edit_mask(edited, src)
    assert mask[:, 37, 32].min() > 0.9 and mask[:, 2, 2].max() < 0.05
    box = wk.mask_bbox(edited, src, pad=0.0, square=False)
    assert box[0] <= 24 and box[2] >= 40 and box[1] <= 30 and box[3] >= 44
    out = wk.annulus_match(edited, src, ring=6, grain=False)
    assert np.array_equal(out[:, 2, 2], edited[:, 2, 2])            # outside: untouched
    # inside: statistics moved toward the surround, but ground truth is never copied
    assert not np.array_equal(out[:, 35:40, 28:36], src[:, 35:40, 28:36])
    comp = wk.mask_composite(np.zeros_like(edited), edited, src)
    assert comp[:, 37, 32].max() < 20 and np.array_equal(comp[:, 2, 2], edited[:, 2, 2])
    key = np.full((64, 64, 3), 255, np.uint8)
    pasted = wk.paste_keyframe(src, 2.0, key, [20, 20, 30, 30], 1.0, 2.0, margin=0, feather=1)
    assert np.array_equal(pasted[0], src[0]) and (pasted[2, 25, 25] == 255).all()
    assert wk.ssim(src[..., 0].astype(np.float32), src[..., 0].astype(np.float32)) == \
        pytest.approx(1.0)


def test_read_write_frames_roundtrip(wk, tmp_path):
    v = _video(tmp_path / "v.mp4", 1.0)
    frames, fps = wk.read_frames(v)
    assert frames.shape == (25, 120, 160, 3) and fps == 25.0
    out = wk.write_frames(frames, fps, tmp_path / "o.mp4")
    again, _ = wk.read_frames(out)
    assert again.shape == frames.shape and np.abs(again.astype(int) - frames).mean() < 6


# ---------------------------------------------------------------- builders (tiny fixtures)

def _media(root: Path) -> None:
    _video(root / "user" / "alice_talk.mp4", 3.5, audio=True)
    _video(root / "user" / "bob_talk.mp4", 3.5, audio=True, src="smptebars")
    (root / "user" / "bob_talk.txt").write_text("hello there general kenobi\n")
    _video(root / "voxceleb2" / "test" / "mp4" / "id00017" / "abcDEF" / "00001.mp4", 3.5,
           audio=True)
    _video(root / "crossling" / "en" / "carol_scene.mp4", 3.0, audio=True)
    _wav(root / "crossling" / "en" / "carol_scene.wav", 3.0)


def test_f1_builders_merge_into_one_pack(arena):
    from bench.arena.builders.f1_crossling import f1_crossling
    from bench.arena.builders.f1_selfreenact import f1_selfreenact

    media = arena / "media"
    _media(media)
    f1_selfreenact("en", n=10, media_dir=str(media), download=False)
    f1_crossling("en", media_dir=str(media))
    items = load_pack("F1", "en")
    kinds = {it.id: it.meta["kind"] for it in items}
    assert sorted(kinds.values()) == ["cross", "self", "self", "self"]
    self_items = [it for it in items if it.meta["kind"] == "self"]
    assert all(it.refs["video"] == it.inputs["video"] for it in self_items)
    assert len({it.group for it in self_items}) == 3             # one clip per speaker
    assert any(it.meta.get("text") == "hello there general kenobi" for it in self_items)
    cross = next(it for it in items if it.meta["kind"] == "cross")
    assert "video" not in cross.refs and Path(cross.inputs["src_audio"]).exists()
    # rebuilding one source replaces only its own rows
    f1_selfreenact("en", n=1, media_dir=str(media), download=False)
    assert sorted(it.meta["kind"] for it in load_pack("F1", "en")) == ["cross", "self"]
    lic = (paths.eval_dir() / "F1" / "en" / "LICENSE.md").read_text()
    assert "self-reenactment" in lic and "cross-lingual" in lic


def test_scene_dub_builder_reads_the_scene_manifest(arena):
    from bench.arena.builders.scene_dub import scene_dub

    scene = paths.eval_dir() / "SCENE" / "ja"
    _video(scene / "clips" / "s1.ja.mp4", 3.0, audio=True)
    _wav(scene / "clips" / "s1.ja.wav", 3.0)
    _wav(scene / "clips" / "s1.en.wav", 3.0, bursts=[(0.5, 2.0)])
    (scene / "refs_asr").mkdir(parents=True)
    (scene / "refs_asr" / "s1.en.json").write_text(json.dumps(
        {"text": "hi", "segments": [{"start": 0.5, "end": 2.0, "text": "hi"}]}))
    write_pack("SCENE", "ja", [{
        "id": "spyfamily-ep1-s1", "group": "spyfamily-ep1",
        "inputs": {"video": "clips/s1.ja.mp4", "audio": "clips/s1.ja.wav"},
        "refs": {"dub_en_audio": "clips/s1.en.wav", "dub_en_asr": "refs_asr/s1.en.json"},
        "meta": {"start_s": 300.0, "end_s": 303.0, "duration_s": 3.0}}], "# scene\n")
    scene_dub("en", n=None)
    f3 = load_pack("F3", "en")
    assert len(f3) == 1 and f3[0].refs["dub_speech"] == [[0.5, 2.0]]
    assert Path(f3[0].inputs["audio"]).exists() and Path(f3[0].inputs["src_audio"]).exists()
    assert f3[0].meta["duration_s"] == pytest.approx(3.0, abs=0.1)  # pre-cut clip, not t=300
    # anime never enters the ranked F1 split
    assert load_pack("F1", "en") == []
    control = load_pack("F1", "en", split="control")
    assert len(control) == 1 and control[0].meta["content"] == "2d_animation"


def test_f4_overlay_builder_and_workers_end_to_end(arena, monkeypatch):
    font = _font()
    if font is None:
        pytest.skip("no TrueType font available")
    from PIL import features

    if not features.check("raqm"):
        pytest.skip("Pillow without Raqm")
    import bench.arena.builders.f4_overlay_synth as b

    monkeypatch.setattr(b, "W", 320)
    monkeypatch.setattr(b, "H", 180)
    monkeypatch.setattr(b, "DUR", 2.0)
    monkeypatch.setattr(b, "SPAN", (0.5, 1.5))
    monkeypatch.setattr(b, "font_path", lambda script, font_dir=None: font)
    b.f4_overlay_synth("ja", n=3)  # ja target ← en source (Latin), renderable with DejaVu
    items = load_pack("F4", "ja")
    assert len(items) == 3 and {it.meta["style"] for it in items} == set(b.STYLES)
    t = items[0].inputs["texts"][0]
    assert t["src_lang"] == "en" and t["tgt_lang"] == "ja" and len(t["box"]) == 4
    assert Path(items[0].refs["clean_video"]).exists()

    # model-free workers on an all-Latin variant (the worker renders the *target* script)
    fonts = arena / "fonts"
    fonts.mkdir()
    shutil.copy(font, fonts / "noto-latin.ttf")
    monkeypatch.setenv("OPENDUB_FONT_DIR", str(fonts))
    rows = []
    for it in items:
        texts = [{**x, "tgt_lang": "en", "tgt_text": "HELLO"} for x in it.inputs["texts"]]
        rows.append({"id": it.id, "group": it.group,
                     "inputs": {"video": it.inputs["video"], "texts": texts},
                     "refs": {"clean_video": it.refs["clean_video"]}, "meta": it.meta})
    write_pack("F4", "en", rows, "# t\n")
    en_items = load_pack("F4", "en")
    cands = [Candidate(id="passthrough", capability="F4", worker="f_passthrough", kind="method"),
             Candidate(id="overlay", capability="F4", worker="f4_overlay", kind="method",
                       params={"plate": True})]
    made = runner.generate("F4", "en", cands, en_items, check_gpu=False, log=lambda _: None)
    assert all(v == 3 for v in made.values()), made
    judges.score("F4", "en", cands, en_items, run_models=False, check_gpu=False,
                 log=lambda _: None)
    con = db.connect()
    vals = {(r["cand_key"], r["metric"]): r["value"] for r in db.scores(con, "F4", "en")
            if r["item"] == en_items[0].id}
    con.close()
    k_pass, k_ovl = cands[0].key, cands[1].key
    assert vals[(k_pass, "psnr_outside")] > 35 and vals[(k_ovl, "psnr_outside")] > 30
    template = next(Path(paths.outputs_dir()).rglob("*.template.json"))
    assert json.loads(template.read_text())["cues"][0]["tgt_text"] == "HELLO"


def test_f2_from_f1_and_model_free_f2_f3_workers(arena):
    from bench.arena.builders.f1_selfreenact import f1_selfreenact
    from bench.arena.builders.f2_from_f1 import f2_from_f1

    media = arena / "media"
    _media(media)
    f1_selfreenact("en", n=2, media_dir=str(media), download=False)
    f1_items = [it for it in load_pack("F1", "en") if it.meta["kind"] == "self"]
    # fake a cached F1 output: the "lip-synced" clip = the original with a grey mouth box
    gen = Candidate(id="latentsync-1.6", capability="F1", worker="f1_latentsync")
    con = db.connect()
    for it in f1_items:
        prefix = runner.output_prefix(gen, it)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        video = prefix.with_suffix(".mp4")
        _ff("-i", it.inputs["video"], "-vf", "drawbox=x=60:y=70:w=40:h=24:color=gray:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video))
        prefix.with_suffix(".json").write_text(json.dumps({"files": {"video": str(video)}}))
        db.put_output(con, key=runner.output_key(gen, it), capability="F1", lang="en",
                      candidate=gen.id, cand_key=gen.key, item=it.id,
                      input_hash=it.input_hash(), path=db.rel_output(prefix), status="ok")
    con.commit()
    con.close()
    f2_from_f1("en")
    f2_items = load_pack("F2", "en")
    assert len(f2_items) == 2 and all(Path(it.inputs["ref_video"]).exists() for it in f2_items)
    cands = [Candidate(id="passthrough", capability="F2", worker="f_passthrough", kind="method"),
             Candidate(id="annulus", capability="F2", worker="f2_annulus", kind="method",
                       params={"ring": 8})]
    made = runner.generate("F2", "en", cands, f2_items, check_gpu=False, log=lambda _: None)
    assert all(v == 2 for v in made.values()), made
    judges.score("F2", "en", cands, f2_items, run_models=False, check_gpu=False,
                 log=lambda _: None)
    con = db.connect()
    psnr = {r["cand_key"]: r["value"] for r in db.scores(con, "F2", "en")
            if r["metric"] == "psnr" and r["item"] == f2_items[0].id}
    con.close()
    assert set(psnr) == {cands[0].key, cands[1].key}

    # F3 decline on an F3-shaped item: picture unchanged, cues exported
    v = _video(arena / "anim.mp4", 3.0)
    dub = _wav(arena / "dub.wav", 3.0, bursts=[(0.5, 2.0)])
    src = _wav(arena / "src.wav", 3.0, bursts=[(1.0, 2.5)])
    write_pack("F3", "en", [{"id": "a1", "inputs": {"video": str(v), "audio": str(dub),
                                                     "src_audio": str(src)},
                             "meta": {"duration_s": 3.0}}], "# t\n")
    f3_items = load_pack("F3", "en")
    dec = Candidate(id="decline", capability="F3", worker="f3_decline", kind="method")
    assert runner.generate("F3", "en", [dec], f3_items, check_gpu=False,
                           log=lambda _: None)[dec.key] == 1
    judges.score("F3", "en", [dec], f3_items, run_models=False, check_gpu=False,
                 log=lambda _: None)
    con = db.connect()
    rows = {r["metric"]: r["value"] for r in db.scores(con, "F3", "en")}
    con.close()
    assert rows["changed_frac"] < 0.05 and rows["cue_f1"] > 0.9 and 0.2 < rows["flap_f1"] < 0.9
