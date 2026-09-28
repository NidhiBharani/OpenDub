"""SPY×FAMILY SCENE builder + scene_projects config: model-free, network-free, ffmpeg-free."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.arena import scene_projects
from bench.arena.builders import BUILDERS
from bench.arena.builders import spyfamily as sf
from bench.arena.packs import load_pack

RATE = sf.RATE


def _onset_signal(seconds: float, seed: int = 0) -> np.ndarray:
    """Noise plus sparse random onsets, like a 100 Hz spectral-flux envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    x = rng.random(n) * 0.3
    hits = rng.choice(n, size=n // 25, replace=False)
    x[hits] += 2 + rng.random(len(hits)) * 3
    return x


@pytest.mark.parametrize("lag_s", [0.0, 1.37, -2.5, 12.0])
def test_estimate_offset_recovers_shift(lag_s):
    a = _onset_signal(200)
    k = round(lag_s * RATE)
    # b[t + lag] == a[t]: b is a delayed (lag > 0) or advanced (lag < 0) copy, plus its own noise
    b = np.roll(a, k) + np.random.default_rng(1).random(len(a)) * 0.3
    lag, conf = sf.estimate_offset(a, b, start=5000, win=3000, search=2000)
    assert lag == pytest.approx(lag_s, abs=0.011)
    assert conf > 10


def test_estimate_offset_low_confidence_on_unrelated_signal():
    _, conf = sf.estimate_offset(_onset_signal(100, 1), _onset_signal(100, 2), start=2000,
                                 win=3000, search=1500)
    assert conf < 10


def test_checkpoints_detect_drift():
    a = _onset_signal(600)
    b = a.copy()
    b[30000:] = np.roll(a, 150)[30000:]  # the dub gains 1.5 s halfway through (an edit)
    cps = sf.offset_checkpoints(a, b, win_s=30, step_s=60, search_s=20)
    early = [c["offset_s"] for c in cps if c["t"] + 30 < 300]
    late = [c["offset_s"] for c in cps if c["t"] >= 300]
    assert all(v == pytest.approx(0, abs=0.02) for v in early)
    assert all(v == pytest.approx(1.5, abs=0.02) for v in late)
    assert sf.robust_offset(cps)["drift_s"] == pytest.approx(1.5, abs=0.03)


def test_band_energies_and_envelope_on_audio():
    sr = sf.SR
    t = np.arange(sr * 3) / sr
    x = (0.01 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
    x[sr:sr + 800] += np.sin(2 * np.pi * 1000 * t[:800]).astype(np.float32)  # a click at 1 s
    env = sf.onset_envelope(sf.band_energies(x))
    assert abs(len(env) - 3 * RATE) <= 5
    assert abs(int(np.argmax(env)) - RATE) <= 3


def test_parse_silences():
    log = ("[silencedetect @ 0x1] silence_start: -0.01\n[silencedetect @ 0x1] silence_end: 2.2 |"
           " silence_duration: 2.2\nsilence_start: 18.08\nsilence_end: 19.25 | x\n"
           "silence_start: 1400.5\n")  # trailing silence without an end is dropped
    assert sf.parse_silences(log) == [(0.0, 2.2), (18.08, 19.25)]


def test_dialogue_profile_songs_and_segments():
    rng = np.random.default_rng(0)
    secs = 900
    bed = rng.normal(0, 3, (secs * RATE, 20))
    ja, en, hi = bed.copy(), bed.copy(), bed.copy()
    talk = np.zeros(secs, bool)
    for a, b in [(100, 260), (400, 560)]:
        talk[a:b] = True
    for arr, seed in ((ja, 1), (en, 2), (hi, 3)):  # each version's own dialogue on top
        r = np.random.default_rng(seed)
        for s in np.flatnonzero(talk):
            arr[s * RATE:(s + 1) * RATE, 3:17] += r.normal(10, 6, (RATE, 14))
    score, loudest = sf.dialogue_profile(ja, [en, hi])
    assert score[talk].mean() > 3 * score[~talk].mean()
    songs = sf.shared_runs(loudest, thresh_db=1.0, min_s=10, merge_s=30, pad_s=5)
    assert songs[0][0] == 0 and songs[-1][1] == secs
    segs = sf.choose_segments(score, loudest, [(259.0, 260.6)], n=2, length_s=120,
                              exclude=[(700, 800)])
    assert len(segs) == 2
    for a, b in segs:
        assert 60 <= b - a <= 180
        assert talk[int(a) + 10:int(b) - 10].mean() > 0.9
    assert segs[0][1] - segs[0][0] <= 180 and segs[1][0] - segs[0][1] >= 30


@pytest.fixture()
def fake_media(tmp_path, monkeypatch):
    """Monkeypatch every network/ffmpeg step: 'videos' are .npy-like stubs, cuts are files."""
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    sr = 8000
    rng = np.random.default_rng(0)
    secs = 600
    bed = (rng.standard_normal(secs * sr) * 0.05).astype(np.float32)
    onsets = rng.choice(secs * sr - 400, size=secs * 3, replace=False)
    for o in onsets:
        bed[o:o + 400] += np.hanning(400).astype(np.float32) * rng.uniform(0.3, 1.0)
    offsets = {"ja": 0.0, "en": 0.0, "hi": 2.0}  # the Hindi upload starts 2 s later
    audio = {}
    for lang, seed in (("ja", 1), ("en", 2), ("hi", 3)):
        x = bed.copy()
        r = np.random.default_rng(seed)
        for a, b in [(150, 290), (380, 520)]:  # dialogue scenes (different per version)
            x[a * sr:b * sr] += (r.standard_normal((b - a) * sr) * 0.4).astype(np.float32)
        shift = int(offsets[lang] * sr)
        audio[lang] = np.concatenate([np.zeros(shift, np.float32), x])
    calls: dict[str, list] = {"download": [], "cut": []}

    def fake_download(url, dest):
        calls["download"].append(url)
        if "dXgq4u3ViFs" in url:
            raise RuntimeError("geo-blocked")  # one dub unavailable: must continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")
        return dest

    def fake_decode(path, sr_=sf.SR):
        lang = Path(path).stem.split("-")[-1]
        x = audio[lang]
        idx = np.arange(0, len(x), sr / sr_).astype(int)
        return x[idx[idx < len(x)]]

    def fake_cut(kind):
        def cut(src, dest, start, duration):
            calls["cut"].append((kind, Path(src).name, round(start, 3), round(duration, 3)))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"clip")
            return dest
        return cut

    monkeypatch.setattr(sf, "download", fake_download)
    monkeypatch.setattr(sf, "decode_mono", fake_decode)
    monkeypatch.setattr(sf, "detect_silences", lambda path: [(289.5, 290.5), (519.4, 520.2)])
    monkeypatch.setattr(sf, "cut_video", fake_cut("video"))
    monkeypatch.setattr(sf, "cut_audio", fake_cut("audio"))
    # the pack-local hi file exists already (e.g. fetched by hand) so both dubs align
    hi = sf.src_dir() / "ep1-hi.mp4"
    hi.parent.mkdir(parents=True, exist_ok=True)
    hi.write_bytes(b"fake")
    return calls


def test_builder_writes_aligned_pack(fake_media):
    assert "spyfamily" in BUILDERS and BUILDERS["spyfamily"].capabilities == ["SCENE"]
    root = sf.spyfamily("ja", n=300, length_s=120, log=lambda _: None)  # --n 300 is capped
    assert len(fake_media["download"]) == 2  # ja + en; hi was already present
    analysis = json.loads((sf.src_dir() / "analysis.json").read_text())
    assert analysis["offsets"]["en"]["offset_s"] == pytest.approx(0.0, abs=0.02)
    assert analysis["offsets"]["hi"]["offset_s"] == pytest.approx(2.0, abs=0.02)

    items = load_pack("SCENE", "ja")
    assert 1 <= len(items) <= 4
    lic = (root / "LICENSE.md").read_text()
    assert "private evaluation only" in lic.lower() and "h_tL16PZ0IE" in lic
    for it in items:
        assert it.group == "spyfamily-ep1"
        assert Path(it.inputs["video"]).is_file() and Path(it.inputs["audio"]).is_file()
        assert set(it.refs) == {"dub_en_audio", "dub_hi_audio"}
        m = it.meta
        assert m["end_s"] - m["start_s"] == pytest.approx(m["duration_s"], abs=0.01)
        assert 60 <= m["duration_s"] <= 180
        assert m["offsets"]["hi"] == pytest.approx(2.0, abs=0.02)
        assert m["sources"]["en"].endswith("ZgyothCzhLA")
        # the hi reference is cut at start + offset, same duration
        hi_cuts = [c for c in fake_media["cut"] if c[1] == "ep1-hi.mp4"]
        assert any(c[2] == pytest.approx(m["start_s"] + 2.0, abs=0.02)
                   and c[3] == pytest.approx(m["duration_s"]) for c in hi_cuts)
    starts = sorted(it.meta["start_s"] for it in items)
    assert any(140 <= s <= 170 for s in starts) and any(370 <= s <= 400 for s in starts)

    # cached analysis, existing clips: a rebuild cuts nothing and keeps transcripts
    manifest = root / "manifest.jsonl"
    rows = [json.loads(ln) for ln in manifest.read_text().splitlines()]
    rows[0]["refs"]["dub_en_text"] = "Hello."
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_cuts = len(fake_media["cut"])
    sf.spyfamily("ja", n=4, log=lambda _: None)
    assert len(fake_media["cut"]) == n_cuts
    assert load_pack("SCENE", "ja")[0].refs["dub_en_text"] == "Hello."


def test_builder_requires_the_original(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDUB_ARENA_DATA", str(tmp_path))
    with pytest.raises(RuntimeError, match="Japanese original"):
        sf.spyfamily("ja", download_missing=False, log=lambda _: None)
    with pytest.raises(ValueError):
        sf.spyfamily("en")


def test_transcribe_refs_uses_worker_job(fake_media, monkeypatch):
    from bench.arena import runner

    root = sf.spyfamily("ja", n=2, log=lambda _: None)
    seen = []

    def fake_job(*, worker, env, params, lang, items, label, **kw):
        seen.append((worker, env, lang, params["max_new_tokens"], len(items)))
        recs = {}
        for it in items:
            assert Path(it["inputs"]["audio"]).name.endswith(f".{lang}.wav")
            Path(it["out"]).with_suffix(".json").write_text(
                json.dumps({"text": f" {lang} words ", "segments": []}))
            recs[it["id"]] = {"id": it["id"], "status": "ok"}
        return runner.JobResult(job="j", records=recs)

    monkeypatch.setattr(runner, "run_worker_job", fake_job)
    monkeypatch.setattr(runner, "ensure_gpu_room", lambda *a: None)
    sf.transcribe_refs(root, candidate="qwen3-asr-1.7b", log=lambda _: None)
    assert [s[:4] for s in seen] == [("asr_qwen3", "qwen3asr", "en", 2048),
                                     ("asr_qwen3", "qwen3asr", "hi", 2048)]
    it = load_pack("SCENE", "ja")[0]
    assert it.refs["dub_en_text"] == "en words" and it.refs["dub_hi_text"] == "hi words"
    assert it.meta["ref_transcripts"]["en"].startswith("qwen3-asr-1.7b@")


def test_scene_projects_config(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "preset": {"preset": "balanced", "runtime": "local"},
        "pipeline": {"tts": {"provider_id": "tts.a"}},
        "per_target": {"hi": {"pipeline": {"tts": {"provider_id": "tts.b"}},
                              "capabilities": {"C2": {"enabled": True}}}}}))
    cfg = scene_projects.load_config(p)
    assert cfg["name"] == "cfg"
    assert scene_projects.config_for(cfg, "en")["pipeline"] == {"tts": {"provider_id": "tts.a"}}
    hi = scene_projects.config_for(cfg, "hi")
    assert hi["pipeline"]["tts"]["provider_id"] == "tts.b" and hi["capabilities"]
    assert hi["preset"]["runtime"] == "local" and hi["stages"] is None
    p.write_text(json.dumps({"presets": {}}))
    with pytest.raises(ValueError, match="unknown key"):
        scene_projects.load_config(p)
