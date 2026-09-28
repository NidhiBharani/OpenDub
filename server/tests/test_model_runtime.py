"""Model lifecycle: the ModelManager and the pipeline's load/unload policy, with fake loaders
(no GPU, no ML packages, no network)."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time

import httpx
import pytest

from app import store
from app.models import Job, Project
from app.pipeline import orchestrator
from app.pipeline import stages as stage_impl
from app.providers import _runtime
from app.providers._runtime import ModelManager, key_str


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Loads:
    """Counts loads/unloads per key; each model is a fresh object."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.unloaded: list[str] = []

    def loader(self, name: str):
        def load():
            self.loaded.append(name)
            return {"name": name}
        return load

    def unloader(self, model) -> None:
        self.unloaded.append(model["name"])


def make(**kwargs) -> tuple[ModelManager, FakeClock, Loads]:
    clock = FakeClock()
    kwargs.setdefault("idle_ttl_s", 60.0)
    return ModelManager(clock=clock, reaper=False, **kwargs), clock, Loads()


def test_refcount_keeps_model_loaded_until_last_release() -> None:
    mm, _clock, loads = make(idle_ttl_s=0)
    key = ("asr.fake", "m")
    with mm.acquire(key, loads.loader("a"), unloader=loads.unloader) as outer:
        with mm.acquire(key, loads.loader("a"), unloader=loads.unloader) as inner:
            assert inner is outer
            assert mm.snapshot()[0]["refcount"] == 2
        assert loads.unloaded == []  # still used by the outer holder
        assert mm.loaded_keys() == [key_str(key)]
    assert loads.loaded == ["a"]
    assert loads.unloaded == ["a"]  # TTL 0: gone as soon as the last user released
    assert mm.snapshot() == []


def test_concurrent_first_use_loads_once() -> None:
    mm, _clock, _loads = make()
    calls = []
    gate = threading.Event()

    def slow_load():
        calls.append(1)
        gate.wait(2)
        return object()

    seen = []

    def worker():
        with mm.acquire(("tts.fake",), slow_load) as model:
            seen.append(model)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join(2)
    assert len(calls) == 1
    assert len(seen) == 4 and all(m is seen[0] for m in seen)


def test_idle_ttl_unloads_after_expiry() -> None:
    mm, clock, loads = make(idle_ttl_s=10)
    with mm.acquire(("asr.fake",), loads.loader("a"), unloader=loads.unloader):
        clock.now += 100  # in use: never expires, however long
        assert mm.reap() == []
    clock.now += 5
    assert mm.reap() == []
    assert mm.snapshot()[0]["idle_s"] == 5.0
    clock.now += 6
    assert mm.reap() == ["asr.fake"]
    assert loads.unloaded == ["a"]
    # Next use loads again.
    with mm.acquire(("asr.fake",), loads.loader("a")):
        pass
    assert loads.loaded == ["a", "a"]


def test_per_acquire_ttl_overrides_default() -> None:
    mm, clock, loads = make(idle_ttl_s=1000)
    with mm.acquire(("asr.fake",), loads.loader("a"), idle_ttl_s=1, unloader=loads.unloader):
        pass
    clock.now += 2
    assert mm.reap() == ["asr.fake"]


def test_reaper_thread_unloads_idle_models() -> None:
    mm = ModelManager(idle_ttl_s=0.05)
    loads = Loads()
    with mm.acquire(("asr.fake",), loads.loader("a"), unloader=loads.unloader):
        pass
    deadline = time.monotonic() + 3
    while loads.unloaded != ["a"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert loads.unloaded == ["a"]
    assert mm.snapshot() == []


def test_lru_eviction_under_gpu_budget() -> None:
    mm, clock, loads = make(gpu_budget_gb=4.0)
    for name in ("a", "b"):
        with mm.acquire(("tts.fake", name), loads.loader(name), est_vram_gb=2.0,
                        unloader=loads.unloader):
            clock.now += 1
    with mm.acquire(("tts.fake", "a"), loads.loader("a"), est_vram_gb=2.0):
        clock.now += 1  # 'a' is now the most recently used
    with mm.acquire(("tts.fake", "c"), loads.loader("c"), est_vram_gb=2.0,
                    unloader=loads.unloader):
        pass
    assert loads.unloaded == ["b"]  # least recently used goes first
    assert sorted(mm.loaded_keys()) == ["tts.fake|a", "tts.fake|c"]
    # A CPU model (0 GB) never evicts anything.
    with mm.acquire(("quality.fake",), loads.loader("cpu"), est_vram_gb=0.0):
        pass
    assert loads.unloaded == ["b"]


def test_models_in_use_are_never_evicted() -> None:
    mm, _clock, loads = make(gpu_budget_gb=3.0)
    with mm.acquire(("tts.fake", "a"), loads.loader("a"), est_vram_gb=2.0,
                    unloader=loads.unloader) as a:
        with mm.acquire(("tts.fake", "b"), loads.loader("b"), est_vram_gb=2.0):
            assert a == {"name": "a"}  # over budget, but 'a' is busy: both stay
        assert loads.unloaded == []


def test_hold_keeps_models_warm_for_the_scope() -> None:
    mm, _clock, loads = make(idle_ttl_s=0)
    with mm.hold():
        for _ in range(3):
            with mm.acquire(("tts.fake",), loads.loader("a"), unloader=loads.unloader):
                pass
        assert loads.loaded == ["a"]
        assert mm.snapshot()[0]["held"] is True
    assert loads.unloaded == ["a"]


async def test_hold_reaches_worker_threads() -> None:
    mm, _clock, loads = make(idle_ttl_s=0)

    def use():
        with mm.acquire(("asr.fake",), loads.loader("a"), unloader=loads.unloader):
            pass

    async with mm.hold_async():
        await asyncio.to_thread(use)
        await asyncio.to_thread(use)
        assert loads.unloaded == []
    assert loads.loaded == ["a"] and loads.unloaded == ["a"]


def test_failing_loader_does_not_leak_entries() -> None:
    mm, _clock, loads = make()

    def broken():
        raise RuntimeError("no weights")

    with mm.hold(), pytest.raises(RuntimeError, match="no weights"), \
            mm.acquire(("asr.fake",), broken, est_vram_gb=2.0):
        pass
    assert mm.snapshot() == []
    with mm.acquire(("asr.fake",), loads.loader("a")) as model:  # a later load still works
        assert model == {"name": "a"}


def test_waiters_on_a_failed_load_get_the_error() -> None:
    mm, _clock, _loads = make()
    gate = threading.Event()

    def broken():
        gate.wait(2)
        raise ValueError("boom")

    errors = []

    def worker():
        try:
            with mm.acquire(("asr.fake",), broken):
                pass
        except ValueError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join(2)
    assert len(errors) == 3
    assert mm.snapshot() == []


def test_unload_busy_model_is_deferred_until_release() -> None:
    mm, _clock, loads = make()
    with mm.acquire(("tts.fake",), loads.loader("a"), unloader=loads.unloader):
        assert mm.unload("tts.fake") is False
        assert loads.unloaded == []
    assert loads.unloaded == ["a"]


def test_unload_all_and_snapshot_fields() -> None:
    mm, clock, loads = make()
    with mm.acquire(("tts.fake",), loads.loader("a"), est_vram_gb=1.5, unloader=loads.unloader):
        clock.now += 3
    (row,) = mm.snapshot()
    assert row["key"] == "tts.fake" and row["family"] == "tts" and row["state"] == "loaded"
    assert row["refcount"] == 0 and row["age_s"] == 3.0 and row["est_vram_gb"] == 1.5
    assert row["last_used"] is not None
    assert mm.unload_all() == ["tts.fake"]
    assert loads.unloaded == ["a"]


def test_ctranslate2_models_are_unloaded_and_torch_is_not_imported() -> None:
    torch_before = "torch" in sys.modules
    calls = []

    class FakeCT2:
        def unload_model(self):
            calls.append("unload_model")

    class FakeWhisperModel:  # faster-whisper keeps its CTranslate2 model at .model
        def __init__(self):
            self.model = FakeCT2()

    mm, _clock, _loads = make(idle_ttl_s=0)
    with mm.acquire(("asr.faster_whisper", "tiny", "cpu", "int8"), FakeWhisperModel):
        pass
    assert calls == ["unload_model"]
    assert ("torch" in sys.modules) == torch_before


async def test_cancelled_async_acquire_gives_its_reference_back() -> None:
    mm, _clock, loads = make(idle_ttl_s=0)
    gate = threading.Event()

    def slow():
        gate.wait(2)
        return loads.loader("a")()

    async def user():
        async with mm.acquire_async(("tts.fake",), slow, unloader=loads.unloader):
            pass

    task = asyncio.create_task(user())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    for _ in range(100):
        if loads.unloaded:
            break
        await asyncio.sleep(0.02)
    assert loads.unloaded == ["a"]
    assert mm.snapshot() == []


# ---- pipeline policy ----------------------------------------------------------------------------


@pytest.fixture
def managed(monkeypatch):
    """Swap the process-wide manager for an isolated one (long TTL: only policy unloads)."""
    mm, _clock, loads = make(idle_ttl_s=1000)
    monkeypatch.setattr(orchestrator, "manager", mm)
    return mm, loads


def _stage(mm: ModelManager, loads: Loads, key: tuple, fail: bool = False):
    async def run(project, job, progress):
        def use():
            with mm.acquire(key, loads.loader(key[0]), unloader=loads.unloader):
                pass
        await asyncio.to_thread(use)
        await asyncio.to_thread(use)
        if fail:
            raise RuntimeError("stage broke")
        return "ok"
    return run


async def _run_job(monkeypatch, mm, loads, stages: dict[str, tuple], fail: str | None = None):
    project = Project(name="lifecycle")
    store.save(project)
    for key, model_key in stages.items():
        monkeypatch.setitem(stage_impl.STAGE_RUNNERS, key,
                            _stage(mm, loads, model_key, fail=key == fail))
    job = Job(project_id=project.id, kind="pipeline", stages=list(stages))
    runner = orchestrator._pipeline_runner(project.id, {k: "pending" for k in stages})
    return project, job, runner


async def test_stage_boundary_releases_models_next_stage_does_not_use(monkeypatch, managed) -> None:
    mm, loads = managed
    unloaded_at_boundary = []
    original = orchestrator._release_models

    async def spy(next_stage, **kwargs):
        await original(next_stage, **kwargs)
        unloaded_at_boundary.append((next_stage, list(loads.unloaded)))

    monkeypatch.setattr(orchestrator, "_release_models", spy)
    _project, job, runner = await _run_job(monkeypatch, mm, loads, {
        "transcribe": ("asr.fake",), "translate": ("llm.ollama", "http://x", "m"),
        "synthesize": ("tts.fake",),
    })
    await runner(job)
    # Loaded once per stage despite two uses; whisper gone before translate, the LLM gone
    # before synthesize, and everything gone when the job ends.
    assert loads.loaded == ["asr.fake", "llm.ollama", "tts.fake"]
    assert unloaded_at_boundary[0] == ("translate", ["asr.fake"])
    assert unloaded_at_boundary[1] == ("synthesize", ["asr.fake", "llm.ollama"])
    assert unloaded_at_boundary[-1][0] is None
    assert mm.snapshot() == []


async def test_models_kept_across_boundary_when_next_stage_uses_them(monkeypatch, managed) -> None:
    mm, loads = managed
    _project, job, runner = await _run_job(monkeypatch, mm, loads, {
        "synthesize": ("quality.fake",), "mix": ("quality.fake",),
    })
    await runner(job)
    assert loads.loaded == ["quality.fake"]  # shared by synthesize and mix, loaded once
    assert loads.unloaded == ["quality.fake"]  # and released at job end


async def test_failing_stage_still_unloads_at_job_end(monkeypatch, managed) -> None:
    mm, loads = managed
    _project, job, runner = await _run_job(
        monkeypatch, mm, loads, {"transcribe": ("asr.fake",)}, fail="transcribe"
    )
    with pytest.raises(RuntimeError, match="stage 'transcribe' failed"):
        await runner(job)
    assert loads.unloaded == ["asr.fake"]
    assert mm.snapshot() == []


async def test_job_end_spares_models_another_job_is_using(monkeypatch, managed) -> None:
    mm, loads = managed
    other = ("tts.other",)
    ctx = mm.acquire(other, loads.loader("other"), unloader=loads.unloader)
    ctx.__enter__()  # a concurrent job's model, mid-use
    _project, job, runner = await _run_job(monkeypatch, mm, loads, {"transcribe": ("asr.fake",)})
    await runner(job)
    assert loads.unloaded == ["asr.fake"]
    ctx.__exit__(None, None, None)  # job end marked it: it goes when its user is done
    assert loads.unloaded == ["asr.fake", "other"]


# ---- Ollama ------------------------------------------------------------------------------------


@pytest.fixture
def ollama(monkeypatch):
    requests: list[httpx.Request] = []
    state = {"is_ollama": True}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            if state["is_ollama"]:
                return httpx.Response(200, json={"models": [{"name": "gemma3:12b"}]})
            return httpx.Response(404, json={"detail": "not found"})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    mm, _clock, _loads = make(idle_ttl_s=1000)
    monkeypatch.setattr(_runtime, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(_runtime, "manager", mm)
    monkeypatch.setattr(_runtime, "_ollama_seen", {})
    return mm, requests, state


async def test_ollama_model_is_unloaded_with_keep_alive_zero(ollama) -> None:
    mm, requests, _state = ollama
    async with _runtime.llm_endpoint("http://gpu-box:8080/v1", "gemma3:12b"):
        assert mm.loaded_keys() == ["llm.ollama|http://gpu-box:8080|gemma3:12b"]
    assert [r.url.path for r in requests] == ["/api/tags"]
    mm.unload_idle()
    unload = requests[-1]
    assert unload.method == "POST" and str(unload.url) == "http://gpu-box:8080/api/generate"
    assert json.loads(unload.content) == {"model": "gemma3:12b", "keep_alive": 0}


async def test_ollama_default_port_skips_the_probe(ollama) -> None:
    mm, requests, _state = ollama
    async with _runtime.llm_endpoint("http://localhost:11434/v1", "qwen2.5:14b"):
        pass
    assert requests == []
    assert mm.unload_all() == ["llm.ollama|http://localhost:11434|qwen2.5:14b"]
    assert [r.url.path for r in requests] == ["/api/generate"]


async def test_non_ollama_servers_and_opt_out_are_no_ops(ollama) -> None:
    mm, requests, state = ollama
    state["is_ollama"] = False
    async with _runtime.llm_endpoint("http://vllm:8000/v1", "qwen"):
        assert mm.snapshot() == []
    async with _runtime.llm_endpoint("http://localhost:11434/v1", "m", unload_after_use=False):
        assert mm.snapshot() == []
    assert [r.url.path for r in requests] == ["/api/tags"]  # probed once, nothing unloaded


def test_ollama_unload_is_best_effort(monkeypatch) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(_runtime, "_TRANSPORT", httpx.MockTransport(down))
    assert _runtime.ollama_unload("http://localhost:11434", "m") is False


def test_server_root_strips_the_openai_suffix() -> None:
    assert _runtime.server_root("http://localhost:11434/v1/") == "http://localhost:11434"
    assert _runtime.server_root("https://h/proxy/v1") == "https://h/proxy"
