"""Process-wide model lifecycle: load on demand, share while in use, unload when idle.

Every in-process model (faster-whisper, F5-TTS, pyannote, the quality verifier / emotion /
speaker models) is obtained through :data:`manager`::

    with manager.acquire(key, loader, est_vram_gb=1.5) as model:
        model.transcribe(...)

- The first ``acquire`` of a key runs ``loader()``; concurrent users of the same key share the
  instance (refcounted) and a second caller waits for an in-flight load instead of loading twice.
- When the refcount drops to 0 the model stays warm for ``idle_ttl_s`` (settings
  ``models.idle_ttl_s``, default 60 s; 0 = unload as soon as the last user releases it), then the
  reaper thread unloads it.
- A load that would push the summed ``est_vram_gb`` of loaded models past ``models.gpu_budget_gb``
  (default: total VRAM minus 1 GB, or no limit when it can't be detected) first evicts idle
  models, least recently used first. Models in use are never evicted.
- ``hold()`` scopes (one per pipeline stage / segment job) keep every model acquired inside them
  warm until the scope ends, whatever the TTL, so a stage doesn't reload its model per line. Holds
  are soft: a held model that is not in use can still be evicted for room or unloaded on request.
- Unloading calls the optional ``unloader``, ``unload_model()`` on CTranslate2 models, drops the
  manager's reference, runs ``gc.collect()`` and, only if torch is already imported, empties the
  CUDA cache. Nothing here imports torch or any ML package.

Keys are tuples whose first element names the model family and provider, e.g.
``("asr.faster_whisper", "large-v3-turbo", "cuda", "float16")``. They must contain everything that
changes the weights, and never secrets: keys are listed by ``GET /api/system/models``.

OpenAI-compatible LLM servers are covered too: when the server is Ollama, :func:`llm_endpoint`
registers the (server, model) pair as an entry whose unloader asks Ollama to evict the model
(``keep_alive: 0``), so it follows the same TTL / stage-boundary / job-end policy as local models.

Like the job engine and store, this is a process-wide singleton: models are a process resource.
"""
from __future__ import annotations

import asyncio
import contextvars
import gc
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Hashable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .. import config

log = logging.getLogger(__name__)

DEFAULT_IDLE_TTL_S = 60.0
GPU_RESERVE_GB = 1.0  # headroom left for CUDA contexts / other processes by the auto budget

Key = tuple[Hashable, ...]
Loader = Callable[[], Any]
Unloader = Callable[[Any], None]


def key_str(key: Key) -> str:
    """Printable form of a key, used by the snapshot and the unload-by-key API."""
    return "|".join(str(part) for part in key)


def family(key: Key) -> str:
    """'asr' for ('asr.faster_whisper', ...): the provider kind, used by the stage policy."""
    return str(key[0]).split(".", 1)[0] if key else ""


# ---- settings ---------------------------------------------------------------------------------


def model_settings() -> dict[str, Any]:
    """The ``models`` settings section, with OPENDUB_MODELS_<KEY> env overrides applied."""
    try:
        out = dict(config.load_settings().get("models") or {})
    except Exception:  # noqa: BLE001 - a broken settings file must not break model loading
        out = {}
    for name in ("idle_ttl_s", "gpu_budget_gb"):
        value = os.environ.get(f"OPENDUB_MODELS_{name.upper()}")
        if value:
            out[name] = value
    return out


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def default_idle_ttl_s() -> float:
    ttl = _as_float(model_settings().get("idle_ttl_s"))
    return DEFAULT_IDLE_TTL_S if ttl is None else max(0.0, ttl)


_total_vram: list[float | None] = []  # detected once: [GB] or [None]


def _nvidia_smi(*query: str) -> list[list[str]]:
    """Rows of `nvidia-smi --query-... --format=csv,noheader,nounits`, or [] if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[cell.strip() for cell in line.split(",")] for line in out.splitlines() if line.strip()]


def total_vram_gb() -> float | None:
    """Total memory of GPU 0: via torch when it is already imported, else nvidia-smi."""
    if not _total_vram:
        total: float | None = None
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:  # noqa: BLE001 - fall through to nvidia-smi
            total = None
        if total is None:
            rows = _nvidia_smi("--query-gpu=memory.total")
            total = _as_float(rows[0][0]) if rows and rows[0] else None
            total = total / 1024 if total else None  # MiB → GiB; "[N/A]" (unified memory) → None
        _total_vram.append(total)
    return _total_vram[0]


def default_gpu_budget_gb() -> float | None:
    """``models.gpu_budget_gb`` when set (> 0), else total VRAM minus the reserve, else None."""
    budget = _as_float(model_settings().get("gpu_budget_gb"))
    if budget is not None and budget > 0:
        return budget
    total = total_vram_gb()
    return max(0.0, total - GPU_RESERVE_GB) if total else None


def gpu_usage() -> dict[str, Any]:
    """nvidia-smi totals plus this process's own VRAM, for the system API. {} without a GPU."""
    gpus = [
        {"index": _as_float(r[0]), "name": r[1], "memory_total_mb": _as_float(r[2]),
         "memory_used_mb": _as_float(r[3])}
        for r in _nvidia_smi("--query-gpu=index,name,memory.total,memory.used") if len(r) >= 4
    ]
    if not gpus:
        return {}
    pid = str(os.getpid())
    mine = [_as_float(r[1]) or 0.0
            for r in _nvidia_smi("--query-compute-apps=pid,used_memory") if len(r) >= 2 and r[0] == pid]
    return {"gpus": gpus, "process_used_mb": sum(mine)}


# ---- freeing memory ---------------------------------------------------------------------------


def _release_native(model: Any) -> None:
    """Free native weights eagerly where the library offers it: CTranslate2 models (and
    faster-whisper's WhisperModel, which wraps one as ``.model``) have ``unload_model()``."""
    for target in (model, getattr(model, "model", None)):
        unload = getattr(target, "unload_model", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:  # noqa: BLE001 - dropping the reference still frees it
                log.debug("unload_model() failed: %s", exc)
            return


def free_memory() -> None:
    """Collect garbage and hand cached CUDA blocks back to the driver. Never imports torch."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # noqa: BLE001 - best effort
        log.debug("emptying the CUDA cache failed: %s", exc)


# ---- manager ----------------------------------------------------------------------------------


@dataclass(eq=False)
class _Entry:
    key: Key
    est_vram_gb: float | None
    idle_ttl_s: float
    unloader: Unloader | None
    model: Any = None
    loaded: bool = False
    error: BaseException | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    refs: int = 0  # active users (hard: never evicted while > 0)
    holds: int = 0  # hold() scopes keeping it warm (soft)
    evict_on_release: bool = False  # unload as soon as it is fully idle
    loaded_at: float = 0.0  # monotonic
    last_used: float = 0.0  # monotonic
    last_used_wall: float = 0.0


class _Hold:
    def __init__(self) -> None:
        self.entries: set[_Entry] = set()
        self.closed = False  # an abandoned worker thread may acquire after the scope ended


_current_hold: contextvars.ContextVar[_Hold | None] = contextvars.ContextVar(
    "opendub_model_hold", default=None
)


class ModelManager:
    def __init__(
        self,
        *,
        idle_ttl_s: float | None = None,
        gpu_budget_gb: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        reaper: bool = True,
    ) -> None:
        # Explicit values are for tests; None reads the settings on every use.
        self._idle_ttl_s = idle_ttl_s
        self._gpu_budget_gb = gpu_budget_gb
        self._clock = clock
        self._reaper_enabled = reaper
        self._reaper: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._entries: dict[Key, _Entry] = {}

    # -- acquire / release --------------------------------------------------------------------

    @contextmanager
    def acquire(
        self,
        key: Key,
        loader: Loader,
        *,
        est_vram_gb: float | None = None,
        idle_ttl_s: float | None = None,
        unloader: Unloader | None = None,
    ) -> Iterator[Any]:
        """Yield the model for ``key``, loading it with ``loader()`` on first use. Blocking:
        call from a worker thread (or use :meth:`acquire_async`)."""
        entry = self._enter(key, loader, est_vram_gb, idle_ttl_s, unloader)
        try:
            yield entry.model
        finally:
            self._exit(entry)

    @asynccontextmanager
    async def acquire_async(
        self,
        key: Key,
        loader: Loader,
        *,
        est_vram_gb: float | None = None,
        idle_ttl_s: float | None = None,
        unloader: Unloader | None = None,
    ) -> AsyncIterator[Any]:
        """:meth:`acquire` for async code: loading and unloading run in a worker thread."""
        entering = asyncio.ensure_future(asyncio.to_thread(
            self._enter, key, loader, est_vram_gb, idle_ttl_s, unloader
        ))
        try:
            entry = await asyncio.shield(entering)
        except asyncio.CancelledError:
            # The load keeps running in its thread; give its reference back once it lands.
            loop = asyncio.get_running_loop()

            def give_back(done: asyncio.Future) -> None:
                if not done.cancelled() and done.exception() is None:
                    loop.run_in_executor(None, self._exit, done.result())

            entering.add_done_callback(give_back)
            raise
        try:
            yield entry.model
        finally:
            await asyncio.shield(asyncio.to_thread(self._exit, entry))

    def _enter(
        self, key: Key, loader: Loader, est_vram_gb: float | None, idle_ttl_s: float | None,
        unloader: Unloader | None,
    ) -> _Entry:
        ttl = idle_ttl_s if idle_ttl_s is not None else (
            self._idle_ttl_s if self._idle_ttl_s is not None else default_idle_ttl_s()
        )
        with self._lock:
            entry = self._entries.get(key)
            owner = entry is None
            if entry is None:
                entry = _Entry(key=key, est_vram_gb=est_vram_gb, idle_ttl_s=max(0.0, ttl),
                               unloader=unloader)
                self._entries[key] = entry
            entry.refs += 1
            entry.evict_on_release = False
        if owner:
            try:
                self._make_room(entry)
                model = loader()
            except BaseException as exc:
                with self._lock:
                    if self._entries.get(key) is entry:
                        del self._entries[key]
                    entry.error = exc
                    entry.refs = 0
                entry.ready.set()
                free_memory()  # a half-built model may have allocated before failing
                raise
            now = self._clock()
            with self._lock:
                entry.model, entry.loaded = model, True
                entry.loaded_at = entry.last_used = now
                entry.last_used_wall = time.time()
            entry.ready.set()
            log.info("loaded model %s", key_str(key))
        else:
            entry.ready.wait()
            if entry.error is not None:
                raise entry.error
        with self._lock:
            entry.last_used, entry.last_used_wall = self._clock(), time.time()
            hold = _current_hold.get()
            if hold is not None and not hold.closed and entry not in hold.entries:
                hold.entries.add(entry)
                entry.holds += 1
        self._ensure_reaper()
        return entry

    def _exit(self, entry: _Entry) -> None:
        with self._lock:
            entry.refs = max(0, entry.refs - 1)
            entry.last_used, entry.last_used_wall = self._clock(), time.time()
            now_idle = self._unload_now_if_idle(entry)
            self._cv.notify_all()
        if now_idle:
            self._finish_unload([entry])

    def _unload_now_if_idle(self, entry: _Entry) -> bool:
        """Under the lock: pop a fully idle entry that should not linger (TTL 0 or marked)."""
        if entry.refs or entry.holds or self._entries.get(entry.key) is not entry:
            return False
        if entry.idle_ttl_s <= 0 or entry.evict_on_release:
            del self._entries[entry.key]
            return True
        return False

    # -- holds --------------------------------------------------------------------------------

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Keep every model acquired in this context (and threads started from it via
        asyncio.to_thread) loaded until the context exits."""
        scope = _Hold()
        token = _current_hold.set(scope)
        try:
            yield
        finally:
            _current_hold.reset(token)
            self._end_hold(scope)

    @asynccontextmanager
    async def hold_async(self) -> AsyncIterator[None]:
        scope = _Hold()
        token = _current_hold.set(scope)
        try:
            yield
        finally:
            _current_hold.reset(token)
            await asyncio.shield(asyncio.to_thread(self._end_hold, scope))

    def _end_hold(self, scope: _Hold) -> None:
        done: list[_Entry] = []
        with self._lock:
            scope.closed = True
            for entry in scope.entries:
                entry.holds = max(0, entry.holds - 1)
                entry.last_used, entry.last_used_wall = self._clock(), time.time()
                if self._unload_now_if_idle(entry):
                    done.append(entry)
            scope.entries.clear()
            self._cv.notify_all()
        self._finish_unload(done)

    # -- unloading ----------------------------------------------------------------------------

    def unload(self, key: Key | str) -> bool:
        """Unload one model now if nobody is using it (holds are overridden). A busy model is
        marked to unload as soon as its last user releases it. Returns True if unloaded now."""
        with self._lock:
            entry = self._find(key)
            if entry is None or not entry.loaded:
                return False
            if entry.refs:
                entry.evict_on_release = True
                return False
            del self._entries[entry.key]
        self._finish_unload([entry])
        return True

    def unload_idle(self, *, keep: Callable[[Key], bool] | None = None,
                    defer_busy: bool = False, respect_holds: bool = True) -> list[str]:
        """Unload every loaded model not in use and not matched by ``keep``. With
        ``defer_busy`` the ones in use are marked to unload when they become fully idle.
        ``respect_holds`` spares models another job's stage keeps warm. Returns unloaded keys."""
        victims: list[_Entry] = []
        with self._lock:
            for entry in list(self._entries.values()):
                if not entry.loaded or (keep is not None and keep(entry.key)):
                    continue
                if entry.refs or (entry.holds and respect_holds):
                    if defer_busy:
                        entry.evict_on_release = True
                    continue
                del self._entries[entry.key]
                victims.append(entry)
        self._finish_unload(victims)
        return [key_str(e.key) for e in victims]

    def unload_all(self) -> list[str]:
        """Unload everything not in use; anything still busy unloads when it is released."""
        return self.unload_idle(defer_busy=True, respect_holds=False)

    def reap(self) -> list[str]:
        """Unload models idle for longer than their TTL. Run by the reaper thread."""
        now = self._clock()
        victims: list[_Entry] = []
        with self._lock:
            for entry in list(self._entries.values()):
                if (entry.loaded and not entry.refs and not entry.holds
                        and now - entry.last_used >= entry.idle_ttl_s):
                    del self._entries[entry.key]
                    victims.append(entry)
        self._finish_unload(victims)
        return [key_str(e.key) for e in victims]

    def _finish_unload(self, entries: list[_Entry]) -> None:
        """Outside the lock: run unloaders, drop references, free memory."""
        if not entries:
            return
        for entry in entries:
            model, entry.model, entry.loaded = entry.model, None, False
            try:
                if entry.unloader is not None:
                    entry.unloader(model)
                else:
                    _release_native(model)
            except Exception as exc:  # noqa: BLE001 - an unloader failure must not leak the entry
                log.warning("unloading %s failed: %s", key_str(entry.key), exc)
            del model
            log.info("unloaded model %s", key_str(entry.key))
        free_memory()

    def _make_room(self, incoming: _Entry) -> None:
        """Evict idle models, least recently used first, until ``incoming`` fits the budget."""
        need = incoming.est_vram_gb or 0.0
        if need <= 0:
            return
        budget = self._gpu_budget_gb if self._gpu_budget_gb is not None else default_gpu_budget_gb()
        if budget is None:
            return
        victims: list[_Entry] = []
        with self._lock:
            others = [e for e in self._entries.values() if e is not incoming]
            used = sum(e.est_vram_gb or 0.0 for e in others)
            idle = sorted((e for e in others if e.loaded and not e.refs and e.est_vram_gb),
                          key=lambda e: e.last_used)
            for entry in idle:
                if used + need <= budget:
                    break
                del self._entries[entry.key]
                victims.append(entry)
                used -= entry.est_vram_gb or 0.0
        if victims:
            log.info("evicting %s to fit %s", [key_str(e.key) for e in victims],
                     key_str(incoming.key))
        self._finish_unload(victims)
        if used + need > budget:
            log.warning("loading %s (~%.1f GB) exceeds the %.1f GB model budget; models in use "
                        "can't be evicted", key_str(incoming.key), need, budget)

    def _find(self, key: Key | str) -> _Entry | None:
        if isinstance(key, str):
            return next((e for e in self._entries.values() if key_str(e.key) == key), None)
        return self._entries.get(key)

    # -- introspection ------------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            return [
                {
                    "key": key_str(e.key),
                    "family": family(e.key),
                    "state": "loaded" if e.loaded else "loading",
                    "refcount": e.refs,
                    "held": e.holds > 0,
                    "age_s": round(now - e.loaded_at, 1) if e.loaded else 0.0,
                    "idle_s": round(now - e.last_used, 1) if e.loaded and not e.refs else 0.0,
                    "last_used": e.last_used_wall or None,
                    "est_vram_gb": e.est_vram_gb,
                    "idle_ttl_s": e.idle_ttl_s,
                }
                for e in self._entries.values()
            ]

    def loaded_keys(self) -> list[str]:
        with self._lock:
            return [key_str(k) for k, e in self._entries.items() if e.loaded]

    # -- reaper -------------------------------------------------------------------------------

    def _ensure_reaper(self) -> None:
        if not self._reaper_enabled or (self._reaper is not None and self._reaper.is_alive()):
            return
        with self._lock:
            if self._reaper is not None and self._reaper.is_alive():
                return
            self._reaper = threading.Thread(
                target=self._reap_forever, name="opendub-model-reaper", daemon=True
            )
            self._reaper.start()

    def _next_expiry(self) -> float | None:
        """Under the lock: seconds until the next idle model expires, None if none will."""
        now = self._clock()
        waits = [e.last_used + e.idle_ttl_s - now for e in self._entries.values()
                 if e.loaded and not e.refs and not e.holds]
        return max(0.0, min(waits)) if waits else None

    def _reap_forever(self) -> None:
        while True:
            with self._cv:
                delay = self._next_expiry()
                if delay is None or delay > 0:
                    # Woken on every release / hold end; re-check at least every 30 s.
                    self._cv.wait(min(delay + 0.05, 30.0) if delay is not None else 30.0)
            try:
                self.reap()
            except Exception:  # the reaper must survive any unloader failure
                log.exception("model reaper failed")


manager = ModelManager()


# ---- LLM endpoints (Ollama) -------------------------------------------------------------------

OLLAMA_PORT = 11434
_OLLAMA_PROBE_TTL_S = 300.0  # re-probe a non-Ollama answer after this long
_ollama_seen: dict[str, tuple[bool, float]] = {}
_TRANSPORT: httpx.BaseTransport | None = None  # tests inject an httpx.MockTransport


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, transport=_TRANSPORT) if _TRANSPORT else httpx.Client(
        timeout=timeout
    )


def server_root(base_url: str) -> str:
    """'http://host:11434/v1' → 'http://host:11434' (Ollama's native API lives at the root)."""
    parts = urlsplit(base_url.strip().rstrip("/"))
    path = parts.path.removesuffix("/v1").rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def is_ollama(root: str, headers: dict[str, str] | None = None) -> bool:
    """Ollama's default port, or a server answering Ollama's ``GET /api/tags``. Cached."""
    if urlsplit(root).port == OLLAMA_PORT:
        return True
    seen = _ollama_seen.get(root)
    if seen is not None and (seen[0] or time.monotonic() - seen[1] < _OLLAMA_PROBE_TTL_S):
        return seen[0]
    try:
        with _client(3.0) as client:
            resp = client.get(f"{root}/api/tags", headers=headers or {})
        found = resp.status_code == 200 and isinstance(resp.json().get("models"), list)
    except (httpx.HTTPError, ValueError, AttributeError):
        found = False
    _ollama_seen[root] = (found, time.monotonic())
    return found


def ollama_unload(root: str, model: str, headers: dict[str, str] | None = None) -> bool:
    """Ask Ollama to evict ``model`` from memory now. Best effort; never raises."""
    try:
        with _client(5.0) as client:
            resp = client.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0},
                               headers=headers or {})
        return resp.status_code < 400
    except httpx.HTTPError:
        return False


@asynccontextmanager
async def llm_endpoint(
    base_url: str, model: str, *, unload_after_use: bool = True,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[None]:
    """Scope one use of an OpenAI-compatible LLM. For Ollama (and ``unload_after_use``) the
    (server, model) pair becomes a manager entry that asks Ollama to unload the model when the
    entry is unloaded: after the idle TTL, at the next stage boundary, or when the job ends.
    A no-op for any other server."""
    if not (unload_after_use and base_url and model):
        yield
        return
    root = server_root(base_url)
    if not await asyncio.to_thread(is_ollama, root, headers):
        yield
        return
    auth = dict(headers or {})

    def unload(_model: Any) -> None:
        ollama_unload(root, model, auth)

    async with manager.acquire_async(
        ("llm.ollama", root, model),
        lambda: model,  # Ollama loads on the first request; nothing to do in-process
        unloader=unload,
    ):
        yield
