"""Shared helpers for the phase-D (voice) workers. Stdlib only at import time.

Every D worker follows the same item contract (``bench/arena/CONTRIBUTING.md``)::

    item.inputs  = {text, ref_audio?, ref_text?, target_s?, emotion?, events?}   (D1–D5, D7)
                 = {audio, ref_audio, text?}                                     (D6)
                 = {audio}                                                       (D8)
    payload      = {"files": {"audio": <abs wav>}, "sample_rate", "duration_s", ...}

Conventions shared by all D workers:

- **Seeds.** ``item["seed"]`` (set by the best-of-N method worker) wins over ``params["seed"]``,
  which wins over a stable hash of the item id, so a take is reproducible and N takes differ.
- **Language.** Forced from the job language; a worker raises ``Unsupported`` for languages its
  model does not declare instead of letting it auto-detect.
- **Native sample rate.** Files keep the model's native rate; judges resample.
- **API spend.** ``"_cost_usd"`` in the payload, from list prices recorded in each worker docstring.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from _sdk import Unsupported

LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "ta": "Tamil",
              "te": "Telugu", "bn": "Bengali"}
LOCALES = {"en": "en-US", "hi": "hi-IN", "ja": "ja-JP", "zh": "zh-CN", "ko": "ko-KR",
           "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "it": "it-IT", "pt": "pt-BR",
           "ru": "ru-RU", "ta": "ta-IN", "te": "te-IN", "bn": "bn-IN"}
WORKERS_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ items

def text_of(item: dict) -> str:
    text = str((item.get("inputs") or {}).get("text") or "").strip()
    if not text:
        raise Unsupported("item has no inputs.text")
    return text


def ref_of(item: dict, required: bool = True) -> tuple[str | None, str]:
    """(reference wav path, its transcript or "")."""
    inp = item.get("inputs") or {}
    ref = inp.get("ref_audio")
    if required and not ref:
        raise Unsupported("item has no inputs.ref_audio (this candidate clones a voice)")
    return ref, str(inp.get("ref_text") or "").strip()


def target_s(item: dict) -> float | None:
    t = (item.get("inputs") or {}).get("target_s")
    return float(t) if t else None


def emotion_of(item: dict) -> str:
    return str((item.get("inputs") or {}).get("emotion") or "").strip()


def src_lang(item: dict) -> str | None:
    meta = item.get("meta") or {}
    return meta.get("src_lang")


def gender_of(item: dict) -> str:
    """"female" | "male" | "" from item meta (FLEURS / IndicVoices-R carry it)."""
    g = str((item.get("meta") or {}).get("gender") or "").lower()
    return "female" if g.startswith("f") else "male" if g.startswith("m") else ""


def require_lang(lang: str, supported: list[str] | tuple[str, ...] | set[str] | str,
                 what: str = "model") -> None:
    if supported != "*" and lang not in supported:
        raise Unsupported(f"{what} does not declare language {lang!r}")


def item_seed(item: dict, params: dict) -> int:
    if item.get("seed") is not None:
        return int(item["seed"])
    if params.get("seed") is not None:
        return int(params["seed"])
    return int(hashlib.sha256(str(item.get("id", "")).encode()).hexdigest()[:8], 16)


def seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np = sys.modules.get("numpy")
    if np is not None:
        np.random.seed(seed % 2**32)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def pick_voice(voices: Any, lang: str, item: dict, default: str | None = None) -> str:
    """``voices`` is a name, {lang: name} or {lang: {female: name, male: name}}; gender from
    item meta picks within a language so a stock voice matches the line's speaker."""
    if isinstance(voices, str):
        return voices
    v = (voices or {}).get(lang, (voices or {}).get("*", default))
    if isinstance(v, dict):
        g = gender_of(item) or "female"
        v = v.get(g) or next(iter(v.values()))
    if not v:
        raise Unsupported(f"no stock voice configured for {lang!r}")
    return str(v)


# ------------------------------------------------------------------ audio files

def wav_path(out: Path) -> Path:
    """``<out>.wav``. Appends rather than ``with_suffix`` so an item id with a dot (``x0.8``) or a
    take suffix never collides with a sibling output."""
    out = Path(out)
    return out if out.suffix == ".wav" else out.with_name(out.name + ".wav")


def write_wav(path: Path, audio: Any, sr: int) -> Path:
    """Write float/int audio (numpy array, torch tensor or list) as a mono 16-bit-safe wav."""
    import numpy as np
    import soundfile as sf

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    arr = arr.astype(np.float32).squeeze()
    if arr.ndim == 2:  # (channels, samples) or (samples, channels) → mono
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[1] else 1)
    path = Path(path)
    sf.write(str(path), arr, int(sr))
    return path


def duration_s(path: Path | str) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except ImportError:
        import wave

        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())


def audio_payload(path: Path, **extra: Any) -> dict:
    path = Path(path)
    if not path.exists() or path.stat().st_size < 64:
        raise RuntimeError(f"model produced no audio at {path}")
    d = duration_s(path)
    try:
        import soundfile as sf

        sr = int(sf.info(str(path)).samplerate)
    except ImportError:
        import wave

        with wave.open(str(path)) as w:
            sr = w.getframerate()
    return {"files": {"audio": str(path.resolve())}, "sample_rate": sr,
            "duration_s": round(d, 4), **extra}


def pcm16_to_wav(pcm: bytes, sr: int, path: Path, channels: int = 1) -> Path:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return Path(path)


def transcode_to_wav(data: bytes, path: Path, suffix: str = ".mp3") -> Path:
    """Decode API audio (mp3/ogg/…) to wav at its native rate with ffmpeg."""
    if data[:4] == b"RIFF":
        Path(path).write_bytes(data)
        return Path(path)
    tmp = Path(path).with_suffix(f".raw{suffix}")
    tmp.write_bytes(data)
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp),
                        "-ac", "1", str(path)], check=True)
    finally:
        tmp.unlink(missing_ok=True)
    return Path(path)


def ffmpeg_to_wav(src: str, dst: Path, sr: int | None = None) -> Path:
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    subprocess.run(cmd + [str(dst)], check=True)
    return Path(dst)


# ------------------------------------------------------------------ HTTP (API workers)

class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url.split('?')[0]}: {body[:400]}")
        self.status = status
        self.body = body


def http(method: str, url: str, *, body: bytes | None = None, headers: dict | None = None,
         timeout: float = 180.0, retries: int = 4, backoff: float = 2.0) -> tuple[bytes, dict]:
    """urllib request with retries on 429 / 5xx / network errors. Never logs headers (keys)."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), dict(resp.headers.items())
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            last = HttpError(exc.code, text, url)
            if exc.code not in (408, 409, 425, 429, 500, 502, 503, 504) or attempt == retries:
                raise last from None
            wait = float(exc.headers.get("Retry-After") or backoff * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if attempt == retries:
                raise
            wait = backoff * 2 ** attempt
        time.sleep(min(wait, 60.0))
    raise RuntimeError(f"request failed: {last}")


def post_json(url: str, payload: Any, headers: dict | None = None, **kw: Any) -> tuple[bytes, dict]:
    h = {"Content-Type": "application/json", **(headers or {})}
    return http("POST", url, body=json.dumps(payload, ensure_ascii=False).encode(), headers=h,
                **kw)


def get_json(url: str, headers: dict | None = None, **kw: Any) -> Any:
    data, _ = http("GET", url, headers=headers, **kw)
    return json.loads(data or b"null")


def multipart(fields: dict[str, Any], files: dict[str, tuple[str, bytes, str]] | list | None = None
              ) -> tuple[bytes, str]:
    """(body, content-type) for multipart/form-data. ``files``: {field: (name, bytes, mime)} or a
    list of (field, (name, bytes, mime)) for repeated fields."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        for one in (v if isinstance(v, list) else [v]):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                         .encode() + str(one).encode() + b"\r\n")
    items = files.items() if isinstance(files, dict) else (files or [])
    for k, (name, data, mime) in items:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; '
                     f'filename="{name}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def b64file(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def msgpack_dumps(obj: Any) -> bytes:
    """Minimal MessagePack encoder (dict, list, str, bytes, int, float, bool, None) for APIs that
    take inline reference audio only as msgpack (Fish Audio)."""
    if obj is None:
        return b"\xc0"
    if obj is True:
        return b"\xc3"
    if obj is False:
        return b"\xc2"
    if isinstance(obj, int):
        if 0 <= obj < 128:
            return struct.pack("B", obj)
        if -32 <= obj < 0:
            return struct.pack("b", obj)
        return b"\xd3" + struct.pack(">q", obj)
    if isinstance(obj, float):
        return b"\xcb" + struct.pack(">d", obj)
    if isinstance(obj, str):
        raw = obj.encode("utf-8")
        n = len(raw)
        head = (struct.pack("B", 0xA0 | n) if n < 32 else b"\xd9" + struct.pack("B", n) if n < 256
                else b"\xda" + struct.pack(">H", n) if n < 65536 else b"\xdb" + struct.pack(">I", n))
        return head + raw
    if isinstance(obj, (bytes, bytearray)):
        n = len(obj)
        head = (b"\xc4" + struct.pack("B", n) if n < 256 else b"\xc5" + struct.pack(">H", n)
                if n < 65536 else b"\xc6" + struct.pack(">I", n))
        return head + bytes(obj)
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        head = (struct.pack("B", 0x90 | n) if n < 16 else b"\xdc" + struct.pack(">H", n))
        return head + b"".join(msgpack_dumps(x) for x in obj)
    if isinstance(obj, dict):
        n = len(obj)
        head = (struct.pack("B", 0x80 | n) if n < 16 else b"\xde" + struct.pack(">H", n))
        return head + b"".join(msgpack_dumps(str(k)) + msgpack_dumps(v) for k, v in obj.items())
    raise TypeError(f"cannot msgpack {type(obj).__name__}")


def char_cost(text: str, usd_per_million_chars: float) -> float:
    return round(len(text) * usd_per_million_chars / 1e6, 6)


# ------------------------------------------------------------------ local servers & repos

def src_dir(name: str) -> Path:
    """Where env recipes clone research repos (``~/.opendub/src/<name>``)."""
    return Path(os.environ.get("OPENDUB_SRC", Path.home() / ".opendub" / "src")) / name


def add_repo_to_path(name: str, *subdirs: str) -> Path:
    root = src_dir(name)
    if not root.exists():
        raise RuntimeError(f"{root} missing: run `python -m bench arena env setup` for this env")
    for sub in (*subdirs, ""):
        p = str(root / sub) if sub else str(root)
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


class LocalServer:
    """A model server (vLLM-Omni, SGLang, fish-speech api_server) started by ``load`` and killed
    when the worker process exits, so its GPU memory is released with the job."""

    def __init__(self, cmd: list[str], health_url: str, *, cwd: str | None = None,
                 env: dict | None = None, timeout: float = 1800.0, log_path: Path | None = None):
        import atexit

        self.health_url = health_url
        log = open(log_path, "ab") if log_path else subprocess.DEVNULL  # noqa: SIM115 - lives with the server
        self.proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})},
                                     stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.stop)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}: {' '.join(cmd)}")
            try:
                with urllib.request.urlopen(health_url, timeout=5):
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                time.sleep(3)
        self.stop()
        raise RuntimeError(f"server not healthy after {timeout:.0f}s: {health_url}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def env_python(env: str) -> str:
    """Interpreter of another arena env, for method workers that call a judge or an inner model
    in a subprocess. "server" is the OpenDub server venv; others live in ~/.opendub/envs."""
    if env == "server":
        venv = WORKERS_DIR.parents[2] / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable
    return str(Path.home() / ".opendub" / "envs" / env / "bin" / "python")


def run_worker_subprocess(worker: str, env: str, params: dict, lang: str,
                          items: list[dict], job_dir: Path, timeout: float = 3600) -> dict:
    """Run another worker as a child job (same protocol as the runner); returns id -> result
    record, with each item's payload under ``"payload"`` when it succeeded."""
    job_dir.mkdir(parents=True, exist_ok=True)
    job = job_dir / f"{worker}-{uuid.uuid4().hex[:8]}.json"
    job.write_text(json.dumps({"params": params, "lang": lang, "items": items},
                              ensure_ascii=False))
    subprocess.run([env_python(env), str(WORKERS_DIR / f"{worker}.py"), "--job", str(job)],
                   cwd=WORKERS_DIR, timeout=timeout, check=False)
    out: dict = {}
    res = job.with_suffix(".results.jsonl")
    if res.exists():
        for line in res.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    it = next(i for i in items if i["id"] == rec["id"])
                    p = Path(it["out"]).with_suffix(".json")
                    rec["payload"] = json.loads(p.read_text()) if p.exists() else {}
                out[rec["id"]] = rec
    return out


class Verifier:
    """ASR verifier subprocess (``d3_asr_verifier.py``) in another env, loaded once per job."""

    def __init__(self, env: str = "server", config: dict | None = None):
        import atexit

        self.proc = subprocess.Popen(
            [env_python(env), str(WORKERS_DIR / "d3_asr_verifier.py"), json.dumps(config or {})],
            cwd=WORKERS_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        atexit.register(self.close)
        ready = self.proc.stdout.readline() if self.proc.stdout else ""
        if not ready or not json.loads(ready).get("ready"):
            raise RuntimeError("ASR verifier failed to start")

    def transcribe(self, audio: str, lang: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"audio": str(audio), "lang": lang}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline() or "{}").get("text", "")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()


def norm_chars(text: str) -> str:
    """NFKC + casefold, punctuation and spaces removed (a CER view that works for ja/hi/en)."""
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold()
                   if not unicodedata.category(ch).startswith(("P", "Z", "S", "C")))


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate (Levenshtein / reference length) on :func:`norm_chars`."""
    r, h = norm_chars(reference), norm_chars(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def import_worker(name: str) -> Any:
    """Import a sibling worker module (its ``load``/``run``) without serving it."""
    import importlib

    if str(WORKERS_DIR) not in sys.path:
        sys.path.insert(0, str(WORKERS_DIR))
    return importlib.import_module(name)
