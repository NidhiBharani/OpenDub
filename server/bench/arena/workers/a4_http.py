"""Shared stdlib HTTP helpers for the A-phase API workers (A1/A2/A4/A5/A6/A8).

Not a worker itself (no ``serve`` call). JSON and multipart requests with retries on 429 / 5xx
and network errors (exponential backoff, ``Retry-After`` honoured); keys are sent only in
headers and never logged. Also a tiny WAV-duration probe for per-minute pricing.
"""
from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def request(method: str, url: str, *, headers: dict[str, str] | None = None,
            data: bytes | None = None, json_body: Any = None, retries: int = 5,
            timeout: float = 600.0, raw: bool = False) -> Any:
    """Send a request; returns parsed JSON (or bytes with ``raw``). Raises with the status and
    the first 500 characters of the body on a non-retryable error."""
    hdrs = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    delay = 2.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if raw:
                    return body
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read()[:500].decode("utf-8", "replace")
            if exc.code in RETRY_STATUS and attempt < retries:
                wait = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    pause = float(wait) if wait else delay
                except ValueError:
                    pause = delay
                time.sleep(min(pause, 120.0))
                delay *= 2
                continue
            raise RuntimeError(f"{method} {url.split('?')[0]} -> HTTP {exc.code}: {body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"{method} {url.split('?')[0]} failed: {exc}") from None
    raise RuntimeError("unreachable")


def multipart(fields: dict[str, Any], files: dict[str, str | Path]) -> tuple[bytes, str]:
    """Encode form fields (lists repeat the field) and files; returns (body, content type)."""
    boundary = uuid.uuid4().hex
    out: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        for v in (value if isinstance(value, list) else [value]):
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            elif isinstance(v, bool):
                v = "true" if v else "false"
            out += [f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(v).encode(), b"\r\n"]
    for name, path in files.items():
        p = Path(path)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        out += [f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{p.name}"\r\n'
                .encode(), f"Content-Type: {ctype}\r\n\r\n".encode(), p.read_bytes(), b"\r\n"]
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def post_form(url: str, fields: dict[str, Any], files: dict[str, str | Path],
              headers: dict[str, str], **kw) -> Any:
    body, ctype = multipart(fields, files)
    return request("POST", url, headers={**headers, "Content-Type": ctype}, data=body, **kw)


def download(url: str, dest: Path, headers: dict[str, str] | None = None) -> Path:
    dest.write_bytes(request("GET", url, headers=headers, raw=True))
    return dest


def audio_seconds(path: str | Path) -> float:
    """Duration for pricing: WAV via the stdlib, anything else via soundfile if present."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        pass
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001 - pricing is best-effort
        return 0.0


def poll(fn, *, interval: float = 3.0, timeout: float = 3600.0):
    """Call ``fn()`` until it returns a non-None value."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        res = fn()
        if res is not None:
            return res
        time.sleep(interval)
        interval = min(interval * 1.5, 30.0)
    raise TimeoutError("job did not finish in time")


def words_to_turns(words: list[dict[str, Any]], *, gap: float = 0.5) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker words (start/end/speaker) into turns."""
    turns: list[dict[str, Any]] = []
    for w in words:
        spk = w.get("speaker")
        if spk is None or w.get("start") is None or w.get("end") is None:
            continue
        s, e = float(w["start"]), float(w["end"])
        if turns and turns[-1]["speaker"] == str(spk) and s - turns[-1]["end"] <= gap:
            turns[-1]["end"] = max(turns[-1]["end"], e)
        else:
            turns.append({"start": s, "end": e, "speaker": str(spk)})
    return turns
