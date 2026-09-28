"""Stdlib helpers shared by the phase-F (picture out) workers. Not a worker itself.

Everything here is stdlib-only so any isolated env can import it (``from f_common import …``).
Video I/O goes through the ``ffmpeg``/``ffprobe`` executables, which every target machine has.

Conventions for F workers (payload contract ``files: {video}``):

- ``item.inputs.video`` is the source picture; ``item.inputs.audio`` the driving (dub) audio.
- Outputs go next to ``out``: ``<out>.mp4`` (the result, with the driving audio muxed in) and
  ``files.audio`` pointing at the driving audio, so ``judgelib.lipsync_score`` sees both streams.
- Research repos are cloned by the env recipe under ``~/.opendub/src/<name>``; a ``repo_dir``
  param overrides the location (e.g. the user's existing ``~/sources/LatentSync``).
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SRC_ROOT = Path.home() / ".opendub" / "src"


def repo_dir(params: dict, name: str) -> Path:
    """Where a cloned research repo lives: ``params.repo_dir`` or ``~/.opendub/src/<name>``."""
    path = Path(params.get("repo_dir") or SRC_ROOT / name).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} repo not found at {path} "
                                f"(run `python -m bench arena env setup <env>`)")
    return path


def run(cmd: list[str], *, cwd: Path | str | None = None, env: dict | None = None,
        timeout: float = 3 * 3600, tail: int = 25) -> str:
    """Run a command, stream nothing, raise with the output tail on failure; returns stdout."""
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=full_env,
                          capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-tail:]
        raise RuntimeError(f"{Path(str(cmd[0])).name} {' '.join(map(str, cmd[1:3]))} … exited "
                           f"{proc.returncode}:\n" + "\n".join(lines))
    return proc.stdout


def probe(path: str | Path) -> dict[str, Any]:
    """Duration (s), fps, width, height and whether an audio stream exists."""
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,width,height,r_frame_rate",
               "-of", "json", str(path)])
    doc = json.loads(out or "{}")
    info: dict[str, Any] = {"duration": float(doc.get("format", {}).get("duration") or 0.0),
                            "has_audio": False, "fps": 0.0, "width": 0, "height": 0}
    for s in doc.get("streams", []):
        if s.get("codec_type") == "video" and not info["width"]:
            num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
            info.update(width=int(s.get("width") or 0), height=int(s.get("height") or 0),
                        fps=float(num) / float(den or 1) if float(den or 1) else 0.0)
        elif s.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def to_wav(src: str | Path, dst: Path, sr: int = 16000, mono: bool = True) -> Path:
    """Decode any audio (or a video's audio track) to PCM WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    run(cmd + ["-c:a", "pcm_s16le", str(dst)])
    return dst


def to_cfr(src: str | Path, dst: Path, fps: float = 25.0, max_height: int | None = None,
           keep_audio: bool = False) -> Path:
    """Re-encode to constant frame rate H.264 (most lip-sync repos assume 25 fps CFR)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = [f"fps={fps:g}"]
    if max_height:
        vf.append(f"scale=-2:'min({max_height},ih)'")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", ",".join(vf),
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac"] if keep_audio else ["-an"]
    run(cmd + [str(dst)])
    return dst


def mux(video: str | Path, audio: str | Path, dst: Path, *, shortest: bool = True) -> Path:
    """Replace the video's audio with ``audio`` (copying the picture stream)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v:0",
           "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if shortest:
        cmd.append("-shortest")
    run(cmd + [str(dst)])
    return dst


def trim_video(src: str | Path, dst: Path, seconds: float) -> Path:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{seconds:.3f}", "-c:v", "libx264",
         "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(dst)])
    return dst


def item_io(item: dict) -> tuple[Path, Path | None]:
    """(source video, driving audio or None) of an F item."""
    inputs = item["inputs"]
    video = Path(inputs["video"])
    audio = Path(inputs["audio"]) if inputs.get("audio") else None
    return video, audio


def driving_audio(item: dict, work: Path, sr: int = 16000) -> Path:
    """The driving audio as mono WAV; for self-reenactment without ``audio`` the video's own."""
    video, audio = item_io(item)
    return to_wav(audio or video, work / f"drive_{sr}.wav", sr=sr)


def workdir(out: Path) -> Path:
    d = out.parent / f"{out.name}.work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup(work: Path, keep: bool = False) -> None:
    if not keep:
        shutil.rmtree(work, ignore_errors=True)


def video_payload(out: Path, produced: Path, audio: Path | None, *, remux: bool = True,
                  **extra: Any) -> dict[str, Any]:
    """Move/mux the produced video to ``<out>.mp4`` and build the standard F payload."""
    final = out.with_suffix(".mp4")
    if audio is not None and remux:
        mux(produced, audio, final)
    elif Path(produced) != final:
        shutil.copyfile(produced, final)
    info = probe(final)
    files: dict[str, Any] = {"video": str(final)}
    if audio is not None:
        files["audio"] = str(audio)
    payload: dict[str, Any] = {"files": files, "duration_s": round(info["duration"], 3),
                               "fps": info["fps"], "width": info["width"],
                               "height": info["height"]}
    payload.update(extra)
    return payload


def keep_audio_copy(audio: Path, out: Path) -> Path:
    """Copy the driving audio next to the output so ``files.audio`` survives cleanup."""
    dst = out.with_suffix(".drive.wav")
    shutil.copyfile(audio, dst)
    return dst


# ------------------------------------------------------------------ HTTP (API workers)

class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status


def http(method: str, url: str, *, headers: dict | None = None, body: Any = None,
         data: bytes | None = None, timeout: float = 300, retries: int = 4) -> Any:
    """JSON-in/JSON-out request with retries on 429/5xx (exponential backoff)."""
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                if "json" in ctype or raw[:1] in (b"{", b"["):
                    return json.loads(raw or b"null")
                return raw
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            raise HttpError(exc.code, text) from None
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            raise
    raise RuntimeError("unreachable")


def multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    """Encode a multipart/form-data body; returns (body, content-type header)."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                     f"{v}\r\n".encode())
    for k, p in files.items():
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; '
                     f'filename="{p.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
        parts.append(p.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def download(url: str, dst: Path, headers: dict | None = None, timeout: float = 600) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dst.open("wb") as f:
        shutil.copyfileobj(resp, f)
    return dst


def poll(fetch, *, done, failed, interval: float = 5.0, timeout: float = 3600.0) -> Any:
    """Call ``fetch()`` until ``done(x)``; raise when ``failed(x)`` or on timeout."""
    t0 = time.time()
    while True:
        x = fetch()
        if done(x):
            return x
        if failed(x):
            raise RuntimeError(f"job failed: {str(x)[:400]}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job still running after {timeout:.0f}s: {str(x)[:200]}")
        time.sleep(interval)


def extract_pngs(video: str | Path, folder: Path, fps: float | None = None) -> Path:
    """Every frame as ``%06d.png`` (for image-only restorers)."""
    folder.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", f"fps={fps:g}"] if fps else []
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), *vf, str(folder / "%06d.png")])
    return folder


def assemble_images(files: list[Path], fps: float, dst: Path) -> Path:
    """Encode an ordered list of image files to H.264 (via an ffconcat list)."""
    lst = dst.with_suffix(".ffconcat")
    lines = ["ffconcat version 1.0"]
    for f in files:
        lines += [f"file '{f}'", f"duration {1.0 / fps:.6f}"]
    lst.write_text("\n".join(lines) + "\n")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-vf", f"fps={fps:g}", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)])
    return dst


def newest(folder: Path, pattern: str = "*.mp4") -> Path:
    found = max(folder.rglob(pattern), key=lambda f: f.stat().st_mtime, default=None)
    if found is None:
        raise RuntimeError(f"no {pattern} produced under {folder}")
    return found
