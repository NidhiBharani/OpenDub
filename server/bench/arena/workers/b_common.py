"""Phase-B (source picture) worker helpers (b_common): video probing, frame grabs, API VLM clients.

Stdlib-only at import time, like ``_sdk``. Frames come from ``ffmpeg`` (a ``tools: [ffmpeg]``
requirement on every B candidate that reads video), so API workers need no imaging library.
Vendor SDKs (``google-genai``, ``anthropic``, ``openai``) are imported inside the client
constructors; they live in the ``b_api`` env.

Coordinates: every box a B worker emits is ``[x1, y1, x2, y2]`` normalised to 0..1.

Pricing: API workers take the list price from ``params`` (``price_in_per_mtok``,
``price_out_per_mtok``, USD per million tokens; the candidate YAML records the source and date)
and bill ``input_tokens × in + output_tokens × out`` from the usage the API reports — output
includes thinking tokens where the vendor bills them as output.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ video

def probe(video: str) -> dict[str, float]:
    """duration_s, fps, width, height, n_frames (estimated) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
         "-of", "json", video], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = (info.get("streams") or [{}])[0]

    def rate(s: str | None) -> float:
        if not s or s in ("0/0", "0"):
            return 0.0
        n, _, d = s.partition("/")
        return float(n) / float(d or 1)

    fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate")) or 25.0
    dur = float((info.get("format") or {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    return {"duration_s": dur, "fps": fps, "width": float(st.get("width") or 0),
            "height": float(st.get("height") or 0),
            "n_frames": float(nb) if nb and str(nb).isdigit() else round(dur * fps)}


def grab_jpeg(video: str, t: float, max_side: int = 1024, quality: int = 3) -> bytes:
    """One frame at ``t`` seconds as JPEG bytes (longest side ≤ ``max_side``)."""
    vf = (f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'")
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", video, "-frames:v", "1",
         "-vf", vf, "-q:v", str(quality), "-f", "image2pipe", "-c:v", "mjpeg", "-"],
        capture_output=True, check=True).stdout


def sample_times(duration: float, n: int, margin: float = 0.03) -> list[float]:
    """``n`` evenly spaced times, skipping the first/last ``margin`` of the video (logos,
    credits)."""
    if duration <= 0 or n <= 0:
        return [0.0]
    lo, hi = duration * margin, duration * (1 - margin)
    if n == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * k / (n - 1) for k in range(n)]


def image_bytes(path: str, max_side: int = 2048) -> tuple[bytes, str]:
    """(bytes, mime) of a still image; re-encoded through ffmpeg when larger than ``max_side``
    or not JPEG/PNG."""
    p = Path(path)
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        p.suffix.lower())
    if mime:
        return p.read_bytes(), mime
    return grab_jpeg(path, 0.0, max_side=max_side), "image/jpeg"


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


# ------------------------------------------------------------------ JSON from model text

def parse_json(text: str) -> Any:
    """First JSON value in a model reply (tolerates ```json fences and prose around it)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        a, b = text.find(open_), text.rfind(close)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in model reply: {text[:200]!r}")


def box_from_1000(box: Any, order: str = "yxyx") -> list[float] | None:
    """A 0–1000 box (Gemini's ``box_2d`` is ``[ymin, xmin, ymax, xmax]``) → normalised xyxy."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        v = [float(x) / 1000.0 for x in box]
    except (TypeError, ValueError):
        return None
    if order == "yxyx":
        v = [v[1], v[0], v[3], v[2]]
    x1, y1, x2, y2 = (min(1.0, max(0.0, x)) for x in v)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


# ------------------------------------------------------------------ API VLM clients

class VlmReply(dict):
    """``{"text", "input_tokens", "output_tokens", "cost_usd", "model"}``."""


def _retry(fn, tries: int = 6, base: float = 2.0):
    """Retry on rate limits / 5xx / transport errors with exponential backoff (never logs keys)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            name = type(exc).__name__
            transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or any(
                k in name for k in ("RateLimit", "Timeout", "Connection", "ServiceUnavailable",
                                    "InternalServer", "ServerError"))
            if not transient or attempt == tries - 1:
                raise
            time.sleep(base * 2 ** attempt)
    raise RuntimeError("unreachable")


class Vlm:
    """One API vision model. ``params``: provider (gemini|anthropic|openai), model, price_in_per_mtok,
    price_out_per_mtok, max_tokens, temperature (where the model allows it), effort
    (anthropic output_config.effort / openai reasoning effort), thinking_level (gemini)."""

    def __init__(self, params: dict[str, Any]):
        from _sdk import api_key

        self.p = params
        self.provider = params["provider"]
        self.model = params["model"]
        if self.provider == "gemini":
            from google import genai

            self.client = genai.Client(api_key=api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"))
        elif self.provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=api_key("OPENAI_API_KEY"))
        else:
            raise ValueError(f"unknown provider {self.provider!r}")

    def cost(self, n_in: int, n_out: int) -> float:
        return (n_in * float(self.p.get("price_in_per_mtok", 0.0))
                + n_out * float(self.p.get("price_out_per_mtok", 0.0))) / 1e6

    def ask(self, prompt: str, images: list[tuple[bytes, str]] | None = None,
            video: str | None = None, schema: dict[str, Any] | None = None) -> VlmReply:
        """Send images (bytes, mime) and/or a video file (Gemini only) with a text prompt."""
        images = images or []
        if self.provider == "gemini":
            return self._gemini(prompt, images, video, schema)
        if video is not None:
            raise ValueError(f"{self.provider} takes frames, not video")
        if self.provider == "anthropic":
            return self._anthropic(prompt, images, schema)
        return self._openai(prompt, images, schema)

    def _gemini(self, prompt, images, video, schema) -> VlmReply:
        from google.genai import types

        parts: list[Any] = []
        if video is not None:
            data = Path(video).read_bytes()
            if len(data) < int(self.p.get("inline_limit_bytes", 18_000_000)):
                parts.append(types.Part.from_bytes(data=data, mime_type="video/mp4"))
            else:  # Files API for large clips; poll until processed
                f = _retry(lambda: self.client.files.upload(file=video))
                while getattr(f.state, "name", str(f.state)) == "PROCESSING":
                    time.sleep(3)
                    f = self.client.files.get(name=f.name)
                parts.append(f)
        parts += [types.Part.from_bytes(data=d, mime_type=m) for d, m in images]
        parts.append(prompt)
        cfg: dict[str, Any] = {"response_mime_type": "application/json"}
        if schema is not None and self.p.get("use_schema"):
            cfg["response_json_schema"] = schema  # google-genai ≥1.2x; JSON mode alone otherwise
        if self.p.get("temperature") is not None:
            cfg["temperature"] = float(self.p["temperature"])
        if self.p.get("media_resolution"):
            cfg["media_resolution"] = self.p["media_resolution"]
        if self.p.get("thinking_level"):
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.p["thinking_level"])
        resp = _retry(lambda: self.client.models.generate_content(
            model=self.model, contents=parts, config=types.GenerateContentConfig(**cfg)))
        um = resp.usage_metadata
        n_in = int(getattr(um, "prompt_token_count", 0) or 0)
        n_out = int(getattr(um, "candidates_token_count", 0) or 0) + int(
            getattr(um, "thoughts_token_count", 0) or 0)
        return VlmReply(text=resp.text or "", input_tokens=n_in, output_tokens=n_out,
                        cost_usd=self.cost(n_in, n_out), model=self.model)

    def _anthropic(self, prompt, images, schema) -> VlmReply:
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": m, "data": b64(d)}}
            for d, m in images]
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {"model": self.model,
                                  "max_tokens": int(self.p.get("max_tokens", 16000)),
                                  "messages": [{"role": "user", "content": content}]}
        out_cfg: dict[str, Any] = {}
        if self.p.get("effort"):
            out_cfg["effort"] = self.p["effort"]
        if schema is not None:
            out_cfg["format"] = {"type": "json_schema", "schema": schema}
        if out_cfg:
            kwargs["output_config"] = out_cfg
        # No server-side refusal fallback on purpose: a fallback would silently swap the model
        # under evaluation. A refusal is recorded as an item error instead.
        resp = _retry(lambda: self.client.messages.create(**kwargs))
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        n_in = int(resp.usage.input_tokens or 0)
        n_out = int(resp.usage.output_tokens or 0)
        return VlmReply(text=text, input_tokens=n_in, output_tokens=n_out,
                        cost_usd=self.cost(n_in, n_out), model=resp.model)

    def _openai(self, prompt, images, schema) -> VlmReply:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        detail = self.p.get("image_detail", "high")
        content += [{"type": "input_image", "image_url": f"data:{m};base64,{b64(d)}",
                     "detail": detail} for d, m in images]
        kwargs: dict[str, Any] = {"model": self.model,
                                  "input": [{"role": "user", "content": content}]}
        if self.p.get("effort"):
            kwargs["reasoning"] = {"effort": self.p["effort"]}
        if schema is not None:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "answer",
                                         "schema": schema, "strict": False}}
        resp = _retry(lambda: self.client.responses.create(**kwargs))
        n_in = int(getattr(resp.usage, "input_tokens", 0) or 0)
        n_out = int(getattr(resp.usage, "output_tokens", 0) or 0)
        return VlmReply(text=resp.output_text or "", input_tokens=n_in, output_tokens=n_out,
                        cost_usd=self.cost(n_in, n_out), model=getattr(resp, "model", self.model))


# ------------------------------------------------------------------ local VLM (transformers)

class LocalVlm:
    """A local image-text-to-text model through transformers (Qwen3-VL, PaddleOCR-VL …).
    ``params``: model, revision, dtype, attn, max_new_tokens, trust_remote_code, device."""

    def __init__(self, params: dict[str, Any]):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.p = params
        kw: dict[str, Any] = {"revision": params.get("revision"),
                              "trust_remote_code": bool(params.get("trust_remote_code", False))}
        dtype = params.get("dtype", "bfloat16")
        self.model = AutoModelForImageTextToText.from_pretrained(
            params["model"], dtype="auto" if dtype == "auto" else getattr(torch, dtype),
            device_map=params.get("device", "cuda:0"),
            attn_implementation=params.get("attn", "sdpa"), **kw)
        self.processor = AutoProcessor.from_pretrained(params["model"], **kw)
        self.model_id = params["model"]

    def ask(self, prompt: str, images: list[tuple[bytes, str]]) -> VlmReply:
        import io

        from PIL import Image

        pil = [Image.open(io.BytesIO(d)).convert("RGB") for d, _m in images]
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in pil]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=int(self.p.get("max_new_tokens",
                                                                          1024)),
                                  do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new, skip_special_tokens=True)
        return VlmReply(text=text, input_tokens=int(inputs["input_ids"].shape[1]),
                        output_tokens=len(new), cost_usd=0.0, model=self.model_id)


def make_vlm(params: dict[str, Any]) -> Vlm | LocalVlm:
    return LocalVlm(params) if params.get("provider", "local") == "local" else Vlm(params)


# ------------------------------------------------------------------ B4 frame → title aggregation

B4_CLASSES = ("live_action", "2d", "3d")


def aggregate_title(frame_probs: list[dict[str, float]], *, pure_share: float = 0.8,
                    mixed_min_share: float = 0.15) -> dict[str, Any]:
    """Title verdict from per-frame class probabilities over live_action / 2d / 3d.

    Each frame votes its argmax. The top class wins outright when it holds ≥ ``pure_share`` of the
    votes; when the top two classes each hold ≥ ``mixed_min_share`` (and the top is below
    ``pure_share``) the title is ``mixed``. Returned ``probs`` put mass ``m = 2·min(share1,
    share2)`` on mixed (0 when not mixed-shaped) and spread the rest by the mean frame
    probabilities; ``confidence`` is the probability of the returned label.
    """
    if not frame_probs:
        return {"abstain": True, "label": None, "probs": {}, "confidence": 0.0}
    votes = {c: 0 for c in B4_CLASSES}
    mean = {c: 0.0 for c in B4_CLASSES}
    for fp in frame_probs:
        total = sum(max(0.0, fp.get(c, 0.0)) for c in B4_CLASSES) or 1.0
        for c in B4_CLASSES:
            mean[c] += max(0.0, fp.get(c, 0.0)) / total / len(frame_probs)
        votes[max(B4_CLASSES, key=lambda c: fp.get(c, 0.0))] += 1
    shares = sorted(((v / len(frame_probs), c) for c, v in votes.items()), reverse=True)
    (s1, c1), (s2, _c2) = shares[0], shares[1]
    mixed = s1 < pure_share and s2 >= mixed_min_share
    m = min(1.0, 2 * s2) if mixed else 0.0
    probs = {c: (1 - m) * mean[c] for c in B4_CLASSES}
    probs["mixed"] = m
    label = "mixed" if mixed else c1
    return {"label": label, "probs": probs, "confidence": probs[label],
            "vote_shares": {c: v / len(frame_probs) for c, v in votes.items()}}


# ------------------------------------------------------------------ B3 frames of an item

def item_frames(item: dict[str, Any], out: Path, max_side: int = 2048) -> list[dict[str, Any]]:
    """The frames a B3 item asks to be read: ``[{"t", "path", "width", "height"}]``.

    ``inputs.image`` → the image itself (t = None); ``inputs.video`` + ``inputs.times`` → JPEGs
    grabbed at those times, written next to ``out`` (kept for the audit viewer). Boxes are
    normalised by the size of the frame the model actually saw."""
    inp = item["inputs"]
    frames = []
    if inp.get("image"):
        path = inp["image"]
        info = probe_image(path)
        if max(info) > max_side or Path(path).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            dst = out.parent / f"{out.name}.frame.jpg"
            dst.write_bytes(grab_jpeg(path, 0.0, max_side=max_side))
            path, info = str(dst), probe_image(str(dst))
        frames.append({"t": None, "path": path, "width": info[0], "height": info[1]})
        return frames
    video = inp["video"]
    times = inp.get("times") or [probe(video)["duration_s"] / 2]
    for k, t in enumerate(times):
        dst = out.parent / f"{out.name}.f{k:03d}.jpg"
        dst.write_bytes(grab_jpeg(video, float(t), max_side=max_side, quality=2))
        w, h = probe_image(str(dst))
        frames.append({"t": float(t), "path": str(dst), "width": w, "height": h})
    return frames


def probe_image(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    st = (json.loads(out).get("streams") or [{}])[0]
    return int(st.get("width") or 0), int(st.get("height") or 0)


def norm_poly(points: Any, width: float, height: float) -> list[float] | None:
    """Pixel polygon / box → normalised xyxy bounding box."""
    try:
        pts = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError):
        try:
            v = [float(x) for x in points]
        except (TypeError, ValueError):
            return None
        pts = [(v[0], v[1]), (v[2], v[3])] if len(v) == 4 else list(zip(v[0::2], v[1::2],
                                                                            strict=False))
    if not pts or not width or not height:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [max(0.0, min(xs) / width), max(0.0, min(ys) / height),
            min(1.0, max(xs) / width), min(1.0, max(ys) / height)]
