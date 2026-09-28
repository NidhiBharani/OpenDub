"""Shared code for the C-phase (text) workers: LLM backends, prompts, parsing. Stdlib only at import.

Not a worker itself. The C-phase LLM workers (``c2_openai_compat``, ``c2_anthropic``,
``c2_openai``, ``c2_gemini``) build a backend here and hand it to :func:`run_task`, so the same
model can compete in several capabilities with the prompt for that capability:

| ``params.task`` | capability | item.inputs                                   | payload                  |
|-----------------|------------|-----------------------------------------------|--------------------------|
| ``translate``   | C2         | text, slot_s?, context_before/after?, speaker?, emotion? | text (+ candidates) |
| ``segment``     | C1         | words [{start, end, word}]                    | lines [{start_word, end_word, start, end, text}] |
| ``paraphrase``  | C3         | text (target line), source_text?, slot_s?     | text, candidates         |
| ``gemba``       | C4 / judge | source, hypothesis, reference?                | metrics {gemba_esa}      |
| ``tn``          | C5 (.tn)   | text                                          | text (spoken form)       |
| ``g2p``         | C5 (.g2p)  | text (one word)                               | text (IPA, space-separated phones) |
| ``kana``        | C5 (.kana) | text                                          | text (katakana reading)  |
| ``condense``    | C6         | cues [{start, end, text, source?}]            | cues, text               |

The pack language is a direction for translation tasks (``"ja-en"``); :func:`direction` splits
it (item ``meta.src_lang``/``meta.tgt_lang`` win when present). Languages are always forced in
the prompt; nothing auto-detects.

Backends return ``(text, usage)`` where ``usage = {"input_tokens", "output_tokens"}``; API
workers turn usage into ``_cost_usd`` with their list-price tables.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import c_speech_rate as rate

LANG_NAMES = {"en": "English", "hi": "Hindi", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
              "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "ru": "Russian", "ta": "Tamil", "te": "Telugu",
              "bn": "Bengali", "ar": "Arabic"}

Messages = list[dict[str, str]]
Complete = Callable[..., tuple[str, dict[str, int]]]


def lang_name(code: str) -> str:
    code = rate.base_lang(code)
    if code not in LANG_NAMES:
        raise ValueError(f"no language name for {code!r}")
    return LANG_NAMES[code]


def direction(lang: str, item: dict | None = None) -> tuple[str, str]:
    """(src, tgt) for a pack language ``"ja-en"``; item meta/inputs override. Monolingual packs
    (``"hi.tn"``) give (hi, hi)."""
    meta = {**((item or {}).get("meta") or {}), **((item or {}).get("inputs") or {})}
    if "-" in (lang or ""):
        src, tgt = lang.split("-", 1)
    else:
        src = tgt = rate.base_lang(lang)
    src = meta.get("src_lang") or src
    tgt = meta.get("tgt_lang") or tgt
    return rate.base_lang(src), rate.base_lang(tgt)


# ------------------------------------------------------------------ HTTP (stdlib)

def http_json(url: str, body: dict | None = None, *, headers: dict | None = None,
              timeout: float = 600.0, retries: int = 4, method: str | None = None) -> Any:
    """POST (or GET when body is None) JSON with retries on 429/5xx/network errors."""
    data = None if body is None else json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode(errors="replace")
            if exc.code == 429 or exc.code >= 500:
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                wait = float(exc.headers.get("retry-after") or 2 ** attempt)
            else:
                raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last, wait = exc, 2 ** attempt
        if attempt < retries:
            time.sleep(min(wait, 60.0))
    raise RuntimeError(f"request to {url} failed: {last}")


def wait_http(url: str, timeout: float, proc: subprocess.Popen | None = None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server process exited with {proc.returncode} before {url} "
                               "answered (see the worker log)")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(2.0)
    raise RuntimeError(f"{url} did not come up within {timeout:.0f}s")


# ------------------------------------------------------------------ local OpenAI-compatible servers

_CLEANUPS: list[Callable[[], None]] = []


def _run_cleanups() -> None:
    while _CLEANUPS:
        fn = _CLEANUPS.pop()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - cleanup must never mask the job result
            print(f"[c_common] cleanup failed: {exc!r}", file=sys.stderr)


def on_exit(fn: Callable[[], None]) -> None:
    """Run ``fn`` when the worker process ends (normal exit, SIGTERM, SIGINT or SIGHUP), so a
    server we started, or a model Ollama keeps resident, never outlives the job."""
    if not _CLEANUPS:
        atexit.register(_run_cleanups)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, lambda s, f: sys.exit(128 + s))
            except (ValueError, OSError):  # not the main thread
                pass
    _CLEANUPS.append(fn)


def _stop_process(proc: subprocess.Popen, grace: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=10)


def ollama_unload(base: str, model: str) -> None:
    """Ollama keeps models resident for minutes; ``keep_alive: 0`` evicts one immediately."""
    http_json(f"{base}/api/generate", {"model": model, "keep_alive": 0}, timeout=60, retries=1)


class OpenAICompatBackend:
    """Chat completions against a local server this worker starts and stops, or an external one.

    ``params``:
      backend      ollama | vllm | vllm_docker | llamacpp | llamacpp_docker | external
      model        Ollama tag, HF repo id (vLLM), or GGUF repo[:quant] / path (llama.cpp)
      served_name  name to request (default: model)
      base_url     external server, or override Ollama's http://127.0.0.1:11434
      port         port for servers we start (default 8011)
      server_args  extra CLI args for vllm serve / llama-server (list)
      image        docker image (vllm_docker: vllm/vllm-openai:<ver> on x86_64,
                   nvcr.io/nvidia/vllm:<tag> on DGX Spark; llamacpp_docker: llama.cpp server-cuda)
      llama_server llama-server binary (default ~/.opendub/src/llama.cpp/build/bin/llama-server)
      startup_timeout  seconds to wait for the server (default 1800: first run downloads weights)
      num_ctx      Ollama context length; extra_body: merged into every request body
      api_key_env  env var holding a bearer token (external servers)
    Everything we start is stopped when the process exits; Ollama models are unloaded with
    ``keep_alive: 0`` (the server itself is the user's).
    """

    def __init__(self, params: dict[str, Any]):
        self.p = params
        self.kind = params.get("backend", "ollama")
        self.model = params["model"]
        self.name = params.get("served_name") or self.model
        self.port = int(params.get("port", 8011))
        self.proc: subprocess.Popen | None = None
        self.headers: dict[str, str] = {}
        if params.get("api_key_env"):
            self.headers["Authorization"] = f"Bearer {os.environ[params['api_key_env']]}"
        timeout = float(params.get("startup_timeout", 1800))
        if self.kind == "ollama":
            self.base = (params.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
            tags = http_json(f"{self.base}/api/tags", timeout=10, retries=1)
            have = {m.get("name") for m in tags.get("models", [])} | \
                   {m.get("model") for m in tags.get("models", [])}
            if self.model not in have and f"{self.model}:latest" not in have:
                if not params.get("pull", False):
                    raise RuntimeError(f"Ollama has no {self.model!r}; run `ollama pull "
                                       f"{self.model}` (or set params.pull: true)")
                http_json(f"{self.base}/api/pull", {"model": self.model, "stream": False},
                          timeout=24 * 3600, retries=0)
            on_exit(lambda: ollama_unload(self.base, self.model))
        elif self.kind == "external":
            self.base = params["base_url"].rstrip("/").removesuffix("/v1")
        else:
            self.base = f"http://127.0.0.1:{self.port}"
            cmd, health = self._server_cmd()
            print(f"[c_common] starting: {' '.join(cmd)}", file=sys.stderr, flush=True)
            self.proc = subprocess.Popen(cmd, start_new_session=True, stdout=sys.stderr,
                                         stderr=sys.stderr)
            if self.kind.endswith("_docker"):
                cname = self._container
                on_exit(lambda: subprocess.run(["docker", "stop", "-t", "30", cname],
                                               capture_output=True, check=False))
            on_exit(lambda: _stop_process(self.proc))
            wait_http(f"{self.base}{health}", timeout, self.proc)

    @property
    def _container(self) -> str:
        return f"opendub-arena-{os.getpid()}"

    def _server_cmd(self) -> tuple[list[str], str]:
        extra = [str(a) for a in self.p.get("server_args", [])]
        hf_cache = str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")))
        if self.kind == "vllm":
            return ([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                     "--model", self.model, "--served-model-name", self.name,
                     "--host", "127.0.0.1", "--port", str(self.port), *extra], "/v1/models")
        if self.kind == "vllm_docker":
            image = self.p.get("image", "vllm/vllm-openai:v0.30.0")
            return (["docker", "run", "--rm", "--name", self._container, "--gpus", "all",
                     "--ipc=host", "-p", f"127.0.0.1:{self.port}:8000", "-v",
                     f"{hf_cache}:/root/.cache/huggingface", "-e", "HF_TOKEN",
                     "--entrypoint", "vllm", image, "serve", self.model,
                     "--served-model-name", self.name, "--host", "0.0.0.0", "--port", "8000",
                     *extra], "/v1/models")
        model_args = (["-m", self.model] if self.model.endswith(".gguf")
                      else ["-hf", self.model])
        if self.kind == "llamacpp":
            binary = os.path.expanduser(self.p.get(
                "llama_server", "~/.opendub/src/llama.cpp/build/bin/llama-server"))
            return ([binary, *model_args, "--alias", self.name, "--host", "127.0.0.1",
                     "--port", str(self.port), "-ngl", "999", *extra], "/health")
        if self.kind == "llamacpp_docker":
            image = self.p.get("image", "ghcr.io/ggml-org/llama.cpp:server-cuda")
            return (["docker", "run", "--rm", "--name", self._container, "--gpus", "all",
                     "-p", f"127.0.0.1:{self.port}:8080", "-v",
                     f"{Path.home() / '.cache' / 'llama.cpp'}:/root/.cache/llama.cpp",
                     "-e", "HF_TOKEN", image, *model_args, "--alias", self.name,
                     "--host", "0.0.0.0", "--port", "8080", "-ngl", "999", *extra], "/health")
        raise ValueError(f"unknown backend {self.kind!r}")

    def complete(self, messages: Messages, *, temperature: float | None = None,
                 max_tokens: int = 2048, json_mode: bool = False,
                 seed: int | None = None) -> tuple[str, dict[str, int]]:
        temp = self.p.get("temperature", 0.3) if temperature is None else temperature
        if self.kind == "ollama":
            body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                    "keep_alive": "10m",
                                    "options": {"temperature": temp,
                                                "num_ctx": int(self.p.get("num_ctx", 8192)),
                                                "num_predict": max_tokens}}
            if seed is not None:
                body["options"]["seed"] = seed
            if json_mode:
                body["format"] = "json"
            if self.p.get("think") is not None:
                body["think"] = self.p["think"]
            body.update(self.p.get("extra_body", {}))
            data = http_json(f"{self.base}/api/chat", body)
            return (strip_thinking(data.get("message", {}).get("content", "")),
                    {"input_tokens": data.get("prompt_eval_count", 0),
                     "output_tokens": data.get("eval_count", 0)})
        body = {"model": self.name, "messages": messages, "temperature": temp,
                "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if json_mode and self.p.get("json_mode", True):
            body["response_format"] = {"type": "json_object"}
        body.update(self.p.get("extra_body", {}))
        data = http_json(f"{self.base}/v1/chat/completions", body, headers=self.headers)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected response: {str(data)[:300]}") from None
        usage = data.get("usage") or {}
        return strip_thinking(text), {"input_tokens": usage.get("prompt_tokens", 0),
                                      "output_tokens": usage.get("completion_tokens", 0)}


# ------------------------------------------------------------------ parsing

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    if "<|channel|>final<|message|>" in text:               # gpt-oss harmony leakage
        text = text.split("<|channel|>final<|message|>")[-1]
    return text.strip()


def parse_json(text: str) -> Any:
    """First JSON object/array in ``text`` (fences, prose and thinking tolerated)."""
    text = _FENCE_RE.sub("", strip_thinking(text)).strip()
    pairs = sorted((("{", "}"), ("[", "]")),
                   key=lambda oc: text.find(oc[0]) if oc[0] in text else len(text))
    for opener, closer in pairs:          # whichever structure starts first
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in response: {text[:200]!r}")


def clean_line(text: str) -> str:
    text = strip_thinking(text).strip()
    text = re.sub(r"^(translation|output)\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'「":
        text = text[1:-1]
    return text.strip()


# ------------------------------------------------------------------ prompts

def opendub_system_prompt(source_lang: str, target_lang: str) -> str:
    """Verbatim copy of ``app/providers/translation/_llm.build_system_prompt`` (the baseline);
    tests/test_arena_c.py fails if the two drift."""
    src = ""
    if source_lang and source_lang.strip().lower() not in ("", "auto"):
        src = f" from {source_lang}"
    return (
        f"You are an expert dubbing translator adapting dialogue{src} into {target_lang} for a "
        "VOICE DUB. The output will be performed by a voice actor or a text-to-speech system, "
        "never read as subtitles.\n\n"
        "Follow these rules for every line:\n"
        "1. Write natural, colloquial SPOKEN dialogue in the target language - how a person would "
        "actually say it out loud - not a stiff, literal, or written-style translation.\n"
        "2. Each line must fit its time budget: you are given `max_chars`, an approximate character "
        "budget (derived from the line's spoken duration, ~15 characters per second of English "
        "speech) that the translation should roughly fit so the dub can be spoken in the time "
        "available. Prefer concise, natural phrasing over padding; it is fine to land a little "
        "under budget, but do not substantially exceed it.\n"
        "3. Preserve each line's emotion, register, tone, and the speaker's intent - match the "
        "energy of a shout, a whisper, sarcasm, hesitation, affection, and so on. Never flatten or "
        "neutralize delivery.\n"
        "4. Keep character names and honorifics natural and consistent for a target-language "
        "audience; never invent, drop, or mistranslate a name.\n"
        "5. Use `context_before` / `context_after` only to resolve ambiguity (pronouns, references, "
        "continuing thoughts) - never translate those context lines themselves.\n"
        "When terminology is supplied, preserve its target spellings consistently. If a "
        "previous_translation and measured_duration are supplied, rewrite that line more concisely "
        "to fit seconds while preserving meaning, names, negation and emotional intent. Do not "
        "drop facts merely to fit. Character budgets are only estimates, not semantic rules.\n"
        "6. Output ONLY the translated line content, inside the required JSON - never add "
        "translator notes, bracketed explanations, parentheticals about tone, or any other "
        "meta-commentary.\n"
    )


def opendub_max_chars(duration: float) -> int:
    """``app/providers/translation/_llm.max_chars_for``: 15 English characters per second."""
    return max(12, round(max(0.0, duration) * 15.0))


def slot_of(inputs: dict, src: str) -> float:
    """The line's time slot: measured when the pack has one, else the predicted source duration."""
    slot = inputs.get("slot_s")
    return float(slot) if slot else rate.predicted_seconds(inputs.get("text", ""), src)


def translate_messages(style: str, item: dict, src: str, tgt: str) -> tuple[Messages, bool]:
    """(messages, expects_json) for one line in the given prompt ``style``.

    - ``opendub``: today's production prompt (15 chars/s English budget for every language);
    - ``dub``: same intent, budget in target-language spoken units (syllables / morae);
    - ``plain``: a bare instruction (no budget), for chat models used as plain MT;
    - ``hunyuan_mt`` / ``seed_x`` / ``translategemma`` / ``tower``: the model card's own
      template (these models are not instruction-tuned for JSON or budgets).
    """
    inp = item["inputs"]
    text = inp["text"]
    s_name, t_name = lang_name(src), lang_name(tgt)
    slot = slot_of(inp, src)
    before, after = list(inp.get("context_before") or []), list(inp.get("context_after") or [])
    if style == "opendub":
        line = {"n": 1, "speaker": inp.get("speaker") or "unknown",
                "emotion": inp.get("emotion") or "neutral", "seconds": round(slot, 2),
                "max_chars": opendub_max_chars(slot), "text": text, "context_before": before,
                "context_after": after, "terminology": inp.get("terminology"),
                "previous_translation": None, "measured_duration": None}
        user = (
            f"Translate the following 1 line(s) of dialogue into {t_name} for dubbing.\n"
            "Each object has: n (line id), speaker, emotion, seconds (slot duration), max_chars "
            "(approximate character budget for the translation so it fits the slot when spoken "
            "aloud), text (the source line), context_before/context_after (neighboring lines for "
            "disambiguation only - do not translate them).\n\n"
            f"{json.dumps([line], ensure_ascii=False)}\n\n"
            'Respond with STRICT JSON ONLY: an array of objects {"n": <int>, "translation": '
            "<string>}, one per input line, covering every n exactly once. No markdown code "
            "fences, no prose before or after the array.")
        return [{"role": "system", "content": opendub_system_prompt(s_name, t_name)},
                {"role": "user", "content": user}], True
    if style == "dub":
        unit = rate.UNIT_NAMES.get(tgt, "syllables")
        budget = rate.budget_units(slot, tgt)
        system = (
            f"You adapt {s_name} film and TV dialogue into {t_name} for a voice dub that will be "
            "spoken by an actor or a TTS voice, never read. Write natural, colloquial spoken "
            f"{t_name}; keep the meaning, names, negation, register and emotion. The line must "
            f"be speakable in {slot:.1f} seconds: aim for about {budget} {unit} of {t_name} "
            f"(within ±10%). Count {unit}, not characters. Never add notes or explanations. "
            'Reply with JSON only: {"translation": "..."}')
        user = json.dumps({"speaker": inp.get("speaker"), "emotion": inp.get("emotion"),
                           "context_before": before, "context_after": after,
                           "terminology": inp.get("terminology"), "line": text,
                           "seconds": round(slot, 2), f"target_{unit}": budget},
                          ensure_ascii=False)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], True
    if style == "plain":
        return [{"role": "user", "content": f"Translate the following {s_name} text into "
                 f"{t_name}. Output only the translation.\n\n{text}"}], False
    if style == "hunyuan_mt":
        # Hunyuan-MT model card: ZH<=>XX uses a Chinese template; XX<=>XX this one.
        return [{"role": "user", "content": f"Translate the following segment into {t_name}, "
                 f"without additional explanation.\n\n{text}"}], False
    if style == "seed_x":
        tag = {"en": "en", "hi": "hi", "ja": "ja"}.get(tgt, tgt)
        return [{"role": "user", "content": f"Translate the following {s_name} sentence into "
                 f"{t_name}:\n{text} <{tag}>"}], False
    if style == "tower":
        return [{"role": "user", "content": f"Translate the following {s_name} source text to "
                 f"{t_name}:\n{s_name}: {text}\n{t_name}: "}], False
    if style == "translategemma":
        # TranslateGemma's chat template takes a structured content item (model card); the
        # OpenAI-compatible server must pass the extra keys through to the template.
        return [{"role": "user", "content": [{"type": "text", "source_lang_code": src,
                                              "target_lang_code": tgt, "text": text}]}], False
    raise ValueError(f"unknown prompt style {style!r}")


def parse_translation(style: str, raw: str) -> str:
    if style == "opendub":
        data = parse_json(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return str(data[0].get("translation", "")).strip()
        if isinstance(data, dict) and "translation" in data:
            return str(data["translation"]).strip()
        raise ValueError(f"malformed opendub response: {raw[:200]!r}")
    if style == "dub":
        data = parse_json(raw)
        if isinstance(data, dict) and "translation" in data:
            return str(data["translation"]).strip()
        raise ValueError(f"malformed dub response: {raw[:200]!r}")
    return clean_line(raw)


GEMBA_ESA = (
    "You are an expert annotator of {src} to {tgt} translations, following the Error Span "
    "Annotation (ESA) protocol. First mark every error span in the translation with a severity: "
    "\"major\" (changes or loses meaning, or is unintelligible/ungrammatical enough to impede "
    "understanding) or \"minor\" (style, fluency, small inaccuracies). Then give an overall "
    "score from 0 (no meaning preserved) to 100 (perfect meaning and grammar), consistent with "
    "the errors you marked.{ref_note}\n\n{src} source:\n{source}\n\n{tgt} translation:\n"
    "{hypothesis}{ref_block}\n\nReply with JSON only: {{\"errors\": [{{\"span\": \"...\", "
    "\"severity\": \"major|minor\"}}], \"score\": <0-100>}}")


def gemba_messages(inputs: dict, src: str, tgt: str) -> Messages:
    ref = inputs.get("reference")
    prompt = GEMBA_ESA.format(
        src=lang_name(src), tgt=lang_name(tgt), source=inputs["source"],
        hypothesis=inputs["hypothesis"],
        ref_note=" A human reference is given for comparison only." if ref else "",
        ref_block=f"\n\n{lang_name(tgt)} reference:\n{ref}" if ref else "")
    return [{"role": "user", "content": prompt}]


# ------------------------------------------------------------------ tasks

def _call(complete: Complete, messages: Messages, params: dict, usage: dict, *,
          json_mode: bool, temperature: float | None = None, retries: int = 1,
          parse: Callable[[str], Any] = lambda s: s) -> Any:
    """complete() + parse with one retry on unparseable output; accumulates token usage."""
    err = ""
    for _ in range(retries + 1):
        msgs = messages if not err else messages + [
            {"role": "user", "content": f"Your previous reply could not be used ({err}). "
                                        "Reply again in exactly the requested format."}]
        text, u = complete(msgs, temperature=temperature, json_mode=json_mode,
                           max_tokens=int(params.get("max_tokens", 2048)))
        for k in ("input_tokens", "output_tokens"):
            usage[k] = usage.get(k, 0) + int(u.get(k) or 0)
        try:
            return parse(text)
        except ValueError as exc:
            err = str(exc)[:200]
    raise ValueError(f"unparseable model output after retry: {err}")


def task_translate(complete: Complete, item: dict, lang: str, params: dict,
                   usage: dict) -> dict:
    src, tgt = direction(lang, item)
    style = params.get("prompt", "dub")
    messages, as_json = translate_messages(style, item, src, tgt)
    n = int(params.get("n_candidates", 1))
    outs = [_call(complete, messages, params, usage, json_mode=as_json,
                  temperature=None if k == 0 else float(params.get("nbest_temperature", 0.8)),
                  parse=lambda s: parse_translation(style, s)) for k in range(n)]
    payload: dict[str, Any] = {"text": outs[0], "src_lang": src, "tgt_lang": tgt}
    if n > 1:
        payload["candidates"] = outs
    return payload


def task_segment(complete: Complete, item: dict, lang: str, params: dict, usage: dict) -> dict:
    src, _ = direction(lang, item)
    words = item["inputs"]["words"]
    max_s = float(params.get("max_line_seconds", 12.0))
    listing = "\n".join(f"{i}\t{w['start']:.2f}-{w['end']:.2f}\t{w['word']}"
                        for i, w in enumerate(words))
    system = (
        f"You split a timed {lang_name(src)} transcript into dub lines: each line will be "
        "translated and re-voiced on its own, so a line must be a complete, speakable phrase, "
        "end where the speaker actually pauses or finishes a thought, never split a "
        f"constituent (name, number, verb phrase), and last at most {max_s:.0f} seconds. "
        "Prefer boundaries at pauses of 150 ms or more. You get one word per row: index, "
        "start-end seconds, word. Reply with JSON only: {\"ends\": [indices of the last word of "
        "every line, ascending, the final index included]}")
    ends = _call(complete, [{"role": "system", "content": system},
                            {"role": "user", "content": listing}], params, usage,
                 json_mode=True, parse=lambda s: [int(x) for x in parse_json(s)["ends"]])
    return {"lines": lines_from_ends(words, ends, src)}


def lines_from_ends(words: list[dict], ends: list[int], lang: str) -> list[dict]:
    """Turn line-end word indices into line records covering every word exactly once."""
    cuts = sorted({e for e in ends if 0 <= e < len(words) - 1})
    starts = [0] + [c + 1 for c in cuts]
    stops = [c + 1 for c in cuts] + [len(words)]
    bounds = list(zip(starts, stops, strict=True))
    joiner = "" if rate.base_lang(lang) in ("ja", "zh", "th") else " "
    lines = []
    for a, b in bounds:
        seg = words[a:b]
        if not seg:
            continue
        text = joiner.join(w["word"].strip() for w in seg).strip()
        lines.append({"start_word": a, "end_word": b, "start": seg[0]["start"],
                      "end": seg[-1]["end"], "text": text})
    return lines


def task_paraphrase(complete: Complete, item: dict, lang: str, params: dict,
                    usage: dict) -> list[str]:
    _, tgt = direction(lang, item)
    inp = item["inputs"]
    n = int(params.get("n_candidates", 16))
    slot = inp.get("slot_s") or rate.predicted_seconds(inp["text"], tgt)
    msg = (f"Give {n} different ways to say this {lang_name(tgt)} dubbing line with the same "
           f"meaning, register and emotion, each speakable in about {slot:.1f} s "
           f"(~{rate.budget_units(slot, tgt)} {rate.UNIT_NAMES.get(tgt, 'syllables')}). Vary "
           "word choice and word order so the mouth shapes differ (open vowels, lip closures "
           "on m/b/p). Reply with JSON only: {\"candidates\": [\"...\"]}\n\n"
           f"Line: {inp['text']}")
    if inp.get("source_text"):
        msg += f"\nOriginal line it translates: {inp['source_text']}"
    return _call(complete, [{"role": "user", "content": msg}], params, usage, json_mode=True,
                 temperature=float(params.get("temperature", 0.9)),
                 parse=lambda s: [str(c) for c in parse_json(s)["candidates"] if str(c).strip()])


def task_gemba(complete: Complete, item: dict, lang: str, params: dict, usage: dict) -> dict:
    src, tgt = direction(lang, item)
    inp = dict(item["inputs"])
    if not params.get("use_reference", False):
        inp.pop("reference", None)

    def parse(s: str) -> dict:
        data = parse_json(s)
        score = float(data["score"])
        if not 0 <= score <= 100:
            raise ValueError(f"score {score} outside 0-100")
        return data

    data = _call(complete, gemba_messages(inp, src, tgt), params, usage, json_mode=True,
                 temperature=0.0, parse=parse)
    errs = data.get("errors") or []
    return {"metrics": {"gemba_esa": float(data["score"]),
                        "gemba_major": float(sum(e.get("severity") == "major" for e in errs)),
                        "gemba_minor": float(sum(e.get("severity") == "minor" for e in errs))},
            "errors": errs}


C5_PROMPTS = {
    "tn": (("Rewrite this {lang} text exactly as a native speaker would read it aloud: spell "
            "out numbers, dates, times, currencies, units, abbreviations, symbols and URLs in "
            "words in {lang}, keep everything else unchanged. Reply with JSON only: "
            "{{\"spoken\": \"...\"}}\n\n{text}"), "spoken"),
    "g2p": (("Give the broad IPA pronunciation of this {lang} word as phones separated by "
             "spaces (no slashes, no stress marks unless phonemic). Reply with JSON only: "
             "{{\"ipa\": \"...\"}}\n\n{text}"), "ipa"),
    "kana": (("Give the reading of this Japanese text in katakana, as it is actually "
              "pronounced in context (resolve every kanji; keep punctuation out). Reply with "
              "JSON only: {{\"kana\": \"...\"}}\n\n{text}"), "kana"),
}


def task_c5(task: str, complete: Complete, item: dict, lang: str, params: dict,
            usage: dict) -> dict:
    template, key = C5_PROMPTS[task]
    base = rate.base_lang(lang)
    msg = template.format(lang=lang_name(base), text=item["inputs"]["text"])
    out = _call(complete, [{"role": "user", "content": msg}], params, usage, json_mode=True,
                temperature=0.0, parse=lambda s: str(parse_json(s)[key]).strip())
    return {"text": out}


def cue_violations(cue: dict, limits: dict) -> list[str]:
    """Which subtitle limits a cue breaks: cps, cpl, lines (limits = {cps, cpl, lines})."""
    lines = [ln for ln in str(cue.get("text", "")).split("\n")]
    chars = sum(len(ln) for ln in lines)
    dur = max(1e-3, float(cue["end"]) - float(cue["start"]))
    bad = []
    if chars / dur > limits["cps"]:
        bad.append("cps")
    if max((len(ln) for ln in lines), default=0) > limits["cpl"]:
        bad.append("cpl")
    if len(lines) > limits.get("lines", 2):
        bad.append("lines")
    return bad


def task_condense(complete: Complete, item: dict, lang: str, params: dict,
                  usage: dict) -> dict:
    """HW-TSC IWSLT 2026 two-pass condenser: only cues that violate CPS/CPL are rewritten —
    pass 1 at temperature 0 (drop redundant words), pass 2 at 0.3 (deeper compression) for cues
    still over budget. The source line is shown so the model compresses, not paraphrases."""
    _, tgt = direction(lang, item)
    limits = item["inputs"].get("limits") or (params.get("limits") or {}).get(tgt)
    if not limits:
        raise ValueError("C6 items need inputs.limits {cps, cpl, lines} (the builders add them)")
    cues = [dict(c) for c in item["inputs"]["cues"]]
    t_name = lang_name(tgt)
    passes = ((1, 0.0, ("remove only redundant words (fillers, auxiliaries, repetitions); "
                        "keep the wording otherwise")),
              (2, 0.3, ("compress harder: shorter synonyms and restructuring are allowed, but "
                        "keep every name, number and negation")))
    for pass_no, temp, how in passes:
        for c in cues:
            if not cue_violations(c, limits):
                continue
            dur = float(c["end"]) - float(c["start"])
            budget = int(limits["cps"] * dur)
            msg = (f"This {t_name} subtitle is too long to read in {dur:.1f} s. Rewrite it in "
                   f"at most {budget} characters, at most {limits.get('lines', 2)} lines of at "
                   f"most {limits['cpl']} characters (use \\n between lines). {how}. Reply with "
                   f"JSON only: {{\"text\": \"...\"}}\n\nSubtitle: {c['text']}")
            if c.get("source"):
                msg += f"\nSource line: {c['source']}"
            new = _call(complete, [{"role": "user", "content": msg}], params, usage,
                        json_mode=True, temperature=temp,
                        parse=lambda s: str(parse_json(s)["text"]).replace("\\n", "\n").strip())
            if new:
                c["text"], c["condensed_pass"] = new, pass_no
    return {"cues": cues, "text": "\n".join(c["text"].replace("\n", " ") for c in cues)}


def run_task(complete: Complete, item: dict, lang: str, params: dict) -> dict:
    """Dispatch ``params.task`` (default translate); returns the payload plus ``_usage``."""
    usage: dict[str, int] = {}
    task = params.get("task", "translate")
    if task == "c5":                      # C5: the pack language names the task (hi.g2p …)
        task = lang.split(".", 1)[1] if "." in lang else "tn"
    if task == "translate":
        out = task_translate(complete, item, lang, params, usage)
    elif task == "segment":
        out = task_segment(complete, item, lang, params, usage)
    elif task == "paraphrase":
        out = {"candidates": task_paraphrase(complete, item, lang, params, usage)}
    elif task == "gemba":
        out = task_gemba(complete, item, lang, params, usage)
    elif task in C5_PROMPTS:
        out = task_c5(task, complete, item, lang, params, usage)
    elif task == "condense":
        out = task_condense(complete, item, lang, params, usage)
    else:
        raise ValueError(f"unknown task {task!r}")
    out["_usage"] = usage
    return out


def check_task_lang(params: dict, lang: str) -> None:
    """Fail fast (at load) when the pack language has no name for the prompts."""
    src, tgt = direction(lang)
    for code in {src, tgt} - {""}:
        lang_name(code)


def cost_usd(usage: dict, price: tuple[float, float] | list[float] | None) -> float | None:
    """List-price spend for ``usage`` at ``price = (USD per 1M input, USD per 1M output)``."""
    if not price:
        return None
    return (usage.get("input_tokens", 0) * float(price[0])
            + usage.get("output_tokens", 0) * float(price[1])) / 1e6


def finish(out: dict, price: Any) -> dict:
    """Pop the usage record into ``_cost_usd`` (and keep token counts for the audit)."""
    usage = out.pop("_usage", {}) or {}
    out["usage"] = usage
    cost = cost_usd(usage, price)
    if cost is not None:
        out["_cost_usd"] = round(cost, 6)
    return out
