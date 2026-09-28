"""Long-lived ASR verifier for method workers (best-of-N, splice): a stdin/stdout JSON-lines loop,
not an arena job worker. Started once per method job by ``d_voice``-style callers, it loads
faster-whisper once and answers ``{"audio": path, "lang": "en"}`` with ``{"text": …}``.

Run in the ``server`` env (faster-whisper + CTranslate2 there). Deliberately NOT the model the
reporting ASR judge uses (large-v3 + Qwen3-ASR): default large-v3-turbo, so the verifier used
for re-ranking is never the reporter (docs/plans/model-ranking.md §4.7) — still Whisper-family;
swap in a non-Whisper verifier when one is available in an env.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    from _sdk import CUDA12_LIBS, preload_pip_cuda_libs

    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    preload_pip_cuda_libs(*CUDA12_LIBS)
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg.get("model", "large-v3-turbo"), device=cfg.get("device", "cuda"),
                         compute_type=cfg.get("compute_type", "float16"))
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            segs, _ = model.transcribe(req["audio"], language=req.get("lang") or None,
                                       beam_size=int(cfg.get("beam_size", 5)),
                                       condition_on_previous_text=False, vad_filter=False)
            print(json.dumps({"text": " ".join(s.text.strip() for s in segs)},
                             ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001 - report and keep serving
            print(json.dumps({"error": repr(exc), "text": ""}), flush=True)


if __name__ == "__main__":
    main()
