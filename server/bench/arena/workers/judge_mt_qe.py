"""MT quality-estimation judge worker (``judgelib.mt_qe``) and the C4 candidate worker.

item.inputs: {source, hypothesis, reference?, src_lang, tgt_lang}. Returns
``{"metrics": {"<mode>_<model>": score}}`` where mode is ``qe`` (reference-free) or ``hyb`` (the
reference was used), so a spec can run both without the metric names colliding.

params:
  model          metricx-24-hybrid-{large,xl,xxl} | cometkiwi-da | cometkiwi-da-xl |
                 cometkiwi-da-xxl | xcomet-xl | xcomet-xxl   (or hf_id: override the repo)
  use_reference  default true (judgelib only passes a reference when asked); C4 candidates set
                 false to rank pure QE
  batch_size     default 8; dtype (MetricX: bfloat16 default)

MetricX-24 (google-research/metricx, Apache-2.0): mT5 encoder with a regression head; the
hybrid checkpoints score with or without a reference. Input text is
``"source: … candidate: …[ reference: …]"``, max 1536 tokens, EOS removed, prediction clipped to
[0, 25] (an error score: lower is better). The repo (pinned at commit fc4978e) is cloned by
the env recipe and put on ``sys.path``; the tokenizer is google/mt5-xl for every size, as in
the MetricX README.

COMET family (unbabel-comet): CometKiwi (wmt22 is InfoXLM-large; wmt23 XL/XXL are XLM-R XL/XXL)
and xCOMET (can run with or without a reference). **CC-BY-NC-SA-4.0: evaluation only**, and the
wmt23 CometKiwi repos are gated on Hugging Face (accept the terms, set HF_TOKEN).

All items of the job are scored in one batched pass at load time (COMET's Lightning predict
per item would be very slow); ``run`` returns the cached score.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from _sdk import serve

MODELS = {
    # the MetricX README uses the google/mt5-xl tokenizer for every size
    "metricx-24-hybrid-large": ("metricx", "google/metricx-24-hybrid-large-v2p6", "google/mt5-xl"),
    "metricx-24-hybrid-xl": ("metricx", "google/metricx-24-hybrid-xl-v2p6", "google/mt5-xl"),
    "metricx-24-hybrid-xxl": ("metricx", "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
                              "google/mt5-xl"),
    "cometkiwi-da": ("comet", "Unbabel/wmt22-cometkiwi-da", None),
    "cometkiwi-da-xl": ("comet", "Unbabel/wmt23-cometkiwi-da-xl", None),
    "cometkiwi-da-xxl": ("comet", "Unbabel/wmt23-cometkiwi-da-xxl", None),
    "xcomet-xl": ("comet", "Unbabel/XCOMET-XL", None),
    "xcomet-xxl": ("comet", "Unbabel/XCOMET-XXL", None),
}
METRICX_SRC = Path(os.environ.get("METRICX_SRC", Path.home() / ".opendub" / "src" / "metricx"))


def _job_items() -> list[dict]:
    """The job's items (argv ``--job``), so load() can batch-score everything once."""
    if "--job" not in sys.argv:
        return []
    job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text())
    return job.get("items", [])


def _examples(items: list[dict], use_ref: bool) -> list[dict]:
    exs = []
    for it in items:
        inp = it["inputs"]
        ref = inp.get("reference") if use_ref else None
        exs.append({"id": it["id"], "src": inp["source"], "mt": inp["hypothesis"],
                    "ref": ref})
    return exs


def _metricx_scores(exs: list[dict], hf_id: str, tok_id: str, params: dict) -> list[float]:
    import torch
    import transformers

    sys.path.insert(0, str(METRICX_SRC))
    from metricx24 import models  # google-research/metricx (cloned by the env recipe)

    dtype = getattr(torch, params.get("dtype", "bfloat16"))
    tok = transformers.AutoTokenizer.from_pretrained(tok_id)
    model = models.MT5ForRegression.from_pretrained(hf_id, torch_dtype=dtype)
    model.to("cuda").eval()
    max_len = int(params.get("max_input_length", 1536))
    texts = [f"source: {e['src']} candidate: {e['mt']}"
             + (f" reference: {e['ref']}" if e["ref"] else "") for e in exs]
    scores: list[float] = []
    bs = int(params.get("batch_size", 8))
    for i in range(0, len(texts), bs):
        enc = [tok(t, max_length=max_len, truncation=True)["input_ids"][:-1]  # drop EOS
               for t in texts[i:i + bs]]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for k, e in enumerate(enc):
            ids[k, :len(e)] = torch.tensor(e)
            mask[k, :len(e)] = 1
        with torch.inference_mode():
            pred = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).predictions
        scores += [float(min(25.0, max(0.0, v))) for v in pred.float().cpu().tolist()]
    return scores


def _comet_scores(exs: list[dict], hf_id: str, params: dict) -> list[float]:
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(hf_id))
    data = [{"src": e["src"], "mt": e["mt"], **({"ref": e["ref"]} if e["ref"] else {})}
            for e in exs]
    out = model.predict(data, batch_size=int(params.get("batch_size", 8)), gpus=1,
                        progress_bar=False)
    return [float(s) for s in out.scores]


def _score(items: list[dict], params: dict) -> dict[str, tuple[str, float]]:
    name = params.get("model", "metricx-24-hybrid-xl")
    family, hf_id, tok_id = MODELS[name]
    hf_id = params.get("hf_id", hf_id)
    use_ref = bool(params.get("use_reference", True))
    if "cometkiwi" in name:
        use_ref = False                           # CometKiwi is reference-free by design
    exs = _examples(items, use_ref)
    out: dict[str, tuple[str, float]] = {}
    # score ref / no-ref groups separately (COMET needs a uniform batch)
    for with_ref in (False, True):
        group = [e for e in exs if bool(e["ref"]) == with_ref]
        if not group:
            continue
        scores = (_metricx_scores(group, hf_id, tok_id, params) if family == "metricx"
                  else _comet_scores(group, hf_id, params))
        mode = "hyb" if with_ref else "qe"
        for e, s in zip(group, scores, strict=True):
            out[e["id"]] = (f"{mode}_{name}", s)
    return out


def load(params: dict, lang: str):
    if params.get("model", "metricx-24-hybrid-xl") not in MODELS:
        raise ValueError(f"unknown QE model {params.get('model')!r}; known: {sorted(MODELS)}")
    items = [it for it in _job_items() if it["inputs"].get("source")
             and it["inputs"].get("hypothesis")]
    return {"params": params, "scores": _score(items, params) if items else {}}


def run(state: dict, item: dict, out: Path) -> dict:
    hit = state["scores"].get(item["id"])
    if hit is None:                                # not in the job file (direct call)
        hit = _score([item], state["params"])[item["id"]]
    name, value = hit
    return {"metrics": {name: value}}


def describe(state: dict) -> dict:
    p = state["params"]
    family, hf_id, tok_id = MODELS[p.get("model", "metricx-24-hybrid-xl")]
    info = {"model": p.get("hf_id", hf_id), "family": family, "tokenizer": tok_id}
    if family == "metricx" and (METRICX_SRC / ".git").exists():
        head = (METRICX_SRC / ".git" / "HEAD").read_text().strip()
        info["metricx_commit"] = head
    return info


if __name__ == "__main__":
    serve(load, run, describe)
