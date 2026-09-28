"""A7 worker: 3D-Speaker (Alibaba DAMO; Apache-2.0) ERes2NetV2 / CAM++ / ERes2Net through the
ModelScope speaker-verification pipeline with ``output_emb=True`` (embeddings returned in
``embs``). Model ids (ModelScope): iic/speech_eres2netv2_sv_zh-cn_16k-common,
iic/speech_campplus_sv_zh-cn_16k-common (Mandarin, 200k speakers), iic/speech_campplus_sv_en_
voxceleb_16k, iic/speech_eres2net_sv_en_voxceleb_16k. The pipeline call shape is from the
3D-Speaker README; ``output_emb`` is not re-verified. params: model.
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve
from a7_common import score_item


def load(params: dict, lang: str):
    from modelscope.pipelines import pipeline

    sv = pipeline(task="speaker-verification",
                  model=params.get("model", "iic/speech_eres2netv2_sv_zh-cn_16k-common"))
    return {"sv": sv, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    import numpy as np

    def embed(path: str):
        res = state["sv"]([path, path], output_emb=True)
        return np.asarray(res["embs"][0], dtype=np.float32)

    return score_item(embed, item, out, state["params"])


if __name__ == "__main__":
    serve(load, run)
