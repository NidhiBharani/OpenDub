"""A2 worker: DeepFilterNet3 (MIT / Apache-2.0; ``deepfilternet`` 0.5.6), 48 kHz only.
``init_df()`` → (model, df_state, _); ``load_audio(path, sr=df_state.sr())``; ``enhance(model,
df_state, audio)``; ``save_audio`` (README checked 2026-09-28). params: atten_lim_db (optional
attenuation limit — keeps some noise instead of over-suppressing).
"""
from __future__ import annotations

from pathlib import Path

from _sdk import serve


def load(params: dict, lang: str):
    from df.enhance import init_df

    model, st, _ = init_df()
    return {"model": model, "st": st, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    from df.enhance import enhance, load_audio, save_audio

    st = state["st"]
    audio, _ = load_audio(item["inputs"]["audio"], sr=st.sr())
    kw = {}
    if state["params"].get("atten_lim_db") is not None:
        kw["atten_lim_db"] = float(state["params"]["atten_lim_db"])
    y = enhance(state["model"], st, audio, **kw)
    dst = out.with_suffix(".wav")
    save_audio(str(dst), y, st.sr())
    return {"files": {"audio": str(dst)}, "sample_rate": st.sr()}


if __name__ == "__main__":
    serve(load, run)
