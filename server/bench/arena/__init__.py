"""Model arena: rank candidate models per capability and per language on frozen eval packs.

Plan: docs/plans/model-ranking.md. The pieces:

- ``registry``   candidate models per capability (``bench/candidates/<ID>.yaml``) + worker envs
- ``packs``      eval packs: frozen inputs + references (``data/eval/<ID>/<lang>/manifest.jsonl``)
- ``runner``     runs each candidate as an out-of-process worker; outputs are cached and kept
- ``judges``     per-item metrics over cached outputs (re-scoring never regenerates)
- ``stats``      paired bootstrap, Holm correction, winner sets
- ``report``     leaderboard markdown + the audit viewer (every output, per stage, per candidate)

Everything the user audits lives under ``data/arena/`` and is never evicted by cache cleanup.
"""
