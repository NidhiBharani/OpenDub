"""SQLite results store. One row per (output, metric) so every statistic stays paired per item."""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS outputs (
  key TEXT PRIMARY KEY,           -- sha(candidate key + item input hash)
  capability TEXT, lang TEXT, candidate TEXT, cand_key TEXT, item TEXT,
  input_hash TEXT, path TEXT,     -- output prefix: <path>.json (+ extra files)
  status TEXT, error TEXT, seconds REAL, job TEXT, created REAL,
  cost_usd REAL, host TEXT        -- API spend for this item; machine that generated it
);
CREATE INDEX IF NOT EXISTS outputs_by_cap ON outputs(capability, lang, cand_key);
CREATE TABLE IF NOT EXISTS jobs (
  job TEXT PRIMARY KEY, capability TEXT, lang TEXT, candidate TEXT, cand_key TEXT,
  items INTEGER, ok INTEGER, info TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS scores (
  output_key TEXT, judge TEXT, metric TEXT, ref_hash TEXT,
  capability TEXT, lang TEXT, cand_key TEXT, item TEXT, grp TEXT,
  value REAL, weight REAL, created REAL,
  PRIMARY KEY (output_key, judge, metric, ref_hash)
);
CREATE INDEX IF NOT EXISTS scores_by_cap ON scores(capability, lang, metric);
CREATE TABLE IF NOT EXISTS audit (
  capability TEXT, lang TEXT, cand_key TEXT, item TEXT, verdict TEXT, note TEXT, ts REAL
);
"""


_MIGRATIONS = {"outputs": {"cost_usd": "REAL", "host": "TEXT"}}


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or paths.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    for table, cols in _MIGRATIONS.items():
        have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    return con


def rel_output(prefix: Path) -> str:
    """Output prefixes are stored relative to the outputs dir so a DB (and its files) can move
    between machines (Thalassa → Spark) and be merged."""
    try:
        return str(prefix.resolve().relative_to(paths.outputs_dir().resolve()))
    except ValueError:
        return str(prefix)


def output_path(row: sqlite3.Row | dict) -> Path:
    """Absolute output prefix for a row (older rows stored absolute paths)."""
    p = Path(row["path"])
    return p if p.is_absolute() else paths.outputs_dir() / p


def merge(other_data_dir: Path) -> dict[str, int]:
    """Fold another machine's arena (``<data>/arena``) into this one: rows + output files.

    Rows are content-addressed, so the same output generated on both machines collapses to one.
    """
    import shutil

    src_arena = Path(other_data_dir) / "arena"
    src_db = src_arena / "arena.sqlite"
    if not src_db.exists():
        raise FileNotFoundError(src_db)
    connect(src_db).close()  # migrate the source schema too
    if (src_arena / "outputs").exists():
        shutil.copytree(src_arena / "outputs", paths.outputs_dir(), dirs_exist_ok=True)
    con = connect()
    con.execute("ATTACH DATABASE ? AS other", (str(src_db),))
    counts = {}
    for table in ("outputs", "jobs", "scores", "audit"):
        before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols = ",".join(r["name"] for r in con.execute(f"PRAGMA table_info({table})"))
        verb = "INSERT INTO" if table == "audit" else "INSERT OR IGNORE INTO"
        con.execute(f"{verb} {table} ({cols}) SELECT {cols} FROM other.{table}")
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] - before
    con.commit()
    con.execute("DETACH DATABASE other")
    con.close()
    return counts


def get_output(con: sqlite3.Connection, key: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM outputs WHERE key=?", (key,)).fetchone()


def put_output(con: sqlite3.Connection, **row: Any) -> None:
    row.setdefault("created", time.time())
    cols = ",".join(row)
    con.execute(f"INSERT OR REPLACE INTO outputs ({cols}) VALUES ({','.join('?' * len(row))})",
                tuple(row.values()))


def put_job(con: sqlite3.Connection, job: str, info: dict[str, Any], **row: Any) -> None:
    con.execute("INSERT OR REPLACE INTO jobs (job, capability, lang, candidate, cand_key, items,"
                " ok, info, created) VALUES (?,?,?,?,?,?,?,?,?)",
                (job, row["capability"], row["lang"], row["candidate"], row["cand_key"],
                 row["items"], row["ok"], json.dumps(info), time.time()))


def has_score(con: sqlite3.Connection, output_key: str, judge: str, ref_hash: str) -> bool:
    return con.execute("SELECT 1 FROM scores WHERE output_key=? AND judge=? AND ref_hash=? LIMIT 1",
                       (output_key, judge, ref_hash)).fetchone() is not None


def put_scores(con: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> None:
    now = time.time()
    con.executemany(
        "INSERT OR REPLACE INTO scores (output_key, judge, metric, ref_hash, capability, lang,"
        " cand_key, item, grp, value, weight, created) VALUES"
        " (:output_key,:judge,:metric,:ref_hash,:capability,:lang,:cand_key,:item,:grp,:value,"
        ":weight,:created)",
        [{**r, "created": now} for r in rows])


def scores(con: sqlite3.Connection, capability: str, lang: str) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM scores WHERE capability=? AND lang=?",
                       (capability, lang)).fetchall()


def latest_job_info(con: sqlite3.Connection, capability: str, lang: str,
                    cand_key: str) -> dict[str, Any]:
    row = con.execute("SELECT info FROM jobs WHERE capability=? AND lang=? AND cand_key=?"
                      " ORDER BY created DESC LIMIT 1", (capability, lang, cand_key)).fetchone()
    return json.loads(row["info"]) if row else {}


def add_audit(con: sqlite3.Connection, capability: str, lang: str, cand_key: str, item: str,
              verdict: str, note: str = "") -> None:
    con.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?)",
                (capability, lang, cand_key, item, verdict, note, time.time()))
    con.commit()


def audit_marks(con: sqlite3.Connection, capability: str, lang: str) -> dict[tuple[str, str], dict]:
    """Latest mark per (cand_key, item)."""
    marks: dict[tuple[str, str], dict] = {}
    for r in con.execute("SELECT * FROM audit WHERE capability=? AND lang=? ORDER BY ts",
                         (capability, lang)):
        marks[(r["cand_key"], r["item"])] = {"verdict": r["verdict"], "note": r["note"]}
    return marks
