"""Tiny audit server: renders audit pages fresh (so marks show up) and stores marks in SQLite.

    GET  /                      index of capability/lang packs that have outputs
    GET  /audit/<ID>/<lang>     audit viewer
    GET  /files/<path>          media under the data dir (Range supported, for seeking)
    POST /api/audit             {"capability","lang","cand_key","item","verdict","note"}
"""
from __future__ import annotations

import html
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from . import db, paths
from .packs import load_pack
from .registry import load_candidates
from .report import audit_data, audit_html, leaderboard


def _ladder_name(p) -> str:
    try:
        d = json.loads(p.read_text())
        return f"{d['name']} ({d['src']}→{d['tgt']})"
    except (OSError, ValueError, KeyError):
        return p.stem


def _index() -> str:
    con = db.connect()
    rows = con.execute("SELECT capability, lang, COUNT(DISTINCT cand_key) c, COUNT(*) n"
                       " FROM outputs GROUP BY capability, lang ORDER BY capability, lang")
    links = "".join(f'<li><a href="/audit/{r["capability"]}/{r["lang"]}">{r["capability"]} · '
                    f'{r["lang"]}</a> — {r["c"]} candidates, {r["n"]} outputs</li>' for r in rows)
    con.close()
    ladders = sorted((paths.arena_dir() / "ladder").glob("prj_*.json"))
    links += "".join(f'<li><a href="/ladder/{p.stem}">ladder · {html.escape(_ladder_name(p))}</a>'
                     f'</li>' for p in ladders)
    return (f"<!doctype html><meta charset=utf-8><title>Arena audit</title>"
            f"<body style='font:15px system-ui;margin:24px'><h1>Arena audit</h1>"
            f"<ul>{links or '<li>no outputs yet</li>'}</ul>")


def _ladder(pid: str) -> str:
    """Saved ladder data (round-trip checks are slow) with the latest audit marks merged in."""
    from . import ladder

    saved = paths.arena_dir() / "ladder" / f"{pid}.json"
    data = json.loads(saved.read_text()) if saved.exists() else ladder.build(pid)
    con = db.connect()
    marks = db.audit_marks(con, f"ladder:{pid}", data["tgt"])
    con.close()
    for row in data["rows"]:
        row["marks"] = {st: marks.get((st, row["id"])) for st in ("A4", "C2", "D1", "E1")}
    return ladder.render(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "arena-audit/1"

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8",
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(self.path.split("?")[0])
        try:
            if path == "/":
                return self._send(200, _index().encode())
            m = re.fullmatch(r"/audit/([A-Z]\d+)/([a-z-]+)", path)
            if m:
                cap, lang = m.groups()
                cands = load_candidates(cap)
                items = load_pack(cap, lang, split=None)
                md, _ = leaderboard(cap, lang, cands, items)
                page = audit_html(audit_data(cap, lang, cands, items), md)
                return self._send(200, page.encode())
            m = re.fullmatch(r"/ladder/(prj_[A-Za-z0-9]+)", path)
            if m:
                return self._send(200, _ladder(m.group(1)).encode())
            if path.startswith("/files/"):
                return self._file(path[len("/files/"):])
            self._send(404, b"not found", "text/plain")
        except FileNotFoundError as e:
            self._send(404, html.escape(str(e)).encode(), "text/plain; charset=utf-8")

    do_HEAD = do_GET

    def _file(self, rel: str) -> None:
        root = paths.data_dir().resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        size = target.stat().st_size
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            end = min(end, size - 1)
            with target.open("rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            return self._send(206, body, ctype, {"Content-Range": f"bytes {start}-{end}/{size}",
                                                 "Accept-Ranges": "bytes"})
        self._send(200, target.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

    def do_POST(self) -> None:
        if self.path != "/api/audit":
            return self._send(404, b"not found", "text/plain")
        try:
            rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            con = db.connect()
            db.add_audit(con, rec["capability"], rec["lang"], rec["cand_key"], rec["item"],
                         rec["verdict"], rec.get("note", ""))
            con.close()
            self._send(200, b'{"ok":true}', "application/json")
        except (KeyError, ValueError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass


def serve(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"arena audit on http://{host}:{port}/  (data: {paths.data_dir()})")
    httpd.serve_forever()

