"""Leaderboards (markdown) and the audit viewer (HTML) for one capability in one language."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from . import db, paths
from .judges import get_spec
from .packs import Item
from .registry import Candidate
from .runner import output_key
from .stats import build_table, corpus_value, rank, rank_corpus


def _fmt(v: float | None, metric: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if metric in ("rtfx",):
        return f"{v:.0f}×"
    if metric in ("wer", "cer", "empty_output") or metric.endswith("_rate"):
        return f"{100 * v:.2f}%"
    return f"{v:.3f}"


class _CorpusTable:
    """Per-candidate (group, prediction, human target) on the commonly scored items; quacks like
    ``stats.MetricTable`` for the leaderboard (``mean`` = the corpus statistic)."""

    def __init__(self, per: dict[str, list[tuple[str, float, float]]], fn):
        self.per, self.fn = per, fn
        self.sums = per  # keys = candidates
        self.n_items = len(next(iter(per.values()))) if per else 0
        self.groups = sorted({g for v in per.values() for g, _, _ in v})

    def mean(self, cand: str) -> float:
        import numpy as np

        v = self.per[cand]
        return corpus_value(self.fn, np.array([p for _, p, _ in v]), np.array([t for _, _, t in v]))


def _corpus_table(rows, cm, keys: list[str], judge_ids: set[str],
                  items: list[Item]) -> _CorpusTable | None:
    target = {it.id: cm.target(it) for it in items}
    per: dict[str, dict[str, tuple[str, float]]] = {}
    for r in rows:
        if r["metric"] == cm.pred and r["cand_key"] in keys and r["judge"] in judge_ids \
                and target.get(r["item"]) is not None:
            per.setdefault(r["cand_key"], {})[r["item"]] = (r["grp"], r["value"])
    if not per:
        return None
    common = set.intersection(*(set(v) for v in per.values()))
    if len(common) < 3:
        return None
    return _CorpusTable({c: [(v[i][0], v[i][1], float(target[i])) for i in sorted(common)]
                         for c, v in per.items()}, cm.fn)


def leaderboard(capability: str, lang: str, candidates: list[Candidate],
                items: list[Item] | int) -> tuple[str, list[dict[str, Any]]]:
    """Markdown table + machine-readable rows (for `select`)."""
    item_list = items if isinstance(items, list) else []
    n_items = len(items) if isinstance(items, list) else items
    spec = get_spec(capability)
    judge_ids = spec.judge_ids(lang)
    con = db.connect()
    rows = db.scores(con, capability, lang)
    by_key = {c.key: c for c in candidates}
    keys = [c.key for c in candidates if c.supports(lang)]
    primary = spec.primary_for(lang)

    def table_for(metric: str, cands: list[str]):
        if metric in spec.corpus:
            return _corpus_table(rows, spec.corpus[metric], cands, judge_ids, item_list)
        return build_table(rows, metric, cands, judge_ids)

    table = table_for(primary, keys)
    if table is None:
        con.close()
        return f"_{capability} {lang}: no scored outputs yet._\n", []
    if primary in spec.corpus:
        ranked = rank_corpus(table.per, spec.corpus[primary].fn,
                             higher_is_better=spec.higher_is_better[primary],
                             threshold=spec.threshold.get(primary, 0.0))
    else:
        ranked = rank(table, spec.higher_is_better[primary], spec.threshold.get(primary, 0.0))
    # Gates (e.g. round-trip CER ≤ 0.2 for TTS): a candidate that fails one cannot win.
    gate_fail: dict[str, list[str]] = {}
    for metric, (op, limit) in spec.gates.items():
        gt = table_for(metric, list(table.sums))
        for ck in table.sums:
            if gt is None or ck not in gt.sums:
                continue
            v = gt.mean(ck)
            if not (v <= limit if op == "<=" else v >= limit):
                gate_fail.setdefault(ck, []).append(f"{metric} {_fmt(v, metric)} fails {op} "
                                                    f"{_fmt(limit, metric)}")
    for r in ranked:
        if r.cand_key in gate_fail:
            r.in_winner_set = False
            r.notes += gate_fail[r.cand_key]
    secondary = [m for m in spec.secondary if m != primary]
    sec_tables = {m: table_for(m, list(table.sums)) for m in secondary}

    direction = "higher" if spec.higher_is_better[primary] else "lower"
    lines = [(f"### {capability} · {lang} — ranked by **{primary}** ({direction} is better) · "
              f"{table.n_items} paired items in {len(table.groups)} groups"),
             "",
             "| # | candidate | " + primary + " (95% CI) | Δ vs best (95% CI) | winner set | "
             + " | ".join(secondary) + " | ok/items | peak VRAM | licence |",
             "|---|---|---|---|---|" + "---|" * len(secondary) + "---|---|---|"]
    out_rows = []
    for n, r in enumerate(ranked, 1):
        cand = by_key[r.cand_key]
        info = db.latest_job_info(con, capability, lang, r.cand_key)
        ok = con.execute("SELECT COUNT(*) FROM outputs WHERE capability=? AND lang=? AND"
                         " cand_key=? AND status='ok'", (capability, lang, r.cand_key)).fetchone()[0]
        vram = info.get("peak_vram_mib")
        secs = [_fmt(t.mean(r.cand_key), m) if t and r.cand_key in t.sums else "–"
                for m, t in sec_tables.items()]
        diff = ("best" if n == 1 else
                f"{_fmt(r.diff_vs_best, primary)} ({_fmt(r.diff_ci[0], primary)}…"
                f"{_fmt(r.diff_ci[1], primary)})")
        lic = cand.license + ("" if cand.ship_ok is not False else " (NC)")
        lines.append(
            f"| {n} | {cand.id}{' (baseline)' if cand.baseline else ''} | "
            f"{_fmt(r.mean, primary)} ({_fmt(r.ci[0], primary)}…{_fmt(r.ci[1], primary)}) | "
            f"{diff} | {'✓' if r.in_winner_set else ''} | " + " | ".join(secs)
            + f" | {ok}/{n_items} | {f'{vram / 1024:.1f} GB' if vram else '–'} | {lic} |")
        out_rows.append({"candidate": cand.id, "cand_key": r.cand_key, "rank": n,
                         "metric": primary, "mean": r.mean, "ci": r.ci,
                         "in_winner_set": r.in_winner_set, "p_adj": r.p_adj,
                         "ship_ok": cand.ship_ok, "peak_vram_mib": vram, "notes": r.notes,
                         "hardware": info.get("hardware"),
                         "secondary": {m: (t.mean(r.cand_key) if t and r.cand_key in t.sums
                                           else None) for m, t in sec_tables.items()}})
    notes = [f"- {by_key[r.cand_key].id}: {'; '.join(r.notes)}" for r in ranked if r.notes]
    if notes:
        lines += ["", "Notes:", *notes]
    skipped = [c.id for c in candidates if not c.supports(lang)]
    if skipped:
        lines += ["", f"Not evaluated in {lang} (language not declared): {', '.join(skipped)}"]
    con.close()
    return "\n".join(lines) + "\n", out_rows


# ---------------------------------------------------------------- audit viewer

def _rel(path: str | Path) -> str | None:
    try:
        return "/files/" + str(Path(path).resolve().relative_to(paths.data_dir().resolve()))
    except ValueError:
        return None


def _media(value: Any) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    ext = Path(value).suffix.lower()
    kind = ("audio" if ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a") else
            "video" if ext in (".mp4", ".webm", ".mkv", ".mov") else None)
    url = _rel(value) if kind and Path(value).exists() else None
    return {"kind": kind, "url": url} if url else None


def audit_data(capability: str, lang: str, candidates: list[Candidate],
               items: list[Item]) -> dict[str, Any]:
    try:
        spec = get_spec(capability)
    except KeyError:
        spec = None
    judge_ids = spec.judge_ids(lang) if spec else set()
    con = db.connect()
    marks = db.audit_marks(con, capability, lang)
    scores: dict[tuple[str, str], dict[str, float]] = {}
    for r in db.scores(con, capability, lang):
        if spec is None or r["judge"] in judge_ids:
            scores.setdefault((r["cand_key"], r["item"]), {})[r["metric"]] = r["value"]
    cands = [c for c in candidates if c.supports(lang)]
    data_items = []
    for it in items:
        entry: dict[str, Any] = {
            "id": it.id, "group": it.group, "meta": it.meta,
            "inputs": {k: _media(v) or {"text": v if not isinstance(v, str) or len(v) < 2000
                                        else v[:2000] + "…"} for k, v in it.inputs.items()},
            "refs": {k: _media(v) or {"text": v} for k, v in it.refs.items()},
            "outputs": {}}
        for c in cands:
            row = db.get_output(con, output_key(c, it))
            if row is None:
                continue
            o: dict[str, Any] = {"status": row["status"], "error": row["error"],
                                 "seconds": row["seconds"],
                                 "metrics": scores.get((c.key, it.id), {}),
                                 "mark": marks.get((c.key, it.id))}
            jpath = db.output_path(row).with_suffix(".json")
            if jpath.exists():
                payload = json.loads(jpath.read_text())
                o["text"] = payload.get("text")
                o["json_url"] = _rel(jpath)
            prefix = db.output_path(row)
            o["files"] = [m for p in sorted(prefix.parent.glob(prefix.name + ".*"))
                          if p.suffix != ".json" and (m := _media(str(p)))]
            entry["outputs"][c.key] = o
        data_items.append(entry)
    con.close()
    return {"capability": capability, "lang": lang,
            "primary": spec.primary_for(lang) if spec else None,
            "higher_is_better": spec.higher_is_better if spec else {},
            "unspaced": lang in ("ja", "zh", "th"),
            "candidates": [{"key": c.key, "id": c.id, "baseline": c.baseline,
                            "license": c.license, "ship_ok": c.ship_ok} for c in cands],
            "items": data_items}


def md_to_html(md: str) -> str:
    """Just enough markdown for leaderboards: headings, pipe tables, paragraphs, **bold**."""
    def inline(t: str) -> str:
        t = html.escape(t)
        parts = t.split("**")
        return "".join(f"<b>{p}</b>" if k % 2 else p for k, p in enumerate(parts))

    out, table = [], []
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                table.append(cells)
            continue
        if table:
            head, *body = table
            out.append("<div class=tw><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>"
                       for c in head) + "</tr></thead><tbody>" + "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                + "</tbody></table></div>")
            table = []
        if line.startswith("#"):
            out.append(f"<h2>{inline(line.lstrip('#').strip())}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    return "\n".join(out)


def audit_html(data: dict[str, Any], leaderboard_md: str = "") -> str:
    title = f"Arena audit · {data['capability']} · {data['lang']}"
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_AUDIT_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__LEADERBOARD__", md_to_html(leaderboard_md))
            .replace("__DATA__", blob))


_AUDIT_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f5fb;--raised:#fff;--line:#e3def0;--ink:#1c1733;--ink2:#5d5780;--ok:#0f8a6e;
--bad:#c62f5b;--del:#c62f5b22;--ins:#e0a10033;--accent:#6b3fd4;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#131022;--raised:#1c1831;
--line:#2e2850;--ink:#efeaff;--ink2:#a9a1cf;--ok:#3ad6ae;--bad:#ff6f98;--del:#ff6f9833;
--ins:#ffc94d33;--accent:#a98bff;color-scheme:dark}}
:root[data-theme=dark]{--bg:#131022;--raised:#1c1831;--line:#2e2850;--ink:#efeaff;--ink2:#a9a1cf;
--ok:#3ad6ae;--bad:#ff6f98;--del:#ff6f9833;--ins:#ffc94d33;--accent:#a98bff;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,sans-serif}
header{padding:16px;border-bottom:1px solid var(--line);background:var(--raised);position:sticky;
top:0;z-index:2;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center}
h1{font-size:18px;margin:0}select,input,button{font:inherit;color:inherit;background:var(--bg);
border:1px solid var(--line);border-radius:6px;padding:4px 8px}
details.lb{margin:12px 16px}.lbody h2{font-size:14px;margin:8px 0}.lbody p{margin:6px 0;
color:var(--ink2);font-size:13px}.tw{overflow-x:auto}.lbody table{border-collapse:collapse;
font-size:12.5px;background:var(--raised);border:1px solid var(--line);border-radius:6px}
.lbody th,.lbody td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;
white-space:nowrap}.lbody th{font-size:11px;color:var(--ink2);font-weight:600}
main{padding:0 16px 64px}.item{background:var(--raised);border:1px solid var(--line);
border-radius:8px;margin:12px 0;padding:12px;overflow:hidden}
.ihead{display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:var(--ink2);font-size:12px}
.ihead b{color:var(--ink);font-size:13px}.ref{margin:6px 0;padding:6px 8px;border-left:3px solid
var(--accent);background:var(--bg);border-radius:0 6px 6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;margin-top:8px}
.cand{border:1px solid var(--line);border-radius:6px;padding:8px;min-width:0}
.cand h3{font-size:12px;margin:0 0 4px;display:flex;gap:6px;justify-content:space-between}
.m{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink2)}
.m .bad{color:var(--bad);font-weight:600}.txt{overflow-wrap:anywhere}
del{background:var(--del);text-decoration:line-through}ins{background:var(--ins);
text-decoration:none}audio,video{width:100%;margin-top:4px}video{max-height:240px}
.marks{display:flex;gap:4px;margin-top:6px;flex-wrap:wrap}.marks button{padding:2px 8px;
font-size:12px}.marks button.on-ok{border-color:var(--ok);color:var(--ok)}
.marks button.on-bad{border-color:var(--bad);color:var(--bad)}.warn{color:var(--bad);font-size:12px}
.err{color:var(--bad);font-size:12px}
</style></head><body>
<header><h1>__TITLE__</h1>
<label>sort <select id="sort"><option value="spread">candidates disagree most</option>
<option value="worst">worst first (primary)</option><option value="id">item id</option></select></label>
<label>show <select id="filter"><option value="all">all items</option>
<option value="unmarked">unmarked</option><option value="bad">marked bad</option></select></label>
<label>search <input id="q" size="16"></label><span id="count" class="m"></span>
<span id="srv" class="warn"></span></header>
<details class="lb" open><summary>Leaderboard</summary><div class="lbody">__LEADERBOARD__</div></details>
<main id="list"></main>
<script>
const D=__DATA__;const P=D.primary;const HIB=D.higher_is_better[P]??false;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let SERVED=location.protocol.startsWith('http');
if(!SERVED)document.getElementById('srv').textContent='opened as a file: marks stay in this browser only';
function toks(s){s=String(s??'').normalize('NFKC').toLowerCase().replace(/[\p{P}]/gu,' ');
 return D.unspaced?[...s.replace(/\s+/g,'')]:s.split(/\s+/).filter(Boolean)}
function diff(ref,hyp){const a=toks(ref),b=toks(hyp),n=a.length,m=b.length;if(n*m>250000)return esc(hyp);
 const d=Array.from({length:n+1},()=>new Int32Array(m+1));
 for(let i=n-1;i>=0;i--)for(let j=m-1;j>=0;j--)d[i][j]=a[i]===b[j]?d[i+1][j+1]+1:Math.max(d[i+1][j],d[i][j+1]);
 let i=0,j=0,o=[],sep=D.unspaced?'':' ';while(i<n||j<m){
  if(i<n&&j<m&&a[i]===b[j]){o.push(esc(b[j]));i++;j++}
  else if(j<m&&(i>=n||d[i][j+1]>=d[i+1][j])){o.push('<ins>'+esc(b[j])+'</ins>');j++}
  else{o.push('<del>'+esc(a[i])+'</del>');i++}}
 return o.join(sep)}
function media(m){if(!m)return'';if(m.kind==='audio')return`<audio controls preload="none" src="${m.url}"></audio>`;
 if(m.kind==='video')return`<video controls preload="none" src="${m.url}"></video>`;
 return m.text!==undefined?`<div class="txt">${esc(typeof m.text==='string'?m.text:JSON.stringify(m.text))}</div>`:''}
function fmt(k,v){if(v==null)return'–';if(k==='rtfx')return v.toFixed(0)+'×';
 if(['wer','cer','empty_output'].includes(k))return(100*v).toFixed(1)+'%';return(+v).toFixed(3)}
const LS='arena-marks-'+D.capability+'-'+D.lang;let local={};try{local=JSON.parse(localStorage.getItem(LS)||'{}')}catch(e){}
function markOf(ck,it,o){return local[ck+'|'+it]||o.mark}
async function mark(ck,it,verdict){let note='';if(verdict==='note'){note=prompt('Note for '+it)||'';if(!note)return}
 const rec={verdict,note};local[ck+'|'+it]=rec;try{localStorage.setItem(LS,JSON.stringify(local))}catch(e){}
 if(SERVED){try{const r=await fetch('/api/audit',{method:'POST',headers:{'content-type':'application/json'},
  body:JSON.stringify({capability:D.capability,lang:D.lang,cand_key:ck,item:it,verdict,note})});
  if(!r.ok)throw 0}catch(e){document.getElementById('srv').textContent='could not save mark to server'}}
 render()}
function prim(it,ck){const o=it.outputs[ck];return o&&o.metrics?o.metrics[P]:undefined}
function spread(it){const v=D.candidates.map(c=>prim(it,c.key)).filter(x=>x!=null);return v.length?Math.max(...v)-Math.min(...v):-1}
function worst(it){const v=D.candidates.map(c=>prim(it,c.key)).filter(x=>x!=null);if(!v.length)return-1e9;return HIB?-Math.min(...v):Math.max(...v)}
function render(){const s=document.getElementById('sort').value,f=document.getElementById('filter').value,
 q=document.getElementById('q').value.toLowerCase();let items=D.items.slice();
 if(q)items=items.filter(it=>JSON.stringify(it).toLowerCase().includes(q));
 if(f!=='all')items=items.filter(it=>D.candidates.some(c=>{const o=it.outputs[c.key];if(!o)return false;
  const mk=markOf(c.key,it.id,o);return f==='bad'?mk&&mk.verdict==='bad':!mk}));
 if(s==='spread')items.sort((a,b)=>spread(b)-spread(a));else if(s==='worst')items.sort((a,b)=>worst(b)-worst(a));
 else items.sort((a,b)=>a.id.localeCompare(b.id));
 document.getElementById('count').textContent=items.length+' items';
 const ref=it=>it.refs.text&&it.refs.text.text;
 document.getElementById('list').innerHTML=items.slice(0,300).map(it=>`<section class="item">
 <div class="ihead"><b>${esc(it.id)}</b><span>group ${esc(it.group)}</span>${it.meta.duration_s?`<span>${(+it.meta.duration_s).toFixed(1)} s</span>`:''}</div>
 ${Object.entries(it.inputs).map(([k,m])=>`<div class="m">${esc(k)}</div>${media(m)}`).join('')}
 ${Object.entries(it.refs).map(([k,m])=>`<div class="ref"><span class="m">ref ${esc(k)}</span>${media(m)}</div>`).join('')}
 <div class="grid">${D.candidates.map(c=>{const o=it.outputs[c.key];if(!o)return`<div class="cand"><h3>${esc(c.id)}</h3><span class="m">not run</span></div>`;
  const mk=markOf(c.key,it.id,o)||{};const ms=Object.entries(o.metrics||{}).map(([k,v])=>`<span class="${k===P&&v>0.3&&!HIB?'bad':''}">${k} ${fmt(k,v)}</span>`).join(' · ');
  return`<div class="cand"><h3><span>${esc(c.id)}${c.baseline?' ·base':''}</span><span class="m">${o.seconds!=null?o.seconds.toFixed(2)+'s':''}</span></h3>
  <div class="m">${ms}</div>${o.status!=='ok'?`<div class="err">${esc(o.error)}</div>`:''}
  ${o.text!=null?`<div class="txt">${ref(it)!=null?diff(ref(it),o.text):esc(o.text)}</div>`:''}
  ${(o.files||[]).map(media).join('')}${o.json_url?`<div class="m"><a href="${o.json_url}" target="_blank">json</a></div>`:''}
  <div class="marks"><button class="${mk.verdict==='ok'?'on-ok':''}" onclick="mark('${c.key}','${esc(it.id)}','ok')">ok</button>
  <button class="${mk.verdict==='bad'?'on-bad':''}" onclick="mark('${c.key}','${esc(it.id)}','bad')">bad</button>
  <button onclick="mark('${c.key}','${esc(it.id)}','note')">note</button>${mk.note?`<span class="m">${esc(mk.note)}</span>`:''}</div></div>`}).join('')}</div></section>`).join('')
 +(items.length>300?'<p class="m">showing first 300; narrow with search or filter</p>':'')}
['sort','filter','q'].forEach(id=>document.getElementById(id).addEventListener('input',render));render();
</script></body></html>
"""
