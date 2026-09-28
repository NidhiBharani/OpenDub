"""Stage ladder: audit one real dub run stage by stage, one row per line, errors carried along.

Columns follow the pipeline (A1 separation → A4/A6/C1 transcript → C2 translation → D1 takes →
E1 timing fit → E4 mix → render), so the user can see where a line first goes wrong and how
that error travels. Per-stage summary metrics reuse the whole-pipeline bench modules; the
optional round-trip check re-transcribes every take and fitted clip against the translation
(intelligibility after each audio stage).

    python -m bench arena ladder prj_6504aae1 [--asr-check]
    → data/arena/ladder/<project>.html  (also served live at /ladder/<project>)
"""
from __future__ import annotations

import html
import json
import wave
from pathlib import Path
from typing import Any

from app import store
from app.providers.translation._llm import max_chars_for

from . import db, paths
from .judges import UNSPACED, normalize_text

LOW_CONFIDENCE = 0.5


def _wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, wave.Error, EOFError):
        return None


def _url(project_dir: Path, rel: str | None) -> str | None:
    if not rel:
        return None
    p = (project_dir / rel).resolve()
    if not p.exists():
        return None
    try:
        return "/files/" + str(p.relative_to(paths.data_dir().resolve()))
    except ValueError:
        return None


def _cer(ref: str, hyp: str, lang: str) -> float | None:
    from app.pipeline.quality import _distance

    r, h = normalize_text(ref).replace(" ", ""), normalize_text(hyp).replace(" ", "")
    return _distance(list(r), list(h)) / len(r) if r else None


class _RoundTrip:
    """Whisper re-transcription of takes (the D1/E1 intelligibility check). Loaded lazily."""

    def __init__(self, model: str = "large-v3-turbo"):
        from app.providers.asr.faster_whisper import _preload_cuda_libs

        _preload_cuda_libs()
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="cuda", compute_type="float16")
        self.name = f"faster-whisper {model}"

    def __call__(self, path: Path, lang: str) -> str:
        segs, _ = self.model.transcribe(str(path), language=lang or None, beam_size=5,
                                        condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


async def _stage_metrics(project, project_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Whole-pipeline bench metrics, grouped by stage label (reference-free ones apply here)."""
    from bench.metrics import MODULES
    from bench.metrics.base import BenchCase, MetricContext

    case = BenchCase(name=project.id, source=project_dir / "source.mp4",
                     source_lang=project.source_lang, target_lang=project.target_lang)
    ctx = MetricContext(project=project, project_dir=project_dir, case=case)
    out: dict[str, list[dict[str, Any]]] = {}
    for label, module in MODULES:
        if label in ("lipsync", "system"):
            continue
        try:
            metrics = await module.score(ctx)
        except Exception as exc:  # noqa: BLE001 - a broken module must not hide the ladder
            metrics = []
            out[label] = [{"name": f"{label}.error", "value": None, "note": repr(exc)}]
        out.setdefault(label, []).extend(m.as_dict() for m in metrics)
    return out


def build(project_id: str, *, asr_check: bool = False, judge: bool = False) -> dict[str, Any]:
    import asyncio

    project = store.load(project_id)
    if project is None:
        raise FileNotFoundError(f"no project {project_id}")
    pdir = store.project_dir(project_id)
    src, tgt = project.source_lang, project.target_lang
    rt = _RoundTrip() if asr_check else None
    speakers = {s.id: s.name for s in project.speakers}
    con = db.connect()
    marks = db.audit_marks(con, f"ladder:{project_id}", tgt)
    con.close()

    rows = []
    for seg in sorted(project.segments, key=lambda s: s.start):
        slot = seg.end - seg.start
        words = [{"t": w.text, "c": w.confidence} for w in (seg.words or [])]
        confs = [w["c"] for w in words if w["c"] is not None]
        low = sum(c < LOW_CONFIDENCE for c in confs)
        seg_dir = f"audio/segments/{seg.id}"
        active = next((t for t in seg.takes if t.id == seg.active_take_id),
                      seg.takes[-1] if seg.takes else None)
        takes = []
        for t in seg.takes:
            d = {"id": t.id, "url": _url(pdir, t.path), "duration": t.duration,
                 "rate_factor": t.rate_factor, "active": active is not None and t.id == active.id,
                 "lang": t.lang}
            if rt and d["url"]:
                hyp = rt(pdir / t.path, tgt)
                d["roundtrip"] = hyp
                d["roundtrip_cer"] = _cer(seg.translated_text, hyp, tgt)
            takes.append(d)
        fitted = pdir / seg_dir / "fitted.wav"
        fitted_s = _wav_seconds(fitted) if fitted.exists() else None
        # E1: the tempo actually applied to the active take, and how much of the slot the dub
        # speech fills afterwards (fitted.wav itself is padded to the slot, so not its length).
        tempo = active.rate_factor if active and active.rate_factor else None
        spoken = active.duration / tempo if active and tempo else None
        fit: dict[str, Any] = {"url": _url(pdir, f"{seg_dir}/fitted.wav"), "seconds": fitted_s,
                               "tempo": tempo,
                               "fill": spoken / slot if spoken and slot > 0 else None}
        if rt and fit["url"]:
            hyp = rt(fitted, tgt)
            fit["roundtrip"], fit["roundtrip_cer"] = hyp, _cer(seg.translated_text, hyp, tgt)
        qrep = pdir / "quality" / "segments" / f"{seg.id}.json"
        n_chars = len(seg.translated_text)
        budget = max_chars_for(slot) if slot > 0 else None
        rows.append({
            "id": seg.id, "start": seg.start, "end": seg.end, "slot": slot,
            "speaker": speakers.get(seg.speaker_id, seg.speaker_id), "skipped": seg.skipped,
            "source_clip": _url(pdir, f"{seg_dir}/source.wav"),
            "transcript": seg.source_text, "words": words,
            "mean_conf": sum(confs) / len(confs) if confs else None, "low_conf_words": low,
            "translation": seg.translated_text, "chars": n_chars, "budget": budget,
            "chars_per_s": n_chars / slot if slot > 0 else None,
            "takes": takes, "fit": fit,
            "quality": json.loads(qrep.read_text()) if qrep.exists() else None,
            "marks": {st: marks.get((st, seg.id)) for st in ("A4", "C2", "D1", "E1")},
        })

    stage_metrics = asyncio.run(_stage_metrics(project, pdir)) if judge else {}
    pl = project.pipeline
    providers = {
        "A1 separation": pl.separation.provider_id, "A4 asr": pl.asr.provider_id,
        "A6 diarization": pl.diarization.provider_id, "C2 translation": pl.translation.provider_id,
        "D1 tts": pl.tts.provider_id, "F1 lipsync": pl.lipsync.provider_id,
    }
    options = {k: getattr(pl, f).options for k, f in (
        ("C2 translation", "translation"), ("D1 tts", "tts"), ("A4 asr", "asr"))}
    full = {k: _url(pdir, v) for k, v in {
        "source video": "playback.mp4", "original audio": "audio/original.wav",
        "A1 dialogue stem": "audio/vocals.wav", "A1 background stem": "audio/background.wav",
        "D1+E1 dub dialogue": "audio/dub_vocals.wav", "E4 dub mix": "audio/dub_mix.wav",
        "dubbed video": "render/dubbed.mp4"}.items()}
    return {"project": project_id, "name": project.name, "src": src, "tgt": tgt,
            "unspaced_src": src in UNSPACED, "providers": providers,
            "options": {k: {o: v for o, v in opt.items() if "key" not in o.lower()}
                        for k, opt in options.items()},
            "stages": {k: {"status": v.status, "detail": v.detail}
                       for k, v in project.stages.items()},
            "full": full, "rows": rows, "stage_metrics": stage_metrics,
            "roundtrip_model": rt.name if rt else None}


def render(data: dict[str, Any]) -> str:
    blob = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    title = f"Stage ladder · {data['name']} ({data['src']}→{data['tgt']})"
    return _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def write(project_id: str, **kw: Any) -> Path:
    out = paths.arena_dir() / "ladder" / f"{project_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build(project_id, **kw)
    out.write_text(render(data))
    (out.with_suffix(".json")).write_text(json.dumps(data, ensure_ascii=False, indent=1,
                                                     default=str))
    return out


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#f7f5fb;--raised:#fff;--line:#e3def0;--ink:#1c1733;--ink2:#5d5780;--ok:#0f8a6e;
--warn:#b87800;--bad:#c62f5b;--accent:#6b3fd4;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#131022;--raised:#1c1831;
--line:#2e2850;--ink:#efeaff;--ink2:#a9a1cf;--ok:#3ad6ae;--warn:#ffc04d;--bad:#ff6f98;
--accent:#a98bff;color-scheme:dark}}
:root[data-theme=dark]{--bg:#131022;--raised:#1c1831;--line:#2e2850;--ink:#efeaff;--ink2:#a9a1cf;
--ok:#3ad6ae;--warn:#ffc04d;--bad:#ff6f98;--accent:#a98bff;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,sans-serif}header{padding:16px;background:var(--raised);
border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 6px}h2{font-size:15px;margin:18px 0 8px}
.m{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--ink2)}main{padding:0 16px 64px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}
.card{background:var(--raised);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
.card b{font-size:12px}audio,video{width:100%}video{max-height:260px;background:#000}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--raised)}
table{border-collapse:collapse;min-width:1100px;width:100%}th,td{border-bottom:1px solid var(--line);
padding:8px;vertical-align:top;text-align:left}th{font-size:11px;text-transform:uppercase;
letter-spacing:.06em;color:var(--ink2);position:sticky;top:0;background:var(--raised)}
td{min-width:160px}td.t{min-width:90px}.low{background:color-mix(in oklab,var(--bad) 22%,transparent);
border-radius:3px}.mid{background:color-mix(in oklab,var(--warn) 20%,transparent);border-radius:3px}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.pill{display:inline-block;
font-size:11px;padding:0 6px;border-radius:9px;border:1px solid var(--line)}.take{margin-top:6px}
.take.active{border-left:3px solid var(--accent);padding-left:6px}.marks button{font:inherit;
font-size:11px;padding:1px 6px;border:1px solid var(--line);border-radius:5px;background:var(--bg);
color:var(--ink);cursor:pointer}.marks{margin-top:4px;display:flex;gap:3px}.on-ok{border-color:var(--ok)!important;
color:var(--ok)!important}.on-bad{border-color:var(--bad)!important;color:var(--bad)!important}
details{margin:8px 0}pre{white-space:pre-wrap;font-size:12px}
</style></head><body><header><h1>__TITLE__</h1><div id="prov" class="m"></div></header><main>
<h2>Whole-file outputs, in stage order</h2><div id="full" class="cards"></div>
<h2>Stage summaries</h2><div id="sums" class="cards"></div>
<h2>Line ladder <span class="m">one row per dub line · red = likely error introduced at that stage</span></h2>
<div class="m" style="margin:0 0 8px">filter <select id="f"><option value="all">all lines</option>
<option value="flag">lines with any flag</option><option value="bad">marked bad</option></select></div>
<div class="tbl"><table><thead><tr><th>line</th><th>A1 source clip</th><th>A4 transcript (confidence)</th>
<th>C2 translation</th><th>D1 takes</th><th>E1 fitted</th></tr></thead><tbody id="rows"></tbody></table></div>
</main><script>
const D=__DATA__;const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const SERVED=location.protocol.startsWith('http');const pct=v=>v==null?'–':(100*v).toFixed(0)+'%';
const cls=(v,a,b)=>v==null?'':v<=a?'ok':v<=b?'warn':'bad';
document.getElementById('prov').innerHTML=Object.entries(D.providers).map(([k,v])=>`${esc(k)}: <b>${esc(v)}</b>`).join(' · ')
 +(D.roundtrip_model?` · round-trip judge: ${esc(D.roundtrip_model)}`:'');
const av=(u)=>!u?'<span class="m">missing</span>':/\.mp4$/.test(u)?`<video controls preload="none" src="${u}"></video>`:`<audio controls preload="none" src="${u}"></audio>`;
document.getElementById('full').innerHTML=Object.entries(D.full).map(([k,u])=>`<div class="card"><b>${esc(k)}</b>${av(u)}</div>`).join('');
const sm=D.stage_metrics||{};const st=Object.entries(D.stages).map(([k,v])=>`<div class="card"><b>${esc(k)}</b> <span class="pill">${esc(v.status)}</span><div class="m">${esc(v.detail||'')}</div>
 ${(sm[{separate:'separation',transcribe:'transcribe',translate:'translate',synthesize:'tts',mix:'mix'}[k]]||[]).map(m=>`<div class="m" title="${esc(m.note)}">${esc(m.name)}: ${m.value==null?'–':esc(typeof m.value==='number'?(+m.value).toFixed(3):m.value)} ${esc(m.unit||'')}</div>`).join('')}</div>`);
document.getElementById('sums').innerHTML=st.join('');
const LS='ladder-'+D.project;let local={};try{local=JSON.parse(localStorage.getItem(LS)||'{}')}catch(e){}
const mk=(r,s)=>local[s+'|'+r.id]||r.marks[s]||{};
async function mark(id,s,v){let note='';if(v==='note'){note=prompt('note')||'';if(!note)return}
 local[s+'|'+id]={verdict:v,note};try{localStorage.setItem(LS,JSON.stringify(local))}catch(e){}
 if(SERVED)fetch('/api/audit',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({capability:'ladder:'+D.project,lang:D.tgt,cand_key:s,item:id,verdict:v,note})});draw()}
const btns=(r,s)=>{const m=mk(r,s);return`<div class="marks"><button class="${m.verdict==='ok'?'on-ok':''}" onclick="mark('${r.id}','${s}','ok')">ok</button><button class="${m.verdict==='bad'?'on-bad':''}" onclick="mark('${r.id}','${s}','bad')">bad</button><button onclick="mark('${r.id}','${s}','note')">note</button>${m.note?`<span class="m">${esc(m.note)}</span>`:''}</div>`};
function flags(r){const f=[];if(r.low_conf_words>0)f.push('A4');if(r.budget&&r.chars>r.budget)f.push('C2');
 if(r.takes.some(t=>t.active&&t.roundtrip_cer>0.2))f.push('D1');if(r.fit.roundtrip_cer>0.2||(r.fit.tempo&&r.fit.tempo>1.15)||(r.fit.fill!=null&&(r.fit.fill<0.6||r.fit.fill>1.15)))f.push('E1');return f}
function draw(){const f=document.getElementById('f').value;let rows=D.rows;
 if(f==='flag')rows=rows.filter(r=>flags(r).length);if(f==='bad')rows=rows.filter(r=>['A4','C2','D1','E1'].some(s=>mk(r,s).verdict==='bad'));
 document.getElementById('rows').innerHTML=rows.map(r=>{const fl=flags(r);
 const words=r.words.length?r.words.map(w=>`<span class="${w.c!=null&&w.c<0.5?'low':w.c!=null&&w.c<0.8?'mid':''}" title="${w.c==null?'':w.c.toFixed(2)}">${esc(w.t)}</span>`).join(D.unspaced_src?'':' '):esc(r.transcript);
 const q=r.quality?`<details><summary class="m">runtime quality</summary><pre>${esc(JSON.stringify(r.quality,null,1)).slice(0,1500)}</pre></details>`:'';
 return`<tr><td class="t"><b>${r.start.toFixed(1)}–${r.end.toFixed(1)}s</b><div class="m">${esc(r.speaker)} · ${r.slot.toFixed(1)}s slot${r.skipped?' · skipped':''}</div>${fl.length?`<div class="bad m">flags: ${fl.join(', ')}</div>`:''}</td>
 <td>${av(r.source_clip)}</td>
 <td><div>${words}</div><div class="m">mean conf ${r.mean_conf==null?'–':r.mean_conf.toFixed(2)} · <span class="${r.low_conf_words?'bad':'ok'}">${r.low_conf_words} low</span></div>${btns(r,'A4')}</td>
 <td><div>${esc(r.translation)}</div><div class="m"><span class="${r.budget&&r.chars>r.budget?'bad':'ok'}">${r.chars}/${r.budget??'–'} chars</span> · ${r.chars_per_s==null?'–':r.chars_per_s.toFixed(1)} ch/s</div>${btns(r,'C2')}</td>
 <td>${r.takes.length?'':'<div class="bad m">no take: synthesis failed or skipped for this line</div>'}${r.takes.map(t=>`<div class="take ${t.active?'active':''}">${av(t.url)}<div class="m">${t.duration?.toFixed(2)}s · rate ×${t.rate_factor?.toFixed(2)}${t.active?' · active':''}</div>
  ${t.roundtrip!=null?`<div class="m">heard: ${esc(t.roundtrip)}</div><div class="m ${cls(t.roundtrip_cer,0.1,0.25)}">round-trip CER ${pct(t.roundtrip_cer)}</div>`:''}</div>`).join('')}${q}${btns(r,'D1')}</td>
 <td>${av(r.fit.url)}<div class="m">tempo <span class="${r.fit.tempo>1.15?'bad':'ok'}">×${r.fit.tempo?.toFixed(2)??'–'}</span> · fills <span class="${r.fit.fill!=null&&(r.fit.fill<0.6||r.fit.fill>1.15)?'bad':'ok'}">${r.fit.fill==null?'–':(100*r.fit.fill).toFixed(0)+'%'}</span> of slot</div>
  ${r.fit.roundtrip!=null?`<div class="m">heard: ${esc(r.fit.roundtrip)}</div><div class="m ${cls(r.fit.roundtrip_cer,0.1,0.25)}">round-trip CER ${pct(r.fit.roundtrip_cer)}</div>`:''}${btns(r,'E1')}</td></tr>`}).join('')}
document.getElementById('f').addEventListener('input',draw);draw();
</script></body></html>
"""
