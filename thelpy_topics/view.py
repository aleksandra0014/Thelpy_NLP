"""Chunks with their topics as an HTML page (cards with a bar of topic shares, filters, search).

    python -m thelpy_topics.view --collection thelpy_turns8                 # what is in Qdrant
    python -m thelpy_topics.view --file data/annomi_64.json --n-turns 8     # without Qdrant: cut the file like thelpy_rag

The page is saved in data/with_topics/ and opened in the browser.
"""
import argparse
import html
import json
import webbrowser

from thelpy_topics import taxonomy as tax
from thelpy_topics import topic_config as config

payload_fields = ["session_id", "chunk_index", "turn_range", "start_seconds", "end_seconds", "text", "topic_main",
                  "topic_all", "topic_shares", "subtopic_all", "topic_none_share", "topic_process_share"]
topic_colors = {
    "substances": "#2a78d6", "physical": "#eb6834", "mental": "#1baf7a", "family": "#eda100", "partner": "#e87ba4",
    "parenting": "#008300", "social": "#6250d6", "school": "#e34948", "work": "#0090b5", "law": "#9b6a00",
    "leisure": "#c25ec9", "eating": "#7f8c00", "sexuality": "#9e2a5c", "pets": "#c47f3f", "opening": "#a9a69b",
    "closing": "#8b9bb0",
}


def qdrant_chunks(collection):
    """Return the payloads of all chunks of a Qdrant collection."""
    from thelpy_rag.store import get_client

    client = get_client()
    payloads = []
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
        payloads = payloads + [point.payload for point in points]
        if offset is None:
            return payloads


def file_chunks(path, n_turns, overlap_turns):
    """Return the payloads of the chunks of a session file, cut like thelpy_rag (no Qdrant needed)."""
    from thelpy_rag.pipeline import resolve_strategy
    from thelpy_topics import integration
    from thelpy_topics.annotate import annotate_files

    annotated = annotate_files([path])[0]
    strategy = resolve_strategy("turns", n_turns=n_turns, overlap_turns=overlap_turns)
    session = integration.load_session_with_topics(annotated)
    chunks = integration.chunk_session_with_topics(session, strategy)
    return [{**chunk.payload(), "session_id": session.session_id} for chunk in chunks]


def clean(payload):
    """Return only the fields shown on the page."""
    return {field: payload.get(field) for field in payload_fields}


def labels():
    """Return English and Polish labels of topics and subtopics for the page."""
    topics = {key: value["label_pl"] for key, value in tax.main_topics.items()}
    subs = {key: value["label_pl"] for key, value in tax.subtopics.items()}
    return {"topics": topics, "subs": subs, "colors": topic_colors}


def build_page(title, payloads):
    """Return the HTML page with the chunks as JSON and the script that draws them."""
    payloads = sorted(map(clean, payloads), key=lambda p: (str(p["session_id"]), p["chunk_index"]))
    data = json.dumps({"chunks": payloads, **labels()}, ensure_ascii=False).replace("</", "<\\/")
    page = page_template.replace("__TITLE__", html.escape(title))
    return page.replace("__DATA__", data)


def save_page(name, page):
    """Save the page in data/with_topics/ and return its path."""
    config.annotated.mkdir(parents=True, exist_ok=True)
    path = config.annotated / f"view_{name}.html"
    path.write_text(page, encoding="utf-8")
    return path


def main():
    """Read the chunks (from Qdrant or from a file), save the page and open it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", help="Qdrant collection, e.g. thelpy_turns8")
    parser.add_argument("--file", help="session .json (instead of Qdrant)")
    parser.add_argument("--n-turns", type=int, default=8, help="with --file: turns in a chunk")
    parser.add_argument("--overlap-turns", type=int, default=0)
    parser.add_argument("--no-open", action="store_true", help="only save the page")
    args = parser.parse_args()
    if args.collection:
        payloads, name = qdrant_chunks(args.collection), args.collection
    elif args.file:
        payloads = file_chunks(args.file, args.n_turns, args.overlap_turns)
        name = f"file_turns{args.n_turns}"
    else:
        raise SystemExit("Give --collection or --file")
    path = save_page(name, build_page(name, payloads))
    print(path)
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())


page_template = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chunki: __TITLE__</title>
<style>
:root{--bg:#f4f3ef;--card:#fff;--ink:#1f1f1d;--soft:#6b6a66;--line:#e3e2dc;--chip:#f0efea;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1f1e1c;--ink:#ecebe6;--soft:#a3a29c;--line:#34332f;--chip:#2a2926}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:14px 16px}
.wrap{max-width:1000px;margin:0 auto}h1{font-size:1.25rem;margin:0 0 10px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.bar input,.bar select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink)}
.bar input{flex:1;min-width:180px}.count{color:var(--soft);font-size:.9rem}
.overview{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:12px 0 2px;cursor:pointer}
.overview div{flex:1;border-right:1px solid var(--bg)}
main{padding:12px 16px 48px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}
.head{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;margin-bottom:8px}
.head b{font-size:1.02rem}.meta{color:var(--soft);font-size:.85rem}
.shares{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--chip);margin:6px 0 8px}
.legend,.subs{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0}
.chip{display:inline-flex;align-items:center;gap:6px;background:var(--chip);border-radius:999px;padding:2px 10px;font-size:.84rem}
.dot{width:9px;height:9px;border-radius:50%}.main-chip{font-weight:600}.out{opacity:.45;text-decoration:line-through}
.text{margin-top:10px;border-top:1px solid var(--line);padding-top:8px;font-size:.93rem;white-space:pre-wrap}
.text .t{color:var(--accent);font-weight:600}.text .c{color:#eb6834;font-weight:600}
details summary{cursor:pointer;color:var(--soft);font-size:.85rem;margin-top:8px}
mark{background:#ffe08a;color:inherit;border-radius:3px}
.empty{color:var(--soft);text-align:center;padding:40px}
</style></head><body>
<header><div class="wrap"><h1>Chunki i tematy: __TITLE__</h1>
<div class="bar"><input class="q" placeholder="Szukaj w tekście…"><select class="topic"></select><select class="session"></select>
<span class="count"></span></div>
<p class="meta" style="margin:8px 0 0">Pasek i procenty: udział tematów w słowach, które mają temat (temat poboczny liczy się za pół). Do chunka (topic_all: filtr i prompt) trafia temat główny i maks. 2 poboczne z udziałem ≥ 15%; przekreślone tematy nie trafiają. Podtematy: pokazane są tylko te, które trafiają. „Słowa z tematem”: jaka część chunka w ogóle ma temat (reszta to np. „mm-hmm”, wtrącenia, sam proces terapii). Pasek na górze: temat główny każdego chunka po kolei, kliknij, aby przejść.</p><div class="overview" title="Temat główny każdego chunka po kolei (kliknij, aby przejść)"></div></div></header>
<main><div class="wrap list"></div></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const $=s=>document.querySelector(s), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const color=t=>D.colors[t]||'#c9c8c1', tl=t=>D.topics[t]||t, sl=s=>D.subs[s]||s;
const mmss=s=>s==null?'':`${Math.floor(s/60)}:${String(Math.round(s%60)).padStart(2,'0')}`;
const topics=[...new Set(D.chunks.flatMap(c=>c.topic_all||[]))], subs=[...new Set(D.chunks.flatMap(c=>c.subtopic_all||[]))];
$('.topic').innerHTML='<option value="">wszystkie tematy</option>'+topics.map(t=>`<option value="${t}">${esc(tl(t))}</option>`).join('')
 +subs.map(s=>`<option value="sub:${s}">&nbsp;&nbsp;– ${esc(sl(s))}</option>`).join('');
const sessions=[...new Set(D.chunks.map(c=>String(c.session_id)))];
$('.session').innerHTML='<option value="">wszystkie sesje</option>'+sessions.map(s=>`<option>${esc(s)}</option>`).join('');
if(sessions.length<2)$('.session').style.display='none';
function textHtml(t,q){let h=esc(t).replace(/^(Therapist):/gm,'<span class="t">Terapeuta:</span>').replace(/^(Client):/gm,'<span class="c">Pacjent:</span>');
 if(q)h=h.replace(new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi'),'<mark>$1</mark>');return h}
function card(c,q){const sh=c.topic_shares||{}, main=c.topic_main;
 const bar=Object.entries(sh).map(([t,v])=>`<div title="${esc(tl(t))} ${Math.round(v*100)}%" style="width:${v*100}%;background:${color(t)}"></div>`).join('');
 const kept=c.topic_all||[];
 const legend=Object.entries(sh).map(([t,v])=>`<span class="chip ${t===main?'main-chip':''} ${kept.includes(t)?'':'out'}" title="${kept.includes(t)?'trafia do chunka (topic_all)':'za mały udział, nie trafia do topic_all'}"><i class="dot" style="background:${color(t)}"></i>${esc(tl(t))} ${Math.round(v*100)}%</span>`).join('');
 const sub=(c.subtopic_all||[]).map((s,i)=>`<span class="chip ${i==0?'main-chip':''}">${esc(sl(s))}</span>`).join('');
 const extra=[c.topic_none_share!=null?`słowa z tematem ${Math.round((1-c.topic_none_share)*100)}%`:'',c.topic_process_share?`proces terapii ${Math.round(c.topic_process_share*100)}%`:''].filter(Boolean).join(' · ');
 const open=q?'open':'';
 return `<div class="card" id="c${esc(c.session_id)}_${c.chunk_index}"><div class="head"><b>#${c.chunk_index} · ${esc(main?tl(main):'brak tematu')}</b>
 <span class="meta">sesja ${esc(c.session_id)} · ${esc(c.turn_range)} · ${mmss(c.start_seconds)}–${mmss(c.end_seconds)}${extra?' · '+extra:''}</span></div>
 <div class="shares">${bar}</div><div class="legend">${legend}</div>${sub?`<div class="subs"><span class="meta">podtematy:</span>${sub}</div>`:''}
 <details ${open}><summary>tekst chunka</summary><div class="text">${textHtml(c.text,q)}</div></details></div>`}
function draw(){const q=$('.q').value.trim(), t=$('.topic').value, s=$('.session').value;
 const shown=D.chunks.filter(c=>(!s||String(c.session_id)===s)&&(!q||(c.text||'').toLowerCase().includes(q.toLowerCase()))
  &&(!t||(t.startsWith('sub:')?(c.subtopic_all||[]).includes(t.slice(4)):(c.topic_all||[]).includes(t))));
 $('.list').innerHTML=shown.map(c=>card(c,q)).join('')||'<div class="empty">Brak chunków dla tych filtrów.</div>';
 $('.count').textContent=`${shown.length} z ${D.chunks.length} chunków`;
 $('.overview').innerHTML=shown.map(c=>`<div title="#${c.chunk_index} ${esc(c.topic_main?tl(c.topic_main):'brak tematu')}" data-id="c${esc(c.session_id)}_${c.chunk_index}" style="background:${color(c.topic_main)}"></div>`).join('')}
$('.overview').addEventListener('click',e=>{const id=e.target.dataset.id;if(id)document.getElementById(id).scrollIntoView({behavior:'smooth',block:'center'})});
['.q','.topic','.session'].forEach(s=>$(s).addEventListener('input',draw));draw();
</script></body></html>
"""


if __name__ == "__main__":
    main()
