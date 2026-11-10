"""Logika endpointów: te same kroki co w scripts/, ale zwracane jako dane zamiast wypisywane.

Tematy (thelpy_topics) są obsługiwane per zapytanie — bez integration.enable_topics(), które podmienia
funkcje thelpy_rag globalnie i w serwerze wielowątkowym mieszałoby zapytania z tematami i bez nich.
Zamiast tego: filtr z tematem przez integration.with_topic, tematy w prompcie przez formatter
w llm.generate, a ingest z tematami jak w scripts/ingest.py.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thelpy_rag import config
from thelpy_rag import keywords as kw
from thelpy_rag.chunking import chunk_session, get_tokenizer
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.llm import generate
from thelpy_rag.loader import iter_session_files, load_session, session_from_dict
from thelpy_rag.pipeline import resolve_strategy
from thelpy_rag.search import Hit, Retriever, build_context, build_filter
from thelpy_rag.sparse import BM25Encoder
from thelpy_rag.store import create_collection, delete_session, get_client, upsert_chunks
from thelpy_rag.summarize import fetch_chunks, summarize
from thelpy_topics import integration, taxonomy
from thelpy_topics import annotate as topic_annotate

from . import schemas

UPLOAD_DIR = config.PROJECT_ROOT / "data" / "uploads"   # sesje przysłane w zapytaniu (JSON / upload)

# logger uvicorna ma już handler i poziom INFO -> postęp długich operacji widać w konsoli serwera
log = logging.getLogger("uvicorn.error")


# --- zasoby współdzielone przez zapytania -------------------------------------
@dataclass
class Resources:
    client: Any
    embedder: OllamaEmbedder
    bm25: BM25Encoder
    retriever: Retriever

    @classmethod
    def create(cls) -> "Resources":
        client, embedder, bm25 = get_client(), OllamaEmbedder(), BM25Encoder()
        return cls(client, embedder, bm25, Retriever(client, embedder, bm25))


_topic_model = None
_topic_lock = threading.Lock()


def topic_model():
    """Model tematów ładowany raz na cały proces (wolny start, potem z pamięci)."""
    global _topic_model
    with _topic_lock:
        if _topic_model is None:
            _topic_model = topic_annotate.TopicModel()
        return _topic_model


# --- pomocnicze ----------------------------------------------------------------
_STRATEGY_RE = re.compile(r"^(turns|fixed)(\d+)(?:_ov(\d+))?$")


def strategy_name(name: str) -> str:
    """Nazwa z config.STRATEGIES albo własna turnsN[_ovM] / fixedN[_ovM] (rejestrowana w locie)."""
    if name in config.STRATEGIES:
        return name
    m = _STRATEGY_RE.match(name)
    if not m:
        raise ValueError(f"Nieznana strategia '{name}'. {schemas.STRATEGY_HELP}")
    kind, n, ov = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if kind == "turns":
        return resolve_strategy("turns", n_turns=n, overlap_turns=ov)
    return resolve_strategy("fixed", size=n, overlap=ov)


def check_topic(topic: str | None) -> None:
    if topic and topic not in taxonomy.main_topics and topic not in taxonomy.subtopics:
        raise ValueError(f"Nieznany temat '{topic}' — klucze: GET /topics")


def apply_topic(flt, topic: str | None):
    check_topic(topic)
    return integration.with_topic(flt, topic) if topic else flt


def formatter(topics: bool):
    return integration.format_context_with_topics if topics else None


def require_collection(res: Resources, collection: str) -> None:
    if not res.client.collection_exists(collection):
        raise LookupError(f"Kolekcja {collection} nie istnieje — najpierw POST /ingest")


def server_path(path: str) -> Path:
    """Ścieżka na serwerze; względne liczone od katalogu projektu."""
    p = Path(path)
    p = p if p.is_absolute() else config.PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Nie ma pliku/katalogu {path}")
    return p


def read_json(path: str) -> Any:
    return json.loads(server_path(path).read_text(encoding="utf-8"))


def save_session(data: dict) -> Path:
    """Waliduje sesję i zapisuje ją do data/uploads/<session_id>.json (ponowne wgranie nadpisuje)."""
    session = session_from_dict(data)   # ValueError -> 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out = UPLOAD_DIR / (re.sub(r"[^\w.-]", "_", session.session_id) + ".json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def annotate(files: list[Path]) -> list[Path]:
    """Jak thelpy_topics.annotate.annotate_files, ale z modelem trzymanym w pamięci serwera."""
    out = []
    for path in files:
        if not topic_annotate.is_session(path):
            continue
        model = None if topic_annotate.is_fresh(path) else topic_model()
        out.append(topic_annotate.annotate_file(path, model))
    return out


def topic_note(payload: dict) -> str | None:
    return integration.topic_note(payload) or None if payload.get("topic_labels") else None


def hit_dict(h: Hit, score_name: str = "score") -> dict:
    return {"id": h.id, score_name: h.score, "ranks": h.ranks, "topic_note": topic_note(h.payload),
            "payload": h.payload}


def source(p: dict) -> dict:
    return {"session_id": p["session_id"], "chunk_index": p["chunk_index"], "turn_range": p.get("turn_range"),
            "session_at": p.get("session_at"), "start_seconds": p.get("start_seconds"),
            "end_seconds": p.get("end_seconds"), "topic_note": topic_note(p),
            "retrieval_rank": p.get("retrieval_rank"), "retrieval_score": p.get("retrieval_score"),
            "retrieval_neighbor": p.get("retrieval_neighbor", False)}


def llm_result(res: dict, include_prompt: bool) -> dict:
    out = {"model": res["model"], "answer": res["answer"],
           "prompt_tokens": res["prompt_tokens"], "answer_tokens": res["answer_tokens"]}
    if include_prompt:
        out["messages"] = res["messages"]
    return out


# --- kolekcje -----------------------------------------------------------------
def create_collections(req: schemas.CreateCollectionsRequest, res: Resources) -> dict:
    dim = req.dim or res.embedder.dimension()
    rows = []
    for s in map(strategy_name, req.strategies):
        name = config.collection_name(s)
        created = create_collection(res.client, name, dim, recreate=req.recreate)
        rows.append({"strategy": s, "collection": name, "created": created, "params": config.STRATEGIES[s]})
    return {"embed_model": config.EMBED_MODEL, "dim": dim, "collections": rows}


def list_collections(res: Resources) -> list[dict]:
    prefix = config.COLLECTION_PREFIX + "_"
    out = []
    for c in res.client.get_collections().collections:
        if c.name.startswith(prefix):
            out.append({"collection": c.name, "strategy": c.name[len(prefix):],
                        "points": res.client.count(c.name, exact=True).count})
    return sorted(out, key=lambda r: r["collection"])


SESSION_FIELDS = ["session_id", "client_id", "session_at", "added_at", "duration_seconds", "source_version",
                  "chunk_index", "turn_ids", "n_tokens", "topic_labels"]


def list_sessions(strategy: str, res: Resources, client_id: str | None = None) -> dict:
    """Metadane sesji w kolekcji: przejście po wszystkich chunkach (tylko pola metadanych, bez tekstu
    i wektorów) i zgrupowanie po session_id."""
    name = config.collection_name(strategy_name(strategy))
    require_collection(res, name)
    flt = build_filter(client_id=client_id)

    sessions: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = res.client.scroll(name, scroll_filter=flt, limit=1024, offset=offset,
                                           with_payload=SESSION_FIELDS, with_vectors=False)
        for pt in points:
            p = pt.payload
            s = sessions.setdefault(p["session_id"], {
                "session_id": p["session_id"], "client_id": p.get("client_id"),
                "session_at": p.get("session_at"), "added_at": p.get("added_at"),
                "duration_seconds": p.get("duration_seconds"), "source_version": p.get("source_version"),
                "n_chunks": 0, "_turns": set(), "_bounds": [], "tokens": 0, "topics": False})
            s["n_chunks"] += 1
            s["tokens"] += p.get("n_tokens") or 0
            s["_turns"].update(p.get("turn_ids") or [])
            if p.get("turn_ids"):
                s["_bounds"].append((p["chunk_index"], p["turn_ids"][0], p["turn_ids"][-1]))
            s["topics"] = s["topics"] or bool(p.get("topic_labels"))
        if offset is None:
            break

    rows = []
    for s in sessions.values():
        bounds = sorted(s.pop("_bounds"))
        s["n_turns"] = len(s.pop("_turns"))
        s["turn_range"] = f"{bounds[0][1]}–{bounds[-1][2]}" if bounds else None
        rows.append(s)
    rows.sort(key=lambda s: (s["session_at"] or "", s["session_id"]))
    return {"collection": name, "client_id": client_id, "count": len(rows), "sessions": rows}


def remove_session(strategy: str, session_id: str, res: Resources) -> dict:
    name = config.collection_name(strategy_name(strategy))
    require_collection(res, name)
    delete_session(res.client, name, session_id)
    return {"collection": name, "session_id": session_id, "deleted": True}


# --- ingest (scripts/ingest.py) -------------------------------------------------
def ingest(req: schemas.IngestRequest, res: Resources) -> dict:
    strategies = [strategy_name(s) for s in req.strategies]
    files = iter_session_files([str(server_path(p)) for p in req.paths])
    files += [save_session(s) for s in req.sessions]
    if not files:
        raise FileNotFoundError("Nie znaleziono plików .json")

    load, chunk = load_session, chunk_session
    if req.topics:
        files = annotate(files)
        integration.add_payload_indexes()
        load, chunk = integration.load_session_with_topics, integration.chunk_session_with_topics

    warnings = [] if get_tokenizer().exact else ["brak tokenizera modelu — rozmiary w tokenach są przybliżone"]
    dim, sessions = None, []
    for path in files:
        session = load(path)
        entry = {"session_id": session.session_id, "file": str(path), "n_turns": len(session.turns),
                 "source_version": session.source_version, "strategies": []}
        for s in strategies:
            chunks = chunk(session, s)
            sizes = [c.n_tokens for c in chunks]
            row = {"strategy": s, "chunks": len(chunks),
                   "tokens": {"mean": round(statistics.mean(sizes)), "min": min(sizes), "max": max(sizes)}}
            if req.dry_run:
                row["preview"] = [{"chunk_index": c.chunk_index, "turn_range": f"{c.turn_ids[0]}–{c.turn_ids[-1]}",
                                   "n_tokens": c.n_tokens, "text": c.text, **c.extra} for c in chunks[:req.show]]
                entry["strategies"].append(row)
                continue

            name = config.collection_name(s)
            if not res.client.collection_exists(name):
                dim = dim or res.embedder.dimension()
                create_collection(res.client, name, dim)
                row["collection_created"] = True
            elif req.topics:
                integration.ensure_indexes(res.client, name)  # kolekcja sprzed tematów: dołóż indeksy

            texts = [c.text for c in chunks]
            dense = res.embedder.embed_documents(texts)
            sparse = [res.bm25.encode_document(t) for t in texts]
            delete_session(res.client, name, session.session_id)
            row["collection"] = name
            row["points"] = upsert_chunks(res.client, name, session, chunks, dense, sparse)
            entry["strategies"].append(row)
        sessions.append(entry)
    return {"dry_run": req.dry_run, "topics": req.topics, "warnings": warnings, "sessions": sessions}


# --- search (scripts/search.py) -------------------------------------------------
def search(req: schemas.SearchRequest, res: Resources) -> dict:
    strategy = strategy_name(req.strategy)
    collection = config.collection_name(strategy)
    require_collection(res, collection)
    flt = apply_topic(build_filter(req.session, req.client, req.role, req.date_from, req.date_to), req.topic)
    kw = {"fusion": req.fusion} if req.mode == "hybrid" else {}
    hits = res.retriever.search(collection, req.query, req.mode, req.k, flt, **kw)

    out = []
    for h in hits:
        d = hit_dict(h)
        if req.neighbors:
            d["neighbors"] = res.retriever.neighbors(collection, h, req.neighbors)
        out.append(d)
    return {"collection": collection, "mode": req.mode, "topic": req.topic, "count": len(out), "hits": out}


# --- wejście sesji (wspólne dla ask / summarize_session / session_keywords) --------
def resolve_source(req: schemas.SessionSource, res: Resources) -> tuple[str | None, str, bool]:
    """(Wgranie) sesji -> (session_id | None, collection, ingested).

    file / session + skip_ingest=false: wgranie do kolekcji strategii; session_id brany z pliku.
    skip_ingest=true albo brak file / session: nic nie jest wgrywane, zostaje filtr session_id (jeśli podany).
    """
    check_topic(req.topic)
    strategy = strategy_name(req.strategy)
    collection = config.collection_name(strategy)
    session_id, ingested = req.session_id, False

    if req.has_source:
        path = save_session(req.session) if req.session else server_path(req.file)
        file_sid = load_session(path).session_id
        if session_id and session_id != file_sid:
            raise ValueError(f"session_id={session_id} nie zgadza się z sesją z file / session ({file_sid})")
        session_id = file_sid
        if not req.skip_ingest:
            ingest(schemas.IngestRequest(paths=[str(path)], strategies=[strategy],
                                         topics=req.topics or bool(req.topic)), res)
            ingested = True

    require_collection(res, collection)
    if session_id and not ingested:
        n = res.client.count(collection, count_filter=build_filter(session_id=session_id), exact=True).count
        if not n:
            raise LookupError(f"Sesji {session_id} nie ma w kolekcji {collection} — wgraj ją (podaj file / session "
                              f"bez skip_ingest albo POST /ingest) lub sprawdź GET /collections/{strategy}/sessions")
    return session_id, collection, ingested


# --- ask (scripts/ask.py) ----------------------------------
def ask(req: schemas.AskRequest, res: Resources) -> dict:
    topics = req.topics or bool(req.topic)
    session_id, collection, ingested = resolve_source(req, res)
    flt = apply_topic(build_filter(session_id, req.client_id, req.role, req.date_from, req.date_to), req.topic)
    kw = {"fusion": req.fusion} if req.mode == "hybrid" else {}
    hits = res.retriever.search(collection, req.question, req.mode, req.k, flt, **kw)

    # kontekst: trafienia z rangą/score (+ ewentualnie sąsiedzi), bez duplikatów, w porządku chronologicznym
    nb = (lambda h: res.retriever.neighbors(collection, h, req.neighbors)) if req.neighbors else None
    ctx = build_context(hits, nb)

    out = {"question": req.question, "session_id": session_id, "client_id": req.client_id,
           "collection": collection, "ingested": ingested, "mode": req.mode, "topic": req.topic,
           "question_tokens": get_tokenizer().count(req.question),
           "hits": [{"score": h.score, **source(h.payload), "text": h.payload["text"]} for h in hits], "sources": [source(p) for p in ctx]}
    if not ctx:
        detail = (f"Brak chunków z tematem '{req.topic}' (kolekcja wgrana z topics? klucz z taxonomy.py?)"
                  if req.topic else "Brak pasujących fragmentów")
        return {**out, "answer": None, "detail": detail + " — LLM nie został wywołany"}
    gen = generate(req.question, ctx, model=req.model, formatter=formatter(topics))
    return {**out, **llm_result(gen, req.include_prompt)}


# --- summarize_session (scripts/summarize_session.py) ---------------------------
def summarize_session(req: schemas.SummarizeSessionRequest, res: Resources) -> dict:
    """Wszystkie chunki sesji (po metadanych, nie wyszukiwaniem) -> map-reduce LLM."""
    topics = req.topics or bool(req.topic)
    session_id, collection, ingested = resolve_source(req, res)

    flt = apply_topic(build_filter(session_id=session_id), req.topic)
    payloads = fetch_chunks(res.client, collection, flt)
    out = {"session_id": session_id, "collection": collection, "ingested": ingested, "topic": req.topic,
           "focus": req.focus, "n_chunks": len(payloads)}
    if not payloads:
        detail = (f"Brak chunków z tematem '{req.topic}' (kolekcja wgrana z topics? klucz z taxonomy.py?)"
                  if req.topic else f"Brak chunków sesji {session_id} w {collection}")
        return {**out, "summary": None, "detail": detail}

    result = summarize(payloads, model=req.model, formatter=formatter(topics), focus=req.focus,
                       max_block_tokens=req.max_block_tokens, length=req.length,
                       log=lambda m: log.info("[summarize %s] %s", session_id, m))
    if not req.include_steps:
        result.pop("steps")
    return {**out, **result}


# --- session_keywords (scripts/session_keywords.py) -----------------------------
def session_keywords(req: schemas.SessionKeywordsRequest, res: Resources) -> dict:
    """Wszystkie chunki sesji z wektorami (z Qdrant, bez ponownego embeddingu) -> keywordy, bez LLM."""
    session_id, collection, ingested = resolve_source(req, res)
    flt = apply_topic(build_filter(session_id=session_id), req.topic)
    payloads = fetch_chunks(res.client, collection, flt, with_vectors=True)
    out = {"session_id": session_id, "collection": collection, "ingested": ingested, "topic": req.topic,
           "n_chunks": len(payloads)}
    if not payloads:
        return {**out, "vocab": [], "free": [], "detail": f"Brak chunków sesji {session_id} w {collection}"}

    vocab = kw.load_vocab(server_path(req.vocab_file) if req.vocab_file else config.KEYWORDS_FILE)
    background = kw.background_vectors(res.client, collection) if req.background else None
    result = kw.session_keywords(payloads, res.embedder, vocab=vocab, background=background, top_n=req.top_n,
                                 free_top_n=req.free_top_n, diversity=req.diversity, client_only=req.client_only)
    return {**out, **result}


# --- benchmark (scripts/benchmark_session.py) -----------------------------------
def records(df) -> list[dict]:
    """DataFrame -> lista słowników zgodna z JSON (NaN -> null)."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


def benchmark(req: schemas.BenchmarkRequest, res: Resources):
    from scripts.benchmark_session import config_frame, run_benchmark

    data = req.benchmark if req.benchmark is not None else read_json(req.benchmark_path)
    if "questions" not in data:
        raise ValueError("Benchmark nie ma sekcji 'questions'")
    strategy = strategy_name(req.strategy)
    collection = config.collection_name(strategy)
    require_collection(res, collection)
    topics = req.topics or bool(req.topic)
    flt = apply_topic(build_filter(session_id=req.session), req.topic)

    df, frag, summary = run_benchmark(data["questions"], req.session, strategy, req.mode, req.k,
                                      req.context_order, req.model, retriever=res.retriever, flt=flt,
                                      formatter=formatter(topics), log=None)
    cfg = config_frame(benchmark=req.benchmark_path or "(w zapytaniu)", session_id=req.session,
                       collection=collection, mode=req.mode, k=req.k, context_order=req.context_order,
                       llm_model=req.model, topics=topics, topic=req.topic or "")
    frames = {"results": df, "fragments": frag, "config": cfg, "summary": summary}
    default_name = f"benchmark_{req.session}_{strategy}_{req.mode}_k{req.k}_{req.context_order}.xlsx"
    return frames, default_name


def benchmark_json(frames: dict, saved: str | None) -> dict:
    out = {name: records(df) for name, df in frames.items() if name != "config"}
    out["config"] = dict(zip(frames["config"]["parameter"], frames["config"]["value"].astype(str)))
    out["saved_to"] = saved
    return out


# --- evaluate (scripts/evaluate.py) -----------------------------------------------
def evaluate(req: schemas.EvaluateRequest, res: Resources) -> dict:
    from scripts.evaluate import evaluate as run_eval, load_queries

    queries = list(req.queries)
    if req.queries_path:
        queries += load_queries(server_path(req.queries_path))
    strategies = [strategy_name(s) for s in req.strategies]
    for s in strategies:
        require_collection(res, config.collection_name(s))
    rows = run_eval(queries, strategies, list(req.modes), req.k, req.scope, retriever=res.retriever, log=None)
    return {"k": req.k, "scope": req.scope, "n_queries": len(queries), "rows": rows}


# --- tematy -------------------------------------------------------------------
def list_topics() -> dict:
    subs: dict[str, list] = {}
    for key, s in taxonomy.subtopics.items():
        subs.setdefault(s["main"], []).append({"key": key, "label_en": s.get("label_en"),
                                               "label_pl": s.get("label_pl")})
    return {"topics": [{"key": key, **t, "subtopics": subs.get(key, [])}
                       for key, t in taxonomy.main_topics.items()]}
