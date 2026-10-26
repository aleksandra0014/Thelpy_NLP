"""Wysokopoziomowe funkcje: embedowanie pliku i przepytywanie w jednym miejscu."""
from __future__ import annotations

from . import config
from .chunking import chunk_session
from .embeddings import OllamaEmbedder
from .llm import answer
from .loader import iter_session_files, load_session
from .search import Retriever, build_filter
from .sparse import BM25Encoder
from .store import create_collection, delete_session, get_client, upsert_chunks


def resolve_strategy(chunking: str, size: int | None = None, overlap: int = 0,
                     n_turns: int | None = None, overlap_turns: int = 0) -> str:
    """Zamienia parametry na nazwę strategii i rejestruje ją w config.STRATEGIES.

    chunking = "fixed"  + size/overlap        -> np. "fixed300_ov50"
    chunking = "turns"  + n_turns/overlap     -> np. "turns6_ov2"
    chunking = gotowa nazwa z config.STRATEGIES (np. "turns4") -> bez zmian
    """
    if chunking in config.STRATEGIES:
        return chunking
    if chunking == "fixed":
        if not size:
            raise ValueError("Dla chunking='fixed' podaj size")
        name = f"fixed{size}" + (f"_ov{overlap}" if overlap else "")
        config.STRATEGIES[name] = {"type": "fixed_tokens", "size": size, "overlap": overlap}
        return name
    if chunking == "turns":
        if not n_turns:
            raise ValueError("Dla chunking='turns' podaj n_turns")
        name = f"turns{n_turns}" + (f"_ov{overlap_turns}" if overlap_turns else "")
        config.STRATEGIES[name] = {"type": "turn_window", "n_turns": n_turns, "overlap_turns": overlap_turns}
        return name
    raise ValueError(f"Nieznany chunking: {chunking}. Użyj 'fixed', 'turns' "
                     f"albo jednej z: {', '.join(config.STRATEGIES)}")


def ingest(paths: list[str], strategy: str, recreate: bool = False,
           client=None, embedder=None, bm25=None, verbose: bool = True) -> int:
    """Chunkuje, embeduje (dense + BM25) i zapisuje sesje do kolekcji strategii."""
    client = client or get_client()
    embedder = embedder or OllamaEmbedder()
    bm25 = bm25 or BM25Encoder()
    name = config.collection_name(strategy)

    if recreate or not client.collection_exists(name):
        create_collection(client, name, embedder.dimension(), recreate=recreate)
        verbose and print(f"[kolekcja] {name} utworzona")

    files = iter_session_files(paths)
    if not files:
        raise FileNotFoundError(f"Brak plików .json w {paths}")

    total = 0
    for path in files:
        session = load_session(path)
        chunks = chunk_session(session, strategy)
        texts = [c.text for c in chunks]
        dense = embedder.embed_documents(texts)
        sparse = [bm25.encode_document(t) for t in texts]
        delete_session(client, name, session.session_id)
        total += upsert_chunks(client, name, session, chunks, dense, sparse)
        verbose and print(f"[ingest] {session.session_id}: {len(chunks)} chunków -> {name}")
    return total


def query(question: str, strategy: str, mode: str = "hybrid", k: int = 5,
          session_id: str | None = None, client_id: str | None = None,
          neighbors: int = 0, fusion: str = "client", ask_llm: bool = False,
          client=None, embedder=None, bm25=None) -> dict:
    """Wyszukuje fragmenty i opcjonalnie generuje odpowiedź LLM."""
    r = Retriever(client or get_client(), embedder, bm25)
    name = config.collection_name(strategy)
    kw = {"fusion": fusion} if mode == "hybrid" else {}
    hits = r.search(name, question, mode, k, build_filter(session_id, client_id), **kw)

    contexts = []
    if neighbors:
        seen = set()
        for h in hits:
            for p in r.neighbors(name, h, neighbors):
                if p["chunk_id"] not in seen:
                    seen.add(p["chunk_id"])
                    contexts.append(p)
        contexts.sort(key=lambda p: (p.get("session_at") or "", p["session_id"], p["chunk_index"]))
    else:
        contexts = [h.payload for h in hits]

    result = {"question": question, "strategy": strategy, "mode": mode, "hits": hits, "contexts": contexts}
    if ask_llm:
        result["answer"] = answer(question, contexts)
    return result


def run(file: str | list[str] | None, questions: str | list[str], chunking: str = "turns",
        mode: str = "hybrid", k: int = 5, size: int | None = None, overlap: int = 0,
        n_turns: int | None = 4, overlap_turns: int = 0, recreate: bool = False,
        skip_ingest: bool = False, session_id: str | None = None, client_id: str | None = None,
        neighbors: int = 0, fusion: str = "client", ask_llm: bool = False) -> list[dict]:
    """Główna funkcja: (1) wgraj plik(i) wg wybranego chunkingu, (2) zadaj pytania."""
    strategy = resolve_strategy(chunking, size, overlap, n_turns, overlap_turns)
    client = get_client()
    embedder, bm25 = OllamaEmbedder(), BM25Encoder()

    if file and not skip_ingest:
        ingest([file] if isinstance(file, str) else file, strategy, recreate, client, embedder, bm25)

    questions = [questions] if isinstance(questions, str) else questions
    return [query(q, strategy, mode, k, session_id, client_id, neighbors, fusion, ask_llm,
                  client, embedder, bm25) for q in questions]
