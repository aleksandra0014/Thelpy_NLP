"""Wyszukiwanie: dense, bm25, hybrid (RRF) + dołączanie sąsiednich chunków."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from qdrant_client import QdrantClient, models

from . import config
from .embeddings import OllamaEmbedder
from .sparse import BM25Encoder

Mode = Literal["dense", "bm25", "hybrid"]


@dataclass
class Hit:
    id: str
    score: float
    payload: dict
    ranks: Optional[dict] = None   # przy hybrid: pozycja w każdej gałęzi (do analizy)


def build_filter(session_id: str | None = None, client_id: str | None = None,
                 speaker_role: str | None = None, date_from: str | None = None,
                 date_to: str | None = None) -> models.Filter | None:
    must = []
    if session_id:
        must.append(models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)))
    if client_id:
        must.append(models.FieldCondition(key="client_id", match=models.MatchValue(value=client_id)))
    if speaker_role:
        must.append(models.FieldCondition(key="speaker_roles", match=models.MatchValue(value=speaker_role)))
    if date_from or date_to:
        must.append(models.FieldCondition(key="session_at", range=models.DatetimeRange(gte=date_from, lte=date_to)))
    return models.Filter(must=must) if must else None


def build_context(hits: list[Hit], neighbors_fn=None, order: str = "chrono") -> list[dict]:
    """Trafienia -> payloady dla LLM (kopie) z informacją o rankingu.

    Każde trafienie dostaje retrieval_rank (1 = najlepsze), retrieval_score i retrieval_k.
    neighbors_fn: opcjonalnie Hit -> list[payload] (np. sąsiedzi); dołożone chunki, które same
    nie są trafieniami, dostają retrieval_neighbor=True. Duplikaty (po chunk_id) są usuwane.
    order: "chrono" (sesja, chunk_index) albo "score" (kolejność rankingu).
    """
    ranked = {h.payload["chunk_id"]: (rank, h.score) for rank, h in enumerate(hits, 1)}
    seen, ctx = set(), []
    for h in hits:
        for p in (neighbors_fn(h) if neighbors_fn else [h.payload]):
            cid = p["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            p = dict(p)
            if cid in ranked:
                p["retrieval_rank"], p["retrieval_score"] = ranked[cid]
                p["retrieval_k"] = len(hits)
            else:
                p["retrieval_neighbor"] = True
            ctx.append(p)
    if order == "chrono":
        ctx.sort(key=lambda p: (p.get("session_at") or "", p["session_id"], p["chunk_index"]))
    elif order != "score":
        raise ValueError(f"Nieznana kolejność kontekstu: {order} (chrono | score)")
    return ctx


def _to_hits(points) -> list[Hit]:
    return [Hit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in points]


def rrf_fuse(rankings: dict[str, list[Hit]], k: int = config.RRF_K, limit: int = 10) -> list[Hit]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank), rank liczony od 1."""
    fused: dict[str, Hit] = {}
    for branch, hits in rankings.items():
        for rank, h in enumerate(hits, start=1):
            if h.id not in fused:
                fused[h.id] = Hit(id=h.id, score=0.0, payload=h.payload, ranks={})
            fused[h.id].score += 1.0 / (k + rank)
            fused[h.id].ranks[branch] = rank
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:limit]


class Retriever:
    def __init__(self, client: QdrantClient, embedder: OllamaEmbedder | None = None,
                 bm25: BM25Encoder | None = None):
        self.client = client
        self.embedder = embedder or OllamaEmbedder()
        self.bm25 = bm25 or BM25Encoder()

    def dense(self, collection: str, query: str, limit: int, flt=None) -> list[Hit]:
        res = self.client.query_points(collection, query=self.embedder.embed_query(query),
                                       using=config.DENSE_VECTOR, query_filter=flt,
                                       limit=limit, with_payload=True)
        return _to_hits(res.points)

    def sparse(self, collection: str, query: str, limit: int, flt=None) -> list[Hit]:
        vec = self.bm25.encode_query(query)
        if not vec.indices:          # zapytanie złożone wyłącznie ze stopwords
            return []
        res = self.client.query_points(collection, query=vec, using=config.SPARSE_VECTOR,
                                       query_filter=flt, limit=limit, with_payload=True)
        return _to_hits(res.points)

    def hybrid(self, collection: str, query: str, limit: int, flt=None,
               fusion: Literal["client", "server"] = "client",
               prefetch_limit: int = config.PREFETCH_LIMIT, rrf_k: int = config.RRF_K) -> list[Hit]:
        """RRF po stronie klienta (pełna kontrola nad k, widoczne rangi gałęzi)
        albo po stronie Qdranta (jedno zapytanie, Query API z prefetch)."""
        if fusion == "client":
            return rrf_fuse({"dense": self.dense(collection, query, prefetch_limit, flt),
                             "bm25": self.sparse(collection, query, prefetch_limit, flt)},
                            k=rrf_k, limit=limit)
        # client (domyślny): pobieramy od Qdranta po 50 kandydatów z każdej gałęzi, a RRF liczymy sami w Pythonie. Kontrolujesz k z .env i widzisz ranks w wynikach.
        # server: jedno zapytanie, w którym Qdrant sam pobiera kandydatów z obu gałęzi i łączy je RRF-em.

        res = self.client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=self.embedder.embed_query(query), using=config.DENSE_VECTOR,
                                filter=flt, limit=prefetch_limit),
                models.Prefetch(query=self.bm25.encode_query(query), using=config.SPARSE_VECTOR,
                                filter=flt, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt, limit=limit, with_payload=True,
        )
        return _to_hits(res.points)

    def search(self, collection: str, query: str, mode: Mode = "hybrid", limit: int = 5,
               flt=None, **kw) -> list[Hit]:
        if mode == "dense":
            return self.dense(collection, query, limit, flt)
        if mode == "bm25":
            return self.sparse(collection, query, limit, flt)
        if mode == "hybrid":
            return self.hybrid(collection, query, limit, flt, **kw)
        raise ValueError(f"Nieznany tryb: {mode}")

    # --- kontekst sąsiadów -------------------------------------------------------
    def neighbors(self, collection: str, hit: Hit, window: int = 1) -> list[dict]:
        """Zwraca payloady chunków [i-window, i+window] z tej samej sesji, posortowane.
        Dla znalezionego fragmentu pobiera chunki o numerach od i − window do i + window z tej samej sesji. Dla fragmentu nr 15 i window=1 dostajesz 14, 15, 16."""
        i = hit.payload["chunk_index"]
        flt = models.Filter(must=[
            models.FieldCondition(key="session_id", match=models.MatchValue(value=hit.payload["session_id"])),
            models.FieldCondition(key="chunk_index", range=models.Range(gte=i - window, lte=i + window)),
        ])
        points, _ = self.client.scroll(collection, scroll_filter=flt, limit=2 * window + 1,
                                       with_payload=True, with_vectors=False)
        return sorted((p.payload for p in points), key=lambda p: p["chunk_index"])
