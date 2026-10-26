"""Operacje na Qdrancie: kolekcje, indeksy payloadu, zapis i usuwanie chunków."""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from . import config
from .schema import Chunk, Session

# stała przestrzeń nazw -> ten sam chunk_id zawsze daje to samo ID punktu (idempotentny upsert)
POINT_NAMESPACE = uuid.UUID("7b1f5c1e-3d2a-4c55-9a0e-5f1e2d3c4b5a")

PAYLOAD_INDEXES = [
    ("session_id", models.PayloadSchemaType.KEYWORD),
    ("client_id", models.PayloadSchemaType.KEYWORD),
    ("strategy", models.PayloadSchemaType.KEYWORD),
    ("source_version", models.PayloadSchemaType.KEYWORD),
    ("speaker_roles", models.PayloadSchemaType.KEYWORD),
    ("speaker_ids", models.PayloadSchemaType.KEYWORD),
    ("turn_ids", models.PayloadSchemaType.KEYWORD),
    ("chunk_index", models.PayloadSchemaType.INTEGER),
    ("session_at", models.PayloadSchemaType.DATETIME),
    ("added_at", models.PayloadSchemaType.DATETIME),
]


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=120)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


# 1. --recreate kasuje całą kolekcję, nie tylko sesję, którą wgrywasz. W main.py połączenie --file S001.json 
# --recreate oznacza: wyczyść kolekcję i wgraj tylko S001. Inne sesje znikają. Do ponownego wgrania jednej sesji
#  wystarczy zwykłe wgranie bez --recreate, bo delete_session zadba o resztę.

# 2. Zmiana modelu embeddingów (inny wymiar) wymaga --recreate, bo kolekcja ma wymiar ustawiony na stałe przy tworzeniu.

def create_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> bool:
    """Tworzy kolekcję z wektorem dense + rzadkim BM25. Zwraca True, jeśli utworzono."""
    if client.collection_exists(name):
        if not recreate:
            return False
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={config.DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={config.SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in PAYLOAD_INDEXES:
        client.create_payload_index(name, field_name=field, field_schema=schema)
    return True


def session_filter(session_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))])


def delete_session(client: QdrantClient, name: str, session_id: str) -> None:
    """Usuwa wszystkie chunki sesji (np. przed wgraniem poprawionej transkrypcji)."""
    client.delete(name, points_selector=models.FilterSelector(filter=session_filter(session_id)), wait=True)


def upsert_chunks(client: QdrantClient, name: str, session: Session, chunks: list[Chunk],
                  dense: list[list[float]], sparse: list[models.SparseVector], batch_size: int = 64) -> int:
    points = [
        models.PointStruct(
            id=point_id(c.chunk_id),
            vector={config.DENSE_VECTOR: d, config.SPARSE_VECTOR: s},
            payload={**session.payload(), **c.payload()},
        )
        for c, d, s in zip(chunks, dense, sparse, strict=True)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(name, points=points[i:i + batch_size], wait=True)
    return len(points)
