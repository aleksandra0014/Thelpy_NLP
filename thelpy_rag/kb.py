"""Baza wiedzy (knowledge base): dokumenty z sekcjami w osobnej kolekcji Qdranta.

Kolekcja ma te same wektory co kolekcje sesji (dense + BM25), więc przeszukuje ją zwykły
search.Retriever (dense / bm25 / hybrid z RRF). Różni się payload: zamiast sesji i tur
każdy punkt opisuje fragment sekcji dokumentu (doc_name, title, source_url, section_index, ...).

Chunkowanie sekcji po akapitach:
1. tekst sekcji dzielony na akapity (pusta linia),
2. akapity sklejane w chunk, dopóki suma tokenów <= max_tokens,
3. akapit dłuższy niż max_tokens cięty oknem tokenowym z overlapem,
4. całe akapity nie są powtarzane między chunkami (overlap tylko przy cięciu za długiego akapitu).

ID punktu = uuid5("kb::<doc_name>::<section_index>::<chunk_index>"), a przed zapisem usuwane są
wszystkie punkty dokumentu — ponowne wgranie nadpisuje dokument zamiast go dublować.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client import QdrantClient, models

from . import config
from .chunking import get_tokenizer
from .store import point_id

PARAGRAPH_RE = re.compile(r"\n\s*\n")


@dataclass
class KbSection:
    text: str
    title: Optional[str] = None


@dataclass
class KbDocument:
    title: str
    sections: list[KbSection]
    source_url: Optional[str] = None
    doc_name: Optional[str] = None          # identyfikator do filtrowania; domyślnie title
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.doc_name or self.title


@dataclass
class KbChunk:
    point_id: str
    embed_text: str      # tekst z nagłówkiem (tytuł dokumentu — tytuł sekcji) do embeddingu i BM25
    payload: dict


# --- chunkowanie -----------------------------------------------------------------
def _split_long(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Akapit dłuższy niż max_tokens -> okna tokenowe z overlapem (cięcie po granicach tokenów)."""
    offs = get_tokenizer().offsets(text)
    n = len(offs)
    # token i zaczyna słowo, jeśli przed nim jest odstęp albo sam zaczyna się od odstępu (offsety
    # sentencepiece, np. bge-m3, obejmują spację: "▁word" -> (7, 12) dla " word"); dzięki temu
    # granica cięcia nie rozrywa słowa na podsłowa
    word_start = [i == 0 or offs[i][0] > offs[i - 1][1] or text[offs[i][0]].isspace() for i in range(n)]

    def snap_back(i: int, lo: int) -> int:
        """Najbliższy początek słowa <= i (ale > lo); gdy słowo jest dłuższe niż okno — i bez zmian."""
        j = i
        while j > lo and not word_start[j]:
            j -= 1
        return j if j > lo else i

    out, start = [], 0
    while start < n:
        end = n if start + max_tokens >= n else snap_back(start + max_tokens, start)
        out.append(text[offs[start][0]:offs[end - 1][1]].strip())
        if end == n:
            break
        nxt = snap_back(max(end - overlap, start + 1), start)   # overlap też od początku słowa
        start = nxt if nxt > start else end
    return out


def chunk_section(text: str, max_tokens: int = config.KB_CHUNK_TOKENS,
                  overlap: int = config.KB_CHUNK_OVERLAP) -> list[str]:
    """Tekst sekcji -> chunki z całych akapitów (długie akapity cięte oknem tokenowym)."""
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"overlap ({overlap}) musi być w zakresie 0..max_tokens-1 ({max_tokens - 1})")
    tok = get_tokenizer()
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            chunks.append("\n\n".join(cur))
        cur, cur_tokens = [], 0

    for para in (p.strip() for p in PARAGRAPH_RE.split(text)):
        if not para:
            continue
        n = tok.count(para)
        if n > max_tokens:
            flush()
            chunks.extend(_split_long(para, max_tokens, overlap))
        elif cur_tokens + n > max_tokens:
            flush()
            cur, cur_tokens = [para], n
        else:
            cur.append(para)
            cur_tokens += n
    flush()
    return chunks


def chunk_document(doc: KbDocument, max_tokens: int = config.KB_CHUNK_TOKENS,
                   overlap: int = config.KB_CHUNK_OVERLAP) -> list[KbChunk]:
    tok = get_tokenizer()
    added_at = datetime.now(timezone.utc).isoformat()
    out = []
    for si, section in enumerate(doc.sections):
        header = doc.title + (f" — {section.title}" if section.title else "")
        for ci, text in enumerate(chunk_section(section.text, max_tokens, overlap)):
            payload = {
                **doc.metadata,                       # pola własne nie nadpisują pól poniżej
                "doc_name": doc.name, "title": doc.title, "source_url": doc.source_url,
                "section_index": si, "section_title": section.title, "chunk_index": ci,
                "n_tokens": tok.count(text), "text": text, "added_at": added_at,
            }
            out.append(KbChunk(point_id=point_id(f"kb::{doc.name}::{si}::{ci}"),
                               embed_text=f"{header}\n\n{text}", payload=payload))
    return out


# --- pliki Markdown / tekst ------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def parse_markdown(text: str, fallback_title: str) -> tuple[str, list[KbSection]]:
    """Markdown / zwykły tekst -> (tytuł, sekcje).

    Tytuł dokumentu = pierwszy nagłówek `# ...` (poziom 1), a bez niego fallback_title (np. nazwa pliku).
    Każdy nagłówek zaczyna nową sekcję o jego tytule; tekst przed pierwszym nagłówkiem i tekst pod
    nagłówkiem-tytułem tworzą sekcję bez tytułu. Puste sekcje są pomijane. Plik bez nagłówków = jedna sekcja.
    """
    text = text.replace("\r\n", "\n")
    heads = list(HEADING_RE.finditer(text))
    h1 = next((h for h in heads if len(h.group(1)) == 1), None)
    title = h1.group(2).strip() if h1 else fallback_title

    sections = []
    bounds = [(None, 0)] + [(h, h.end()) for h in heads]
    for i, (h, start) in enumerate(bounds):
        end = heads[i].start() if i < len(heads) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        sec_title = None if h is None or h is h1 else h.group(2).strip()
        sections.append(KbSection(text=body, title=sec_title))
    return title, sections


# --- Qdrant ----------------------------------------------------------------------
def doc_filter(doc_names: list[str] | None) -> models.Filter | None:
    if not doc_names:
        return None
    return models.Filter(must=[models.FieldCondition(key="doc_name", match=models.MatchAny(any=doc_names))])


def delete_document(client: QdrantClient, name: str, doc_name: str) -> None:
    """Usuwa wszystkie fragmenty dokumentu (przed ponownym wgraniem albo na żądanie)."""
    client.delete(name, points_selector=models.FilterSelector(filter=doc_filter([doc_name])), wait=True)


def upsert_kb(client: QdrantClient, name: str, chunks: list[KbChunk], dense: list[list[float]],
              sparse: list[models.SparseVector], batch_size: int = 64) -> int:
    points = [
        models.PointStruct(id=c.point_id, vector={config.DENSE_VECTOR: d, config.SPARSE_VECTOR: s},
                           payload=c.payload)
        for c, d, s in zip(chunks, dense, sparse, strict=True)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(name, points=points[i:i + batch_size], wait=True)
    return len(points)
