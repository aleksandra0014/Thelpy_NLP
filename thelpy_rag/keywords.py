"""Keywordy sesji bez LLM — dwie metody, obie na wektorach chunków już zapisanych w Qdrant.

A. Słownik (data/keywords.yaml): dla każdego hasła dwa sygnały
   1. leksykalny  — czy hasło/alias pada dosłownie (po stemmingu: "drinks" == "drink"),
   2. znaczeniowy — cosinus embeddingu hasła z embeddingiem każdego chunka, znormalizowany
                    (patrz semantic_scores), agregowany po najlepszych chunkach.
B. Wolne frazy (KeyBERT-style): n-gramy z wypowiedzi klienta, ranking po podobieństwie do
   wektora sesji (średnia wektorów chunków) + MMR dla różnorodności.

Opis krok po kroku: thelpy_rag/KEYWORDS.md.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import config
from .sparse import ENGLISH_STOPWORDS, TOKEN_RE, get_stemmer

# wtrącenia rozmowy, które nie są keywordami (dokładane do stopwords przy wolnych frazach)
FILLERS = set("""
yes yeah yep nope no okay ok alright right well um uh huh mm hmm mhm oh ah eh
suppose guess kinda sorta dunno gonna wanna gotta cause cos stuff thing things something anything
everything nothing maybe quite pretty actually obviously basically really just like know think mean
kind sort said say saying says tell told got get getting go going goes went come coming want lot
bit little bad good sure probably definitely exactly totally crosstalk inaudible laughs laughing
sounds sound way time times day today people person doesn don didn isn wasn aren ll ve re
""".split())
KEYWORD_TEMPLATE = "The client talks about {}."   # dodatkowy tekst hasła — kontekst zdania stabilizuje embedding
CLIENT_ROLES = ("Client:", "Patient:")
LINE_RE = re.compile(r"^([A-Z][\w ]{0,20}):\s?", re.MULTILINE)


@dataclass
class Keyword:
    key: str
    category: str
    label: str
    aliases: list[str] = field(default_factory=list)
    desc: str | None = None

    def texts(self) -> list[str]:
        """Teksty, których embeddingi uśredniamy w jeden wektor hasła."""
        out = [self.label, KEYWORD_TEMPLATE.format(self.label), *self.aliases]
        return out + ([self.desc] if self.desc else [])


# --- słownik ----------------------------------------------------------------------
def load_vocab(path: str | Path = config.KEYWORDS_FILE) -> list[Keyword]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    vocab = []
    for category, items in data.items():
        for key, v in (items or {}).items():
            v = v or {}
            vocab.append(Keyword(key=str(key), category=str(category), label=str(v.get("label") or key),
                                 aliases=[str(a) for a in v.get("aliases") or []], desc=v.get("desc")))
    if not vocab:
        raise ValueError(f"Słownik {path} jest pusty")
    return vocab


_stemmer = get_stemmer(config.BM25_STEMMER)


def terms(text: str) -> list[str]:
    """Tokeny po stemmingu, BEZ usuwania stopwords (żeby frazy typu "split up" dało się dopasować)."""
    toks = TOKEN_RE.findall(text.lower())
    return _stemmer.stemWords(toks) if _stemmer else toks


def count_phrase(tokens: list[str], phrase: tuple[str, ...]) -> int:
    n = len(phrase)
    return sum(tuple(tokens[i:i + n]) == phrase for i in range(len(tokens) - n + 1)) if n else 0


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


_vocab_cache: dict[str, np.ndarray] = {}


def vocab_matrix(vocab: list[Keyword], embedder) -> np.ndarray:
    """Macierz K×D: wektor hasła = znormalizowana średnia embeddingów jego tekstów.
    Cache w pamięci i na dysku (data/keyword_cache/), klucz = treść słownika + model."""
    texts = [k.texts() for k in vocab]
    key = hashlib.md5((repr(texts) + embedder.model).encode()).hexdigest()[:12]
    if key in _vocab_cache:
        return _vocab_cache[key]
    path = config.PROJECT_ROOT / "data" / "keyword_cache" / f"vocab_{key}.npy"
    if path.exists():
        mat = np.load(path)
    else:
        flat = [t for ts in texts for t in ts]
        emb = _normalize(np.array(embedder.embed_documents(flat), dtype=np.float32))
        rows, i = [], 0
        for ts in texts:
            rows.append(emb[i:i + len(ts)].mean(axis=0))
            i += len(ts)
        mat = _normalize(np.stack(rows))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mat)
    _vocab_cache[key] = mat
    return mat


# --- pomocnicze: chunki ------------------------------------------------------------
def chunk_vectors(payloads: list[dict]) -> np.ndarray:
    if any("_dense" not in p for p in payloads):
        raise ValueError("Chunki bez wektorów — pobierz je przez fetch_chunks(..., with_vectors=True)")
    return _normalize(np.array([p["_dense"] for p in payloads], dtype=np.float32))


def turn_ranges(payloads: list[dict]) -> list[str]:
    """Zakresy tur sąsiednich chunków sklejone: [T001–T008, T009–T016] -> T001–T016."""
    out: list[list] = []
    for p in payloads:
        ids = p.get("turn_ids") or [str(p["chunk_index"])]
        if out and out[-1][2] == p["chunk_index"] - 1:
            out[-1][1], out[-1][2] = ids[-1], p["chunk_index"]
        else:
            out.append([ids[0], ids[-1], p["chunk_index"]])
    return [a if a == b else f"{a}–{b}" for a, b, _ in out]


def speaker_text(payload: dict, roles: tuple[str, ...] | None) -> str:
    """Tekst chunka; z roles tylko wypowiedzi tych mówiących (linie "Client: ...")."""
    text = payload["text"]
    if not roles:
        return LINE_RE.sub("", text)
    parts, keep = [], False
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            keep = f"{m.group(1)}:" in roles
            line = line[m.end():]
        if keep:
            parts.append(line)
    return "\n".join(parts)


def background_vectors(client, collection: str, limit: int = 2000) -> np.ndarray:
    """Wektory dense do `limit` chunków całej kolekcji (wszystkie sesje) — tło do normalizacji."""
    vecs, offset = [], None
    while len(vecs) < limit:
        points, offset = client.scroll(collection, limit=min(256, limit - len(vecs)), offset=offset,
                                       with_payload=False, with_vectors=[config.DENSE_VECTOR])
        vecs.extend(p.vector[config.DENSE_VECTOR] for p in points)
        if offset is None:
            break
    return _normalize(np.array(vecs, dtype=np.float32)) if vecs else np.zeros((0, 0), dtype=np.float32)


# --- A. słownik ----------------------------------------------------------------------
def semantic_scores(chunk_vecs: np.ndarray, vocab_vecs: np.ndarray,
                    background: np.ndarray | None = None) -> np.ndarray:
    """Macierz n×K "jak bardzo chunk i jest o haśle k", w jednostkach odchylenia standardowego.

    Surowy cosinus jest mało czytelny: wszystkie hasła mają 0.40–0.60, a niektóre ("anxiety")
    są wysoko przy KAŻDYM chunku terapii. Dlatego normalizacja dwustopniowa:
    1. w wierszu (chunk): z-score po wszystkich hasłach — które hasła wyróżniają się dla tego chunka,
    2. w kolumnie (hasło): odjęcie średniej tego hasła na tle (background = chunki całej kolekcji),
       czyli „ile bardziej niż zwykle”. Bez tła (None) krok 2 jest pomijany."""
    def row_z(s):
        return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)

    z = row_z(chunk_vecs @ vocab_vecs.T)
    if background is not None and len(background):
        z = z - row_z(background @ vocab_vecs.T).mean(axis=0, keepdims=True)
    return z


def score_vocab(payloads: list[dict], vocab: list[Keyword], vocab_vecs: np.ndarray,
                background: np.ndarray | None = None, top_n: int = 10, top_frac: float = 0.15,
                chunk_threshold: float = config.KW_CHUNK_Z, lex_weight: float = config.KW_LEX_WEIGHT,
                min_score: float = 0.0) -> list[dict]:
    """Ranking haseł słownika dla jednej sesji (payloads z "_dense", po chunk_index)."""
    z = semantic_scores(chunk_vectors(payloads), vocab_vecs, background)          # n × K
    m = max(1, math.ceil(top_frac * len(payloads)))
    semantic = np.sort(z, axis=0)[-m:].mean(axis=0)                                # średnia z m najlepszych chunków

    chunk_terms = [terms(speaker_text(p, None)) for p in payloads]     # bez etykiet "Therapist:" / "Client:"
    out = []
    for k, kw in enumerate(vocab):
        phrases = {tuple(terms(a)) for a in kw.aliases + [kw.label]} - {()}
        per_chunk = [sum(count_phrase(t, ph) for ph in phrases) for t in chunk_terms]
        hits = sum(per_chunk)
        score = float(semantic[k]) + lex_weight * math.log1p(hits)
        present = [p for p, h, s in zip(payloads, per_chunk, z[:, k]) if h or s >= chunk_threshold]
        out.append({"keyword": kw.key, "label": kw.label, "category": kw.category, "score": round(score, 3),
                    "semantic": round(float(semantic[k]), 3), "lexical_hits": hits,
                    "coverage": round(len(present) / len(payloads), 3), "turns": turn_ranges(present)})
    out.sort(key=lambda r: r["score"], reverse=True)
    return [r for r in out if r["score"] >= min_score][:top_n]


# --- B. wolne frazy (KeyBERT-style) ---------------------------------------------------
def candidate_phrases(texts: list[str], ngram_range=(1, 3), max_candidates: int = 300,
                      min_count: int = 2) -> list[tuple[str, int]]:
    """N-gramy bez stopwords (sklearn CountVectorizer) z liczbą wystąpień, najczęstsze pierwsze."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    stop = sorted(ENGLISH_STOPWORDS | set(ENGLISH_STOP_WORDS) | FILLERS)
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    try:
        counts = vec.fit_transform(texts).sum(axis=0).A1
    except ValueError:      # pusty słownik (np. sesja bez wypowiedzi klienta)
        return []
    pairs = [(w, int(c)) for w, c in zip(vec.get_feature_names_out(), counts) if c >= min_count]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs[:max_candidates]


def mmr_rel(rel: np.ndarray, cands: np.ndarray, top_n: int, diversity: float) -> list[int]:
    """Maximal Marginal Relevance: kolejno fraza najtrafniejsza (rel), ale daleka od już wybranych.
    diversity=0 -> czysty ranking trafności, 1 -> maksymalna różnorodność."""
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)      # trafność i podobieństwo w podobnej skali
    chosen = [int(rel.argmax())]
    sim = cands @ cands.T
    while len(chosen) < min(top_n, len(cands)):
        rest = [i for i in range(len(cands)) if i not in chosen]
        mmr_score = (1 - diversity) * rel[rest] - diversity * sim[np.ix_(rest, chosen)].max(axis=1)
        chosen.append(rest[int(mmr_score.argmax())])
    return chosen


def extract_free(payloads: list[dict], embedder, top_n: int = 10, diversity: float = 0.5,
                 client_only: bool = True, max_candidates: int = 300,
                 background: np.ndarray | None = None) -> list[dict]:
    """background: wektory chunków całej kolekcji; jeśli są w niej też inne sesje, trafność frazy to
    cos(fraza, sesja) − cos(fraza, tło) — fraza typowa dla KAŻDEJ rozmowy terapeutycznej spada."""
    roles = CLIENT_ROLES if client_only else None
    texts = [speaker_text(p, roles) for p in payloads]
    cands = candidate_phrases(texts, max_candidates=max_candidates)
    if not cands:
        return []
    doc = _normalize(chunk_vectors(payloads).mean(axis=0))                     # wektor całej sesji
    emb = _normalize(np.array(embedder.embed_documents([c for c, _ in cands]), dtype=np.float32))
    rel = emb @ doc
    if background is not None and len(background) > len(payloads):
        rel = rel - emb @ _normalize(background.mean(axis=0))
    out = []
    for i in mmr_rel(rel, emb, top_n, diversity):
        out.append({"phrase": cands[i][0], "count": cands[i][1], "relevance": round(float(rel[i]), 3)})
    return out


# --- całość ------------------------------------------------------------------------------
def session_keywords(payloads: list[dict], embedder, vocab: list[Keyword] | None = None,
                     background: np.ndarray | None = None, top_n: int = 10, free_top_n: int = 10,
                     diversity: float = 0.5, client_only: bool = True) -> dict:
    vocab = vocab or load_vocab()
    vocab_vecs = vocab_matrix(vocab, embedder)
    return {"vocab": score_vocab(payloads, vocab, vocab_vecs, background, top_n=top_n),
            "free": extract_free(payloads, embedder, top_n=free_top_n, diversity=diversity,
                                 client_only=client_only, background=background) if free_top_n else [],
            "vocab_size": len(vocab), "background_chunks": 0 if background is None else len(background)}
