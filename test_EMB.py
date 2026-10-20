#!/usr/bin/env python3
"""
Demo (wersja ANGIELSKA): SPARSE (BM25) vs DENSE (qwen3-embedding) na zdaniach o psychoterapii.

Wymagania:
    pip install numpy requests
    ollama pull qwen3-embedding:0.6b-q8_0
"""

import math
import re
from collections import Counter

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3-embedding:0.6b-q8_0"
TASK = "Given a question about psychotherapy, retrieve relevant passages"

DOCS = [
    "The patient feels intense anxiety before public speaking.",
    "The therapist teaches the patient relaxation and breathing techniques.",
    "Depression shows up as sadness, low energy and loss of interest.",
    "In session we talk about childhood and the relationship with parents.",
    "Panic attacks can cause a racing heart and shortness of breath.",
    "A good relationship with the therapist helps treatment.",
    "The patient has trouble sleeping and wakes up at night.",
]

# (zapytanie, numer oczekiwanego dokumentu, komentarz)
QUERIES = [
    ("anxiety",          0, "to samo słowo jest w dokumencie -> BM25 działa"),
    ("fear",             0, "synonim, tego słowa nie ma nigdzie -> BM25 nic nie znajdzie"),
    ("sad",              2, "w dokumencie jest 'sadness' (inna forma) -> BM25 pudłuje"),
    ("insomnia",         6, "jedno słowo vs opis 'trouble sleeping' -> dense może się pomylić"),
    ("trouble sleeping", 6, "to samo, ale opisowo -> BM25 i dense powinny trafić"),
]


# ============================================================================
# SPARSE (BM25)
# ============================================================================
def tokenize(text):
    return re.findall(r"\w+", text.lower())


class SparseBM25:
    """Dokument = wektor o długości słownika, prawie same zera.
    Trzymamy go jako słownik {słowo: waga} (tylko niezerowe wartości)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        tokens = [tokenize(d) for d in docs]
        self.vocab = sorted({t for doc in tokens for t in doc})
        N = len(docs)
        avgdl = sum(len(t) for t in tokens) / N
        df = Counter(t for doc in tokens for t in set(doc))
        self.idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}

        self.doc_vecs = []
        for doc in tokens:
            tf = Counter(doc)
            dl = len(doc)
            self.doc_vecs.append({
                t: self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
                for t, f in tf.items()
            })

    def search(self, query):
        q_terms = tokenize(query)
        return np.array([sum(v.get(t, 0.0) for t in q_terms) for v in self.doc_vecs])


# ============================================================================
# DENSE (Ollama)
# ============================================================================
def embed(texts):
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # dot = cosine


def embed_query(q):
    return embed([f"Instruct: {TASK}\nQuery:{q}"])[0]


# ============================================================================
# Pomocnicze
# ============================================================================
def rrf(score_lists, k=60):
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        order = [i for i in np.argsort(-s) if s[i] != 0]  # zera = brak trafienia
        for rank, idx in enumerate(order):
            fused[idx] += 1 / (k + rank + 1)
    return fused


def rank_of(scores, idx):
    if scores[idx] == 0:
        return None
    return int(np.where(np.argsort(-scores) == idx)[0][0]) + 1


def top(scores, n=2):
    out = []
    for i in np.argsort(-scores)[:n]:
        out.append(f"#{i} ({scores[i]:.2f})" if scores[i] > 1e-9 else "— brak trafień —")
    return ", ".join(out)


def header(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main():
    header("DOKUMENTY")
    for i, d in enumerate(DOCS):
        print(f"  #{i}  {d}")

    sparse = SparseBM25(DOCS)
    dense_docs = embed(DOCS)

    header("1) SPARSE – jak wygląda dokument w BM25")
    print(f"Słownik ma {len(sparse.vocab)} słów, więc każdy dokument to wektor o {len(sparse.vocab)} pozycjach.")
    print("Niezerowe są tylko słowa, które w dokumencie faktycznie występują.\n")
    for i in (0, 4):
        vec = sparse.doc_vecs[i]
        print(f"Dokument #{i}: {DOCS[i]}")
        print(f"  niezerowych pozycji: {len(vec)} z {len(sparse.vocab)}  "
              f"({100 * len(vec) / len(sparse.vocab):.0f}%), reszta to zera")
        for t, w in sorted(vec.items(), key=lambda x: -x[1]):
            print(f"    {t:<14} {w:.2f}")
        print()
    print("Rzadkie słowa dostają wyższą wagę (IDF), np. 'anxiety' > 'the'.")
    print("Wynik = suma wag słów z zapytania w dokumencie.")

    header("2) DENSE – jak wygląda dokument w embeddingu")
    v = dense_docs[0]
    print(f"Dokument #0: {DOCS[0]}")
    print(f"  wymiar wektora: {len(v)} liczb (zawsze tyle, niezależnie od długości tekstu)")
    print(f"  pierwsze 8 liczb: {np.round(v[:8], 3).tolist()}")
    print(f"  zer w wektorze: {int(np.sum(v == 0))} z {len(v)}  -> wektor 'gęsty' (dense)")

    print("\nPodobieństwo kosinusowe między dokumentami (1.0 = identyczne):")
    sim = dense_docs @ dense_docs.T
    print("      " + "".join(f"#{j:<5}" for j in range(len(DOCS))))
    for i in range(len(DOCS)):
        print(f"  #{i}  " + "".join(f"{sim[i, j]:<6.2f}" for j in range(len(DOCS))))

    header("3) PORÓWNANIE na zapytaniach")
    for q, expected, comment in QUERIES:
        s_sparse = sparse.search(q)
        s_dense = dense_docs @ embed_query(q)
        s_hybrid = rrf([s_sparse, s_dense])
        print(f"\nZapytanie: '{q}'   <- {comment}")
        print(f"  BM25 (sparse):     {top(s_sparse)}")
        print(f"  Embedding (dense): {top(s_dense)}")
        print(f"  Hybryda (RRF):     {top(s_hybrid)}")
        print(f"  Oczekiwany dokument #{expected}: \"{DOCS[expected]}\"")
        for name, s in (("BM25", s_sparse), ("dense", s_dense), ("hybryda", s_hybrid)):
            r = rank_of(s, expected)
            print(f"    miejsce wg {name:<8}: {r if r else 'brak (wynik 0)'}")

    header("PODSUMOWANIE")
    print("  SPARSE (BM25):  szuka tych samych słów; nie zna synonimów ani odmian ('sad' != 'sadness').")
    print("  DENSE (embed.): szuka podobnego znaczenia; potrafi się mylić przy bardzo krótkich zapytaniach.")
    print("  HYBRYDA:        łączy oba rankingi.")


if __name__ == "__main__":
    main()