#!/usr/bin/env python3
"""
Wizualizacja embeddingów (dense) wypowiedzi z transkryptu sesji: t-SNE + UMAP.

Wymagania:
    pip install numpy requests scikit-learn umap-learn plotly
    ollama pull qwen3-embedding:0.6b-q8_0

Użycie:
    python visualize_dialog_embeddings.py transcript.json
    python visualize_dialog_embeddings.py transcript.json --color time
    python visualize_dialog_embeddings.py transcript.json --min-words 3     # bez "Okay.", "Mm-hmm."
    python visualize_dialog_embeddings.py transcript.json --nn 20           # najbliższe sąsiedzi wypowiedzi #20

Wynik: dialog_embeddings.html (interaktywny - najedź na punkt, żeby zobaczyć tekst).
"""

import argparse
import html
import json
import sys
import textwrap

import numpy as np
import plotly.graph_objects as go
import requests
import umap
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3-embedding:0.6b-q8_0"

SPEAKER_STYLE = {
    "therapist": {"color": "#2a6fdb", "symbol": "circle"},
    "patient": {"color": "#e0662b", "symbol": "diamond"},
}
DEFAULT_STYLE = {"color": "#777777", "symbol": "square"}


# ---------------------------------------------------------------------------
# Dane
# ---------------------------------------------------------------------------
def load_utterances(path, min_words):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for turn in data["dialog"]:
        text = turn["text"].strip()
        if len(text.split()) < min_words:
            continue
        items.append({"timestamp": turn["timestamp"], "speaker": turn["speaker"], "text": text})
    return data.get("video_title", ""), items


# ---------------------------------------------------------------------------
# Embeddingi (dense)
# ---------------------------------------------------------------------------
def embed(texts, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": MODEL, "input": texts[i : i + batch]},
            timeout=300,
        )
        r.raise_for_status()
        vecs.extend(r.json()["embeddings"])
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # dot = cosine


# ---------------------------------------------------------------------------
# Redukcja wymiarów
# ---------------------------------------------------------------------------
def run_tsne(X, perplexity):
    n = len(X)
    perplexity = max(2, min(perplexity, (n - 1) // 3))  # perplexity musi być << liczby próbek
    return TSNE(
        n_components=2, perplexity=perplexity, metric="cosine",
        init="pca", learning_rate="auto", random_state=42,
    ).fit_transform(X)


def run_umap(X, n_neighbors):
    n_neighbors = max(2, min(n_neighbors, len(X) - 1))
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.2,
        metric="cosine", random_state=42,
    ).fit_transform(X)


# ---------------------------------------------------------------------------
# Wykres
# ---------------------------------------------------------------------------
def hover_text(i, u):
    wrapped = "<br>".join(textwrap.wrap(html.escape(u["text"]), 60))
    return f"<b>#{i}</b> [{u['timestamp']}] {u['speaker']}<br>{wrapped}"


def add_panel(fig, coords, items, col, color_mode, show_legend):
    speakers = sorted({u["speaker"] for u in items})
    n = len(items)
    for sp in speakers:
        idx = [i for i, u in enumerate(items) if u["speaker"] == sp]
        style = SPEAKER_STYLE.get(sp, DEFAULT_STYLE)
        marker = {"size": 11, "symbol": style["symbol"], "line": {"width": 0.5, "color": "white"}}
        if color_mode == "time":
            marker.update(color=idx, colorscale="Viridis", cmin=0, cmax=n - 1)
        else:
            marker.update(color=style["color"])
        fig.add_trace(
            go.Scatter(
                x=coords[idx, 0], y=coords[idx, 1],
                mode="markers+text",
                text=[str(i) for i in idx], textposition="top center",
                textfont={"size": 9, "color": "#555"},
                marker=marker,
                name=sp, legendgroup=sp, showlegend=show_legend,
                hovertext=[hover_text(i, items[i]) for i in idx],
                hoverinfo="text",
            ),
            row=1, col=col,
        )


def make_figure(title, items, tsne_xy, umap_xy, color_mode):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("t-SNE", "UMAP"))
    add_panel(fig, tsne_xy, items, col=1, color_mode=color_mode, show_legend=True)
    add_panel(fig, umap_xy, items, col=2, color_mode=color_mode, show_legend=False)
    suffix = "kolor = czas rozmowy (ciemny = początek)" if color_mode == "time" else "kolor = mówiący"
    fig.update_layout(
        title=f"{title}<br><sup>Embeddingi {MODEL}, {len(items)} wypowiedzi, {suffix}</sup>",
        template="plotly_white", height=650, hoverlabel={"align": "left"},
    )
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Najbliżsi sąsiedzi (szybka kontrola jakości embeddingów)
# ---------------------------------------------------------------------------
def print_neighbors(X, items, i, k=3):
    sims = X @ X[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    print(f"\nNajbliżsi sąsiedzi wypowiedzi #{i}: \"{items[i]['text'][:90]}\"")
    for j in order:
        print(f"  {sims[j]:.2f}  #{j} ({items[j]['speaker']}) {items[j]['text'][:90]}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--min-words", type=int, default=0,
                    help="pomiń wypowiedzi krótsze niż N słów (np. 'Okay.', 'Mm-hmm.')")
    ap.add_argument("--color", choices=["speaker", "time"], default="speaker")
    ap.add_argument("--perplexity", type=int, default=10, help="t-SNE")
    ap.add_argument("--n-neighbors", type=int, default=8, help="UMAP")
    ap.add_argument("--nn", type=int, default=None, help="pokaż najbliższych sąsiadów wypowiedzi o tym numerze")
    ap.add_argument("--out", default="dialog_embeddings.html")
    args = ap.parse_args()

    title, items = load_utterances(args.json_path, args.min_words)
    if len(items) < 5:
        sys.exit("Za mało wypowiedzi do wizualizacji.")
    print(f"Wczytano {len(items)} wypowiedzi.")

    try:
        X = embed([u["text"] for u in items])  # bez prefiksu instrukcji: to dokumenty, nie zapytania
    except requests.RequestException as e:
        sys.exit(f"Błąd połączenia z Ollamą ({OLLAMA_URL}): {e}")
    print(f"Embeddingi: {X.shape[0]} x {X.shape[1]}")

    print("Liczenie t-SNE...")
    tsne_xy = run_tsne(X, args.perplexity)
    print("Liczenie UMAP...")
    umap_xy = run_umap(X, args.n_neighbors)

    fig = make_figure(title, items, tsne_xy, umap_xy, args.color)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"Zapisano: {args.out}")

    if args.nn is not None and 0 <= args.nn < len(items):
        print_neighbors(X, items, args.nn)


if __name__ == "__main__":
    main()