#!/usr/bin/env python3
"""
t-SNE embeddingów tekstu z CSV (wypowiedzi + cechy akustyczne), kolor = wybrany wskaźnik.

Trzy panele:
  1) terapeuta  (osobny t-SNE tylko z jego wypowiedzi)
  2) pacjent    (osobny t-SNE tylko z jego wypowiedzi)
  3) wspólny    (jeden t-SNE dla wszystkich, kształt = mówiący)

Pozycja punktu zależy WYŁĄCZNIE od znaczenia tekstu (embedding), a kolor od wskaźnika
akustycznego. Lista rozwijana nad wykresem przełącza wskaźnik we wszystkich panelach.

Wymagania:
    pip install numpy pandas requests scikit-learn plotly
    ollama pull qwen3-embedding:0.6b-q8_0

Użycie:
    python visualize_csv_tsne.py dane.csv
    python visualize_csv_tsne.py dane.csv --features arousal f0_mean silence_ratio
    python visualize_csv_tsne.py dane.csv --offline      # plotly.js wbudowany w HTML
"""

import argparse
import html
import sys
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3-embedding:0.6b-q8_0"

DEFAULT_FEATURES = [
    "arousal", "f0_mean", "loudness_mean", "words_per_min", "silence_ratio",
    "jitter", "shimmer", "hnr", "alpha_ratio",
]
DIVERGING = {"arousal"}  # cecha ze znakiem: skala niebieski - biały - czerwony, symetryczna wokół 0
SEQ_SCALE = "Viridis"
DIV_SCALE = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]


# ---------------------------------------------------------------------------
# Embeddingi (dense, Ollama)
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
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def tsne_2d(X, perplexity):
    n = len(X)
    if n < 4:
        return np.zeros((n, 2))
    perp = max(2, min(perplexity, (n - 1) // 3))  # perplexity musi być mniejsze niż liczba próbek
    return TSNE(
        n_components=2, perplexity=perp, metric="cosine",
        init="pca", learning_rate="auto", random_state=42,
    ).fit_transform(X)


# ---------------------------------------------------------------------------
# Kolory
# ---------------------------------------------------------------------------
def color_range(values, feature, clip_pct):
    """Zakres koloru z obcięciem skrajnych wartości (żeby 1-2 outliery nie spłaszczyły skali)."""
    lo, hi = np.percentile(values, [clip_pct, 100 - clip_pct])
    if feature in DIVERGING:
        m = max(abs(lo), abs(hi)) or 1.0
        return -m, m, DIV_SCALE
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi), SEQ_SCALE


# ---------------------------------------------------------------------------
# Wykres
# ---------------------------------------------------------------------------
def hover_text(i, row, features):
    wrapped = "<br>".join(textwrap.wrap(html.escape(str(row["text"])), 60))
    vals = " | ".join(f"{f}: {row[f]:.3g}" for f in features)
    return f"<b>#{i}</b> [{row['time_label']}] {row['speaker']}<br>{wrapped}<br><i>{vals}</i>"


def build_figure(df, X_coords, features, args, labels):
    therapist, patient = labels
    idx_t = np.where(df["speaker"] == therapist)[0]
    idx_p = np.where(df["speaker"] == patient)[0]

    # specyfikacja trace'ów: (wiersze, współrzędne, kolumna panelu, mówiący, pokazać w legendzie)
    specs = [
        (idx_t, X_coords["t"], 1, therapist, False),
        (idx_p, X_coords["p"], 2, patient, False),
        (idx_t, X_coords["all"][idx_t], 3, therapist, True),
        (idx_p, X_coords["all"][idx_p], 3, patient, True),
    ]

    first = features[0]
    lo, hi, scale = color_range(df[first].values, first, args.clip_pct)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f"Terapeuta ({therapist})", f"Pacjent ({patient})", "Wspólny"),
    )
    for rows, xy, col, sp, legend in specs:
        symbol = "circle" if sp == therapist else "diamond"
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0], y=xy[:, 1], mode="markers+text",
                text=[str(i) for i in rows], textposition="top center",
                textfont={"size": 9, "color": "#555"},
                marker={
                    "size": 13, "symbol": symbol, "coloraxis": "coloraxis",
                    "color": df[first].values[rows],
                    "line": {"width": 0.8, "color": "#333"},
                },
                name=f"{sp} ({'kółko' if sp == therapist else 'romb'})",
                legendgroup=sp, showlegend=legend,
                hovertext=[hover_text(i, df.iloc[i], features) for i in rows],
                hoverinfo="text",
            ),
            row=1, col=col,
        )

    # przyciski: zmiana koloru wszystkich czterech trace'ów + zakres/skala wspólnego colorbara
    buttons = []
    for f in features:
        flo, fhi, fscale = color_range(df[f].values, f, args.clip_pct)
        buttons.append({
            "label": f,
            "method": "update",
            "args": [
                {"marker.color": [df[f].values[rows].tolist() for rows, *_ in specs]},
                {
                    "coloraxis.cmin": flo, "coloraxis.cmax": fhi,
                    "coloraxis.colorscale": fscale,
                    "coloraxis.colorbar.title.text": f,
                },
            ],
        })

    fig.update_layout(
        title=(f"Embeddingi tekstu ({MODEL}) + t-SNE, kolor = wskaźnik akustyczny"
               f"<br><sup>{len(df)} wypowiedzi; zakres koloru obcięty do percentyli "
               f"{args.clip_pct}-{100 - args.clip_pct}</sup>"),
        template="plotly_white", height=680, hoverlabel={"align": "left"},
        coloraxis={
            "cmin": lo, "cmax": hi, "colorscale": scale,
            "colorbar": {"title": {"text": first}},
        },
        updatemenus=[{
            "type": "dropdown", "buttons": buttons, "x": 0.0, "xanchor": "left",
            "y": 1.18, "yanchor": "top", "showactive": True,
        }],
        annotations=list(fig.layout.annotations) + [{
            "text": "Kolor =", "x": -0.005, "xref": "paper", "y": 1.155, "yref": "paper",
            "xanchor": "right", "showarrow": False,
        }],
        legend={"orientation": "h", "y": -0.05},
    )
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                    help="kolumny CSV do kolorowania (pierwsza jest startowa)")
    ap.add_argument("--perplexity", type=int, default=10, help="t-SNE (skrypt sam przytnie przy małej liczbie punktów)")
    ap.add_argument("--clip-pct", type=float, default=2.0, help="obcięcie skrajnych wartości na skali koloru")
    ap.add_argument("--offline", action="store_true", help="wbuduj plotly.js w HTML (większy plik, działa bez internetu)")
    ap.add_argument("--out", default="csv_embeddings_tsne.html")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    for col in ("speaker", "text"):
        if col not in df.columns:
            sys.exit(f"Brak kolumny '{col}' w CSV.")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()

    features = [f for f in args.features if f in df.columns]
    missing = [f for f in args.features if f not in df.columns]
    if missing:
        print(f"Uwaga: pomijam brakujące kolumny: {missing}")
    if not features:
        sys.exit("Brak poprawnych kolumn ze wskaźnikami.")
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        df[f] = df[f].fillna(df[f].median())

    time_col = "time" if "time" in df.columns else ("start" if "start" in df.columns else None)
    df["time_label"] = df[time_col].round(1).astype(str) + "s" if time_col else ""

    speakers = list(df["speaker"].unique())
    therapist = "therapist" if "therapist" in speakers else speakers[0]
    others = [s for s in speakers if s != therapist]
    if not others:
        sys.exit("W CSV jest tylko jeden mówiący.")
    patient = others[0]  # w tych danych: 'client'

    print(f"Wczytano {len(df)} wypowiedzi: " +
          ", ".join(f"{s}={int((df.speaker == s).sum())}" for s in (therapist, patient)))
    print("\nŚrednie wskaźników wg mówiącego:")
    print(df.groupby("speaker")[features].mean().round(3).T.to_string())

    try:
        X = embed(df["text"].tolist())
    except requests.RequestException as e:
        sys.exit(f"Błąd połączenia z Ollamą ({OLLAMA_URL}): {e}")
    print(f"\nEmbeddingi: {X.shape[0]} x {X.shape[1]}")

    idx_t = np.where(df["speaker"] == therapist)[0]
    idx_p = np.where(df["speaker"] == patient)[0]
    print("Liczenie t-SNE (3 razy: terapeuta, pacjent, wspólny)...")
    coords = {
        "t": tsne_2d(X[idx_t], args.perplexity),
        "p": tsne_2d(X[idx_p], args.perplexity),
        "all": tsne_2d(X, args.perplexity),
    }

    fig = build_figure(df, coords, features, args, (therapist, patient))
    fig.write_html(args.out, include_plotlyjs=True if args.offline else "cdn")
    print(f"Zapisano: {args.out}")


if __name__ == "__main__":
    main()