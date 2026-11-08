"""Keywordy jednej sesji bez LLM: hasła ze słownika (data/keywords.yaml) + wolne frazy (KeyBERT-style).

Wektory chunków są brane z Qdrant (te z ingestu) — sesja nie jest embedowana ponownie.
Jak powstaje wynik: thelpy_rag/KEYWORDS.md.

    python -m scripts.session_keywords data/annomi_64.json --skip-ingest
    python -m scripts.session_keywords data/annomi_121.json --skip-ingest --top-n 20 --free-top-n 15 --diversity 0.7
    python -m scripts.session_keywords data/annomi_64.json --skip-ingest --vocab moj_slownik.yaml --no-background
"""
import argparse

from thelpy_rag import config
from thelpy_rag import keywords as kw
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.loader import load_session
from thelpy_rag.pipeline import ingest, resolve_strategy
from thelpy_rag.search import build_filter
from thelpy_rag.store import get_client
from thelpy_rag.summarize import fetch_chunks


def header(title: str) -> None:
    print(f"\n{'=' * 30} {title} {'=' * 30}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="plik .json sesji")
    ap.add_argument("--n-turns", type=int, default=8, help="liczba tur w chunku")
    ap.add_argument("--overlap-turns", type=int, default=0)
    ap.add_argument("--skip-ingest", action="store_true", help="sesja jest już w kolekcji")
    ap.add_argument("--vocab", default=str(config.KEYWORDS_FILE), help="słownik .yaml")
    ap.add_argument("--top-n", type=int, default=10, help="ile haseł ze słownika")
    ap.add_argument("--free-top-n", type=int, default=10, help="ile wolnych fraz (0 = pomiń)")
    ap.add_argument("--diversity", type=float, default=0.5, help="MMR: 0 = trafność, 1 = różnorodność")
    ap.add_argument("--all-speakers", action="store_true", help="wolne frazy także z wypowiedzi terapeuty")
    ap.add_argument("--no-background", action="store_true", help="bez normalizacji na tle całej kolekcji")
    args = ap.parse_args()

    session = load_session(args.file)
    strategy = resolve_strategy("turns", n_turns=args.n_turns, overlap_turns=args.overlap_turns)
    collection = config.collection_name(strategy)
    client, embedder = get_client(), OllamaEmbedder()
    if not args.skip_ingest:
        ingest([args.file], strategy, client=client, embedder=embedder)

    payloads = fetch_chunks(client, collection, build_filter(session_id=session.session_id), with_vectors=True)
    if not payloads:
        print(f"Brak chunków sesji {session.session_id} w {collection}")
        return
    background = None if args.no_background else kw.background_vectors(client, collection)
    vocab = kw.load_vocab(args.vocab)
    print(f"Sesja {session.session_id}: {len(payloads)} chunków w {collection}; słownik {len(vocab)} haseł; "
          f"tło: {0 if background is None else len(background)} chunków "
          f"(pierwsze użycie słownika liczy embeddingi haseł ~30 s, potem cache)")

    res = kw.session_keywords(payloads, embedder, vocab=vocab, background=background, top_n=args.top_n,
                              free_top_n=args.free_top_n, diversity=args.diversity,
                              client_only=not args.all_speakers)

    header("HASŁA ZE SŁOWNIKA")
    print(f"{'hasło':28s} {'score':>6s} {'semant.':>7s} {'trafienia':>9s} {'pokrycie':>8s}  tury")
    for r in res["vocab"]:
        turns = ", ".join(r["turns"][:4]) + (" …" if len(r["turns"]) > 4 else "")
        print(f"{r['keyword']:28s} {r['score']:6.2f} {r['semantic']:7.2f} {r['lexical_hits']:9d} "
              f"{r['coverage']:8.0%}  {turns}")

    if res["free"]:
        header("WOLNE FRAZY (KeyBERT-style)")
        for r in res["free"]:
            print(f"  {r['phrase']:30s} x{r['count']:<3d} trafność {r['relevance']:+.3f}")


if __name__ == "__main__":
    main()
