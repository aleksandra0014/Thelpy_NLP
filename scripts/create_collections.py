"""Tworzy kolekcje Qdranta — po jednej na strategię chunkingu.

    python -m scripts.create_collections                 # wszystkie strategie
    python -m scripts.create_collections -s turns4 turns8
    python -m scripts.create_collections --recreate      # UWAGA: kasuje dane
"""
import argparse

from thelpy_rag import config
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.store import create_collection, get_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--strategies", nargs="+", default=list(config.STRATEGIES),
                    choices=list(config.STRATEGIES))
    ap.add_argument("--recreate", action="store_true", help="usuń i utwórz od nowa")
    ap.add_argument("--dim", type=int, help="wymiar wektora (domyślnie odczytany z modelu)")
    args = ap.parse_args()

    dim = args.dim or OllamaEmbedder().dimension()
    print(f"Model {config.EMBED_MODEL}: wymiar {dim}")

    client = get_client()
    for s in args.strategies:
        name = config.collection_name(s)
        created = create_collection(client, name, dim, recreate=args.recreate)
        print(f"  {name:28s} {'utworzona' if created else 'już istnieje (pomijam)'}   {config.STRATEGIES[s]}")


if __name__ == "__main__":
    main()
