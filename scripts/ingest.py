"""Wgrywa sesje do kolekcji (każda sesja trafia do każdej wybranej strategii).

    python -m scripts.ingest data/sessions/
    python -m scripts.ingest data/sessions/S001.json -s turns4 fixed256
    python -m scripts.ingest data/sessions/ --dry-run      # tylko statystyki chunków, bez zapisu
    python -m scripts.ingest data/annomi_64.json -s turns8 --topics   # z tematami w payloadzie (thelpy_topics)

Ponowne wgranie tej samej sesji najpierw usuwa jej stare chunki, więc poprawiona
transkrypcja (nowe `source_version`) zastępuje poprzednią zamiast się dublować.
"""
import argparse
import statistics

from thelpy_rag import config
from thelpy_rag.chunking import chunk_session, get_tokenizer
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.loader import iter_session_files, load_session
from thelpy_rag.sparse import BM25Encoder
from thelpy_rag.store import create_collection, delete_session, get_client, upsert_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="pliki .json lub katalogi")
    ap.add_argument("-s", "--strategies", nargs="+", default=list(config.STRATEGIES),
                    choices=list(config.STRATEGIES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="przy --dry-run: wypisz N pierwszych chunków")
    ap.add_argument("--topics", action="store_true",
                    help="dodaj tematy do chunków (sesje są najpierw anotowane do data/with_topics/)")
    args = ap.parse_args()

    files = iter_session_files(args.paths)
    if not files:
        raise SystemExit("Nie znaleziono plików .json")
    load, chunk = load_session, chunk_session
    if args.topics:
        from thelpy_topics import integration
        from thelpy_topics.annotate import annotate_files

        files = annotate_files(files)
        integration.add_payload_indexes()
        load, chunk = integration.load_session_with_topics, integration.chunk_session_with_topics
    if not get_tokenizer().exact:
        print("UWAGA: brak tokenizera modelu — rozmiary w tokenach są przybliżone.")

    embedder, bm25 = OllamaEmbedder(), BM25Encoder()
    client = None if args.dry_run else get_client()
    dim = None

    for path in files:
        session = load(path)
        print(f"\n{session.session_id} ({len(session.turns)} tur, wersja {session.source_version})")
        for s in args.strategies:
            chunks = chunk(session, s)
            sizes = [c.n_tokens for c in chunks]
            print(f"  {s:16s} chunków: {len(chunks):4d}   tokeny: śr. {statistics.mean(sizes):.0f}, "
                  f"min {min(sizes)}, max {max(sizes)}")
            if args.dry_run:
                for c in chunks[: args.show]:
                    print(f"    [{c.chunk_index}] {c.turn_ids[0]}–{c.turn_ids[-1]} | {c.text[:120]!r}")
                continue

            name = config.collection_name(s)
            if not client.collection_exists(name):
                dim = dim or embedder.dimension()
                create_collection(client, name, dim)
                print(f"    utworzono kolekcję {name}")
            elif args.topics:
                integration.ensure_indexes(client, name)  # kolekcja sprzed tematów: dołóż indeksy

            texts = [c.text for c in chunks]
            dense = embedder.embed_documents(texts)
            sparse = [bm25.encode_document(t) for t in texts]
            delete_session(client, name, session.session_id)
            n = upsert_chunks(client, name, session, chunks, dense, sparse)
            print(f"    zapisano {n} punktów do {name}")


if __name__ == "__main__":
    main()
