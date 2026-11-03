"""Pytanie do jednej sesji: wgranie JSON-a, wyszukiwanie dense, odpowiedź LLM.

Pokazuje znalezione fragmenty ze scorami embeddingów (cosine), pełny prompt,
odpowiedź modelu oraz liczbę tokenów pytania i odpowiedzi.

    python -m scripts.ask_session data/annomi_64.json "Why did Ricky come to therapy?"
    python -m scripts.ask_session data/annomi_64.json "What drugs does he use?" --skip-ingest

Z tematami (thelpy_topics): --topics wgrywa sesję z tematami i pokazuje je LLM-owi,
--topic szuka tylko w chunkach z tym tematem albo podtematem (klucz z thelpy_topics/taxonomy.py).

    python -m scripts.ask_session data/annomi_64.json "What does he say about his friends?" --skip-ingest --topic friends
"""
import argparse
import textwrap

from thelpy_rag import config
from thelpy_rag.chunking import get_tokenizer
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.llm import generate
from thelpy_rag.loader import load_session
from thelpy_rag.pipeline import ingest, resolve_strategy
from thelpy_rag.search import Retriever, build_filter
from thelpy_rag.store import get_client


def header(title: str) -> None:
    print(f"\n{'=' * 30} {title} {'=' * 30}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="plik .json sesji")
    ap.add_argument("question")
    ap.add_argument("--n-turns", type=int, default=8, help="liczba tur w chunku")
    ap.add_argument("--overlap-turns", type=int, default=0)
    ap.add_argument("-k", type=int, default=5, help="liczba fragmentów przekazanych do LLM")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--skip-ingest", action="store_true", help="sesja jest już w kolekcji")
    ap.add_argument("--topics", action="store_true", help="wgraj sesję z tematami i pokaż tematy LLM-owi")
    ap.add_argument("--topic", help="szukaj tylko w chunkach z tym tematem/podtematem, np. social, friends "
                                    "(kolekcja musi być wgrana z --topics; włącza --topics)")
    args = ap.parse_args()

    topics = args.topics or bool(args.topic)
    if topics:
        from thelpy_topics import integration
        from thelpy_topics.annotate import annotate_files

        integration.enable_topics()

    session = load_session(args.file)
    strategy = resolve_strategy("turns", n_turns=args.n_turns, overlap_turns=args.overlap_turns)
    collection = config.collection_name(strategy)
    client, embedder = get_client(), OllamaEmbedder()

    if not args.skip_ingest:
        files = [str(path) for path in annotate_files([args.file])] if topics else [args.file]
        ingest(files, strategy, client=client, embedder=embedder)
        if topics:
            integration.ensure_indexes(client, collection)

    # --- retrieval (dense, tylko ta sesja) ---
    r = Retriever(client, embedder)
    flt = build_filter(session_id=session.session_id)
    if args.topic:
        flt = integration.with_topic(flt, args.topic)
    hits = r.search(collection, args.question, "dense", args.k, flt)

    header(f"RETRIEVAL  dense | {collection} | top {args.k}" + (f" | topic={args.topic}" if args.topic else ""))
    if not hits and args.topic:
        print(f"\nBrak chunków z tematem '{args.topic}' (kolekcja wgrana z --topics? klucz z taxonomy.py?)")
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"\n{i}. cosine={h.score:.4f}  chunk={p['chunk_index']}  {p.get('turn_range')}  "
              f"[{p.get('start_seconds')}–{p.get('end_seconds')} s]  tokeny={p.get('n_tokens')}")
        if topics:
            print(f"   ({integration.topic_note(p) or 'brak tematu'})")
        print(textwrap.indent(p["text"], "   "))

    # --- LLM ---
    contexts = sorted((h.payload for h in hits), key=lambda p: p["chunk_index"])
    res = generate(args.question, contexts, model=args.model)

    header("PROMPT")
    for m in res["messages"]:
        print(f"\n[{m['role']}]\n{m['content']}")

    header(f"ODPOWIEDŹ  {res['model']}")
    print(res["answer"])

    header("TOKENY")
    print(f"pytanie:            {get_tokenizer().count(args.question)}  (tokenizer {config.TOKENIZER_NAME})")
    print(f"cały prompt:        {res['prompt_tokens']}  (Ollama, prompt_eval_count)")
    print(f"odpowiedź:          {res['answer_tokens']}  (Ollama, eval_count)")


if __name__ == "__main__":
    main()
