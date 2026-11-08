"""Podsumowanie jednej sesji: wgranie JSON-a, wszystkie chunki po kolei (bez wyszukiwania), map-reduce LLM.

Jeśli cała sesja mieści się w oknie kontekstu — jedno wywołanie (stuff); jeśli nie — streszczenia
bloków sąsiednich chunków (map) łączone hierarchicznie (reduce).

    python -m scripts.summarize_session data/annomi_64.json
    python -m scripts.summarize_session data/annomi_64.json --skip-ingest --focus "client's motivation to change"
    python -m scripts.summarize_session data/annomi_64.json --skip-ingest --max-block-tokens 800 --show-steps

Z tematami (thelpy_topics): --topics wgrywa sesję z tematami i pokazuje je LLM-owi,
--topic podsumowuje tylko chunki z tym tematem albo podtematem (klucz z thelpy_topics/taxonomy.py).

    python -m scripts.summarize_session data/annomi_64.json --skip-ingest --topic alcohol
"""
import argparse
import textwrap

from thelpy_rag import config
from thelpy_rag.embeddings import OllamaEmbedder
from thelpy_rag.loader import load_session
from thelpy_rag.pipeline import ingest, resolve_strategy
from thelpy_rag.search import build_filter
from thelpy_rag.store import get_client
from thelpy_rag.summarize import fetch_chunks, summarize


def header(title: str) -> None:
    print(f"\n{'=' * 30} {title} {'=' * 30}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="plik .json sesji")
    ap.add_argument("--n-turns", type=int, default=8, help="liczba tur w chunku")
    ap.add_argument("--overlap-turns", type=int, default=0)
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--skip-ingest", action="store_true", help="sesja jest już w kolekcji")
    ap.add_argument("--length", choices=["full", "brief"], default="full",
                    help="full: Markdown w sekcjach (~600 słów); brief: 3-5 zdań")
    ap.add_argument("--focus", help="na czym skupić podsumowanie")
    ap.add_argument("--max-block-tokens", type=int, help="zmniejsza blok map (wymusza map-reduce)")
    ap.add_argument("--show-steps", action="store_true", help="pokaż streszczenia cząstkowe")
    ap.add_argument("--topics", action="store_true", help="wgraj sesję z tematami i pokaż tematy LLM-owi")
    ap.add_argument("--topic", help="podsumuj tylko chunki z tym tematem/podtematem (włącza --topics)")
    args = ap.parse_args()

    topics = args.topics or bool(args.topic)
    formatter = None
    if topics:
        from thelpy_topics import integration
        from thelpy_topics.annotate import annotate_files

        integration.enable_topics()
        formatter = integration.format_context_with_topics

    session = load_session(args.file)
    strategy = resolve_strategy("turns", n_turns=args.n_turns, overlap_turns=args.overlap_turns)
    collection = config.collection_name(strategy)
    client = get_client()

    if not args.skip_ingest:
        files = [str(path) for path in annotate_files([args.file])] if topics else [args.file]
        ingest(files, strategy, client=client, embedder=OllamaEmbedder())
        if topics:
            integration.ensure_indexes(client, collection)

    # --- wszystkie chunki sesji po metadanych ---
    flt = build_filter(session_id=session.session_id)
    if args.topic:
        flt = integration.with_topic(flt, args.topic)
    payloads = fetch_chunks(client, collection, flt)

    header(f"CHUNKI  {collection} | sesja {session.session_id}" + (f" | topic={args.topic}" if args.topic else ""))
    if not payloads:
        print("\nBrak chunków" + (f" z tematem '{args.topic}'" if args.topic else "") + " — nic do podsumowania.")
        return
    for p in payloads:
        print(f"  chunk={p['chunk_index']:>3}  {p.get('turn_range')}  tokeny={p.get('n_tokens')}")

    header("PRZEBIEG")
    res = summarize(payloads, model=args.model, formatter=formatter, focus=args.focus,
                    max_block_tokens=args.max_block_tokens, length=args.length, log=lambda m: print(m, flush=True))
    for w in res["warnings"]:
        print(f"\nUWAGA: {w}")

    if args.show_steps and res["steps"]:
        header("KROKI MAP-REDUCE")
        for s in res["steps"]:
            print(f"\n[poziom {s['level']}] tury {s['turn_range']} ({s['n_inputs']} wejść)")
            print(textwrap.indent(s["summary"], "   "))

    header(f"PODSUMOWANIE  {res['model']} | {res['method']} | poziomy={res['levels']}")
    print(res["summary"])

    header("STATYSTYKI")
    print(f"chunki:             {res['n_chunks']}  (tury {res['turn_range']})")
    print(f"budżet bloku:       {res['block_budget']} tokenów")
    print(f"wywołania LLM:      {res['llm_calls']}")
    print(f"tokeny promptów:    {res['prompt_tokens']}  (Ollama, suma)")
    print(f"tokeny odpowiedzi:  {res['answer_tokens']}  (Ollama, suma)")


if __name__ == "__main__":
    main()
