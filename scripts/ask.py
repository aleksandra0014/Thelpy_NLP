"""Pełny RAG: wyszukanie fragmentów + odpowiedź modelu LLM (config.LLM_MODEL).

    python -m scripts.ask "Jakie strategie radzenia sobie ze stresem omawiano?" -s turns8 --client C017
"""
import argparse

from thelpy_rag import config
from thelpy_rag.llm import answer
from thelpy_rag.search import Retriever, build_filter
from thelpy_rag.store import get_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-s", "--strategy", default="turns4", choices=list(config.STRATEGIES))
    ap.add_argument("-m", "--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--session"), ap.add_argument("--client")
    ap.add_argument("--neighbors", type=int, default=0)
    args = ap.parse_args()

    r = Retriever(get_client())
    collection = config.collection_name(args.strategy)
    hits = r.search(collection, args.question, args.mode, args.k, build_filter(args.session, args.client))

    # kontekst: trafienia (+ ewentualnie sąsiedzi), bez duplikatów, w porządku chronologicznym
    seen, ctx = set(), []
    for h in hits:
        for p in (r.neighbors(collection, h, args.neighbors) if args.neighbors else [h.payload]):
            if p["chunk_id"] not in seen:
                seen.add(p["chunk_id"])
                ctx.append(p)
    ctx.sort(key=lambda p: (p.get("session_at") or "", p["session_id"], p["chunk_index"]))

    print(answer(args.question, ctx))
    print("\nŹródła:")
    for p in ctx:
        print(f"  {p['session_id']}  {p.get('turn_range')}  ({p.get('session_at')})")


if __name__ == "__main__":
    main()
