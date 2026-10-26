"""Wyszukiwanie z linii poleceń.

    python -m scripts.search "jak klient opisuje relację z matką" -s turns4 -m hybrid
    python -m scripts.search "bezsenność" -s fixed256 -m bm25 --session S001
    python -m scripts.search "lęk w pracy" -m dense --client C017 --neighbors 1
"""
import argparse
import textwrap

from thelpy_rag import config
from thelpy_rag.search import Retriever, build_filter
from thelpy_rag.store import get_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-s", "--strategy", default="turns4", choices=list(config.STRATEGIES))
    ap.add_argument("-m", "--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--fusion", default="client", choices=["client", "server"],
                    help="hybrid: RRF liczone lokalnie (k=RRF_K) albo w Qdrancie")
    ap.add_argument("--session"), ap.add_argument("--client"), ap.add_argument("--role")
    ap.add_argument("--from", dest="date_from"), ap.add_argument("--to", dest="date_to")
    ap.add_argument("--neighbors", type=int, default=0, help="dołącz N sąsiednich chunków z każdej strony")
    args = ap.parse_args()

    r = Retriever(get_client())
    collection = config.collection_name(args.strategy)
    flt = build_filter(args.session, args.client, args.role, args.date_from, args.date_to)
    kw = {"fusion": args.fusion} if args.mode == "hybrid" else {}
    hits = r.search(collection, args.query, args.mode, args.k, flt, **kw)

    print(f"{collection} | {args.mode} | {len(hits)} wyników\n")
    for i, h in enumerate(hits, 1):
        p = h.payload
        ranks = f"  rangi={h.ranks}" if h.ranks else ""
        print(f"{i}. score={h.score:.4f}  {p['session_id']}  {p.get('turn_range')}  "
              f"[{p.get('start_seconds')}–{p.get('end_seconds')} s]  {p.get('session_at')}{ranks}")
        if args.neighbors:
            for n in r.neighbors(collection, h, args.neighbors):
                mark = ">>" if n["chunk_index"] == p["chunk_index"] else "  "
                print(textwrap.indent(n["text"], f"   {mark} "))
        else:
            print(textwrap.indent(textwrap.shorten(p["text"], 400), "   "))
        print()


if __name__ == "__main__":
    main()
