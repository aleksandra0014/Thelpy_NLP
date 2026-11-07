"""Porównanie strategii chunkingu i trybów wyszukiwania.

Plik z pytaniami (JSONL), jedna linia = jedno pytanie:
  {"qid": "q1", "query": "...", "session_id": "S001", "relevant_turn_ids": ["T063", "T064"]}

Złoty standard jest zapisany na poziomie TUR, a nie chunków, dlatego ten sam zbiór pytań
działa dla każdej strategii. Chunk jest trafny, jeśli zawiera choć jedną trafną turę.

Metryki dla top-k:
  hit@k       — czy którykolwiek chunk zawiera trafną turę
  recall@k    — jaka część trafnych tur jest pokryta przez top-k
  mrr         — 1 / pozycja pierwszego trafnego chunka
  ctx_tokens  — ile tokenów kontekstu trzeba przekazać (duże chunki "kupują" recall tokenami!)

    python -m scripts.evaluate data/eval/queries.jsonl -k 5
    python -m scripts.evaluate data/eval/queries.jsonl -s turns4 fixed256 -m hybrid --scope session
"""
import argparse
import csv
import json
import statistics

from thelpy_rag import config
from thelpy_rag.search import Retriever, build_filter
from thelpy_rag.store import get_client


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(hits, q, k):
    gold = set(q["relevant_turn_ids"])
    covered, first = set(), None
    for rank, h in enumerate(hits[:k], 1):
        if h.payload["session_id"] != q["session_id"]:
            continue
        found = gold & set(h.payload["turn_ids"])
        if found and first is None:
            first = rank
        covered |= found
    return {
        "hit": float(first is not None),
        "recall": len(covered) / len(gold) if gold else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "ctx_tokens": sum(h.payload.get("n_tokens", 0) for h in hits[:k]),
    }


def evaluate(queries: list[dict], strategies: list[str], modes: list[str], k: int = 5,
             scope: str = "all", retriever: Retriever | None = None, log=print) -> list[dict]:
    """Średnie metryki dla każdej pary (strategia, tryb)."""
    r = retriever or Retriever(get_client())
    rows = []
    for s in strategies:
        coll = config.collection_name(s)
        for mode in modes:
            per_q = []
            for q in queries:
                flt = None
                if scope == "session":
                    flt = build_filter(session_id=q["session_id"])
                elif scope == "client" and q.get("client_id"):
                    flt = build_filter(client_id=q["client_id"])
                per_q.append(score(r.search(coll, q["query"], mode, k, flt), q, k))
            row = {"strategy": s, "mode": mode, "k": k, "n": len(queries)}
            for m in ("hit", "recall", "mrr", "ctx_tokens"):
                row[m] = round(statistics.mean(x[m] for x in per_q), 4) if per_q else 0.0
            rows.append(row)
            log and log(f"{s:16s} {mode:7s} hit@{k}={row['hit']:.3f}  recall@{k}={row['recall']:.3f}  "
                        f"mrr={row['mrr']:.3f}  ctx_tokens={row['ctx_tokens']:.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("-s", "--strategies", nargs="+", default=list(config.STRATEGIES),
                    choices=list(config.STRATEGIES))
    ap.add_argument("-m", "--modes", nargs="+", default=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--scope", choices=["all", "session", "client"], default="all",
                    help="filtr podczas wyszukiwania: cała baza / tylko sesja pytania / tylko klient")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    rows = evaluate(load_queries(args.queries), args.strategies, args.modes, args.k, args.scope)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nZapisano {args.out}")


if __name__ == "__main__":
    main()
