"""Benchmark jednej sesji: pytania z pliku benchmarku -> RAG -> porównanie w Excelu.

Sesja musi być już wgrana do kolekcji (np. przez scripts.ask --file albo scripts.ingest). Dla każdego
pytania z sekcji "questions" skrypt wyszukuje fragmenty, generuje odpowiedź LLM
i sprawdza, które elementy "evidence" (znaczniki hh:mm:ss lub zakresy a–b) są
pokryte przez pobrane chunki. Ocenę merytoryczną (required_info / forbidden_additions)
wpisuje się ręcznie w puste kolumny arkusza.

    python -m scripts.benchmark_session data/benchamrk_064.json --session 64
    python -m scripts.benchmark_session data/benchamrk_064.json --session 64 -m hybrid -k 8 --context-order chrono
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from thelpy_rag import config
from thelpy_rag.llm import generate
from thelpy_rag.loader import to_seconds
from thelpy_rag.search import Retriever, build_context, build_filter
from thelpy_rag.store import get_client


# --- evidence ------------------------------------------------------------------
def parse_evidence(ev: str) -> tuple[float, float]:
    """'00:21:46' -> (1306, 1306); '00:21:46–00:22:57' -> (1306, 1377)."""
    parts = [p.strip() for p in re.split(r"[–—]|\s-\s|(?<=\d)-(?=\d)", ev) if p.strip()]
    a = to_seconds(parts[0])
    b = to_seconds(parts[-1])
    return a, b


def covers(payload: dict, ev: tuple[float, float]) -> bool:
    """Czy chunk zachodzi na evidence; przedział chunka półotwarty, bo jego end = start następnego."""
    start, end = payload.get("start_seconds"), payload.get("end_seconds")
    if start is None or end is None:
        return False
    a, b = ev
    if a == b:
        return start <= a < end or start == end == a
    return start <= b and end > a


def fmt_s(sec) -> str:
    return "" if sec is None else f"{sec:.0f}"


# --- Excel ---------------------------------------------------------------------
WIDTHS = {"question": 40, "answer": 70, "sources": 32, "evidence": 22, "evidence_found": 26,
          "required_info": 50, "forbidden_additions": 40, "text": 90, "notes": 30}


def format_sheet(ws, df: pd.DataFrame) -> None:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for i, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(i)].width = WIDTHS.get(col, max(10, len(str(col)) + 2))
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def run_benchmark(questions: list[dict], session: str, strategy: str = "turns8", mode: str = "dense",
                  k: int = 5, context_order: str = "score", model: str = config.LLM_MODEL,
                  retriever: Retriever | None = None, flt=None, formatter=None,
                  log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pytania -> RAG -> (wyniki, fragmenty, podsumowanie). flt nadpisuje filtr sesji (np. z tematem)."""
    collection = config.collection_name(strategy)
    r = retriever or Retriever(get_client())
    flt = flt if flt is not None else build_filter(session_id=session)
    results, fragments = [], []

    for n, q in enumerate(questions, 1):
        log and log(f"{q['id']} [{n}/{len(questions)}] {q['question']}")
        hits = r.search(collection, q["question"], mode, k, flt)
        if not hits:
            raise LookupError(f"Brak wyników: sesja {session} nie jest w kolekcji {collection}?")

        contexts = build_context(hits, order=context_order)
        res = generate(q["question"], contexts, model=model, formatter=formatter)

        evidence = q.get("evidence") or []
        ev_ranges = [parse_evidence(e) for e in evidence]
        ev_found = [any(covers(h.payload, ev) for h in hits) for ev in ev_ranges]

        for rank, h in enumerate(hits, 1):
            p = h.payload
            fragments.append({
                "id": q["id"], "rank": rank, "score": round(h.score, 4),
                "chunk_index": p["chunk_index"], "turn_range": p.get("turn_range"),
                "start_s": p.get("start_seconds"), "end_s": p.get("end_seconds"),
                "covers_evidence": ", ".join(e for e, ev in zip(evidence, ev_ranges) if covers(p, ev)),
                "text": p["text"],
            })

        results.append({
            "id": q["id"], "type": q.get("type"), "question": q["question"], "answer": res["answer"],
            "sources": "\n".join(f"{h.payload.get('turn_range')} ({h.score:.2f}) "
                                 f"[{fmt_s(h.payload.get('start_seconds'))}–{fmt_s(h.payload.get('end_seconds'))} s]"
                                 for h in hits),
            "evidence": "\n".join(evidence),
            "evidence_found": "\n".join(f"{e} {'✓' if ok else '✗'}" for e, ok in zip(evidence, ev_found)),
            "evidence_hit": (float(any(ev_found)) if evidence else None),
            "evidence_recall": (round(sum(ev_found) / len(evidence), 3) if evidence else None),
            "required_info": "\n".join(f"• {x}" for x in q.get("required_info", [])),
            "min_required_for_pass": q.get("min_required_for_pass"),
            "forbidden_additions": "\n".join(f"• {x}" for x in q.get("forbidden_additions", [])),
            "prompt_tokens": res["prompt_tokens"], "answer_tokens": res["answer_tokens"],
            "required_covered": None, "forbidden_present": None, "pass": None, "notes": None,
        })

    df = pd.DataFrame(results)
    frag = pd.DataFrame(fragments)
    by_type = df.groupby("type", sort=False, dropna=False).agg(
        n=("id", "count"), evidence_hit=("evidence_hit", "mean"), evidence_recall=("evidence_recall", "mean"),
        prompt_tokens=("prompt_tokens", "sum"), answer_tokens=("answer_tokens", "sum")).reset_index()
    total = pd.DataFrame([{"type": "ALL", "n": len(df), "evidence_hit": df["evidence_hit"].mean(),
                           "evidence_recall": df["evidence_recall"].mean(),
                           "prompt_tokens": df["prompt_tokens"].sum(), "answer_tokens": df["answer_tokens"].sum()}])
    summary = pd.concat([by_type, total], ignore_index=True).round(3)
    return df, frag, summary


def config_frame(**params) -> pd.DataFrame:
    rows = list(params.items()) + [("embed_model", config.EMBED_MODEL),
                                   ("run_at", datetime.now().isoformat(timespec="seconds"))]
    return pd.DataFrame(rows, columns=["parameter", "value"])


def write_excel(out: Path, df: pd.DataFrame, frag: pd.DataFrame, cfg: pd.DataFrame, summary: pd.DataFrame) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, d in [("wyniki", df), ("fragmenty", frag), ("konfiguracja", cfg), ("podsumowanie", summary)]:
            d.to_excel(xw, sheet_name=name, index=False)
            format_sheet(xw.sheets[name], d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", help="plik JSON benchmarku (sekcja 'questions')")
    ap.add_argument("--session", required=True, help="session_id w Qdrancie, np. 64")
    ap.add_argument("-s", "--strategy", default="turns8", choices=list(config.STRATEGIES))
    ap.add_argument("-m", "--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--context-order", default="score", choices=["score", "chrono"],
                    help="kolejność fragmentów w prompcie: od najwyższego score albo po chunk_index")
    ap.add_argument("--model", default=config.LLM_MODEL)
    ap.add_argument("--out")
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    collection = config.collection_name(args.strategy)
    out = Path(args.out or f"benchmark_{args.session}_{args.strategy}_{args.mode}_k{args.k}_{args.context_order}.xlsx")

    try:
        df, frag, summary = run_benchmark(questions, args.session, args.strategy, args.mode, args.k,
                                          args.context_order, args.model,
                                          log=lambda m: print(m, flush=True))
    except LookupError as e:
        raise SystemExit(str(e))
    cfg = config_frame(benchmark=args.benchmark, session_id=args.session, collection=collection,
                       mode=args.mode, k=args.k, context_order=args.context_order, llm_model=args.model)
    write_excel(out, df, frag, cfg, summary)
    print(f"\n{summary.to_string(index=False)}\n\nZapisano {out}")


if __name__ == "__main__":
    main()
