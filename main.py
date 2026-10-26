"""Główny punkt wejścia: embeduj plik i od razu przepytaj.

Przykłady:
  # okno 4 tur, hybrid
  python main.py --file data/sessions/S001.json --chunking turns --n-turns 4 --mode hybrid \
                 --query "problemy ze snem"

  # stałe 256 tokenów z overlapem 64, dense, kilka pytań, odpowiedź LLM
  python main.py --file data/sessions/ --chunking fixed --size 256 --overlap 64 --mode dense \
                 --query "relacja z matką" --query "praca domowa" --ask

  # pytania z pliku (jedno w linii), bez ponownego embedowania
  python main.py --chunking turns --n-turns 8 --mode bm25 --questions pytania.txt --skip-ingest

  # gotowa strategia z config.py
  python main.py --file data/sessions/ --chunking fixed512_ov128 --query "lęk"
"""
import argparse
import textwrap

from thelpy_rag.pipeline import run


def parse_args():
    ap = argparse.ArgumentParser(description="Thelpy RAG: ingest + wyszukiwanie")
    g = ap.add_argument_group("dane")
    g.add_argument("--file", nargs="+", help="plik(i) .json sesji lub katalog")
    g.add_argument("--skip-ingest", action="store_true", help="nie embeduj, tylko pytaj")
    g.add_argument("--recreate", action="store_true", help="wyczyść kolekcję przed wgraniem")

    g = ap.add_argument_group("chunking")
    g.add_argument("--chunking", default="turns",
                   help="'fixed', 'turns' albo nazwa z config.STRATEGIES (np. turns4, fixed256_ov64)")
    g.add_argument("--size", type=int, help="fixed: rozmiar chunka w tokenach")
    g.add_argument("--overlap", type=int, default=0, help="fixed: overlap w tokenach")
    g.add_argument("--n-turns", type=int, default=4, help="turns: liczba tur w chunku")
    g.add_argument("--overlap-turns", type=int, default=0, help="turns: wspólne tury między oknami")

    g = ap.add_argument_group("wyszukiwanie")
    g.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    g.add_argument("--fusion", default="client", choices=["client", "server"], help="hybrid: gdzie liczyć RRF")
    g.add_argument("-k", type=int, default=5)
    g.add_argument("--query", action="append", default=[], help="pytanie (można powtarzać)")
    g.add_argument("--questions", help="plik tekstowy z pytaniami, jedno w linii")
    g.add_argument("--session", help="filtr: session_id")
    g.add_argument("--client", help="filtr: client_id")
    g.add_argument("--neighbors", type=int, default=0, help="dołącz N sąsiednich chunków")
    g.add_argument("--ask", action="store_true", help="wygeneruj odpowiedź qwen3:8b")
    return ap.parse_args()


def main():
    a = parse_args()
    questions = list(a.query)
    if a.questions:
        with open(a.questions, encoding="utf-8") as f:
            questions += [l.strip() for l in f if l.strip()]
    if not questions:
        raise SystemExit("Podaj --query albo --questions")
    if not a.file and not a.skip_ingest:
        raise SystemExit("Podaj --file albo użyj --skip-ingest")

    results = run(a.file, questions, chunking=a.chunking, mode=a.mode, k=a.k,
                  size=a.size, overlap=a.overlap, n_turns=a.n_turns, overlap_turns=a.overlap_turns,
                  recreate=a.recreate, skip_ingest=a.skip_ingest, session_id=a.session,
                  client_id=a.client, neighbors=a.neighbors, fusion=a.fusion, ask_llm=a.ask)

    for res in results:
        print(f"\n=== {res['question']}   [{res['strategy']} | {res['mode']}]")
        for i, h in enumerate(res["hits"], 1):
            p = h.payload
            ranks = f"  rangi={h.ranks}" if h.ranks else ""
            print(f"{i}. {h.score:.4f}  {p['session_id']}  {p.get('turn_range')}{ranks}")
            print(textwrap.indent(textwrap.shorten(p["text"], 300), "   "))
        if "answer" in res:
            print(f"\n--- odpowiedź ---\n{res['answer']}")


if __name__ == "__main__":
    main()
