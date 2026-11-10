"""Pytanie + odpowiedź LLM ze źródłami — to samo co POST /ask (ta sama logika: thelpy_api.service.ask).

Wejście sesji jak w API:
- --file + bez --skip-ingest: sesja jest wgrywana do kolekcji --strategy, jej session_id staje się filtrem,
- --skip-ingest albo bez --file: nic nie jest wgrywane, wyszukiwanie po filtrach
  (--session-id, --client-id, --role, --date-from / --date-to, --topic).

    python -m scripts.ask "Why did Ricky come to therapy?" --file data/annomi_64.json
    python -m scripts.ask "What drugs does he use?" --file data/annomi_64.json --skip-ingest --show-prompt
    python -m scripts.ask "What does he say about his friends?" --session-id 64 --topic friends
    python -m scripts.ask "How has his drug use changed?" --client-id 1 -m hybrid -k 8 --neighbors 1
"""
import argparse
import sys
import textwrap

from thelpy_rag import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")

    src = ap.add_argument_group("sesja")
    src.add_argument("--file", help="plik .json sesji (wgrywany, chyba że --skip-ingest)")
    src.add_argument("--skip-ingest", action="store_true", help="nie wgrywaj --file, tylko odczytaj z niego session_id")
    src.add_argument("-s", "--strategy", default="turns8",
                     help="kolekcja: turns4, turns8, fixed512_ov128 albo własna turnsN[_ovM] / fixedN[_ovM]")
    src.add_argument("--topics", action="store_true", help="wgraj z tematami i pokaż tematy LLM-owi")
    src.add_argument("--topic", help="tylko chunki z tym tematem/podtematem, np. friends (włącza --topics)")

    flt = ap.add_argument_group("filtry")
    flt.add_argument("--session-id", "--session", dest="session_id", help="jedna sesja")
    flt.add_argument("--client-id", "--client", dest="client_id", help="wszystkie sesje klienta")
    flt.add_argument("--role", help="chunki z wypowiedzią tej roli")
    flt.add_argument("--date-from", help="ISO, session_at >=")
    flt.add_argument("--date-to", help="ISO, session_at <=")

    ret = ap.add_argument_group("wyszukiwanie i LLM")
    ret.add_argument("-m", "--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    ret.add_argument("-k", type=int, default=5, help="liczba trafień przekazanych do LLM")
    ret.add_argument("--fusion", default="client", choices=["client", "server"], help="przy hybrid")
    ret.add_argument("--neighbors", type=int, default=0, help="dołącz N sąsiednich chunków z każdej strony")
    ret.add_argument("--model", default=config.LLM_MODEL)
    ret.add_argument("--show-prompt", action="store_true", help="pokaż pełny prompt (z tekstem fragmentów)")
    args = ap.parse_args()

    # import po parse_args: --help działa bez ładowania modeli i połączeń
    from pydantic import ValidationError

    from thelpy_api import schemas, service

    fields = {k: v for k, v in vars(args).items() if k != "show_prompt" and v is not None}
    try:
        req = schemas.AskRequest(**fields, include_prompt=args.show_prompt)
        out = service.ask(req, service.Resources.create())
    except ValidationError as e:
        sys.exit(f"Błędne argumenty: {e}")
    except (ValueError, LookupError) as e:          # w tym FileNotFoundError, PromptTooLongError
        sys.exit(f"Błąd: {e}")

    print_result(out)


def header(title: str) -> None:
    print(f"\n{'=' * 30} {title} {'=' * 30}")


def fmt_s(x) -> str:
    return "?" if x is None else f"{x:.0f}"


def print_result(out: dict) -> None:
    filters = ", ".join(f"{k}={out[k]}" for k in ("session_id", "client_id", "topic") if out.get(k))
    header(f"RETRIEVAL  {out['mode']} | {out['collection']} | top {len(out['hits'])}"
           + (f" | {filters}" if filters else "") + (" | wgrano sesję" if out["ingested"] else ""))
    for rank, h in enumerate(out["hits"], 1):
        print(f"{rank}. score={h['score']:.4f}  sesja={h['session_id']}  chunk={h['chunk_index']}  "
              f"{h['turn_range']}  [{fmt_s(h['start_seconds'])}–{fmt_s(h['end_seconds'])} s]"
              + (f"\n   ({h['topic_note']})" if h.get("topic_note") else ""))
        print(textwrap.indent(h["text"], "   ") + "\n")
    extra = [s for s in out["sources"] if s["retrieval_neighbor"]]
    if extra:
        print(f"+ {len(extra)} sąsiednich chunków jako kontekst: "
              + ", ".join(f"{s['session_id']}/{s['turn_range']}" for s in extra))

    if out.get("messages"):
        header("PROMPT")
        for m in out["messages"]:
            print(f"\n[{m['role']}]\n{m['content']}")

    header(f"ODPOWIEDŹ  {out.get('model', '')}")
    print(out["answer"] if out.get("answer") is not None else out.get("detail"))

    if out.get("answer") is not None:
        header("TOKENY")
        print(f"pytanie:      {out['question_tokens']}  (tokenizer {config.TOKENIZER_NAME})")
        print(f"cały prompt:  {out['prompt_tokens']}  (Ollama)")
        print(f"odpowiedź:    {out['answer_tokens']}  (Ollama)")


if __name__ == "__main__":
    main()
