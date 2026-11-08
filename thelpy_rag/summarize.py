"""Podsumowanie całej sesji metodą map-reduce (tree summarize).

Chunki nie są wyszukiwane wektorowo — pobieramy wszystkie chunki sesji po metadanych
(session_id, opcjonalnie temat) i układamy po chunk_index.

1. stuff  — cała sesja mieści się w oknie kontekstu: jedno wywołanie LLM.
2. map    — sąsiednie chunki grupowane w bloki mieszczące się w oknie, każdy blok streszczany.
3. reduce — streszczenia cząstkowe łączone; jeśli dalej się nie mieszczą, grupowane i streszczane
            ponownie (kolejny poziom), aż zmieszczą się w jednym, końcowym wywołaniu.
"""
from __future__ import annotations

import time

from qdrant_client import QdrantClient

from . import config
from .chunking import get_tokenizer
from .llm import TEMPLATE_OVERHEAD, chat, check_prompt_fits, format_context

SEPARATOR = "\n\n---\n\n"
MAX_LEVELS = 6               # bezpiecznik dla reduce hierarchicznego
# długość streszczeń cząstkowych jest proporcjonalna do wejścia (prompt podaje cel w słowach,
# num_predict to tylko bezpiecznik z zapasem — długością steruje prompt, nie ucinanie)
MAP_RATIO = 0.20             # map: streszczenie ~20% tokenów bloku transkryptu
REDUCE_RATIO = 0.50          # reduce: łączy już skrócone streszczenia, więc ściska mniej
MIN_WORDS, MAX_WORDS = 120, 500
TOKENS_PER_WORD = 1.4        # angielski, tokenizer Qwena (przybliżenie)
HARD_LIMIT_FACTOR = 2.0      # num_predict = cel × 2 (model nie trzyma się limitów słów dokładnie)
PART_TOKENS = int(MAX_WORDS * TOKENS_PER_WORD * HARD_LIMIT_FACTOR) + 64   # maks. limit kroku map/reduce
PROMPT_MARGIN = 32           # zapas na rozbieżności tokenizera
MIN_BLOCK_TOKENS = 600       # mniejszy blok nie mieści nawet 3 najkrótszych streszczeń -> reduce nie zbiega

SYSTEM_PROMPT = (
    "You are an assistant summarising psychotherapy session transcripts. "
    "Use only the information in the provided text; do not invent facts or interpret beyond it. "
    "Refer to the source turns in the format [turns X–Y]."
)

MAP_TASK = (
    "Summarise this part of the session (turns {range}) in at most {words} words, as bullet points. Cover: "
    "(1) concrete facts the client shares, (2) their feelings and attitudes, (3) what the therapist does — "
    "questions, reflections, summaries, suggestions, (4) anything agreed or planned. Add turn references."
)
REDUCE_TASK = (
    "Below are chronological partial summaries of consecutive parts of one session (turns {range}). "
    "Merge them into one coherent chronological summary of at most {words} words, without repetition, "
    "keeping the client's facts and feelings, the therapist's interventions, agreements and turn references."
)
FROM_CHUNKS = "Below is the transcript of a psychotherapy session (turns {range}). "
FROM_SUMMARIES = "Below are chronological partial summaries of one psychotherapy session (turns {range}). "
FOCUS_NOTE = "\nPay particular attention to: {focus}."

# warianty końcowego podsumowania: words = cel w prompcie, tokens = twardy limit odpowiedzi (num_predict)
LENGTHS = {
    "full": {
        "words": 600,
        "tokens": max(config.LLM_ANSWER_RESERVE, 1536),
        "task": ("Write the final summary of the session in Markdown, at most {words} words, with these sections:\n"
                 "## Reason for the session\n## Main topics\n## Client's state and emotions\n"
                 "## Therapist's interventions\n## Agreements and next steps\n"
                 "Keep turn references. If a section has no information, write 'Not discussed.'"),
    },
    "brief": {
        "words": 80,
        "tokens": 400,
        "task": ("Write a brief summary of the whole session in 3-5 sentences (at most {words} words), as plain "
                 "prose: no headings, no bullet points, no turn references. Say why the client came, the main "
                 "issue, the client's attitude, and the key outcome or next step."),
    },
}


# --- pobranie chunków ---------------------------------------------------------
def fetch_chunks(client: QdrantClient, collection: str, flt, page: int = 256) -> list[dict]:
    """Wszystkie chunki pasujące do filtra (np. build_filter(session_id=...)), po chunk_index."""
    out, offset = [], None
    while True:
        points, offset = client.scroll(collection, scroll_filter=flt, limit=page, offset=offset,
                                       with_payload=True, with_vectors=False)
        out.extend(p.payload for p in points)
        if offset is None:
            break
    return sorted(out, key=lambda p: p["chunk_index"])


# --- pomocnicze ----------------------------------------------------------------
def _turn_bounds(p: dict) -> tuple[str, str]:
    ids = p.get("turn_ids") or [str(p.get("chunk_index"))]
    return str(ids[0]), str(ids[-1])


def _range(items: list[dict]) -> str:
    return f"{items[0]['first']}–{items[-1]['last']}"


def pack(items: list[dict], budget: int) -> list[list[dict]]:
    """Grupuje SĄSIEDNIE elementy w bloki o łącznej liczbie tokenów <= budget, o możliwie równych
    rozmiarach (np. 17+16 zamiast 28+5). Element większy niż budget trafia do osobnego bloku."""
    greedy = _pack_greedy(items, budget)
    biggest = max(it["tokens"] for it in items)
    if len(greedy) < 2 or biggest > budget:
        return greedy
    # najmniejszy limit bloku, przy którym liczba bloków się nie zwiększa (wyszukiwanie binarne)
    lo, hi = biggest, budget
    while lo < hi:
        mid = (lo + hi) // 2
        if len(_pack_greedy(items, mid)) <= len(greedy):
            hi = mid
        else:
            lo = mid + 1
    return _pack_greedy(items, lo)


def _pack_greedy(items: list[dict], budget: int) -> list[list[dict]]:
    groups, cur, used = [], [], 0
    for it in items:
        if cur and used + it["tokens"] > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += it["tokens"]
    if cur:
        groups.append(cur)
    return groups


def overlap_warnings(payloads: list[dict]) -> list[str]:
    seen, dup = set(), 0
    for p in payloads:
        for t in p.get("turn_ids") or []:
            dup += t in seen
            seen.add(t)
    return [f"{dup} tur występuje w kilku chunkach (overlap) — streszczenie może je powtarzać; "
            f"użyj strategii bez overlapu"] if dup else []


# --- map-reduce ------------------------------------------------------------------
def summarize(payloads: list[dict], model: str = config.LLM_MODEL, formatter=None, focus: str | None = None,
              max_block_tokens: int | None = None, num_ctx: int = config.LLM_NUM_CTX,
              length: str = "full", log=None) -> dict:
    """payloads: chunki jednej sesji w kolejności chronologicznej.
    length: "full" (Markdown w sekcjach, ~600 słów) albo "brief" (3-5 zdań) — patrz LENGTHS.
    formatter: payloads -> tekst (domyślnie llm.format_context; z tematami: integration.format_context_with_topics).
    max_block_tokens: zmniejsza budżet bloku (np. żeby wymusić map-reduce do testów).
    log: opcjonalna funkcja str -> None (np. print, logger.info) — postęp kolejnych kroków."""
    if not payloads:
        raise ValueError("Brak chunków do podsumowania")
    log = log or (lambda _msg: None)
    t_start = time.perf_counter()
    tok = get_tokenizer()
    fmt = formatter or format_context
    focus_note = FOCUS_NOTE.format(focus=focus) if focus else ""

    if length not in LENGTHS:
        raise ValueError(f"Nieznana długość '{length}' — dostępne: {', '.join(LENGTHS)}")
    final = LENGTHS[length]
    final_tokens = final["tokens"]

    # budżet na tekst w jednym wywołaniu: okno - najdłuższa odpowiedź (finał albo map/reduce)
    # - prompt systemowy - najdłuższa instrukcja
    longest_task = max((MAP_TASK, REDUCE_TASK, FROM_SUMMARIES + final["task"], FROM_CHUNKS + final["task"]),
                       key=len)
    overhead = (tok.count(SYSTEM_PROMPT) + tok.count(longest_task + focus_note)
                + TEMPLATE_OVERHEAD + PROMPT_MARGIN)
    budget = num_ctx - max(final_tokens, PART_TOKENS) - overhead
    if max_block_tokens:
        budget = min(budget, max_block_tokens)
    if budget < MIN_BLOCK_TOKENS:
        raise ValueError(f"Budżet bloku {budget} tokenów jest za mały — minimum {MIN_BLOCK_TOKENS}, żeby blok "
                         f"zmieścił kilka streszczeń (każde ≥ {MIN_WORDS} słów) i reduce zbiegał "
                         f"(num_ctx={num_ctx}, max_block_tokens={max_block_tokens})")
    sep = tok.count(SEPARATOR)

    def target(items: list[dict], ratio: float) -> tuple[int, int]:
        """(docelowa liczba słów, twardy limit tokenów odpowiedzi) dla bloku wejść."""
        words = int(sum(it["tokens"] for it in items) * ratio / TOKENS_PER_WORD)
        words = max(MIN_WORDS, min(MAX_WORDS, words))
        return words, min(PART_TOKENS, int(words * TOKENS_PER_WORD * HARD_LIMIT_FACTOR) + 64)

    stats = {"llm_calls": 0, "prompt_tokens": 0, "answer_tokens": 0}
    steps: list[dict] = []

    def call(task: str, items: list[dict], num_predict: int, label: str, words: int) -> str:
        text = SEPARATOR.join(it["text"] for it in items)
        instruction = task.format(range=_range(items), words=words)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{instruction}{focus_note}\n\n{text}"}]
        est = check_prompt_fits(messages, num_ctx, reserve=num_predict)
        log(f"  {label}: tury {_range(items)}, {len(items)} wejść, ~{est} tok promptu "
            f"(cel ~{words} słów, limit odpowiedzi {num_predict} tok) …")
        t0 = time.perf_counter()
        r = chat(messages, model=model, num_ctx=num_ctx, num_predict=num_predict)
        stats["llm_calls"] += 1
        stats["prompt_tokens"] += r["prompt_tokens"]
        stats["answer_tokens"] += r["answer_tokens"]
        log(f"    gotowe w {time.perf_counter() - t0:.1f} s: prompt {r['prompt_tokens']} tok, "
            f"odpowiedź {r['answer_tokens']} tok / {len(r['answer'].split())} słów"
            + ("  [OBCIĘTA]" if r.get("truncated") else ""))
        if r.get("truncated"):
            out["warnings"].append(f"Odpowiedź dla tur {_range(items)} obcięta na limicie {num_predict} tokenów")
        return r["answer"].strip()

    def done(method: str, levels: int, summary: str) -> dict:
        log(f"Gotowe: {method}, poziomy={levels}, wywołania LLM={stats['llm_calls']}, "
            f"tokeny prompt/odpowiedź={stats['prompt_tokens']}/{stats['answer_tokens']}, "
            f"czas {time.perf_counter() - t_start:.1f} s")
        return {**out, "method": method, "levels": levels, **stats, "summary": summary, "steps": steps}

    def item(text: str, first: str, last: str) -> dict:
        return {"text": text, "first": first, "last": last, "tokens": tok.count(text) + sep}

    items = [item(fmt([p]), *_turn_bounds(p)) for p in payloads]
    out = {"model": model, "length": length, "n_chunks": len(payloads), "turn_range": _range(items),
           "block_budget": budget, "warnings": overlap_warnings(payloads)}

    total = sum(it["tokens"] for it in items)
    log(f"Podsumowanie: {len(items)} chunków (tury {out['turn_range']}), ~{total} tok tekstu, "
        f"budżet bloku {budget} tok, model {model}")
    for w in out["warnings"]:
        log(f"  UWAGA: {w}")

    # 1. stuff: wszystko w jednym oknie
    if total <= budget:
        log("Tryb stuff: cała sesja mieści się w jednym wywołaniu")
        summary = call(FROM_CHUNKS + final["task"], items, final_tokens, f"finał ({length})", final["words"])
        return done("stuff", 0, summary)

    # 2-3. map, potem reduce hierarchiczny
    level = 0
    while True:
        groups = pack(items, budget)
        if level > 0 and len(groups) == 1:
            break
        if level > 0 and len(items) <= 2:   # nie mieszczą się w budżecie bloku, ale finał i tak je połączy
            break                           # (check_prompt_fits pilnuje okna num_ctx)
        if level > 0 and len(groups) == len(items):     # brak postępu: łącz parami
            groups = [items[i:i + 2] for i in range(0, len(items), 2)]
        if level >= MAX_LEVELS:
            raise ValueError(f"Reduce nie zbiegł po {MAX_LEVELS} poziomach — zwiększ max_block_tokens")
        task, name, ratio = (MAP_TASK, "map", MAP_RATIO) if level == 0 else (REDUCE_TASK, "reduce", REDUCE_RATIO)
        log(f"Poziom {level} ({name}): {len(items)} wejść -> {len(groups)} bloków "
            f"[{', '.join(str(len(g)) for g in groups)}]")
        new = []
        for i, g in enumerate(groups, 1):
            words, limit = target(g, ratio)
            s = call(task, g, limit, f"{name} {i}/{len(groups)}", words)
            rng = _range(g)
            steps.append({"level": level, "turn_range": rng, "n_inputs": len(g), "target_words": words,
                          "words": len(s.split()), "summary": s})
            new.append(item(f"[turns {rng}]\n{s}", g[0]["first"], g[-1]["last"]))
        items, level = new, level + 1

    log(f"Finał: łączenie {len(items)} streszczeń (~{sum(it['tokens'] for it in items)} tok)")
    summary = call(FROM_SUMMARIES + final["task"], items, final_tokens, f"finał ({length})", final["words"])
    return done("map_reduce", level, summary)
