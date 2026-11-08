# Podsumowanie sesji (map-reduce): jak to działa

Opis modułu [`thelpy_rag/summarize.py`](summarize.py) oraz tego, jak korzystają z niego
`POST /summarize-session` ([`thelpy_api/service.py`](../thelpy_api/service.py)) i
`scripts/summarize_session.py`.

## Spis treści

1. [Po co i czym różni się od pytań RAG](#1-po-co-i-czym-różni-się-od-pytań-rag)
2. [Przepływ w skrócie](#2-przepływ-w-skrócie)
3. [Krok 1: przygotowanie sesji i pobranie chunków](#3-krok-1-przygotowanie-sesji-i-pobranie-chunków)
4. [Krok 2: tekst chunków i liczenie tokenów](#4-krok-2-tekst-chunków-i-liczenie-tokenów)
5. [Krok 3: budżet bloku](#5-krok-3-budżet-bloku)
6. [Krok 4: wybór trybu (stuff albo map-reduce)](#6-krok-4-wybór-trybu-stuff-albo-map-reduce)
7. [Krok 5: pakowanie w bloki](#7-krok-5-pakowanie-w-bloki)
8. [Krok 6: pętla map → reduce](#8-krok-6-pętla-map--reduce)
9. [Krok 7: finał](#9-krok-7-finał)
10. [Długość odpowiedzi: cel w słowach i twardy limit](#10-długość-odpowiedzi-cel-w-słowach-i-twardy-limit)
11. [Prompty](#11-prompty)
12. [Pilnowanie okna kontekstu](#12-pilnowanie-okna-kontekstu)
13. [Ostrzeżenia](#13-ostrzeżenia)
14. [Logi](#14-logi)
15. [Wynik](#15-wynik)
16. [Przykład liczbowy: sesja 64](#16-przykład-liczbowy-sesja-64)
17. [Parametry i stałe](#17-parametry-i-stałe)
18. [Ograniczenia i uproszczenia](#18-ograniczenia-i-uproszczenia)
19. [Jak testować](#19-jak-testować)

---

## 1. Po co i czym różni się od pytań RAG

`/ask-session` odpowiada na **pytanie**. Pytanie jest zamieniane na embedding, a wyszukiwanie dense
zwraca top-k (np. 5) najbardziej podobnych chunków. Do pytania o konkretny fakt to wystarcza, ale
do podsumowania już nie, bo 5 chunków to tylko kawałek sesji, a reszta w ogóle nie trafia do modelu.

`/summarize-session` przyjmuje sesję tak samo (plik, JSON albo samo `session_id` już wgranej sesji,
`n_turns`, `topics`, `topic`), ale:

- **nie wyszukuje**. Pobiera **wszystkie** chunki sesji po metadanych (`session_id`, opcjonalnie
  temat) i układa je w kolejności rozmowy (`chunk_index`),
- streszcza je metodą **map-reduce** (w LangChain/LlamaIndex: `map_reduce` / `tree_summarize`),
  bo cała sesja zwykle nie mieści się w oknie modelu (domyślnie 8192 tokeny).

| | `/ask-session` | `/summarize-session` |
|---|---|---|
| wejście | sesja + `question` + `k` | sesja (+ opcjonalnie `focus`) |
| wybór chunków | wyszukiwanie dense, top-k | wszystkie chunki sesji, po `chunk_index` |
| wywołania LLM | 1 | 1 (stuff) albo kilka (map-reduce) |
| wynik | odpowiedź z odwołaniami | podsumowanie w stałych sekcjach + kroki pośrednie |

---

## 2. Przepływ w skrócie

```
sesja (plik / JSON / session_id)
        │  prepare_session: ewentualny ingest do thelpy_turns<n>
        ▼
Qdrant scroll, filtr session_id (+ temat)  ──►  wszystkie chunki, sortowanie po chunk_index
        │
        ▼
tekst każdego chunka (formatter) + liczba tokenów
        │
        ├── suma tokenów ≤ budżet bloku ──► STUFF: 1 wywołanie z promptem finału ──► podsumowanie
        │
        └── suma > budżet ──► MAP-REDUCE
                 │
                 ▼
          poziom 0 (map):    [c0..c15] [c16..c32]       → streszczenie każdego bloku
                 │
                 ▼
          poziom 1+ (reduce): jeśli streszczenia nadal się nie mieszczą,
                              grupuj sąsiednie i streszczaj ponownie (kolejny poziom)
                 │
                 ▼
          FINAŁ: wszystkie streszczenia ostatniego poziomu → podsumowanie w sekcjach
```

Kolejność rozmowy jest zachowana na każdym poziomie. Bloki to zawsze **sąsiednie** fragmenty,
a streszczenia są łączone w porządku chronologicznym.

---

## 3. Krok 1: przygotowanie sesji i pobranie chunków

**Przygotowanie** (`prepare_session` w `thelpy_api/service.py`, wspólne z `/ask-session`):

1. Strategia chunkingu to `turns<n_turns>` (z `_ov<overlap_turns>`, jeśli overlap > 0), a kolekcja
   to `thelpy_turns<n_turns>`.
2. Sesja przysłana jako JSON jest zapisywana do `data/uploads/<session_id>.json`. Sesja podana jako
   `file` jest czytana z dysku serwera.
3. Bez `skip_ingest` sesja z `file` / `session` jest wgrywana do kolekcji (z `topics: true` najpierw
   anotowana tematami).
   **Samo `session_id`** (bez `file` i `session`) oznacza użycie gotowych chunków z kolekcji, bez wgrywania.
   Jeśli sesji tam nie ma, endpoint zwraca 404 z podpowiedzią (`GET /collections/{strategy}/sessions`).
4. Jeśli kolekcja nie istnieje, endpoint zwraca błąd 404.

**Pobranie chunków** (`fetch_chunks`):

- filtr `build_filter(session_id=...)`; z `topic` dodatkowo `integration.with_topic(...)`
  (chunk ma ten temat albo podtemat jako główny lub poboczny),
- `client.scroll(...)` stronami po 256 punktów, bez wektorów (tylko payload), aż Qdrant zwróci
  `next_page_offset = None`,
- sortowanie lokalne po `chunk_index`.

Nie ma tu wyszukiwania wektorowego ani embeddingu zapytania, tylko odczyt po metadanych.
Bez chunków (np. temat nie występuje w sesji) LLM nie jest wywoływany, a odpowiedź ma
`summary: null` i wyjaśnienie w `detail`.

---

## 4. Krok 2: tekst chunków i liczenie tokenów

Każdy chunk zamienia się w **element** (`item`):

```python
{"text": ..., "first": "T001", "last": "T008", "tokens": 75}
```

- `text` powstaje przez **formatter**, ten sam co w `/ask`:
  - domyślnie `llm.format_context`:
    ```
    [64, T001–T008, session date: 2024-01-15]
    Therapist: ...
    Client: ...
    ```
  - z `topics: true` używany jest `integration.format_context_with_topics`, który dokłada linię
    `(topics: Friends and social life 59%, ...; subtopics: ...)`.
- `first` / `last` to pierwsza i ostatnia tura chunka (z `turn_ids`). Z nich budowane są zakresy
  `T001–T128` w promptach, logach i wyniku.
- `tokens` to liczba tokenów tekstu (tokenizer `TOKENIZER_NAME`, domyślnie bge-m3) plus tokeny
  separatora `\n\n---\n\n`, którym bloki są sklejane.

Streszczenie z poziomu map/reduce też staje się elementem, z nagłówkiem zakresu:
```
[turns T001–T128]
- The client, Ricky, ...
```

---

## 5. Krok 3: budżet bloku

**Budżet bloku** to maksymalna liczba tokenów *tekstu wejściowego* (chunków albo streszczeń)
w jednym wywołaniu LLM:

```
budżet = LLM_NUM_CTX − max(limit finału, PART_TOKENS) − narzut
narzut = tokeny(SYSTEM_PROMPT) + tokeny(najdłuższa instrukcja + focus) + TEMPLATE_OVERHEAD (64) + PROMPT_MARGIN (32)
```

Przy domyślnych ustawieniach: `full` daje `8192 − 1536 − ok. 245 ≈ 6410` tokenów, a `brief` daje
`8192 − 1464 − ok. 230 ≈ 6500` tokenów.

- `LLM_NUM_CTX` (8192) to okno Ollamy, wspólne dla promptu **i** odpowiedzi.
- Odjęta jest najdłuższa możliwa odpowiedź: limit finału (`full` 1536, `brief` 400) albo limit
  kroku map/reduce (`PART_TOKENS` = 1464), zależnie od tego, który jest większy.
  Przy `length: "all"` brany jest największy limit spośród wariantów (czyli jak dla `full`).
- Narzut jest liczony dla najdłuższej z instrukcji, więc budżet jest bezpieczny dla każdego kroku.
- `focus` wydłuża instrukcję, więc nieco zmniejsza budżet.
- `max_block_tokens` może budżet tylko **zmniejszyć**. Przydaje się, żeby wymusić map-reduce
  i kilka poziomów na krótkiej sesji.
- Budżet poniżej 600 tokenów (`MIN_BLOCK_TOKENS`) daje błąd 400 jeszcze przed pierwszym wywołaniem LLM: mniejszy blok nie mieści nawet 3 najkrótszych streszczeń, więc reduce by nie zbiegał.

---

## 6. Krok 4: wybór trybu (stuff albo map-reduce)

```
suma tokenów wszystkich chunków ≤ budżet  →  STUFF
w przeciwnym razie                         →  MAP-REDUCE
```

- **stuff**: jedno wywołanie. Model dostaje cały transkrypt i od razu pisze końcowe podsumowanie
  (prompt `FROM_CHUNKS` + instrukcja wariantu z `LENGTHS`). To najlepsza jakość, bo model widzi wszystko naraz, a nie streszczenia.
- **map-reduce**: patrz kroki 5–7.

Większe `LLM_NUM_CTX` (np. 16384) sprawia, że więcej sesji mieści się w trybie stuff. Kosztem jest
pamięć i wolniejsze przetwarzanie długiego promptu.

---

## 7. Krok 5: pakowanie w bloki

`pack(items, budget)` dzieli listę elementów na **sąsiednie** grupy, z których każda mieści się
w budżecie.

**1. Zachłannie** (`_pack_greedy`): elementy są dokładane do bieżącego bloku, dopóki suma tokenów
nie przekroczy budżetu, potem zaczyna się nowy blok. Tak powstaje minimalna liczba bloków `n`.
Ostatni blok bywa jednak dużo mniejszy, np. 25 + 8 chunków (ok. 6200 + 1700 tokenów).

**2. Wyrównanie**: wyszukiwanie binarne szuka najmniejszego limitu bloku z przedziału
`[największy element, budżet]`, przy którym podział zachłanny nadal daje `n` bloków. Pakowanie
z tym limitem daje bloki o podobnych rozmiarach, np. **16 + 17** (ok. 4100 + 3800 tokenów) zamiast 25 + 8.

Dlaczego to ważne: przy stałej długości streszczeń duży blok byłby mocniej ściśnięty niż mały,
a końcówka sesji (zwykle ustalenia) dostałaby osobne, niewielkie streszczenie. Równe bloki dają
równą szczegółowość w całej sesji.

Przypadki brzegowe:
- element większy niż budżet (jeden bardzo długi chunk) trafia do osobnego bloku, a wyrównanie
  jest wtedy pomijane,
- jeśli wszystko mieści się w jednym bloku, kod zwraca jeden blok.

---

## 8. Krok 6: pętla map → reduce

```python
level = 0
while True:
    groups = pack(items, budget)
    if level > 0 and len(groups) == 1:    break   # wszystko zmieści się w finale
    if level > 0 and len(items) <= 2:     break   # 2 streszczenia: finał połączy je bez dodatkowego reduce
    if level > 0 and len(groups) == len(items):   # pakowanie nic nie połączyło -> łącz parami
        groups = pary
    if level >= MAX_LEVELS (6):           błąd "Reduce nie zbiegł"
    dla każdej grupy: streszczenie (map na poziomie 0, reduce na wyższych) -> nowy element
    items = nowe elementy; level += 1
→ FINAŁ
```

- **Poziom 0 (map)**: wejściem są chunki transkryptu, prompt `MAP_TASK`. Model streszcza fragment
  w punktach: fakty od klienta, jego uczucia i postawy, działania terapeuty, ustalenia, z odwołaniami do tur.
- **Poziom 1+ (reduce)**: wejściem są streszczenia z poprzedniego poziomu, prompt `REDUCE_TASK`.
  Model łączy je w jedno chronologiczne streszczenie bez powtórzeń.
- Każdy poziom zmniejsza liczbę elementów. Kod kończy, gdy zostaje jedna grupa albo najwyżej dwa
  streszczenia, i przechodzi do finału.
- **Bezpieczniki**:
  - „łącz parami”: gdy streszczenia są tak długie względem budżetu, że pakowanie nie połączyło
    żadnych dwóch, kod wymusza pary, żeby zagwarantować postęp,
  - `MAX_LEVELS = 6`: gdy nie zbiega (np. absurdalnie mały `max_block_tokens`), zwraca czytelny
    błąd zamiast pętli bez końca.

Przy domyślnym budżecie (ok. 6.4k) jeden poziom reduce łączy ok. 8 streszczeń, więc nawet bardzo
długie sesje kończą się na 1–2 poziomach.

---

## 9. Krok 7: finał

Ostatnie wywołanie dostaje wszystkie streszczenia z ostatniego poziomu (`FROM_SUMMARIES`), a w trybie
stuff cały transkrypt (`FROM_CHUNKS`), plus instrukcję wybranego wariantu (`length`, słownik `LENGTHS`).

| `length` | Format | Cel | Twardy limit |
|---|---|---|---|
| `full` (domyślnie) | Markdown w 5 sekcjach, z odwołaniami do tur | ok. 600 słów | 1536 tok |
| `brief` | 3–5 zdań zwykłej prozy, bez nagłówków i odwołań: po co klient przyszedł, główny problem, jego nastawienie, kluczowe ustalenie | ok. 80 słów | 400 tok |
| `all` | oba powyższe: map/reduce **raz**, potem osobny finał dla każdego wariantu (z tych samych wejść) | | |

Kroki map/reduce są **takie same** w obu wariantach. Brief powstaje ze szczegółowych streszczeń, więc
jest trafny, ale trwa prawie tyle samo co `full`, bo czas zajmuje głównie map. Dlatego, gdy potrzebne
są oba, `length: "all"` jest dużo tańsze niż dwa osobne zapytania: kosztuje tylko jedno dodatkowe
wywołanie LLM (finał `brief`). Brief nie powstaje z gotowego `full`, tylko z tych samych streszczeń
(albo transkryptu w stuff), żeby błędy jednego finału nie przechodziły do drugiego. W trybie stuff `all`
to po prostu dwa wywołania na całym transkrypcie.

Sekcje wariantu `full`:

```
## Reason for the session
## Main topics
## Client's state and emotions
## Therapist's interventions
## Agreements and next steps
```

W `full` odwołania do tur są zachowane, a sekcja bez informacji dostaje „Not discussed.”.

---

## 10. Długość odpowiedzi: cel w słowach i twardy limit

Każde wywołanie ma dwa limity:

1. **cel w słowach**: wpisany w prompt („in at most N words”). To on steruje długością.
2. **twardy limit tokenów** (`num_predict` w Ollamie): po jego osiągnięciu Ollama **ucina**
   odpowiedź. To tylko bezpiecznik, ustawiony z zapasem.

Mały model (qwen3.5:4b) trzyma się liczby słów tylko z grubsza, często pisze o 20–30% więcej
lub mniej. Gdyby twardy limit był równy celowi, odpowiedzi byłyby ucinane w pół zdania.

**Map i reduce: długość proporcjonalna do wejścia** (`target()`):

```
słowa = tokeny_wejścia × ratio / TOKENS_PER_WORD          (ratio: map 0.20, reduce 0.50)
słowa = obcięte do przedziału [MIN_WORDS=120, MAX_WORDS=500]
limit = min(PART_TOKENS, słowa × TOKENS_PER_WORD × HARD_LIMIT_FACTOR + 64)
```

- map ściska transkrypt ok. 5 razy (20%). Blok 1400 tokenów daje ok. 200 słów, blok 4000 tokenów
  daje 500 słów (górna granica).
- reduce ściska tylko 2 razy (50%), bo jego wejście to już streszczenia i większość treści jest
  istotna. Mocne ściskanie na każdym poziomie gubiłoby szczegóły.
- `TOKENS_PER_WORD = 1.4` to przybliżenie dla angielskiego.
- `HARD_LIMIT_FACTOR = 2` oznacza twardy limit dwa razy większy od celu.

Przykład: cel 500 słów daje limit `500 × 1.4 × 2 + 64 = 1464` tokenów (to też `PART_TOKENS`, czyli maksimum).

**Finał**: cel i limit zależą od wariantu `length`: `full` ok. 600 słów i 1536 tokenów, `brief` ok. 80 słów i 400 tokenów.

W `steps` każdego kroku są `target_words` (cel) i `words` (faktyczna długość), więc widać,
jak model się zachowuje.

---

## 11. Prompty

Wszystkie po angielsku (transkrypcje są po angielsku). Wspólny prompt systemowy:

> You are an assistant summarising psychotherapy session transcripts. Use only the information in
> the provided text; do not invent facts or interpret beyond it. Refer to the source turns in the
> format [turns X–Y].

Wiadomość użytkownika = `instrukcja` + (opcjonalnie) `focus` + pusta linia + teksty elementów
rozdzielone `---`.

| Prompt | Kiedy | Treść (skrót) |
|---|---|---|
| `MAP_TASK` | poziom 0 | streść część sesji (turns X–Y) w maks. N słowach, w punktach: (1) fakty od klienta, (2) uczucia i postawy, (3) działania terapeuty (pytania, odzwierciedlenia, podsumowania, sugestie), (4) ustalenia; z odwołaniami do tur |
| `REDUCE_TASK` | poziom 1+ | połącz chronologiczne streszczenia częściowe w jedno, maks. N słów, bez powtórzeń, zachowując fakty, uczucia, interwencje, ustalenia i odwołania |
| `FROM_CHUNKS` + `LENGTHS[length].task` | stuff | transkrypt sesji → podsumowanie w wybranym wariancie |
| `FROM_SUMMARIES` + `LENGTHS[length].task` | finał map-reduce | streszczenia częściowe → podsumowanie w wybranym wariancie |
| `LENGTHS["full"].task` | | Markdown w 5 sekcjach, maks. 600 słów, z odwołaniami do tur |
| `LENGTHS["brief"].task` | | 3–5 zdań, maks. 80 słów, bez nagłówków, punktów i odwołań |
| `FOCUS_NOTE` | gdy podano `focus` | „Pay particular attention to: …”, doklejane do **każdej** instrukcji, żeby wątek nie zginął już na etapie map |

Punkt (3) w `MAP_TASK` („what the therapist does”) jest celowo wymieniony wprost. Bez niego model
streszczał prawie wyłącznie wypowiedzi klienta, a w finale sekcja *Therapist's interventions*
kończyła się na „Not discussed”.

---

## 12. Pilnowanie okna kontekstu

Przed **każdym** wywołaniem `check_prompt_fits(messages, num_ctx, reserve=limit_odpowiedzi)`
liczy tokeny promptu i sprawdza:

```
tokeny promptu + limit odpowiedzi ≤ LLM_NUM_CTX
```

Jeśli warunek nie jest spełniony, rzuca `PromptTooLongError` (HTTP 400), zamiast wysłać zapytanie.
Ollama przy za długim prompcie **po cichu obcina** jego początek, więc model streszczałby niepełny
tekst, a nikt by o tym nie wiedział.

Budżet bloku jest **miękką** granicą pakowania. Twardą granicą jest okno `num_ctx`. Dlatego finał
może połączyć dwa streszczenia, które razem przekraczają mały budżet testowy: i tak mieszczą się
w oknie.

Po wywołaniu `llm.chat` sprawdza jeszcze odpowiedź Ollamy:
- `done_reason == "length"` oznacza, że odpowiedź została ucięta na limicie (`truncated: true`),
- `prompt_eval_count ≥ num_ctx` oznacza, że prompt prawdopodobnie został obcięty (ostrzeżenie Pythona).

---

## 13. Ostrzeżenia

Lista `warnings` w wyniku (w CLI jako `UWAGA: …`, w API jako pole JSON):

| Ostrzeżenie | Kiedy | Co zrobić |
|---|---|---|
| `N tur występuje w kilku chunkach (overlap)…` | strategia z `overlap_turns > 0`: te same tury są w dwóch chunkach i model może je streścić dwa razy | użyj `overlap_turns: 0` (domyślne) |
| `Odpowiedź dla tur X–Y obcięta na limicie N tokenów` | model doszedł do twardego limitu w kroku map/reduce/finał, więc koniec tego streszczenia przepadł | zwiększ `HARD_LIMIT_FACTOR` / `tokens` w `LENGTHS` albo zmniejsz `MAX_WORDS` |

Poza listą `warnings` Python wypisuje na stderr ostrzeżenia z `llm.py` (`warnings.warn`).
Domyślnie pokazuje je raz na proces.

---

## 14. Logi

`summarize(..., log=funkcja)` wywołuje `log(tekst)` na każdym etapie. CLI przekazuje `print`, a serwer
logger `uvicorn.error` (z prefiksem `[summarize <session_id>]`). Bez `log` funkcja nic nie loguje.

```
Podsumowanie: 33 chunków (tury T001–T262), ~7847 tok tekstu, budżet bloku 6411 tok, model qwen3.5:4b
Poziom 0 (map): 33 wejść -> 2 bloków [16, 17]
  map 1/2: tury T001–T128, 16 wejść, ~4240 tok promptu (cel ~500 słów, limit odpowiedzi 1464 tok) …
    gotowe w 31.2 s: prompt 4381 tok, odpowiedź 725 tok / 418 słów
  map 2/2: tury T129–T262, 17 wejść, ~3950 tok promptu (cel ~500 słów, limit odpowiedzi 1464 tok) …
    gotowe w 20.0 s: prompt 4161 tok, odpowiedź 779 tok / 469 słów
Finał: łączenie 2 streszczeń (~1555 tok)
  finał: tury T001–T262, 2 wejść, ~1763 tok promptu (cel ~600 słów, limit odpowiedzi 1536 tok) …
    gotowe w 27.8 s: prompt 1687 tok, odpowiedź 1175 tok / 742 słów
Gotowe: map_reduce, poziomy=1, wywołania LLM=3, tokeny prompt/odpowiedź=10229/2679, czas 82.1 s
```

- `Poziom 0 (map): 33 wejść -> 2 bloków [16, 17]`: ile elementów weszło i jak zostały podzielone
  (liczba elementów w każdym bloku).
- `~4240 tok promptu` to szacunek tokenizerem bge-m3 przed wysłaniem. `prompt 4381 tok` to faktyczny
  licznik Ollamy (tokenizer Qwena). Różnica kilku procent jest normalna.
- `[OBCIĘTA]` na końcu linii oznacza, że odpowiedź doszła do twardego limitu (patrz ostrzeżenia).

---

## 15. Wynik

```json
{
  "session_id": "64", "collection": "thelpy_turns8", "ingested": false,
  "topic": null, "focus": null, "n_chunks": 33,
  "model": "qwen3.5:4b", "turn_range": "T001–T262", "block_budget": 6411,
  "warnings": [],
  "method": "map_reduce", "levels": 1,
  "llm_calls": 3, "prompt_tokens": 10229, "answer_tokens": 2679,
  "summary": "## Reason for the session\n...",
  "summaries": {"full": "## Reason for the session\n..."},
  "steps": [
    {"level": 0, "turn_range": "T001–T128", "n_inputs": 16, "target_words": 500, "words": 418, "summary": "..."},
    {"level": 0, "turn_range": "T129–T262", "n_inputs": 17, "target_words": 500, "words": 469, "summary": "..."}
  ]
}
```

| Pole | Znaczenie |
|---|---|
| `method` | `stuff` (jedno wywołanie) albo `map_reduce` |
| `levels` | liczba poziomów map/reduce przed finałem (`0` = stuff, `1` = sam map, `2` = map + reduce, …) |
| `block_budget` | budżet bloku z kroku 3 |
| `llm_calls` | liczba wywołań LLM łącznie z finałem |
| `prompt_tokens` / `answer_tokens` | sumy liczników Ollamy ze wszystkich wywołań |
| `steps` | kroki pośrednie (bez finału); `n_inputs` = ile chunków/streszczeń weszło do kroku |
| `summary` | końcowe podsumowanie (Markdown); przy `length: "all"` jest `null`, bo warianty są w `summaries` |
| `summaries` | zawsze: `{wariant: tekst}`, np. `{"full": …, "brief": …}` przy `all` |
| `ingested` | czy sesja była wgrywana w tym wywołaniu |

`include_steps: false` usuwa `steps`, a reszta zostaje bez zmian.

---

## 16. Przykład liczbowy: sesja 64

Sesja `data/annomi_64.json`: 262 tury, strategia `turns8` daje 33 chunki po 8 tur (70–662 tokenów),
łącznie ok. 7847 tokenów.

**Domyślnie** (`LLM_NUM_CTX=8192`, budżet 6411):
1. 7847 > 6411, więc map-reduce.
2. Zachłannie wyszłyby 2 bloki (25 + 8 chunków, ok. 6200 + 1700 tokenów). Po wyrównaniu są 2 bloki: 16 + 17 (ok. 4100 + 3800 tokenów).
3. Map: cel `4100 × 0.2 / 1.4 ≈ 585`, ograniczony do 500 słów; limit 1464 tokeny. Model napisał 418 i 469 słów.
4. Dwa streszczenia (ok. 1555 tokenów) mieszczą się razem, więc finał. Wyszło 742 słowa.
5. Razem 3 wywołania, ok. 80 s.

**Z `max_block_tokens: 1500`** (test map-reduce):
1. 6 bloków `[7, 4, 5, 6, 5, 6]`, każdy ok. 1200–1400 tokenów tekstu, cel ok. 175–196 słów.
2. 6 streszczeń się nie mieści, więc reduce: 2 bloki po 3 streszczenia, cel ok. 415–420 słów.
3. Zostają 2 streszczenia, więc finał.
4. Razem 9 wywołań, ok. 105 s.

---

## 17. Parametry i stałe

**Parametry zapytania** (`SummarizeSessionRequest` / flagi CLI):

| API | CLI | Domyślnie | Opis |
|---|---|---|---|
| `file` / `session` / `session_id` | `file` (pozycyjny) | — | sesja |
| `n_turns` | `--n-turns` | 8 | tur w chunku (kolekcja `thelpy_turns<n>`) |
| `overlap_turns` | `--overlap-turns` | 0 | overlap (niezalecany przy podsumowaniu) |
| `model` | `--model` | `LLM_MODEL` | model Ollamy |
| `skip_ingest` | `--skip-ingest` | false | przy `file` / `session`: nie wgrywaj ponownie (w API samo `session_id` i tak nie wgrywa) |
| `topics` | `--topics` | false | wgraj z tematami, pokaż tematy w nagłówkach chunków |
| `topic` | `--topic` | — | podsumuj tylko chunki z tym tematem (włącza `topics`) |
| `length` | `--length` | full | `full` (sekcje, ok. 600 słów) albo `brief` (3–5 zdań) |
| `focus` | `--focus` | — | na czym się skupić, doklejane do każdego promptu |
| `max_block_tokens` | `--max-block-tokens` | — | zmniejsza budżet bloku (min. 600) |
| `include_steps` | `--show-steps` | true / false | kroki pośrednie w wyniku / na ekranie |

**Konfiguracja** ([`config.py`](config.py), `.env`):

| Zmienna | Domyślnie | Wpływ |
|---|---|---|
| `LLM_NUM_CTX` | 8192 | okno modelu, a z nim budżet bloku i to, kiedy wystarcza stuff |
| `LLM_ANSWER_RESERVE` | 1024 | limit finału `full` = max(to, 1536) |
| `LLM_TEMPERATURE` | 0.2 | losowość odpowiedzi |
| `TOKENIZER_NAME` | BAAI/bge-m3 | liczenie tokenów do budżetów |

**Stałe w kodzie** ([`summarize.py`](summarize.py)):

| Stała | Wartość | Znaczenie |
|---|---|---|
| `MAP_RATIO` | 0.20 | map: streszczenie ok. 20% tokenów bloku |
| `REDUCE_RATIO` | 0.50 | reduce: ok. 50% wejścia |
| `MIN_WORDS`, `MAX_WORDS` | 120, 500 | granice celu dla map/reduce |
| `TOKENS_PER_WORD` | 1.4 | przelicznik słowa ↔ tokeny |
| `HARD_LIMIT_FACTOR` | 2.0 | twardy limit = cel × 2 |
| `LENGTHS` | full: 600 słów / 1536 tok; brief: 80 słów / 400 tok | wariant finału: cel, twardy limit, instrukcja |
| `PART_TOKENS` | 1464 | maks. twardy limit kroku map/reduce |
| `MAX_LEVELS` | 6 | bezpiecznik liczby poziomów reduce |
| `PROMPT_MARGIN` | 32 | zapas na różnice tokenizerów |

---

## 18. Ograniczenia i uproszczenia

- **Streszczenie streszczeń traci szczegóły.** Każdy poziom map/reduce coś gubi, więc najlepszą
  jakość daje stuff. Jeśli sprzęt pozwala, większe `LLM_NUM_CTX` jest najprostszą poprawą jakości.
- **Map nie widzi reszty sesji.** Każdy blok jest streszczany osobno, więc wątek zaczęty w jednym
  bloku i dokończony w drugim łączy dopiero reduce albo finał.
- **Jeden budżet dla wszystkich kroków.** Budżet odejmuje najdłuższą możliwą odpowiedź (finał
  albo map), więc przy `full` bloki map mogłyby być o ok. 70 tokenów większe. To świadome
  uproszczenie, bo jeden budżet jest prostszy i bezpieczny.
- **Szacunki tokenów są przybliżone.** Budżety liczy tokenizer bge-m3, a Ollama liczy tokenizerem
  modelu. `PROMPT_MARGIN` i zapas w twardych limitach to pokrywają. Jeśli tokenizer bge-m3 się nie
  wczytał (fallback na regex), szacunki są mniej dokładne.
- **Limity słów są miękkie.** Model może napisać ±30% (np. 742 słowa przy celu 600). Twarde
  limity są ustawione z zapasem.
- **Filtr `topic` daje fragmenty nieciągłe.** Zakres `T001–T262` w promptach oznacza wtedy
  „od pierwszego do ostatniego wybranego chunka”, a nie całą ciągłą rozmowę.
- **Overlap powtarza tury.** Przy `overlap_turns > 0` te same wypowiedzi trafiają do dwóch
  chunków (jest na to ostrzeżenie).
- **Czas** rośnie z liczbą wywołań. Wywołania na jednym poziomie są wykonywane po kolei, nie równolegle.

---

## 19. Jak testować

```bash
# cała sesja; przy domyślnym oknie sesja 64 idzie przez map-reduce (2 bloki + finał)
python -m scripts.summarize_session data/annomi_64.json --skip-ingest

# wymuszenie kilku poziomów + podgląd kroków pośrednich
python -m scripts.summarize_session data/annomi_64.json --skip-ingest --max-block-tokens 1500 --show-steps

# z naciskiem na wątek
python -m scripts.summarize_session data/annomi_64.json --skip-ingest --focus "client's motivation to change"

# tylko chunki z tematem (kolekcja wgrana z --topics)
python -m scripts.summarize_session data/annomi_64.json --skip-ingest --topic alcohol
```

```bash
curl -X POST localhost:8000/summarize-session -H 'Content-Type: application/json' \
  -d '{"session_id": "64", "max_block_tokens": 1500}'
```

Na co patrzeć:
- `method` i `levels` zgadzają się z oczekiwaniem (stuff, gdy sesja się mieści),
- zakresy `turn_range` w `steps` są ciągłe i pokrywają całą sesję,
- `warnings` jest puste (brak obciętych odpowiedzi),
- `words` jest bliskie `target_words`,
- w podsumowaniu wypełnione są wszystkie sekcje, łącznie z *Therapist's interventions*
  i *Agreements and next steps*.
