# Thelpy RAG API — opis endpointów

Serwer FastAPI nad `thelpy_rag` (Qdrant + Ollama) i `thelpy_topics` (tematy rozmowy).
Każdy endpoint robi to samo co jeden ze skryptów w `scripts/`, ale przyjmuje argumenty jako JSON
i zwraca wynik jako JSON zamiast wypisywać go w terminalu.

- interaktywna dokumentacja z formularzami: **http://127.0.0.1:8000/docs**
- uruchomienie: `docker compose up -d --build` albo lokalnie `python -m thelpy_api`

## Spis treści

1. [Pojęcia](#1-pojęcia)
2. [Przegląd endpointów](#2-przegląd-endpointów)
3. [Informacje](#3-informacje): `/health`, `/config`, `/topics`
4. [Kolekcje](#4-kolekcje): `/collections`
5. [Wgrywanie sesji](#5-wgrywanie-sesji): `/ingest`, `/ingest/upload`
6. [Zapytania](#6-zapytania): `/search`, `/ask`, `/ask-session`
7. [Ewaluacja](#7-ewaluacja): `/benchmark-session`, `/evaluate`
8. [Tematy](#8-tematy)
9. [Błędy](#9-błędy)
10. [Typowy przebieg](#10-typowy-przebieg)

---

## 1. Pojęcia

| Pojęcie | Znaczenie |
|---|---|
| **sesja** | jeden plik JSON z transkrypcją (`session_id`, `client_id`, `session_at`, `turns`); format w `thelpy_rag/loader.py` |
| **chunk** | fragment sesji zapisany w Qdrancie jako jeden punkt (tekst + wektory + metadane) |
| **strategia** | sposób cięcia sesji na chunki. Gotowe: `fixed256`, `fixed256_ov64`, `fixed512`, `fixed512_ov128`, `turns4`, `turns8`. Można też podać własną: `turnsN[_ovM]` (okno N tur, M tur wspólnych) albo `fixedN[_ovM]` (N tokenów, M tokenów overlapu) |
| **kolekcja** | jedna strategia = jedna kolekcja Qdranta o nazwie `thelpy_<strategia>`, np. `thelpy_turns8` |
| **tryb wyszukiwania** (`mode`) | `dense` — embeddingi (bge-m3, podobieństwo cosinusowe); `bm25` — słowa kluczowe; `hybrid` — oba połączone RRF |
| **fusion** | tylko przy `hybrid`: `client` — RRF liczony w Pythonie (widać `ranks` z obu gałęzi); `server` — RRF liczy Qdrant |
| **sąsiedzi** (`neighbors`) | N chunków przed i po każdym trafieniu z tej samej sesji (szerszy kontekst) |
| **temat** (`topic`) | klucz tematu lub podtematu z `thelpy_topics/taxonomy.py`, np. `social`, `friends`, `alcohol` (lista: `GET /topics`) |

**Ścieżki plików** (`paths`, `file`, `benchmark_path`, `queries_path`, `out`) dotyczą dysku serwera.
Względne są liczone od katalogu projektu, np. `data/annomi_64.json`. W Dockerze `./data`
jest zamontowane jako `/app/data`, więc podajesz je tak samo.

---

## 2. Przegląd endpointów

| Metoda i ścieżka | Co robi | Skrypt |
|---|---|---|
| `GET /health` | czy Qdrant i Ollama odpowiadają | — |
| `GET /config` | aktywna konfiguracja i strategie | — |
| `GET /topics` | lista tematów i podtematów | — |
| `GET /collections` | kolekcje i liczba punktów | — |
| `POST /collections` | tworzy puste kolekcje | `create_collections.py` |
| `DELETE /collections/{strategy}/sessions/{session_id}` | usuwa jedną sesję z kolekcji | — |
| `POST /ingest` | wgrywa sesje (ścieżki lub JSON) | `ingest.py` |
| `POST /ingest/upload` | wgrywa sesje wysłane jako pliki | `ingest.py` |
| `POST /search` | wyszukuje fragmenty, bez LLM | `search.py` |
| `POST /ask` | wyszukuje fragmenty i generuje odpowiedź LLM | `ask.py` |
| `POST /ask-session` | pytanie do jednej sesji z pełną diagnostyką | `ask_session.py` |
| `POST /benchmark-session` | benchmark pytań do jednej sesji (JSON lub Excel) | `benchmark_session.py` |
| `POST /evaluate` | metryki wyszukiwania dla strategii × trybów | `evaluate.py` |

---

## 3. Informacje

### `GET /health`

Sprawdza połączenie z Qdrantem i Ollamą. Wywołaj najpierw, gdy coś nie działa.

```json
{"qdrant": "ok", "ollama": "ok", "status": "ok"}
```

Gdy któraś usługa nie odpowiada, zamiast `"ok"` jest opis błędu, a `status` to `"degraded"`.

### `GET /config`

Zwraca konfigurację z `.env` / `thelpy_rag/config.py` (bez klucza API): adresy usług, modele
(`llm_model`, `embed_model`), parametry BM25 i RRF oraz wszystkie zarejestrowane strategie
z ich parametrami.

### `GET /topics`

Lista tematów głównych z podtematami. Klucze (`key`) podajesz w parametrze `topic`.

```json
{"topics": [
  {"key": "substances", "label_pl": "Substancje i uzależnienia", "label_en": "Substances and addictions",
   "kind": "life", "subtopics": [{"key": "alcohol", "label_en": "alcohol", "label_pl": "alkohol"}, "..."]},
  "..."
]}
```

`kind`: `life` — tematy z życia klienta, `frame` — otwarcie i zakończenie rozmowy,
`process` — proces terapii.

---

## 4. Kolekcje

### `GET /collections`

Kolekcje z prefiksem projektu i liczba punktów (chunków) w każdej.

```json
[{"collection": "thelpy_turns8", "strategy": "turns8", "points": 33}]
```

### `POST /collections` — `scripts.create_collections`

Tworzy puste kolekcje (wektor dense + rzadki BM25 + indeksy metadanych). Zwykle nie jest
potrzebny, bo `/ingest` sam tworzy brakującą kolekcję.

| Pole | Domyślnie | Opis |
|---|---|---|
| `strategies` | wszystkie gotowe | lista strategii |
| `recreate` | `false` | **usuwa kolekcję z danymi** i tworzy ją od nowa (np. po zmianie modelu embeddingów) |
| `dim` | z modelu | wymiar wektora; domyślnie odczytany z modelu w Ollamie |

Odpowiedź: `embed_model`, `dim` i dla każdej strategii `created` (`false` = już istniała, pominięta).

### `DELETE /collections/{strategy}/sessions/{session_id}`

Usuwa wszystkie chunki jednej sesji z kolekcji danej strategii. Pozostałe sesje zostają.

```bash
curl -X DELETE localhost:8000/collections/turns8/sessions/64
```

---

## 5. Wgrywanie sesji

### `POST /ingest` — `scripts.ingest`

Dla każdej sesji i każdej strategii: tnie sesję na chunki, liczy embeddingi (Ollama) i wektory
BM25 i zapisuje je do kolekcji. Ponowne wgranie tej samej sesji najpierw usuwa jej stare chunki,
więc poprawiona transkrypcja zastępuje poprzednią i nic się nie dubluje.

| Pole | Domyślnie | Opis |
|---|---|---|
| `paths` | `[]` | pliki `.json` lub katalogi na serwerze |
| `sessions` | `[]` | sesje JSON wprost w zapytaniu; są zapisywane w `data/uploads/<session_id>.json` |
| `strategies` | wszystkie gotowe | do których kolekcji wgrać |
| `topics` | `false` | dodaj tematy do chunków (zob. [Tematy](#8-tematy)) |
| `dry_run` | `false` | tylko policz chunki i tokeny, nic nie zapisuj |
| `show` | `0` | przy `dry_run`: zwróć N pierwszych chunków każdej strategii (tekst + tematy) |

Trzeba podać `paths` albo `sessions` (albo oba).

```bash
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"paths": ["data/annomi_64.json"], "strategies": ["turns8"], "topics": true}'
```

```json
{"dry_run": false, "topics": true, "warnings": [],
 "sessions": [{"session_id": "64", "file": "/app/data/with_topics/annomi_64.json", "n_turns": 262,
   "source_version": "v1",
   "strategies": [{"strategy": "turns8", "chunks": 33, "tokens": {"mean": 213, "min": 70, "max": 662},
                   "collection": "thelpy_turns8", "points": 33}]}]}
```

- `warnings` zawiera ostrzeżenie, gdy nie udało się wczytać tokenizera modelu (rozmiary w tokenach są wtedy przybliżone).
- `collection_created: true` pojawia się, gdy kolekcja została właśnie utworzona.

### `POST /ingest/upload`

To samo co `/ingest`, ale sesje wysyła się jako pliki (`multipart/form-data`), np. z formularza.

```bash
curl -X POST localhost:8000/ingest/upload \
  -F files=@data/annomi_64.json -F strategies=turns8 -F topics=true
```

Pola formularza: `files` (jeden lub więcej), `strategies` (można powtarzać), `topics`, `dry_run`, `show`.

---

## 6. Zapytania

### `POST /search` — `scripts.search`

Wyszukuje fragmenty i zwraca je ze scorami. **Nie wywołuje LLM**, więc jest szybki i nadaje się
do sprawdzania, czy retrieval znajduje właściwe miejsca.

| Pole | Domyślnie | Opis |
|---|---|---|
| `query` | **wymagane** | tekst zapytania |
| `strategy` | `turns4` | w której kolekcji szukać |
| `mode` | `hybrid` | `dense` / `bm25` / `hybrid` |
| `k` | `5` | liczba wyników (1–100) |
| `fusion` | `client` | przy `hybrid`: `client` / `server` |
| `session`, `client` | — | filtr: tylko ta sesja / ten klient |
| `role` | — | filtr: chunki, w których mówi dana rola (`therapist`, `client`) |
| `date_from`, `date_to` | — | filtr po dacie sesji (ISO, np. `2026-03-01`) |
| `neighbors` | `0` | dołącz N sąsiednich chunków z każdej strony |
| `topic` | — | tylko chunki z tym tematem lub podtematem |

```json
{"collection": "thelpy_turns8", "mode": "hybrid", "topic": "friends", "count": 2,
 "hits": [{"id": "…", "score": 0.0328, "ranks": {"dense": 1, "bm25": 1},
           "topic_note": "topics: Friends and social life 100%; subtopics: friends",
           "payload": {"session_id": "64", "chunk_index": 5, "turn_range": "T041–T048", "text": "…", "…": "…"}}]}
```

- `score`: przy `dense` to podobieństwo cosinusowe, przy `bm25` score BM25, przy `hybrid` suma RRF (małe liczby, liczy się kolejność).
- `ranks`: pozycja chunka w każdej gałęzi (tylko `hybrid` z `fusion: client`).
- `topic_note`: tematy fragmentu w czytelnej postaci; `null`, gdy kolekcja jest bez tematów.
- `payload`: wszystkie metadane chunka (czasy `start_seconds`/`end_seconds`, `turn_ids`, mówiący, tematy…).
- `neighbors`: lista payloadów sąsiednich chunków (tylko gdy `neighbors > 0`).

### `POST /ask` — `scripts.ask`

Pełny RAG po całej bazie lub po wybranym kliencie/sesji: wyszukuje fragmenty, opcjonalnie
dokłada sąsiadów, usuwa duplikaty, układa fragmenty chronologicznie i wysyła je do LLM. Model
odpowiada wyłącznie na podstawie fragmentów, z odwołaniami `[sesja, tury]`.

| Pole | Domyślnie | Opis |
|---|---|---|
| `question` | **wymagane** | pytanie |
| `strategy`, `mode`, `k`, `fusion` | jak w `/search` | |
| `session`, `client` | — | filtry |
| `neighbors` | `0` | sąsiednie chunki jako dodatkowy kontekst |
| `model` | `LLM_MODEL` z `.env` | model Ollamy |
| `topics` | `false` | pokaż LLM-owi tematy każdego fragmentu (w nagłówku fragmentu) |
| `topic` | — | szukaj tylko w chunkach z tym tematem; włącza `topics` |
| `include_prompt` | `false` | zwróć też pełny prompt (`messages`) |

```json
{"question": "What drugs does he use?", "collection": "thelpy_turns8", "mode": "hybrid", "topic": null,
 "hits":    [{"score": 0.03, "session_id": "64", "chunk_index": 12, "turn_range": "T097–T104", "…": "…"}],
 "sources": [{"session_id": "64", "chunk_index": 12, "turn_range": "T097–T104", "session_at": "…",
              "start_seconds": 635.0, "end_seconds": 713.0, "topic_note": "…"}],
 "model": "qwen3.5:4b", "answer": "…", "prompt_tokens": 1122, "answer_tokens": 299}
```

- `hits`: trafienia wyszukiwania ze scorami; `sources`: to, co faktycznie trafiło do promptu (z sąsiadami), chronologicznie.
- Gdy nic nie znaleziono, LLM nie jest wywoływany: `answer: null` i wyjaśnienie w `detail`.

### `POST /ask-session` — `scripts.ask_session`

Pytanie do **jednej sesji**, z diagnostyką do analizy działania RAG. Opcjonalnie najpierw wgrywa
sesję, potem szuka trybem `dense` tylko w niej i zwraca znalezione fragmenty ze scorami
cosinusowymi, pełny prompt, odpowiedź i liczby tokenów.

| Pole | Domyślnie | Opis |
|---|---|---|
| `question` | **wymagane** | pytanie |
| `file` | — | plik sesji na serwerze, np. `data/annomi_64.json` |
| `session` | — | albo sesja JSON wprost w zapytaniu |
| `session_id` | — | albo samo ID, gdy sesja jest już wgrana (`skip_ingest: true`) |
| `n_turns` | `8` | liczba tur w chunku (kolekcja `thelpy_turns<n_turns>`) |
| `overlap_turns` | `0` | tury wspólne między chunkami |
| `k` | `5` | liczba fragmentów przekazanych do LLM |
| `model` | `LLM_MODEL` | model Ollamy |
| `skip_ingest` | `false` | nie wgrywaj, sesja już jest w kolekcji |
| `topics` | `false` | wgraj z tematami i pokaż tematy LLM-owi |
| `topic` | — | szukaj tylko w chunkach z tym tematem; włącza `topics` |

```bash
curl -X POST localhost:8000/ask-session -H 'Content-Type: application/json' \
  -d '{"session_id": "64", "skip_ingest": true, "question": "What does he say about his friends?", "topic": "friends"}'
```

```json
{"question": "…", "session_id": "64", "collection": "thelpy_turns8", "ingested": false, "topic": "friends",
 "hits": [{"id": "…", "cosine": 0.535, "ranks": null, "topic_note": "…", "payload": {"…": "…"}}],
 "question_tokens": 8, "tokenizer": "BAAI/bge-m3",
 "model": "qwen3.5:4b", "answer": "…", "prompt_tokens": 1122, "answer_tokens": 299,
 "messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "Fragments: …"}]}
```

- `question_tokens`: liczba tokenów samego pytania (tokenizer bge-m3).
- `prompt_tokens` / `answer_tokens`: liczby podane przez Ollamę (cały prompt / odpowiedź).
- Fragmenty w prompcie są ułożone po kolejności w sesji (`chunk_index`).
- Bez trafień (np. temat nie występuje w sesji): `answer: null` i wyjaśnienie w `detail`.

> Bez `skip_ingest` sesja jest wgrywana przy każdym wywołaniu. Przy kolejnych pytaniach do tej
> samej sesji używaj `skip_ingest: true` i `session_id`.

---

## 7. Ewaluacja

### `POST /benchmark-session` — `scripts.benchmark_session`

Uruchamia RAG dla każdego pytania z benchmarku (sekcja `questions`) na jednej, wcześniej wgranej
sesji. Sprawdza, czy pobrane chunki pokrywają miejsca z `evidence` (znaczniki `hh:mm:ss` lub
zakresy `a–b`). Ocenę merytoryczną (`required_info`, `forbidden_additions`) wpisuje się ręcznie
w puste kolumny Excela.

| Pole | Domyślnie | Opis |
|---|---|---|
| `session` | **wymagane** | `session_id` w Qdrancie, np. `64` |
| `benchmark_path` | — | plik benchmarku na serwerze, np. `data/benchamrk_064.json` |
| `benchmark` | — | albo benchmark JSON wprost w zapytaniu |
| `strategy` | `turns8` | kolekcja |
| `mode` | `dense` | tryb wyszukiwania |
| `k` | `5` | fragmentów na pytanie |
| `context_order` | `score` | kolejność fragmentów w prompcie: `score` (od najlepszego) albo `chrono` (wg kolejności w sesji) |
| `model` | `LLM_MODEL` | model Ollamy |
| `topics`, `topic` | — | jak w `/ask` |
| `out` | — | zapisz też Excela na serwerze, np. `data/benchmarks/b64.xlsx` |
| `xlsx` | `false` | zamiast JSON-a zwróć plik `.xlsx` do pobrania |

Odpowiedź JSON ma te same cztery części co arkusze Excela:

| Klucz | Arkusz | Zawartość |
|---|---|---|
| `results` | `wyniki` | jedno pytanie = jeden wiersz: odpowiedź, źródła, `evidence_found` (✓/✗), `evidence_hit`, `evidence_recall`, tokeny, puste kolumny do ręcznej oceny |
| `fragments` | `fragmenty` | każdy pobrany chunk: ranga, score, czasy, które evidence pokrywa, tekst |
| `config` | `konfiguracja` | parametry uruchomienia |
| `summary` | `podsumowanie` | średnie `evidence_hit` / `evidence_recall` i suma tokenów wg typu pytania i łącznie (`ALL`) |

Dodatkowo `saved_to` zawiera ścieżkę zapisanego pliku, jeśli podano `out`.

```bash
# plik Excela do pobrania
curl -X POST localhost:8000/benchmark-session -H 'Content-Type: application/json' \
  -d '{"session": "64", "benchmark_path": "data/benchamrk_064.json", "k": 8, "xlsx": true}' -o wynik.xlsx
```

Uwaga: każde pytanie to jedno wywołanie LLM, więc cały benchmark może trwać kilka minut.

### `POST /evaluate` — `scripts.evaluate`

Porównuje strategie chunkingu i tryby wyszukiwania na zbiorze pytań ze złotym standardem.
**Nie używa LLM**, ocenia tylko retrieval. Złoty standard jest zapisany na poziomie tur, więc
ten sam zbiór pytań działa dla każdej strategii: chunk jest trafny, jeśli zawiera choć jedną
trafną turę.

| Pole | Domyślnie | Opis |
|---|---|---|
| `queries_path` | — | plik JSONL na serwerze (jedna linia = jedno pytanie) |
| `queries` | `[]` | albo lista pytań w zapytaniu |
| `strategies` | wszystkie gotowe | które kolekcje porównać (wszystkie muszą istnieć) |
| `modes` | wszystkie trzy | które tryby porównać |
| `k` | `5` | top-k |
| `scope` | `all` | `all` — cała baza; `session` — tylko sesja pytania; `client` — tylko klient pytania |

Format pytania: `{"qid": "q1", "query": "…", "session_id": "64", "relevant_turn_ids": ["T097", "T098"]}`
(przy `scope: client` także `client_id`).

```json
{"k": 5, "scope": "session", "n_queries": 1,
 "rows": [{"strategy": "turns8", "mode": "dense", "k": 5, "n": 1,
           "hit": 1.0, "recall": 0.5, "mrr": 0.5, "ctx_tokens": 1200}]}
```

| Metryka | Znaczenie |
|---|---|
| `hit` | czy którykolwiek z top-k chunków zawiera trafną turę |
| `recall` | jaka część trafnych tur jest pokryta przez top-k |
| `mrr` | 1 / pozycja pierwszego trafnego chunka |
| `ctx_tokens` | ile tokenów kontekstu dałoby top-k (duże chunki „kupują” recall tokenami) |

Wartości są średnimi po wszystkich pytaniach.

---

## 8. Tematy

Tematy pochodzą z `thelpy_topics`. Działają w dwóch krokach:

1. **Wgranie z tematami** — `/ingest` (albo `/ask-session`) z `topics: true`. Sesja jest najpierw
   anotowana modelem tematów (kopia z tematami każdej wypowiedzi trafia do `data/with_topics/`),
   a każdy chunk dostaje w payloadzie tematy swoich wypowiedzi: `topic_main`, `topic_all`,
   `topic_labels`, `topic_shares`, `subtopic_main`, `subtopic_all`, `subtopic_labels`,
   `subtopic_shares` i inne. Do pól tematów dokładane są indeksy w Qdrancie.
   Anotacja jest liczona raz na sesję: jeśli plik się nie zmienił, kolejne wgrania biorą gotową
   kopię. Pierwsza anotacja po starcie serwera ładuje model (kilka–kilkanaście sekund).
2. **Użycie tematów w zapytaniach:**
   - `topic: "<klucz>"` — szukaj tylko w chunkach, w których ten temat lub podtemat występuje
     jako główny albo poboczny. Działa w `/search`, `/ask`, `/ask-session`, `/benchmark-session`.
   - `topics: true` — LLM widzi przy każdym fragmencie linię z jego tematami, np.
     `(topics: Friends and social life 59%, Substances and addictions 31%; subtopics: …)`.
     Działa w `/ask`, `/ask-session`, `/benchmark-session` (`topic` włącza to automatycznie).

Nieznany klucz tematu daje błąd 400. Jeśli kolekcja była wgrana **bez** tematów, filtr `topic`
nie znajdzie niczego: wgraj sesję ponownie z `topics: true`.

Tematy są włączane osobno dla każdego zapytania. Zapytania z tematami i bez nich mogą działać
równolegle na tym samym serwerze.

---

## 9. Błędy

Błędy mają postać `{"detail": "opis", "error": "NazwaWyjątku"}`.

| Kod | Kiedy |
|---|---|
| `400` | zły parametr: nieznana strategia lub temat, zły format sesji, prompt nie mieści się w `LLM_NUM_CTX` (zmniejsz `k` / `neighbors`) |
| `404` | nie ma kolekcji (najpierw `/ingest`), pliku na serwerze albo sesji w kolekcji (benchmark) |
| `422` | błędny JSON zapytania: brak wymaganego pola, zły typ, brak `paths`/`sessions` itp. (szczegóły w `detail`) |
| `502` | Ollama albo Qdrant nie odpowiada lub zwrócił błąd (sprawdź `GET /health`) |

---

## 10. Typowy przebieg

```bash
API=localhost:8000; H='Content-Type: application/json'

# 1. czy wszystko działa
curl $API/health

# 2. wgraj sesję z tematami do kolekcji turns8
curl -X POST $API/ingest -H "$H" -d '{"paths": ["data/annomi_64.json"], "strategies": ["turns8"], "topics": true}'

# 3. sprawdź, co znajduje wyszukiwanie
curl -X POST $API/search -H "$H" -d '{"query": "friends", "strategy": "turns8", "session": "64", "topic": "social"}'

# 4. zapytaj LLM
curl -X POST $API/ask-session -H "$H" \
  -d '{"session_id": "64", "skip_ingest": true, "question": "What does he say about his friends?", "topics": true}'

# 5. benchmark do Excela
curl -X POST $API/benchmark-session -H "$H" \
  -d '{"session": "64", "benchmark_path": "data/benchamrk_064.json", "xlsx": true}' -o benchmark_64.xlsx
```
