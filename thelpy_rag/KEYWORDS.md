# Keywordy sesji: jak powstają

Moduł [`thelpy_rag/keywords.py`](keywords.py), endpoint `POST /session-keywords`
([`thelpy_api/service.py`](../thelpy_api/service.py)), skrypt `scripts/session_keywords.py`.
Nie ma tu LLM. Całość trwa kilka sekund. Wyjątkiem jest pierwsze użycie słownika, kiedy liczą się
embeddingi haseł (ok. 30 s), a potem są brane z cache.

```bash
python -m scripts.session_keywords data/annomi_64.json --skip-ingest
curl -X POST localhost:8000/session-keywords -H 'Content-Type: application/json' -d '{"session_id": "64"}'
```

## Spis treści
1. [Dwie metody w skrócie](#1-dwie-metody-w-skrócie)
2. [Skąd są wektory](#2-skąd-są-wektory)
3. [Metoda A: słownik](#3-metoda-a-słownik)
4. [Metoda B: wolne frazy (KeyBERT-style)](#4-metoda-b-wolne-frazy-keybert-style)
5. [Wynik](#5-wynik)
6. [Jak rozbudować słownik](#6-jak-rozbudować-słownik)
7. [Parametry](#7-parametry)
8. [Ograniczenia](#8-ograniczenia)

---

## 1. Dwie metody w skrócie

| | A. słownik (`vocab`) | B. wolne frazy (`free`) |
|---|---|---|
| skąd hasła | z pliku [`data/keywords.yaml`](../data/keywords.yaml), np. matka, ojciec, praca, alkohol | z samego transkryptu (n-gramy z wypowiedzi klienta) |
| co mówi | które **z Twoich haseł** pojawiają się w sesji, jak mocno i w których turach | o czym była sesja **jej własnymi słowami**, także o rzeczach spoza słownika |
| porównywalność między sesjami | tak, zawsze ten sam zestaw haseł | nie, każda sesja ma inne frazy |
| przykład (sesja 64) | narkotyki, życie towarzyskie, pieniądze, praca, przyjaciele | scene, drugs, used money, palmy, party |

Obie metody się uzupełniają. Słownik daje stałe, porównywalne etykiety. Wolne frazy pokazują,
czego w słowniku brakuje, i to dobra podpowiedź, co do niego dopisać.

---

## 2. Skąd są wektory

**Embedding** to wektor liczb (dla bge-m3 jest ich 1024), który opisuje *znaczenie* tekstu.
Teksty o podobnym znaczeniu mają podobne wektory. Podobieństwo mierzymy **cosinusem**: im bliżej 1,
tym bardziej podobne. bge-m3 jest wielojęzyczny, więc polskie „ojciec” wypada blisko angielskiego
„my dad never calls”.

Przy ingeście każdy chunk (np. 8 kolejnych tur) dostał embedding, który jest zapisany w Qdrant.
Endpoint pobiera **wszystkie** chunki sesji **razem z wektorami** (`fetch_chunks(..., with_vectors=True)`),
więc **sesji nie embedujemy ponownie**. Embedujemy tylko hasła słownika (raz, z cache) i kandydatów
na wolne frazy.

Dodatkowo pobierane jest **tło**: wektory chunków całej kolekcji, czyli wszystkich wgranych sesji
(`background_vectors`, maks. 2000). Tło służy do pytania „czy to hasło jest tu ważniejsze **niż zwykle**?”.

---

## 3. Metoda A: słownik

### 3.1 Hasło w słowniku

```yaml
rodzina:
  ojciec: {label: father, aliases: [father, dad, daddy, papa, my father, tata, ojciec]}
uzależnienia behawioralne:
  hazard: {label: gambling, aliases: [gamble, gambling, bet, casino, poker, ...],
           desc: "the client gambles and loses money on bets, casino or poker machines"}
```

- **klucz** (`ojciec`) to nazwa zwracana w wyniku,
- **`label`** to krótka nazwa po angielsku (transkrypty są po angielsku),
- **`aliases`** to słowa i frazy szukane **dosłownie**,
- **`desc`** (opcjonalnie) to zdanie opisujące temat. Pomaga, gdy temat jest abstrakcyjny
  (np. „ambiwalencja”), bo samo słowo słabo oddaje znaczenie.

### 3.2 Sygnał 1: trafienia dosłowne (leksykalne)

Czy słowo albo alias **pada w tekście**?

1. Z tekstu chunka usuwane są etykiety mówiących (`Therapist:` / `Client:`), bo inaczej alias
   „therapist” trafiałby w każdej linii.
2. Tekst i aliasy są tokenizowane i **stemowane** (Snowball, jak w BM25), więc „drinks”, „drinking”
   i „drink” to to samo, a „mothers” to „mother”.
3. Alias wielowyrazowy („split up”, „my father”) musi wystąpić jako ciągła fraza.
4. `lexical_hits` to łączna liczba trafień wszystkich aliasów hasła w sesji (u obu mówiących).

Dla **konkretnych słów** (mama, tata, piwo, cukrzyca) to najpewniejszy sygnał. Embedding jednego
krótkiego słowa jest dość nieprecyzyjny.

### 3.3 Sygnał 2: podobieństwo znaczeniowe (embedding)

Ten sygnał łapie temat, **nawet gdy słowo nie pada**: „he never showed up for my birthday” jest
o ojcu i relacjach, choć słowa „father” w tekście nie ma.

**Wektor hasła**: embedujemy `label`, zdanie `"The client talks about <label>."`, każdy alias
i `desc`, a potem uśredniamy wszystko w jeden wektor. Wynik jest w cache: w pamięci oraz w
`data/keyword_cache/vocab_<hash>.npy`. Hash liczony jest z treści słownika i modelu, więc po zmianie
słownika liczy się automatycznie od nowa.

**Podobieństwo**: liczymy cosinus każdego hasła z każdym chunkiem sesji, co daje macierz
*chunki × hasła* (33 × 158 dla sesji 64).

**Problem**: surowe cosinusy są mało czytelne. W eksperymencie na sesji 64 wszystkie hasła wypadały
między 0.40 a 0.59, a niektóre („anxiety”) są wysoko przy **każdym** chunku terapii, bo to hasło
pasuje do rozmowy terapeutycznej w ogóle. Dlatego wartości są normalizowane w dwóch krokach
(`semantic_scores`):

1. **W wierszu (chunk)**: z-score po wszystkich hasłach, czyli
   `(cos − średnia cos tego chunka) / odchylenie`. Wynik mówi, które hasła **wyróżniają się** dla
   tego chunka na tle pozostałych haseł. Np. „narkotyki = 2.5” oznacza 2.5 odchylenia
   standardowego powyżej przeciętnego hasła.
2. **W kolumnie (hasło)**: od wyniku odejmujemy średni wynik tego hasła **na tle** (chunki całej
   kolekcji). Hasło typowe dla każdej sesji spada wtedy do ok. 0, a hasło charakterystyczne dla
   tej sesji zostaje wysoko.

**Agregacja po sesji**: `semantic` = średnia z **najlepszych ~15% chunków** (dla 33 chunków to 5).
- sam max złapałby jedną przypadkową wzmiankę,
- średnia ze wszystkich chunków rozmyłaby temat, który zajmuje tylko część sesji
  (np. 10 minut o ojcu w godzinnej rozmowie o alkoholu).

### 3.4 Wynik końcowy hasła

```
score = semantic + KW_LEX_WEIGHT × ln(1 + lexical_hits)          (KW_LEX_WEIGHT = 0.5)
```

Logarytm sprawia, że pierwsze trafienia znaczą dużo, a kolejne coraz mniej:
1 trafienie daje +0.35, 10 daje +1.2, a 50 daje +2.0. Hasło powtórzone 50 razy nie dominuje
całego rankingu.

Dla każdego hasła liczymy też:
- **`coverage`**: odsetek chunków, w których hasło „jest”, czyli ma trafienie dosłowne **albo**
  semantyczny z-score ≥ `KW_CHUNK_Z` (1.5),
- **`turns`**: zakresy tur tych chunków, sąsiednie sklejone (`T097–T120`). Dzięki temu łatwo
  sprawdzić w transkrypcie, skąd wynik się wziął.

Zwracamy `top_n` haseł z najwyższym `score`.

### 3.5 Przykład: sesja 64

| hasło | score | semantic | trafienia | pokrycie | dlaczego |
|---|---|---|---|---|---|
| narkotyki | 3.90 | 2.29 | 24 | 46% | „meth” 13×, „drugs” 11×, a temat ciągnie się przez pół sesji |
| życie towarzyskie | 3.86 | 2.67 | 10 | 18% | „party scene”, wychodzenie, imprezy |
| pieniądze | 3.22 | 1.81 | 16 | 18% | „money” 13×, w turach T137–T168 i T201–T216 |
| interwencja bliskich | 2.21 | 1.40 | 4 | 15% | przyjaciele „sat me down”; mało słów, ale wysoki sygnał znaczeniowy |

---

## 4. Metoda B: wolne frazy (KeyBERT-style)

To algorytm [KeyBERT](https://github.com/MaartenGr/KeyBERT) napisany w kilkudziesięciu linijkach.
Paczka `keybert` nie jest używana, bo ciągnęłaby własny model, a ten dawałby inną przestrzeń
wektorów niż chunki w Qdrant. Tu wszystko liczy ten sam bge-m3.

1. **Kandydaci**: z wypowiedzi **klienta** (`client_only`, bo terapeuta wnosi frazy typu
   „how do you feel about that”) wyciągamy frazy 1–3-wyrazowe (sklearn `CountVectorizer`), bez
   stopwords i wtrąceń (`yeah`, `mm-hmm`, `suppose`, `stuff`…). Zostają frazy, które padły
   ≥ 2 razy, maks. 300 najczęstszych.
2. **Wektor sesji**: średnia wektorów wszystkich chunków, czyli „o czym jest cała sesja”.
3. **Trafność frazy** = `cos(fraza, sesja) − cos(fraza, tło)`. Odjęcie tła działa jak w metodzie A:
   fraza pasująca do *każdej* rozmowy („talk”, „feel good”) spada, a fraza charakterystyczna
   dla tej sesji („scene”, „insulin”) idzie w górę. Bez innych sesji w kolekcji ten krok jest
   pomijany.
4. **MMR (Maximal Marginal Relevance)**: wybieramy frazy po kolei. Każda następna to ta, która jest
   **trafna**, ale **niepodobna do już wybranych**:
   ```
   MMR = (1 − diversity) × trafność − diversity × max podobieństwo do wybranych
   ```
   Bez MMR top-10 to byłyby „diabetes”, „diabetes feel”, „diagnosed diabetes”, „diabetic”…
   `diversity = 0` daje czystą trafność, a `1` maksymalną różnorodność (domyślnie 0.5).

`relevance` w wyniku to trafność z kroku 3. Liczy się kolejność, nie wartość bezwzględna.
Wartości są małe (0.01–0.03), bo to różnica dwóch cosinusów.

Przykład (sesja 121, cukrzyca): *diabetes, cut sugar coffee, breakfast, diagnosed diabetes, insulin,
milk sugar, form exercise, sugars, hereditary*.

---

## 5. Wynik

```json
{
  "session_id": "64", "collection": "thelpy_turns8", "ingested": false, "topic": null, "n_chunks": 33,
  "vocab_size": 158, "background_chunks": 108,
  "vocab": [
    {"keyword": "narkotyki", "label": "illegal drugs", "category": "substancje", "score": 3.9,
     "semantic": 2.29, "lexical_hits": 24, "coverage": 0.46,
     "turns": ["T009–T016", "T049–T056", "T065–T072", "T097–T120"]}
  ],
  "free": [{"phrase": "scene", "count": 9, "relevance": 0.023}]
}
```

| Pole | Znaczenie |
|---|---|
| `score` | wynik końcowy (semantic + bonus za trafienia), według niego jest ranking |
| `semantic` | znormalizowane podobieństwo znaczeniowe (w odchyleniach std, ok. 0 = „jak zwykle”) |
| `lexical_hits` | ile razy hasło/alias padło dosłownie |
| `coverage` | odsetek chunków sesji, w których hasło występuje |
| `turns` | gdzie w sesji, zakresy tur |
| `count` (free) | ile razy fraza padła u klienta |
| `background_chunks` | z ilu chunków policzono tło (0 = bez normalizacji na tle) |

---

## 6. Jak rozbudować słownik

- Dopisz linijkę do [`data/keywords.yaml`](../data/keywords.yaml). Kod się nie zmienia, a cache
  embeddingów przeliczy się sam.
- Aliasy: wypisz **konkretne** formy, które padają w rozmowie (mum, mom, mummy). Unikaj słów zbyt
  ogólnych (`stop`, `less`, `looks`, `plan`), bo dają fałszywe trafienia dosłowne.
- Hasło abstrakcyjne (ambiwalencja, opór, motywacja) wymaga dobrego `desc`. Wtedy przeważa sygnał
  znaczeniowy.
- Podpowiedzi, czego brakuje, dają wolne frazy z kilku sesji.
- Własny słownik bez podmieniania domyślnego: `vocab_file` w API, `--vocab` w skrypcie, albo
  `KEYWORDS_FILE` w `.env`.

---

## 7. Parametry

| Parametr | Domyślnie | Gdzie | Znaczenie |
|---|---|---|---|
| `top_n` | 10 | API / `--top-n` | liczba haseł ze słownika |
| `free_top_n` | 10 | API / `--free-top-n` | liczba wolnych fraz (0 = pomiń) |
| `diversity` | 0.5 | API / `--diversity` | MMR |
| `client_only` | true | API / `--all-speakers` | wolne frazy tylko od klienta |
| `background` | true | API / `--no-background` | normalizacja na tle kolekcji |
| `KEYWORDS_FILE` | `data/keywords.yaml` | `.env` | słownik |
| `KW_CHUNK_Z` | 1.5 | `.env` | od jakiego z-score chunk „zawiera” hasło (`coverage`, `turns`) |
| `KW_LEX_WEIGHT` | 0.5 | `.env` | waga trafień dosłownych |
| `top_frac` | 0.15 | kod | część najlepszych chunków uśrednianych w `semantic` |

---

## 8. Ograniczenia

- **Tło zależy od kolekcji.** Z jedną sesją w kolekcji tło to ona sama, więc krok „ile bardziej
  niż zwykle” niewiele daje (wolne frazy go wtedy pomijają). Im więcej sesji, tym lepsza
  normalizacja. Przy niewielu sesjach `background: false` może dać czytelniejszy ranking.
- **Negacja**: „I don't drink at all” daje trafienie dla alkoholu. Słownik mówi, *o czym* była
  mowa, a nie *co* klient o tym myśli.
- **Język**: trafienia dosłowne działają dla aliasów w języku transkryptu. Stemmer to
  `BM25_STEMMER` (domyślnie english). Sygnał znaczeniowy jest wielojęzyczny.
- **Progi** (`KW_CHUNK_Z`, `KW_LEX_WEIGHT`) ustawione na 2 sesjach AnnoMI. Warto je sprawdzić na
  większej liczbie sesji.
