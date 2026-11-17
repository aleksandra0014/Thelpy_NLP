"""Modele zapytań i odpowiedzi API (odpowiedniki argumentów ze scripts/)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from thelpy_rag import config

Mode = Literal["dense", "bm25", "hybrid"]
Fusion = Literal["client", "server"]

STRATEGY_HELP = (f"nazwa strategii: {', '.join(config.STRATEGIES)} "
                 f"albo własna w formacie turnsN[_ovM] / fixedN[_ovM]")
TOPIC_HELP = ("szukaj tylko w chunkach z tym tematem albo podtematem (klucz z thelpy_topics/taxonomy.py, "
              "np. social, friends, alcohol; kolekcja musi być wgrana z topics=true)")


# --- kolekcje ----------------------------------------------------------------
class CreateCollectionsRequest(BaseModel):
    """scripts/create_collections.py"""
    strategies: list[str] = Field(default_factory=lambda: list(config.STRATEGIES), description=STRATEGY_HELP)
    recreate: bool = Field(False, description="UWAGA: usuwa i tworzy kolekcje od nowa (kasuje dane)")
    dim: Optional[int] = Field(None, description="wymiar wektora (domyślnie odczytany z modelu)")


# --- ingest ------------------------------------------------------------------
class IngestRequest(BaseModel):
    """scripts/ingest.py — sesje jako ścieżki na serwerze i/lub JSON-y wprost w zapytaniu."""
    paths: list[str] = Field(default_factory=list, description="pliki .json lub katalogi na serwerze")
    sessions: list[dict[str, Any]] = Field(default_factory=list,
                                           description="sesje w formacie thelpy_rag/loader.py")
    strategies: list[str] = Field(default_factory=lambda: list(config.STRATEGIES), description=STRATEGY_HELP)
    dry_run: bool = Field(False, description="tylko statystyki chunków, bez zapisu")
    show: int = Field(0, ge=0, description="przy dry_run: zwróć N pierwszych chunków każdej strategii")
    topics: bool = Field(False, description="dodaj tematy do chunków (anotacja do data/with_topics/)")

    @model_validator(mode="after")
    def _need_input(self):
        if not self.paths and not self.sessions:
            raise ValueError("Podaj paths albo sessions")
        return self


# --- search / ask ------------------------------------------------------------
class SearchRequest(BaseModel):
    """scripts/search.py"""
    query: str
    strategy: str = Field("turns4", description=STRATEGY_HELP)
    mode: Mode = "hybrid"
    k: int = Field(5, ge=1, le=100)
    fusion: Fusion = Field("client", description="hybrid: RRF liczone lokalnie albo w Qdrancie")
    session: Optional[str] = None
    client: Optional[str] = None
    role: Optional[str] = None
    date_from: Optional[str] = Field(None, description="ISO data/czas, filtr session_at >=")
    date_to: Optional[str] = Field(None, description="ISO data/czas, filtr session_at <=")
    neighbors: int = Field(0, ge=0, description="dołącz N sąsiednich chunków z każdej strony")
    topic: Optional[str] = Field(None, description=TOPIC_HELP)


THINK_HELP = ("tryb rozumowania qwena (think): zwykle lepsze odpowiedzi, ale wolniej i więcej tokenów; "
              "rezerwa odpowiedzi rośnie o LLM_THINK_RESERVE, myślenie zwracane w polu thinking")
LANGUAGE_HELP = ('język odpowiedzi: kod ("pl", "en", "de", ...) albo nazwa ("Polish"); puste = bez wymuszania '
                 '(model zwykle odpowiada w języku pytania / instrukcji)')


def _blank_to_none(v):
    """Swagger (/docs) wysyła puste "" i {} dla nieużywanych pól — traktuj je jak niepodane."""
    return None if v in ("", {}) or (isinstance(v, str) and not v.strip()) else v


class SessionSource(BaseModel):
    """Wspólne wejście sesji dla /ask, /summarize-session i /session-keywords.

    - file albo session + skip_ingest=false (domyślnie): sesja jest wgrywana do kolekcji strategii,
      a jej session_id staje się filtrem,
    - skip_ingest=true albo brak file/session: nic nie jest wgrywane, praca po filtrach (session_id, ...).
    """
    file: Optional[str] = Field(None, description="ścieżka pliku .json sesji na serwerze")
    session: Optional[dict[str, Any]] = Field(None, description="sesja JSON wprost w zapytaniu")
    skip_ingest: bool = Field(False, description="true = nie wgrywaj file / session, pracuj na tym, co już jest "
                                                 "w kolekcji (session_id brany z pliku)")
    session_id: Optional[str] = Field(None, description="filtr: jedna sesja (przy file / session brany z pliku)")
    strategy: str = Field("turns8", description=STRATEGY_HELP)
    topics: bool = Field(False, description="wgraj sesję z tematami i pokaż tematy LLM-owi")
    topic: Optional[str] = Field(None, description=TOPIC_HELP + "; włącza topics")

    @field_validator("file", "session", "session_id", "topic", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return _blank_to_none(v)

    @model_validator(mode="after")
    def _one_source(self):
        if self.file and self.session:
            raise ValueError("Podaj file albo session, nie oba")
        return self

    @property
    def has_source(self) -> bool:
        return bool(self.file or self.session)


class SingleSessionSource(SessionSource):
    """Operacje na jednej sesji (podsumowanie, keywordy): potrzebne session_id albo file / session."""

    @model_validator(mode="after")
    def _need_session(self):
        if not (self.has_source or self.session_id):
            raise ValueError("Podaj session_id (sesja już wgrana) albo file / session")
        return self


class AskRequest(SessionSource):
    """Pytanie + odpowiedź LLM ze źródłami: opcjonalne wgranie sesji, potem wyszukanie po filtrach."""
    question: str
    client_id: Optional[str] = Field(None, description="filtr: wszystkie sesje jednego klienta")
    role: Optional[str] = Field(None, description="filtr: chunki z wypowiedzią tej roli (np. client)")
    date_from: Optional[str] = Field(None, description="ISO data/czas, filtr session_at >=")
    date_to: Optional[str] = Field(None, description="ISO data/czas, filtr session_at <=")
    mode: Mode = "dense"
    k: int = Field(5, ge=1, le=100, description="liczba trafień (fragmentów) przekazanych do LLM")
    fusion: Fusion = Field("client", description="hybrid: RRF liczone lokalnie albo w Qdrancie")
    neighbors: int = Field(0, ge=0, description="dołącz N sąsiednich chunków z każdej strony trafienia")
    model: str = config.LLM_MODEL
    language: Optional[str] = Field(None, description=LANGUAGE_HELP)
    think: bool = Field(False, description=THINK_HELP)
    include_prompt: bool = Field(False, description="zwróć pełny prompt (messages)")

    @field_validator("client_id", "role", "date_from", "date_to", "language", mode="before")
    @classmethod
    def _empty_filter(cls, v):
        return _blank_to_none(v)


class SummarizeSessionRequest(SingleSessionSource):
    """scripts/summarize_session.py — podsumowanie całej sesji (map-reduce po chunkach w kolejności)."""
    model: str = config.LLM_MODEL
    length: Literal["full", "brief", "all"] = Field(
        "full", description="full: Markdown w 5 sekcjach (~600 słów) z odwołaniami do tur; brief: 3-5 zdań "
                            "(~80 słów); all: oba — map-reduce raz, osobny finał na wariant (pole summaries)")
    focus: Optional[str] = Field(None, description="na czym skupić podsumowanie, np. \"client's motivation\"")
    language: Optional[str] = Field(None, description=LANGUAGE_HELP)
    think: bool = Field(False, description=THINK_HELP)
    max_block_tokens: Optional[int] = Field(None, ge=600, description="zmniejsza blok map (wymusza map-reduce); "
                                            "zwykle zostaw puste, do testów np. 1500")
    include_steps: bool = Field(True, description="zwróć streszczenia cząstkowe z każdego poziomu")

    @field_validator("focus", "language", mode="before")
    @classmethod
    def _empty_focus(cls, v):
        return None if isinstance(v, str) and not v.strip() else v


class SessionKeywordsRequest(SingleSessionSource):
    """scripts/session_keywords.py — keywordy sesji bez LLM: hasła ze słownika + wolne frazy (KeyBERT-style)."""
    top_n: int = Field(10, ge=1, le=200, description="ile haseł ze słownika zwrócić")
    free_top_n: int = Field(10, ge=0, le=100, description="ile wolnych fraz zwrócić (0 = pomiń, szybciej)")
    diversity: float = Field(0.5, ge=0.0, le=1.0, description="MMR dla wolnych fraz: 0 = sama trafność, "
                                                                "1 = maksymalna różnorodność")
    client_only: bool = Field(True, description="wolne frazy tylko z wypowiedzi klienta")
    vocab_file: Optional[str] = Field(None, description="własny słownik .yaml na serwerze "
                                                        "(domyślnie KEYWORDS_FILE = data/keywords.yaml)")
    background: bool = Field(True, description="normalizuj na tle chunków całej kolekcji (inne sesje) — "
                                               "obniża hasła i frazy typowe dla każdej rozmowy")

    @field_validator("vocab_file", mode="before")
    @classmethod
    def _empty_vocab(cls, v):
        return None if isinstance(v, str) and not v.strip() else v


# --- baza wiedzy (thelpy_rag/kb.py) --------------------------------------------
class KbSectionIn(BaseModel):
    title: Optional[str] = Field(None, description="tytuł sekcji (doklejany do tekstu przy embeddingu)")
    text: str = Field(..., min_length=1, description="treść sekcji; akapity oddzielone pustą linią")

    @field_validator("title", mode="before")
    @classmethod
    def _empty_title(cls, v):
        return _blank_to_none(v)


class KbDocumentIn(BaseModel):
    title: str = Field(..., min_length=1)
    source_url: Optional[str] = Field(None, description="adres źródła (URL, DOI, ISBN, ...)")
    doc_name: Optional[str] = Field(None, description="identyfikator dokumentu do filtrowania i nadpisywania; "
                                                      "domyślnie title")
    sections: list[KbSectionIn] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict, description="dodatkowe pola payloadu (np. author, year)")

    @field_validator("source_url", "doc_name", mode="before")
    @classmethod
    def _empty_fields(cls, v):
        return _blank_to_none(v)


class KbIngestRequest(BaseModel):
    """Wgrywa dokumenty do kolekcji bazy wiedzy (KB_COLLECTION). Ponowne wgranie dokumentu o tym samym
    doc_name zastępuje wszystkie jego stare fragmenty."""
    documents: list[KbDocumentIn] = Field(..., min_length=1)
    max_tokens: int = Field(config.KB_CHUNK_TOKENS, ge=32, le=8192,
                            description="maks. tokenów w chunku; akapity sklejane do tej granicy")
    overlap: int = Field(config.KB_CHUNK_OVERLAP, ge=0,
                         description="overlap (tokeny) przy cięciu akapitu dłuższego niż max_tokens")
    recreate: bool = Field(False, description="UWAGA: usuwa całą kolekcję bazy wiedzy i tworzy ją od nowa")
    dry_run: bool = Field(False, description="tylko podział na chunki, bez embeddingów i zapisu")
    show: int = Field(0, ge=0, description="przy dry_run: zwróć N pierwszych chunków każdego dokumentu")

    @model_validator(mode="after")
    def _overlap_lt_max(self):
        if self.overlap >= self.max_tokens:
            raise ValueError("overlap musi być mniejszy niż max_tokens")
        return self


SCORE_THRESHOLD_HELP = ("odrzuć trafienia ze score poniżej progu; skala zależy od trybu: dense = cosinus 0–1 "
                        "(np. 0.5), bm25 = suma wag BM25×IDF (bez górnej granicy), hybrid = RRF Σ 1/(RRF_K+rank) "
                        "(maks. ≈ 0.033 przy RRF_K=60)")


class KbSearchRequest(BaseModel):
    """Wyszukiwanie w bazie wiedzy (ten sam Retriever co /search)."""
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=100)
    mode: Mode = "hybrid"
    fusion: Fusion = Field("client", description="hybrid: RRF liczone lokalnie albo w Qdrancie")
    documents: Optional[list[str]] = Field(None, description="szukaj tylko w tych dokumentach (doc_name); "
                                                             "lista: GET /kb/documents")
    score_threshold: Optional[float] = Field(None, description=SCORE_THRESHOLD_HELP)

    @field_validator("documents", mode="before")
    @classmethod
    def _empty_docs(cls, v):
        if isinstance(v, list):
            v = [d for d in v if isinstance(d, str) and d.strip()] or None
        return _blank_to_none(v)


# --- benchmark / evaluate ----------------------------------------------------
class BenchmarkRequest(BaseModel):
    """scripts/benchmark_session.py"""
    session: str = Field(..., description="session_id w Qdrancie, np. 64")
    benchmark_path: Optional[str] = Field(None, description="plik JSON benchmarku na serwerze")
    benchmark: Optional[dict[str, Any]] = Field(None, description="benchmark JSON (sekcja 'questions')")
    strategy: str = Field("turns8", description=STRATEGY_HELP)
    mode: Mode = "dense"
    k: int = Field(5, ge=1, le=100)
    context_order: Literal["score", "chrono"] = "score"
    model: str = config.LLM_MODEL
    topics: bool = Field(False, description="pokaż LLM-owi tematy każdego fragmentu")
    topic: Optional[str] = Field(None, description=TOPIC_HELP + "; włącza topics")
    out: Optional[str] = Field(None, description="zapisz też Excela na serwerze pod tą ścieżką")
    xlsx: bool = Field(False, description="zamiast JSON-a zwróć plik .xlsx do pobrania")

    @model_validator(mode="after")
    def _need_benchmark(self):
        if not self.benchmark_path and not self.benchmark:
            raise ValueError("Podaj benchmark_path albo benchmark")
        return self


class EvaluateRequest(BaseModel):
    """scripts/evaluate.py"""
    queries_path: Optional[str] = Field(None, description="plik JSONL z pytaniami na serwerze")
    queries: list[dict[str, Any]] = Field(default_factory=list,
                                          description='[{"qid", "query", "session_id", "relevant_turn_ids"}]')
    strategies: list[str] = Field(default_factory=lambda: list(config.STRATEGIES), description=STRATEGY_HELP)
    modes: list[Mode] = Field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    k: int = Field(5, ge=1, le=100)
    scope: Literal["all", "session", "client"] = "all"

    @model_validator(mode="after")
    def _need_queries(self):
        if not self.queries_path and not self.queries:
            raise ValueError("Podaj queries_path albo queries")
        return self
