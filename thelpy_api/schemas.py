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


class AskRequest(BaseModel):
    """scripts/ask.py — wyszukanie + odpowiedź LLM."""
    question: str
    strategy: str = Field("turns4", description=STRATEGY_HELP)
    mode: Mode = "hybrid"
    k: int = Field(5, ge=1, le=100)
    fusion: Fusion = "client"
    session: Optional[str] = None
    client: Optional[str] = None
    neighbors: int = Field(0, ge=0)
    model: str = config.LLM_MODEL
    topics: bool = Field(False, description="pokaż LLM-owi tematy każdego fragmentu")
    topic: Optional[str] = Field(None, description=TOPIC_HELP + "; włącza topics")
    include_prompt: bool = Field(False, description="zwróć pełny prompt (messages)")


class SessionInput(BaseModel):
    """Wspólne wejście dla operacji na jednej sesji (ask-session, summarize-session)."""
    file: Optional[str] = Field(None, description="ścieżka pliku .json sesji na serwerze")
    session: Optional[dict[str, Any]] = Field(None, description="sesja JSON wprost w zapytaniu")
    session_id: Optional[str] = Field(None, description="sesja już wgrana: samo session_id (bez file/session) "
                                      "bierze gotowe chunki z kolekcji, bez wgrywania")
    n_turns: int = Field(8, ge=1, description="liczba tur w chunku")
    overlap_turns: int = Field(0, ge=0)
    model: str = config.LLM_MODEL
    skip_ingest: bool = Field(False, description="przy file / session: nie wgrywaj ponownie (sesja już jest w "
                                                 "kolekcji); przy samym session_id bez znaczenia")
    topics: bool = Field(False, description="wgraj sesję z tematami i pokaż tematy LLM-owi")
    topic: Optional[str] = Field(None, description=TOPIC_HELP + "; włącza topics")

    @field_validator("file", "session", "session_id", "topic", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        """Swagger (/docs) wysyła puste "" i {} dla nieużywanych pól — traktuj je jak niepodane."""
        return None if v in ("", {}) or (isinstance(v, str) and not v.strip()) else v

    @model_validator(mode="after")
    def _need_session(self):
        if not (self.file or self.session or self.session_id):
            raise ValueError("Podaj session_id (sesja już wgrana) albo file / session (wgranie i użycie)")
        return self

    @property
    def uses_existing(self) -> bool:
        """Samo session_id (bez file/session): użyj chunków już wgranych do kolekcji, bez ingestu."""
        return not (self.file or self.session)


class AskSessionRequest(SessionInput):
    """scripts/ask_session.py — (wgranie) jednej sesji, dense, odpowiedź LLM z diagnostyką."""
    question: str
    k: int = Field(5, ge=1, le=100, description="liczba fragmentów przekazanych do LLM")


class SummarizeSessionRequest(SessionInput):
    """scripts/summarize_session.py — podsumowanie całej sesji (map-reduce po chunkach w kolejności)."""
    length: Literal["full", "brief"] = Field("full", description="full: Markdown w 5 sekcjach (~600 słów) "
                                             "z odwołaniami do tur; brief: 3-5 zdań (~80 słów)")
    focus: Optional[str] = Field(None, description="na czym skupić podsumowanie, np. \"client's motivation\"")
    max_block_tokens: Optional[int] = Field(None, ge=600, description="zmniejsza blok map (wymusza map-reduce); "
                                            "zwykle zostaw puste, do testów np. 1500")
    include_steps: bool = Field(True, description="zwróć streszczenia cząstkowe z każdego poziomu")

    @field_validator("focus", mode="before")
    @classmethod
    def _empty_focus(cls, v):
        return None if isinstance(v, str) and not v.strip() else v


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
