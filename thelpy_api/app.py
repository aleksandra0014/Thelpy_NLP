"""Serwer FastAPI dla Thelpy RAG — endpointy odpowiadają skryptom z scripts/.

    uvicorn thelpy_api.app:app --host 127.0.0.1 --port 8000
    python -m thelpy_api                       # to samo, port z THELPY_API_PORT (domyślnie 8000)

Dokumentacja i formularze do testowania: http://127.0.0.1:8000/docs

Endpointy przyjmują ścieżki plików na serwerze (np. data/annomi_64.json) — wystawiaj serwer
tylko na localhost albo za uwierzytelnianiem.
"""
from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import Field, create_model
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from starlette.background import BackgroundTask

from thelpy_rag import config

from . import schemas, service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.res = service.Resources.create()
    yield
    app.state.res.client.close()


app = FastAPI(title="Thelpy RAG API", version="0.1.0", lifespan=lifespan,
              description="Wyszukiwanie i odpowiedzi LLM na transkrypcjach sesji terapeutycznych "
                          "(Qdrant + Ollama), z tematami z thelpy_topics.")


def _openapi_with_file_inputs():
    """FastAPI generuje OpenAPI 3.1, gdzie plik to {"type": "string", "contentMediaType": ...}.
    Swagger UI (/docs) rozpoznaje pole pliku tylko po "format": "binary" — bez tego list[UploadFile]
    wyświetla się jako zwykłe pole tekstowe. Dopisujemy format do takich schematów."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
                node["format"] = "binary"
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_file_inputs


def res(request: Request) -> service.Resources:
    return request.app.state.res


# --- błędy -> kody HTTP --------------------------------------------------------
def _error(status: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": str(exc), "error": type(exc).__name__})


@app.exception_handler(ValueError)          # w tym PromptTooLongError, błędy formatu sesji
async def _value_error(_, exc):
    return _error(400, exc)


@app.exception_handler(LookupError)
async def _lookup_error(_, exc):
    return _error(404, exc)


@app.exception_handler(FileNotFoundError)
async def _not_found(_, exc):
    return _error(404, exc)


@app.exception_handler(requests.RequestException)
async def _ollama_error(_, exc):
    return _error(502, RuntimeError(f"Ollama ({config.OLLAMA_URL}): {exc}"))


@app.exception_handler(UnexpectedResponse)
@app.exception_handler(ResponseHandlingException)
async def _qdrant_error(_, exc):
    return _error(502, RuntimeError(f"Qdrant ({config.QDRANT_URL}): {exc}"))


# --- informacje ------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["info"])
def health(request: Request):
    """Czy Qdrant i Ollama odpowiadają."""
    out = {}
    try:
        res(request).client.get_collections()
        out["qdrant"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["qdrant"] = f"błąd: {e}"
    try:
        requests.get(f"{config.OLLAMA_URL.rstrip('/')}/api/tags", timeout=5).raise_for_status()
        out["ollama"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["ollama"] = f"błąd: {e}"
    out["status"] = "ok" if all(v == "ok" for v in out.values()) else "degraded"
    return out


@app.get("/config", tags=["info"])
def get_config():
    """Aktywna konfiguracja (bez klucza API) i zarejestrowane strategie chunkingu."""
    return {"summary": config.summary(), "llm_model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL,
            "strategies": config.STRATEGIES}


@app.get("/topics", tags=["info"])
def topics():
    """Klucze tematów i podtematów (do parametru `topic`)."""
    return service.list_topics()


# --- kolekcje (scripts/create_collections.py) --------------------------------------
@app.get("/collections", tags=["collections"])
def list_collections(request: Request):
    return service.list_collections(res(request))


@app.post("/collections", tags=["collections"])
def create_collections(body: schemas.CreateCollectionsRequest, request: Request):
    return service.create_collections(body, res(request))


@app.get("/collections/{strategy}/sessions", tags=["collections"])
def list_sessions(strategy: str, request: Request, client_id: Optional[str] = None):
    """Metadane wszystkich sesji w kolekcji strategii (np. turns8): ID, pacjent, daty, czas trwania,
    liczba chunków / tur / tokenów, zakres tur, czy wgrana z tematami. `client_id` zawęża do pacjenta."""
    return service.list_sessions(strategy, res(request), client_id)


@app.delete("/collections/{strategy}/sessions/{session_id}", tags=["collections"])
def delete_session(strategy: str, session_id: str, request: Request):
    """Usuwa wszystkie chunki jednej sesji z kolekcji strategii."""
    return service.remove_session(strategy, session_id, res(request))


# --- ingest (scripts/ingest.py) -------------------------------------------------
@app.post("/ingest", tags=["ingest"])
def ingest(body: schemas.IngestRequest, request: Request):
    """Wgrywa sesje (ścieżki na serwerze i/lub JSON-y w `sessions`) do kolekcji wybranych strategii.
    Z `topics=true` sesje są najpierw anotowane tematami (pierwsze wywołanie ładuje model tematów)."""
    return service.ingest(body, res(request))


def read_json_upload(f: UploadFile) -> dict:
    try:
        return json.loads(f.file.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(400, f"{f.filename}: niepoprawny JSON ({e})")


@app.post("/ingest/upload", tags=["ingest"])
def ingest_upload(request: Request, files: list[UploadFile] = File(..., description="pliki .json sesji"),
                  strategies: Optional[list[str]] = Form(None, description="puste = wszystkie; "
                                                         + schemas.STRATEGY_HELP),
                  topics: bool = Form(False), dry_run: bool = Form(False), show: int = Form(0)):
    """To samo co /ingest, ale z plikami wysłanymi jako multipart/form-data."""
    strategies = [s for s in (strategies or []) if s.strip()] or list(config.STRATEGIES)
    sessions = [read_json_upload(f) for f in files]
    body = schemas.IngestRequest(sessions=sessions, strategies=strategies, topics=topics,
                                 dry_run=dry_run, show=show)
    return service.ingest(body, res(request))


# --- zapytania ------------------------------------------------------------------
@app.post("/search", tags=["query"])
def search(body: schemas.SearchRequest, request: Request):
    """scripts/search.py — dense / bm25 / hybrid, filtry metadanych i tematu, sąsiednie chunki."""
    return service.search(body, res(request))


@app.post("/ask", tags=["query"])
def ask(body: schemas.AskRequest, request: Request):
    """Pytanie + odpowiedź LLM ze źródłami. Z `file` / `session` sesja jest najpierw wgrywana
    (chyba że `skip_ingest=true`); potem wyszukanie po filtrach (`session_id`, `client_id`, daty, `topic`)."""
    return service.ask(body, res(request))


@app.post("/summarize-session", tags=["query"])
def summarize_session(body: schemas.SummarizeSessionRequest, request: Request):
    """scripts/summarize_session.py — (wgranie, jak w /ask) jednej sesji i podsumowanie map-reduce wszystkich chunków
    w kolejności chronologicznej (bez wyszukiwania); opcjonalnie tylko chunki z tematem `topic`."""
    return service.summarize_session(body, res(request))


@app.post("/session-keywords", tags=["query"])
def session_keywords(body: schemas.SessionKeywordsRequest, request: Request):
    """scripts/session_keywords.py — (wgranie, jak w /ask) keywordy sesji bez LLM: hasła ze słownika (dosłowne trafienia +
    podobieństwo embeddingów do chunków) oraz wolne frazy z transkryptu (KeyBERT-style, MMR)."""
    return service.session_keywords(body, res(request))


# --- te same zapytania z plikiem sesji wysłanym jako multipart/form-data -------------
def upload_form(model: type) -> type:
    """Formularz multipart: pole `upload` (plik .json sesji) + wszystkie pola `model` poza file / session."""
    fields = {n: (f.annotation, f) for n, f in model.model_fields.items() if n not in ("file", "session")}
    return create_model(model.__name__.removesuffix("Request") + "Upload",
                        upload=(UploadFile, Field(description="plik .json sesji (format thelpy_rag/loader.py)")),
                        **fields)


def from_upload(model: type, form):
    """Formularz -> zwykłe zapytanie JSON-owe z sesją z pliku (walidatory modelu działają jak w JSON)."""
    return model(session=read_json_upload(form.upload), **form.model_dump(exclude={"upload"}))


MULTIPART = "multipart/form-data"
AskUpload = upload_form(schemas.AskRequest)
SummarizeSessionUpload = upload_form(schemas.SummarizeSessionRequest)
SessionKeywordsUpload = upload_form(schemas.SessionKeywordsRequest)


@app.post("/ask/upload", tags=["query"])
def ask_upload(form: Annotated[AskUpload, Form(media_type=MULTIPART)], request: Request):
    """To samo co /ask, ale sesja jako załączony plik .json (z `skip_ingest=true` plik służy tylko do
    odczytania session_id)."""
    return service.ask(from_upload(schemas.AskRequest, form), res(request))


@app.post("/summarize-session/upload", tags=["query"])
def summarize_session_upload(form: Annotated[SummarizeSessionUpload, Form(media_type=MULTIPART)], request: Request):
    """To samo co /summarize-session, ale sesja jako załączony plik .json."""
    return service.summarize_session(from_upload(schemas.SummarizeSessionRequest, form), res(request))


@app.post("/session-keywords/upload", tags=["query"])
def session_keywords_upload(form: Annotated[SessionKeywordsUpload, Form(media_type=MULTIPART)], request: Request):
    """To samo co /session-keywords, ale sesja jako załączony plik .json."""
    return service.session_keywords(from_upload(schemas.SessionKeywordsRequest, form), res(request))


# --- ewaluacja ------------------------------------------------------------------
@app.post("/benchmark-session", tags=["evaluation"])
def benchmark_session(body: schemas.BenchmarkRequest, request: Request):
    """scripts/benchmark_session.py — JSON z wynikami albo (xlsx=true) plik Excela do pobrania."""
    from scripts.benchmark_session import write_excel

    frames, default_name = service.benchmark(body, res(request))
    sheets = (frames["results"], frames["fragments"], frames["config"], frames["summary"])
    saved = None
    if body.out:
        out = Path(body.out)
        saved = str(write_excel(out if out.is_absolute() else config.PROJECT_ROOT / out, *sheets))
    if not body.xlsx:
        return service.benchmark_json(frames, saved)

    if saved:
        return FileResponse(saved, filename=Path(saved).name)
    tmp = Path(tempfile.mkstemp(suffix=".xlsx")[1])
    write_excel(tmp, *sheets)
    return FileResponse(tmp, filename=default_name, background=BackgroundTask(tmp.unlink, missing_ok=True),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.post("/evaluate", tags=["evaluation"])
def evaluate(body: schemas.EvaluateRequest, request: Request):
    """scripts/evaluate.py — hit@k, recall@k, MRR, ctx_tokens dla strategii × trybów."""
    return service.evaluate(body, res(request))
