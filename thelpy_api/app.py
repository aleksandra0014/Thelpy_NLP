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

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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


@app.post("/ingest/upload", tags=["ingest"])
def ingest_upload(request: Request, files: list[UploadFile] = File(..., description="pliki .json sesji"),
                  strategies: list[str] = Form(default_factory=lambda: list(config.STRATEGIES)),
                  topics: bool = Form(False), dry_run: bool = Form(False), show: int = Form(0)):
    """To samo co /ingest, ale z plikami wysłanymi jako multipart/form-data."""
    sessions = []
    for f in files:
        try:
            sessions.append(json.loads(f.file.read().decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"{f.filename}: niepoprawny JSON ({e})")
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
    """scripts/ask.py — wyszukanie + odpowiedź LLM ze źródłami."""
    return service.ask(body, res(request))


@app.post("/ask-session", tags=["query"])
def ask_session(body: schemas.AskSessionRequest, request: Request):
    """scripts/ask_session.py — (wgranie) jednej sesji, dense, prompt, odpowiedź i tokeny."""
    return service.ask_session(body, res(request))


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
