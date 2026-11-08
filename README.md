# Thelpy_NLP

## Serwer API (FastAPI)

Opis wszystkich endpointów: [thelpy_api/README.md](thelpy_api/README.md).

W kontenerze (razem z Qdrantem):

```bash
docker compose up -d --build    # qdrant + backend, API: http://127.0.0.1:8000/docs
docker compose logs -f backend
```

- `backend` używa sieci hosta, więc widzi Qdranta (`127.0.0.1:6333`) i Ollamę (`localhost:11434`)
  tak jak lokalne skrypty; inne adresy: `BACKEND_QDRANT_URL`, `BACKEND_OLLAMA_URL` w `.env`.
- `./data` jest zamontowane w `/app/data` — ścieżki w zapytaniach podawaj jak lokalnie, np. `data/annomi_64.json`.
- modele HF (tokenizer bge-m3, model tematów) są w wolumenie `hf_cache`, pobierane przy pierwszym użyciu.
- po zmianie kodu: `docker compose up -d --build backend`.

Lokalnie, bez Dockera:

```bash
uv pip install -r requirements.txt
python -m thelpy_api            # http://127.0.0.1:8000/docs  (albo: uvicorn thelpy_api.app:app)
```

| Endpoint | Odpowiednik | Tematy |
|---|---|---|
| `GET /health`, `/config`, `/topics`, `/collections` | — | `/topics` = klucze do `topic` |
| `POST /collections` | `scripts.create_collections` | |
| `GET /collections/{strategy}/sessions` | — | metadane sesji w kolekcji, `?client_id=` |
| `DELETE /collections/{strategy}/sessions/{id}` | — | |
| `POST /ingest`, `POST /ingest/upload` | `scripts.ingest` | `topics: true` |
| `POST /search` | `scripts.search` | `topic` |
| `POST /ask` | `scripts.ask` | `topics`, `topic` |
| `POST /ask-session` | `scripts.ask_session` | `topics`, `topic` |
| `POST /summarize-session` | `scripts.summarize_session` (map-reduce całej sesji) | `topics`, `topic` |
| `POST /benchmark-session` | `scripts.benchmark_session` (`xlsx: true` → plik) | `topics`, `topic` |
| `POST /evaluate` | `scripts.evaluate` | |

```bash
curl -X POST localhost:8000/ask-session -H 'Content-Type: application/json' \
  -d '{"session_id": "64", "question": "What does he say about his friends?", "topic": "friends"}'
```
