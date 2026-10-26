"""Embeddingi dense przez Ollama (/api/embed)."""
from __future__ import annotations

import requests

from . import config


class OllamaEmbedder:
    def __init__(self, url: str = config.OLLAMA_URL, model: str = config.EMBED_MODEL,
                 batch_size: int = 32, timeout: int = 300):
        self.url = url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            r = requests.post(f"{self.url}/api/embed",
                              json={"model": self.model, "input": texts[i:i + self.batch_size]},
                              timeout=self.timeout)
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Dokumenty (chunki) — bez instrukcji."""
        return self._embed(texts)

    def embed_query(self, query: str, instruction: str = config.QUERY_INSTRUCTION) -> list[float]:
        """Zapytanie — z instrukcją w formacie wymaganym przez Qwen3-Embedding."""
        return self._embed([f"Instruct: {instruction}\nQuery: {query}"])[0]

    def dimension(self) -> int:
        return len(self._embed(["test"])[0])
