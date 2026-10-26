"""Centralna konfiguracja.

Wartości są czytane z pliku .env w katalogu projektu.
Kolejność ważności: zmienna ustawiona w terminalu > .env > wartość domyślna poniżej.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")   # wczytuje .env; nie nadpisuje zmiennych z terminala


def _env(name: str, default: str) -> str:
    """Zwraca zmienną środowiskową; pusta wartość w .env = wartość domyślna."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


# --- usługi -----------------------------------------------------------------
QDRANT_URL = _env("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = _env("EMBED_MODEL", "qwen3-embedding:0.6b-q8_0")
LLM_MODEL = _env("LLM_MODEL", "qwen3:8b")
LLM_NUM_CTX = int(_env("LLM_NUM_CTX", "8192"))          # okno kontekstu Ollamy (prompt + odpowiedź)
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.2"))
LLM_ANSWER_RESERVE = int(_env("LLM_ANSWER_RESERVE", "1024"))  # tokeny zostawione na odpowiedź

TOKENIZER_NAME = _env("TOKENIZER_NAME", "Qwen/Qwen3-Embedding-0.6B")

QUERY_INSTRUCTION = _env(
    "QUERY_INSTRUCTION",
    "Given a question about a psychotherapy session, retrieve transcript fragments that answer it",
)

# --- nazwy wektorów w kolekcji ----------------------------------------------
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# --- BM25 -------------------------------------------------------------------
BM25_K1 = float(_env("BM25_K1", "1.2"))
BM25_B = float(_env("BM25_B", "0.75"))
BM25_AVG_LEN = float(_env("BM25_AVG_LEN", "150"))
BM25_STEM_PREFIX = int(_env("BM25_STEM_PREFIX", "0")) or None   # 0 = wyłączone
BM25_STOPWORDS = _env("BM25_STOPWORDS", "english").lower()        # english | polish | none
BM25_STEMMER = _env("BM25_STEMMER", "english").lower()            # język Snowball (english, polish, ...) | none

# --- RRF ----------------------------------------------------------------------
RRF_K = int(_env("RRF_K", "60"))
PREFETCH_LIMIT = int(_env("PREFETCH_LIMIT", "50"))

# --- strategie chunkingu (jedna strategia = jedna kolekcja) -----------------
COLLECTION_PREFIX = _env("COLLECTION_PREFIX", "thelpy")

STRATEGIES = {
    "fixed256":       {"type": "fixed_tokens", "size": 256, "overlap": 0},
    "fixed256_ov64":  {"type": "fixed_tokens", "size": 256, "overlap": 64},
    "fixed512":       {"type": "fixed_tokens", "size": 512, "overlap": 0},
    "fixed512_ov128": {"type": "fixed_tokens", "size": 512, "overlap": 128},
    "turns4":         {"type": "turn_window", "n_turns": 4, "overlap_turns": 0},
    "turns8":         {"type": "turn_window", "n_turns": 8, "overlap_turns": 0},
}


def collection_name(strategy: str) -> str:
    return f"{COLLECTION_PREFIX}_{strategy}"


def summary() -> str:
    """Podgląd aktywnej konfiguracji (bez klucza API)."""
    return (f"Qdrant: {QDRANT_URL} (klucz: {'tak' if QDRANT_API_KEY else 'nie'}) | "
            f"Ollama: {OLLAMA_URL} | embed: {EMBED_MODEL} | llm: {LLM_MODEL} "
            f"(num_ctx={LLM_NUM_CTX}, temp={LLM_TEMPERATURE}) | "
            f"BM25 k1={BM25_K1} b={BM25_B} avg_len={BM25_AVG_LEN} stem={BM25_STEM_PREFIX} stopwords={BM25_STOPWORDS} stemmer={BM25_STEMMER} | "
            f"RRF k={RRF_K} prefetch={PREFETCH_LIMIT}")