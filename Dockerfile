# Backend Thelpy RAG (FastAPI): thelpy_rag + thelpy_topics + thelpy_api
#
# Szybkie przebudowy:
# - obraz bazowy przypięty do digestu: sam tag python:3.14-slim co jakiś czas wskazuje nowy obraz na
#   Docker Hubie, a nowa warstwa FROM unieważnia cache wszystkich kolejnych (8 GB torcha od nowa).
#   Aktualizacja: docker buildx imagetools inspect python:3.14-slim  -> wklej nowy Digest.
# - torch w osobnej warstwie przed requirements.txt: zmiana requirements nie pobiera torcha ponownie.
# - --mount=type=cache: pobrane paczki pip zostają w cache BuildKita (poza obrazem), więc nawet
#   po unieważnieniu warstwy instalacja nie pobiera ich z sieci. Czyszczenie: docker builder prune.
FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    THELPY_API_HOST=0.0.0.0 \
    THELPY_API_PORT=8000

WORKDIR /app

# Wariant torcha (indeks https://download.pytorch.org/whl/<TORCH_INDEX>):
#   cu128 + 2.11.0 — GPU, działa ze sterownikiem NVIDIA >= 570 (CUDA 12.8; najnowszy torch z buildem cu128)
#   cu130 + 2.14.1 — GPU, wymaga sterownika >= 580
#   cpu   + 2.15.0 — bez GPU, mniejszy obraz
# torch z requirements.txt jest pomijany, żeby pip nie podmienił go wersją z PyPI.
ARG TORCH_INDEX=cu128
ARG TORCH_VERSION=2.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v -i '^torch==' requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt "torch==${TORCH_VERSION}"

COPY thelpy_rag/ thelpy_rag/
COPY thelpy_topics/ thelpy_topics/
COPY thelpy_api/ thelpy_api/
COPY scripts/ scripts/

# użytkownik bez roota; uid 1000 = właściciel plików w zamontowanym ./data
RUN useradd --uid 1000 --create-home app \
 && mkdir -p /app/data /cache/huggingface \
 && chown -R app:app /app/data /cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"THELPY_API_PORT\"]}/docs', timeout=4)"

CMD ["python", "-m", "thelpy_api"]
