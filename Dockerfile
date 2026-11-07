# Backend Thelpy RAG (FastAPI): thelpy_rag + thelpy_topics + thelpy_api
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
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
COPY requirements.txt .
RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
 && grep -v -i '^torch==' requirements.txt > /tmp/requirements.txt \
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
