# Thelpy — skróty do docker compose (qdrant + backend) i lokalnego serwera.
#   make            lista komend
#   make rebuild    po zmianie kodu: przebuduj obraz backendu, podmień kontener i pokaż logi
#
# Kod (thelpy_rag, thelpy_api, ...) jest kopiowany do obrazu przy budowaniu, więc po zmianie kodu
# sam restart nie wystarczy — potrzebny jest rebuild. Restart wystarcza po zmianie .env.

COMPOSE ?= docker compose
BACKEND := backend
PORT    ?= 8000
PY      ?= .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help up down build rebuild restart logs logs-qdrant logs-all ps health shell dev

help:  ## lista komend
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

up:  ## uruchom qdrant + backend w tle (zbuduje obraz, jeśli go nie ma)
	$(COMPOSE) up -d

down:  ## zatrzymaj i usuń kontenery (dane w qdrant_storage/ i data/ zostają)
	$(COMPOSE) down

build:  ## przebuduj obraz backendu (bez uruchamiania)
	$(COMPOSE) build $(BACKEND)

rebuild:  ## po zmianie kodu: przebuduj backend, podmień kontener, pokaż logi
	$(COMPOSE) up -d --build $(BACKEND)
	$(COMPOSE) logs -f --tail=50 $(BACKEND)

restart:  ## restart backendu bez przebudowy (np. po zmianie .env)
	$(COMPOSE) up -d --force-recreate $(BACKEND)
	$(COMPOSE) logs -f --tail=50 $(BACKEND)

logs:  ## logi backendu na żywo (Ctrl+C kończy podgląd, kontener działa dalej)
	$(COMPOSE) logs -f --tail=100 $(BACKEND)

logs-qdrant:  ## logi qdranta na żywo
	$(COMPOSE) logs -f --tail=100 qdrant

logs-all:  ## logi wszystkich usług na żywo
	$(COMPOSE) logs -f --tail=100

ps:  ## stan kontenerów
	$(COMPOSE) ps

health:  ## czy API, Qdrant i Ollama odpowiadają
	@curl -fsS localhost:$(PORT)/health && echo || echo "API nie odpowiada na porcie $(PORT)"

shell:  ## powłoka w kontenerze backendu
	$(COMPOSE) exec $(BACKEND) bash

dev:  ## serwer lokalnie z .venv, z auto-przeładowaniem po zmianie kodu (zatrzymaj wcześniej kontener backendu: port 8000)
	THELPY_API_RELOAD=1 THELPY_API_PORT=$(PORT) $(PY) -m thelpy_api
