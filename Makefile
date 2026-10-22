# RAG Chatbot — common development and deployment tasks.
#
# Local workflows use uv (https://docs.astral.sh/uv/); Docker workflows wrap
# compose.yaml (dev) and compose.prod.yaml (prod).

SHELL := /bin/bash
.DEFAULT_GOAL := help

HOST ?= 127.0.0.1
PORT ?= 8000
ARGS ?=

COMPOSE_DEV := docker compose
COMPOSE_PROD := docker compose -f compose.prod.yaml

.PHONY: help install lock env run dev test eval clean \
	dev-up dev-down dev-logs dev-shell dev-test \
	prod-up prod-build prod-down prod-logs prod-shell

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## --- Local development ------------------------------------------------------

install: ## Install/sync dependencies with uv
	uv sync

lock: ## Regenerate uv.lock
	uv lock

env: ## Create .env from .env.example if it does not exist
	@test -f .env && echo ".env already exists" || cp .env.example .env

run: ## Run the app (HOST=, PORT=)
	HOST=$(HOST) PORT=$(PORT) ./run.sh

dev: ## Run the app with auto-reload
	uv run uvicorn app.main:app --reload --reload-dir app --host $(HOST) --port $(PORT)

test: ## Run the test suite
	uv run pytest -q

eval: ## Run the evaluation suite (ARGS="--mode retrieval --top-k 5")
	uv run python -c 'from app.evaluation.cli import main; main()' $(ARGS)

clean: ## Remove Python/test caches (keeps .venv and data)
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -path ./.venv -prune -o -name '*.py[co]' -type f -delete 2>/dev/null || true

## --- Docker: development ----------------------------------------------------

dev-up: ## Build and start the dev stack (foreground)
	$(COMPOSE_DEV) up --build

dev-down: ## Stop the dev stack (keeps data and model volumes)
	$(COMPOSE_DEV) down

dev-logs: ## Follow dev stack logs
	$(COMPOSE_DEV) logs -f app

dev-shell: ## Open a shell in the dev container
	$(COMPOSE_DEV) exec app sh

dev-test: ## Run tests inside the dev container
	$(COMPOSE_DEV) run --rm app pytest -q

## --- Docker: production -----------------------------------------------------

prod-up: env ## Build and start the prod stack in the background
	$(COMPOSE_PROD) up -d --build

prod-build: ## Build the production image
	$(COMPOSE_PROD) build

prod-down: ## Stop the prod stack (keeps volumes)
	$(COMPOSE_PROD) down

prod-logs: ## Follow prod stack logs
	$(COMPOSE_PROD) logs -f app

prod-shell: ## Open a shell in the prod container
	$(COMPOSE_PROD) exec app sh
