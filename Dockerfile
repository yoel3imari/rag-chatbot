# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Base: Python + uv + runtime libraries for the vendored crispembed .so files.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS base

# Pinned to the uv release used to generate uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    CRISPEMBED_CACHE_DIR=/models \
    PATH="/opt/venv/bin:${PATH}"

# libstdc++6 is required by the vendored crispembed/ggml shared libraries.
# curl is used by crispembed to download GGUF models on first use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/app --shell /usr/sbin/nologin app

WORKDIR /app

# The lock file and the vendored crispembed package are needed to resolve deps.
COPY pyproject.toml uv.lock PROGRESS.md ./
COPY vendor/crispembed ./vendor/crispembed

# ---------------------------------------------------------------------------
# Production dependencies (no pytest).
# ---------------------------------------------------------------------------
FROM base AS deps-prod
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Development image: source is bind-mounted by compose, deps include pytest.
# ---------------------------------------------------------------------------
FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
RUN mkdir -p /app/data /models \
    && chown -R app:app /app/data /models

USER app
EXPOSE 8000

# --reload-dir is scoped to app/ so SQLite/LanceDB writes under /app/data and
# the host .venv never trigger a reload loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/meta', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/app"]

# ---------------------------------------------------------------------------
# Production image: non-root, model pre-baked, no dev dependencies.
# ---------------------------------------------------------------------------
FROM deps-prod AS prod

ARG EMBEDDING_MODEL=all-MiniLM-L6-v2

# Bake the default GGUF model in so the first boot does not need the network.
RUN python -c "from crispembed import CrispEmbed; CrispEmbed('${EMBEDDING_MODEL}')"

COPY --chown=app:app app ./app
COPY --chown=app:app static ./static
COPY --chown=app:app run.sh ./

RUN mkdir -p /app/data /models \
    && chown -R app:app /app/data /models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/meta', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
