# Docker

Two stacks are provided:

| File | Purpose | Image target | Data |
| :--- | :--- | :--- | :--- |
| `compose.yaml` | Local development with hot reload | `dev` | bind-mounted `./data` |
| `compose.prod.yaml` | Production deployment | `prod` | named volumes |

Both use the single multi-stage `Dockerfile` and run as the non-root `app`
user. The vendored `crispembed` shared libraries are **linux/amd64 only**, so
compose forces `platform: linux/amd64` (on Apple Silicon this runs under
emulation).

## Requirements

- Docker Engine 24+ with Compose v2.24+ (`docker compose version`)
- An LLM API key (or a local OpenAI-compatible endpoint), configured either in
  `.env` or in the UI

## Development

```bash
cp .env.example .env        # optional: add an API key
docker compose up --build
```

Open <http://127.0.0.1:8000>.

- The whole repository is bind-mounted at `/app`; edits under `app/` trigger a
  reload. Static files are served directly — refresh the browser after
  changing `static/`.
- Runtime state (SQLite, LanceDB, uploads, generated `.secret_key`) is written
  to the host's `./data` directory, exactly like running locally.
- The CrispEmbed model cache lives in the `rag-chatbot-models` volume, so the
  default 19 MB model is downloaded only once.
- Do not run the container and `./run.sh` at the same time against the same
  `./data`: SQLite tolerates it, but LanceDB writes are not designed for two
  writers.

Run the test suite inside the dev image:

```bash
docker compose run --rm app pytest -q
```

Stop the stack:

```bash
docker compose down          # keeps ./data and the model volume
```

Rebuild after changing `pyproject.toml` / `uv.lock` (or when dependencies are
added):

```bash
docker compose build --pull
```

### Ports

`APP_BIND` defaults to `127.0.0.1` and `APP_PORT` to `8000`. To expose the dev
server on the LAN:

```bash
APP_BIND=0.0.0.0 docker compose up
```

### macOS / Windows

If hot reload misses changes on a bind mount, uncomment
`WATCHFILES_FORCE_POLLING: "true"` in `compose.yaml`.

## Production

1. Create the environment file and generate a stable encryption key:

   ```bash
   cp .env.example .env
   echo "RAG_SECRET_KEY=$(openssl rand -hex 32)" >> .env
   # add OPENROUTER_API_KEY / OPENAI_API_KEY / ... to .env
   ```

   Keep `.env` out of version control (it is already git-ignored). If
   `RAG_SECRET_KEY` is unset, a key file is generated at
   `/app/data/.secret_key` inside the data volume. Changing the key makes
   previously stored encrypted API keys unreadable.

2. Build and start:

   ```bash
   docker compose -f compose.prod.yaml up -d --build
   ```

3. Check status and logs:

   ```bash
   docker compose -f compose.prod.yaml ps
   docker compose -f compose.prod.yaml logs -f app
   docker compose -f compose.prod.yaml exec app id -u   # prints 1000
   ```

The production image pre-downloads the default embedding model, installs no
test dependencies, drops all Linux capabilities and is healthchecked against
`GET /api/meta` every 30 seconds.

### Network exposure

By default the app binds to `127.0.0.1:8000` on the host. Deploy it behind a
TLS-terminating reverse proxy (Caddy, nginx, Traefik, a cloud load balancer) or
set `APP_BIND=0.0.0.0` to expose it directly:

```bash
APP_BIND=0.0.0.0 APP_PORT=8000 docker compose -f compose.prod.yaml up -d
```

### Update / stop

```bash
docker compose -f compose.prod.yaml up -d --build   # rebuild + rolling recreate
docker compose -f compose.prod.yaml down            # keeps volumes
```

`down -v` also deletes the data and model volumes — use with care.

## Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `openrouter` | Default provider preset |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Default model ID |
| `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, ... | – | Provider keys (see `app/config/providers.py` for the full mapping) |
| `RAG_SECRET_KEY` | generated file | AES-256-GCM master key for stored credentials |
| `RAG_MAX_UPLOAD_MB` | `20` | Maximum upload size |
| `APP_BIND` | `127.0.0.1` | Host interface to publish the port on |
| `APP_PORT` | `8000` | Host port |
| `APP_UID` / `APP_GID` | `1000` | Container user ID (match your host user for dev bind mounts) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model baked into the prod image |

Values in `.env` are passed to the container by compose. Provider keys can also
be entered in the UI, in which case they are encrypted at rest inside the data
volume.

## Data & backups

Production volumes:

- `rag-chatbot-data` → `/app/data` (SQLite `app.db`, LanceDB, uploads,
  `.secret_key`)
- `rag-chatbot-models` → `/models` (CrispEmbed GGUF cache)

Back up the data volume while the stack is stopped:

```bash
docker compose -f compose.prod.yaml stop
docker run --rm \
  -v rag-chatbot-data:/data:ro \
  -v "$PWD":/backup \
  alpine tar czf "/backup/rag-chatbot-data-$(date +%F).tgz" -C /data .
docker compose -f compose.prod.yaml start
```

Restore into a fresh volume:

```bash
docker compose -f compose.prod.yaml stop
docker run --rm \
  -v rag-chatbot-data:/data \
  -v "$PWD":/backup \
  alpine sh -c 'find /data -mindepth 1 -delete && tar xzf /backup/rag-chatbot-data-YYYY-MM-DD.tgz -C /data'
docker compose -f compose.prod.yaml start
```

To seed the production volume with an existing local `./data` directory (only
when the volume is empty):

```bash
docker run --rm \
  -v rag-chatbot-data:/dest \
  -v "$PWD/data":/src:ro \
  alpine sh -c 'cp -a /src/. /dest/'
```

## Operational notes

- **One worker.** SQLite, LanceDB, the rate limiter and idempotency caches are
  process-local. Do not add `--workers N`; scale vertically, or move those
  components to shared services first.
- **Embedded storage.** The data volume must stay attached to a single app
  container. Removing the volume loses sessions, documents and encrypted keys.
- **Model cache.** Switching the embedding model in the UI downloads the new
  GGUF into `/models`; the volume keeps it across restarts. The default model
  is already in the production image.
- **Healthcheck.** `GET /api/meta` is also subject to the in-memory rate
  limiter; two requests per minute from the healthcheck are well below the
  limit.
