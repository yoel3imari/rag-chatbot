#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
exec uv run uvicorn app.main:app --host "$HOST" --port "$PORT"
