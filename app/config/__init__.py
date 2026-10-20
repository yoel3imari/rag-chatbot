"""Application configuration, re-exported for backwards-compatible `app.config` imports."""

from app.config.crypto import decrypt_secret, encrypt_secret, mask_secret
from app.config.defaults import DEFAULT_SETTINGS, DEFAULT_SYSTEM_PROMPT
from app.config.paths import (
    BASE_DIR,
    DATA_DIR,
    EMBEDDING_DIM,
    LANCEDB_DIR,
    MAX_UPLOAD_BYTES,
    SECRET_KEY_PATH,
    SQLITE_PATH,
    UPLOADS_DIR,
)
from app.config.providers import (
    EMBEDDING_MODELS,
    KEYLESS_PROVIDERS,
    PROVIDER_ENV_KEY_MAP,
    PROVIDER_PRESETS,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DEFAULT_SETTINGS",
    "DEFAULT_SYSTEM_PROMPT",
    "EMBEDDING_DIM",
    "EMBEDDING_MODELS",
    "KEYLESS_PROVIDERS",
    "LANCEDB_DIR",
    "MAX_UPLOAD_BYTES",
    "PROVIDER_ENV_KEY_MAP",
    "PROVIDER_PRESETS",
    "SECRET_KEY_PATH",
    "SQLITE_PATH",
    "UPLOADS_DIR",
    "decrypt_secret",
    "encrypt_secret",
    "mask_secret",
]
