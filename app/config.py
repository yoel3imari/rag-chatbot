import base64
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BASE_DIR = Path(__file__).resolve().parent.parent
# Tests (tests/conftest.py) redirect all runtime data into a temp dir via RAG_DATA_DIR.
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR") or (BASE_DIR / "data"))
LANCEDB_DIR = DATA_DIR / "lancedb"
UPLOADS_DIR = DATA_DIR / "uploads"
SQLITE_PATH = DATA_DIR / "app.db"
SECRET_KEY_PATH = DATA_DIR / ".secret_key"

MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_MB", "20")) * 1024 * 1024
EMBEDDING_DIM = 384

for d in (DATA_DIR, LANCEDB_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _get_or_create_aes_key() -> bytes:
    env_secret = os.environ.get("RAG_SECRET_KEY")
    if env_secret:
        return hashlib.sha256(env_secret.encode("utf-8")).digest()
    if SECRET_KEY_PATH.exists():
        raw = SECRET_KEY_PATH.read_bytes()
        if len(raw) == 32:
            return raw
    key = AESGCM.generate_key(bit_length=256)
    SECRET_KEY_PATH.write_bytes(key)
    try:
        os.chmod(SECRET_KEY_PATH, 0o600)
    except OSError:
        pass
    return key


_AES_KEY = _get_or_create_aes_key()


def encrypt_secret(plaintext: str) -> str:
    """Encrypts a secret string using AES-256-GCM."""
    if not plaintext:
        return ""
    aesgcm = AESGCM(_AES_KEY)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + ct).decode("ascii")


def decrypt_secret(token: str) -> str:
    """Decrypts an AES-256-GCM encrypted string."""
    if not token:
        return ""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        nonce, ct = raw[:12], raw[12:]
        aesgcm = AESGCM(_AES_KEY)
        return aesgcm.decrypt(nonce, ct, None).decode("utf-8")
    except Exception:
        return ""


def mask_secret(plaintext: str) -> str:
    """Masks an API key so plaintext is never returned in API responses."""
    if not plaintext:
        return ""
    if len(plaintext) <= 8:
        return "••••••••"
    return f"{plaintext[:4]}••••{plaintext[-4:]}"


PROVIDER_ENV_KEY_MAP: Dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "grok": "XAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together": "TOGETHER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "custom": "CUSTOM_LLM_API_KEY",
}

KEYLESS_PROVIDERS = {"ollama", "lmstudio", "custom"}

PROVIDER_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "x-ai/grok-2-1212",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "requires_key": True,
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o3-mini"],
        "requires_key": True,
    },
    {
        "id": "grok",
        "name": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "models": ["grok-2-latest", "grok-3-beta", "grok-3-mini-beta"],
        "requires_key": True,
    },
    {
        "id": "google",
        "name": "Google Gemini (OpenAI Compat)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "requires_key": True,
    },
    {
        "id": "groq",
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "mixtral-8x7b-32768",
        ],
        "requires_key": True,
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "requires_key": True,
    },
    {
        "id": "together",
        "name": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
        ],
        "requires_key": True,
    },
    {
        "id": "mistral",
        "name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
        "requires_key": True,
    },
    {
        "id": "ollama",
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1:8b",
        "models": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "deepseek-r1:8b", "gemma2:9b"],
        "requires_key": False,
    },
    {
        "id": "lmstudio",
        "name": "LM Studio (Local)",
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
        "models": ["local-model"],
        "requires_key": False,
    },
    {
        "id": "custom",
        "name": "Custom OpenAI-Compatible",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": [],
        "requires_key": False,
    },
]

EMBEDDING_MODELS: List[Dict[str, Any]] = [
    {
        "id": "all-MiniLM-L6-v2",
        "name": "all-MiniLM-L6-v2 (Default · 384d · Fast CPU)",
        "dim": 384,
    },
    {
        "id": "gte-small",
        "name": "gte-small (384d · BERT English)",
        "dim": 384,
    },
    {
        "id": "arctic-embed-xs",
        "name": "arctic-embed-xs (384d · Snowflake CLS)",
        "dim": 384,
    },
    {
        "id": "multilingual-e5-small",
        "name": "multilingual-e5-small (384d · 100+ Languages)",
        "dim": 384,
    },
]

DEFAULT_SYSTEM_PROMPT = (
    "You are an enterprise RAG knowledge assistant operating under a ReAct "
    "(Reasoning + Acting) protocol. Answer accurately, concisely, and with high "
    "information density. When a user asks about company policies, uploaded documents, "
    "ingested URLs, or domain-specific facts, invoke the `search_knowledge_base` tool first. "
    "When the user engages in general chitchat, math, or pure logic that does not depend on "
    "internal documents, respond directly without calling retrieval tools. Always cite the "
    "source document or URL name in brackets (e.g., [policy.pdf]) when using retrieved context."
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": os.environ.get("LLM_PROVIDER", "openrouter"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    "model": os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
    "temperature": 0.2,
    "max_tokens": 2048,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "similarity_threshold": 0.15,
    "dedup_threshold": 0.92,
    "theme": "dark",
    "compaction_mode": "Auto-Summarize",  # Sliding Window | Auto-Summarize | Strict Truncation
    "max_context_tokens": 8192,
    "reserved_output_tokens": 2048,
    "rag_token_ceiling": 1500,
    "sliding_window_turns": 10,
}
