"""LLM provider presets and embedding model catalogue."""

from typing import Any, Dict, List

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
