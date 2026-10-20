"""AES-256-GCM helpers for credentials stored at rest."""

import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config.paths import SECRET_KEY_PATH


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
