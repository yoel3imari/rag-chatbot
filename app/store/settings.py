"""User settings and provider-isolated encrypted credential storage."""

import json
import os
import time
from typing import Any, Dict

from app.config import (
    DEFAULT_SETTINGS,
    PROVIDER_ENV_KEY_MAP,
    decrypt_secret,
    encrypt_secret,
    mask_secret,
)


class SettingsMixin:
    def get_settings(
        self, user_id: str = "default", include_decrypted_key: bool = False
    ) -> Dict[str, Any]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT settings_json, encrypted_keys_json FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            settings = dict(DEFAULT_SETTINGS)
            encrypted_keys: Dict[str, str] = {}
            if row:
                try:
                    settings.update(json.loads(row["settings_json"]))
                except Exception:
                    pass
                try:
                    encrypted_keys = json.loads(row["encrypted_keys_json"])
                except Exception:
                    pass

            # Strictly isolate keys by provider ID — never alias across providers.
            provider = settings.get("provider", "openrouter")
            enc_key = encrypted_keys.get(provider, "")
            raw_key = decrypt_secret(enc_key) if enc_key else ""
            if not raw_key:
                env_var = PROVIDER_ENV_KEY_MAP.get(provider)
                if env_var:
                    raw_key = os.environ.get(env_var, "")

            settings["api_key_masked"] = mask_secret(raw_key)
            settings["has_api_key"] = bool(raw_key)

            provider_key_status = {}
            for provider_id, encrypted_value in encrypted_keys.items():
                if provider_id.startswith("_"):
                    continue
                decrypted = decrypt_secret(encrypted_value)
                if decrypted:
                    provider_key_status[provider_id] = mask_secret(decrypted)
            settings["provider_keys_masked"] = provider_key_status

            if include_decrypted_key:
                settings["_decrypted_api_key"] = raw_key
            return settings

    def update_settings(self, updates: Dict[str, Any], user_id: str = "default") -> Dict[str, Any]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT settings_json, encrypted_keys_json FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            current = dict(DEFAULT_SETTINGS)
            encrypted_keys: Dict[str, str] = {}
            if row:
                try:
                    current.update(json.loads(row["settings_json"]))
                except Exception:
                    pass
                try:
                    encrypted_keys = json.loads(row["encrypted_keys_json"])
                except Exception:
                    pass

            # Remove any legacy _default key if present.
            encrypted_keys.pop("_default", None)

            new_api_key = updates.pop("api_key", None)
            for key, value in updates.items():
                if value is not None and key in DEFAULT_SETTINGS:
                    current[key] = value

            provider = current.get("provider", "openrouter")
            if new_api_key is not None:
                cleaned = new_api_key.strip()
                if "••••" not in cleaned:
                    if cleaned == "":
                        encrypted_keys.pop(provider, None)
                    else:
                        encrypted_keys[provider] = encrypt_secret(cleaned)

            now = time.time()
            conn.execute(
                """
                INSERT INTO user_settings (user_id, settings_json, encrypted_keys_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    encrypted_keys_json = excluded.encrypted_keys_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, json.dumps(current), json.dumps(encrypted_keys), now),
            )
        return self.get_settings(user_id=user_id, include_decrypted_key=False)
