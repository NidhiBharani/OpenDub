"""Paths + persisted settings (provider options / defaults).

Settings live in configs/settings.yaml (runtime-generated, gitignored). Environment variables
override stored values: OPENDUB_<PROVIDER_ID upper, dots→underscores>_<FIELD_KEY upper>,
e.g. OPENDUB_TRANSLATION_ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("OPENDUB_ROOT", Path(__file__).resolve().parents[2].parent))
DATA_DIR = Path(os.environ.get("OPENDUB_DATA_DIR", ROOT / "data"))
PROJECTS_DIR = DATA_DIR / "projects"
CONFIGS_DIR = Path(os.environ.get("OPENDUB_CONFIGS_DIR", ROOT / "configs"))
SETTINGS_PATH = CONFIGS_DIR / "settings.yaml"
WEB_DIST = ROOT / "web" / "dist"

SECRET_MASK = "•••"

_lock = threading.Lock()


def _ensure_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    _ensure_dirs()
    if SETTINGS_PATH.exists():
        with SETTINGS_PATH.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def save_settings(settings: dict[str, Any]) -> None:
    _ensure_dirs()
    with _lock, SETTINGS_PATH.open("w") as f:
        yaml.safe_dump(settings, f, sort_keys=True, allow_unicode=True)


def _env_key(provider_id: str, field_key: str) -> str:
    return f"OPENDUB_{provider_id.replace('.', '_').upper()}_{field_key.upper()}"


def provider_options(provider_id: str) -> dict[str, Any]:
    """Stored options for a provider, with env-var overrides applied."""
    stored = (load_settings().get("providers") or {}).get(provider_id) or {}
    out = dict(stored)
    prefix = f"OPENDUB_{provider_id.replace('.', '_').upper()}_"
    for env_name, value in os.environ.items():
        if env_name.startswith(prefix) and value:
            out[env_name[len(prefix) :].lower()] = value
    return out


def set_provider_options(provider_id: str, options: dict[str, Any], secret_keys: set[str]) -> None:
    """Persist options. A secret field arriving as SECRET_MASK means 'leave unchanged'."""
    settings = load_settings()
    providers = settings.setdefault("providers", {})
    current = providers.setdefault(provider_id, {})
    for k, v in options.items():
        if k in secret_keys and v == SECRET_MASK:
            continue
        if v in (None, ""):
            current.pop(k, None)
        else:
            current[k] = v
    save_settings(settings)


def mask_options(options: dict[str, Any], secret_keys: set[str]) -> dict[str, Any]:
    return {k: (SECRET_MASK if k in secret_keys and v else v) for k, v in options.items()}
