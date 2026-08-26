"""
placement_ai/llm/registry.py
----------------------------
Secret resolution and provider selection.

A key can arrive three ways depending on where the app runs: an environment
variable (a container), a ``.env`` file (a laptop), or ``st.secrets`` (Streamlit
Community Cloud, which has no way to set env vars). All three are checked, in
that order, so the same code deploys to all three without a branch.

When no key is found anywhere, ``get_provider()`` returns None. That is a
supported mode, not an error: the heuristic planner takes over and the product
still trains and predicts, just without generated reasoning.
"""

from __future__ import annotations

import os
from functools import lru_cache

from placement_ai.config import (
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    GROK_API_KEY_ENV,
    GROK_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    REPO_ROOT,
)
from placement_ai.llm.base import LLMProvider
from placement_ai.llm.gemini import GeminiProvider
from placement_ai.llm.grok import GrokProvider

PROVIDER_CLASSES: dict[str, tuple[type[LLMProvider], str, str]] = {
    "gemini": (GeminiProvider, GEMINI_API_KEY_ENV, GEMINI_MODEL),
    "grok": (GrokProvider, GROK_API_KEY_ENV, GROK_MODEL),
}


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    """Fold .env into os.environ if python-dotenv is installed.

    Optional on purpose — a missing dotenv package must not stop the app, it
    just means only real environment variables are visible.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _from_streamlit_secrets(name: str) -> str | None:
    """Read st.secrets[name] without requiring Streamlit or a secrets file.

    Touching st.secrets raises when no secrets.toml exists, so every access is
    guarded; this function is also called from plain pytest runs.
    """
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        value = st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None
    return str(value) if value else None


def resolve_secret(name: str) -> str:
    """Look up one secret across env, .env and Streamlit secrets."""
    _load_dotenv_once()
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    from_secrets = _from_streamlit_secrets(name)
    return from_secrets.strip() if from_secrets else ""


def build_provider(kind: str) -> LLMProvider | None:
    """Instantiate one named provider, or None when its key is absent."""
    entry = PROVIDER_CLASSES.get(kind)
    if entry is None:
        return None
    provider_cls, key_env, model = entry
    api_key = resolve_secret(key_env)
    if not api_key:
        return None
    return provider_cls(api_key=api_key, model=model, timeout=LLM_TIMEOUT_SECONDS)


def get_provider(preferred: str | None = None) -> LLMProvider | None:
    """Pick a provider.

    ``preferred`` (or LLM_PROVIDER) may name one explicitly; "off" disables
    generation entirely, which is how CI and the offline demo run. "auto" takes
    the first configured provider in PROVIDER_CLASSES order.
    """
    choice = (preferred or LLM_PROVIDER or "auto").strip().lower()

    if choice in {"off", "none", "offline", "heuristic", "disabled"}:
        return None
    if choice in PROVIDER_CLASSES:
        return build_provider(choice)

    for kind in PROVIDER_CLASSES:
        provider = build_provider(kind)
        if provider is not None:
            return provider
    return None


def provider_status() -> dict[str, bool]:
    """Which providers currently hold a usable key — for the settings panel."""
    return {kind: bool(resolve_secret(key_env)) for kind, (_, key_env, _) in PROVIDER_CLASSES.items()}
