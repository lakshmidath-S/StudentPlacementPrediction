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
    OPENROUTER_API_KEY_ALIASES,
    OPENROUTER_MODEL,
    REPO_ROOT,
)
from placement_ai.llm.base import LLMProvider
from placement_ai.llm.gemini import GeminiProvider
from placement_ai.llm.grok import GrokProvider
from placement_ai.llm.openrouter import OpenRouterProvider

# name -> (class, accepted key env names, default model). Order is the
# preference order "auto" walks: direct vendor APIs first, then the gateway.
PROVIDER_CLASSES: dict[str, tuple[type[LLMProvider], tuple[str, ...], str]] = {
    "gemini": (GeminiProvider, (GEMINI_API_KEY_ENV,), GEMINI_MODEL),
    "grok": (GrokProvider, (GROK_API_KEY_ENV,), GROK_MODEL),
    "openrouter": (OpenRouterProvider, OPENROUTER_API_KEY_ALIASES, OPENROUTER_MODEL),
}

_OFF_VALUES = {"off", "none", "offline", "heuristic", "disabled"}


def _detect_dotenv_problem() -> str | None:
    """Fold .env into os.environ, returning a description of any problem.

    python-dotenv is an optional dependency, but treating a missing one as
    silence was a real trap: a user with a correctly written .env saw only
    "running on built-in rules" and no reason. If the file exists and cannot be
    read, that is now something the UI can say out loud.

    Kept separate from the cached wrapper below so tests can exercise it
    directly without reaching into an lru_cache.
    """
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        if env_path.exists():
            return (
                "A .env file is present but python-dotenv is not installed, so it "
                "cannot be read. Run `pip install python-dotenv` (or `pip install "
                "-r requirements.txt`) and restart."
            )
        return None

    if env_path.exists():
        load_dotenv(env_path, override=False)
    return None


@lru_cache(maxsize=1)
def _dotenv_problem() -> str | None:
    """Cached so .env is read once per process, not once per secret lookup."""
    return _detect_dotenv_problem()


def dotenv_status() -> str | None:
    """A message to surface when a .env exists but could not be loaded."""
    return _dotenv_problem()


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


def resolve_secret(*names: str) -> str:
    """Look up a secret across env, .env and Streamlit secrets.

    Several names may be given as accepted spellings of the same secret; the
    first that resolves wins.
    """
    _dotenv_problem()
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    for name in names:
        from_secrets = _from_streamlit_secrets(name)
        if from_secrets and from_secrets.strip():
            return from_secrets.strip()
    return ""


def build_provider(kind: str) -> LLMProvider | None:
    """Instantiate one named provider, or None when its key is absent."""
    entry = PROVIDER_CLASSES.get(kind)
    if entry is None:
        return None
    provider_cls, key_names, model = entry
    api_key = resolve_secret(*key_names)
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

    if choice in _OFF_VALUES:
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
    return {
        kind: bool(resolve_secret(*key_names))
        for kind, (_, key_names, _) in PROVIDER_CLASSES.items()
    }


def provider_key_names() -> dict[str, tuple[str, ...]]:
    """The env var name(s) each provider accepts, for the settings panel."""
    return {kind: key_names for kind, (_, key_names, _) in PROVIDER_CLASSES.items()}
