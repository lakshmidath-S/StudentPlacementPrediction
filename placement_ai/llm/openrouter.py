"""
placement_ai/llm/openrouter.py
------------------------------
OpenRouter — a gateway in front of several hundred models, including a rotating
set offered free.

Two things differ from a direct vendor API and both matter here.

Free model IDs churn: a model that exists today may be retired next month, and
the request then fails with a 404 that says nothing useful. The default is
therefore configurable, the error message names the setting to change, and
``list_free_models()`` exists so the UI can show what is currently available.

Requests may be routed. OpenRouter answers with the model that actually served
the request, which is what the shared transport records in provenance — so the
model card names the model that really planned the run, not the one asked for.
"""

from __future__ import annotations

from typing import Any

import requests

from placement_ai.config import (
    OPENROUTER_ENDPOINT,
    OPENROUTER_MODELS_ENDPOINT,
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
)
from placement_ai.llm.base import LLMError
from placement_ai.llm.openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    endpoint = OPENROUTER_ENDPOINT
    # Optional attribution headers. OpenRouter uses them for its public
    # rankings; sending them is polite rather than required.
    extra_headers = {"HTTP-Referer": OPENROUTER_REFERER, "X-Title": OPENROUTER_TITLE}

    def complete_json(self, *args: Any, **kwargs: Any):
        try:
            return super().complete_json(*args, **kwargs)
        except LLMError as exc:
            # A retired or misspelled free model is the single most likely
            # failure here, and "HTTP 404" alone sends people to the wrong place.
            if "404" in str(exc) or "not a valid model" in str(exc).lower():
                raise LLMError(
                    f"OpenRouter does not recognise the model {self.model!r}. "
                    "Free model IDs are retired regularly — set OPENROUTER_MODEL "
                    "to a current one (see openrouter.ai/models?max_price=0)."
                ) from exc
            raise


def list_free_models(api_key: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Free models that can be asked for JSON, widest context first.

    Used by the settings panel so a broken default is a visible, fixable thing
    rather than a training run that quietly falls back to the rules.
    """
    try:
        response = requests.get(
            OPENROUTER_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
    except (requests.RequestException, ValueError) as exc:
        raise LLMError(f"Could not list OpenRouter models: {exc}") from exc

    free: list[dict[str, Any]] = []
    for model in models:
        pricing = model.get("pricing") or {}
        if str(pricing.get("prompt", "1")) not in {"0", "0.0", "-1"}:
            continue
        if "response_format" not in (model.get("supported_parameters") or []):
            continue
        free.append(
            {
                "id": model.get("id", ""),
                "name": model.get("name", ""),
                "context_length": int(model.get("context_length") or 0),
                "structured_outputs": "structured_outputs"
                in (model.get("supported_parameters") or []),
            }
        )

    free.sort(key=lambda m: (-int(m["structured_outputs"]), -m["context_length"]))
    return free
