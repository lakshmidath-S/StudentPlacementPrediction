"""
The HTTP layer: retries, and the OpenRouter gateway.

Free tiers fail transiently far more than paid ones — 429 when a shared model is
busy, 503 when a hosted one is briefly overloaded — and both clear in seconds.
These tests pin which failures are worth retrying and which are not, because
retrying a permanent failure just delays the fallback and stalls a user watching
a progress bar.

Every test patches `time.sleep`, so the suite asserts on backoff behaviour
without ever waiting for it.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from placement_ai.llm import http as http_module
from placement_ai.llm.base import LLMError
from placement_ai.llm.grok import GrokProvider
from placement_ai.llm.http import TRANSIENT_STATUS, post_with_retry
from placement_ai.llm.openrouter import OpenRouterProvider, list_free_models


class _Response:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def chat_body(content='{"ok": true}', model="some/model"):
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"total_tokens": 10},
    }


@pytest.fixture(autouse=True)
def _no_real_sleeping():
    """Assert on backoff without serving it."""
    with patch.object(http_module.time, "sleep") as sleeper:
        yield sleeper


# ── what gets retried ────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", sorted(TRANSIENT_STATUS))
def test_a_transient_status_is_retried_and_can_succeed(status, _no_real_sleeping):
    responses = [_Response(status, text="busy"), _Response(200, chat_body())]
    with patch("requests.post", side_effect=responses) as post:
        result = GrokProvider("k", "m", 30).complete_json("s", "u")
    assert result.data == {"ok": True}
    assert post.call_count == 2
    assert _no_real_sleeping.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_permanent_status_is_not_retried(status):
    """A bad key or a retired model fails identically forever."""
    with (
        patch("requests.post", return_value=_Response(status, text="nope")) as post,
        pytest.raises(LLMError),
    ):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    assert post.call_count == 1


def test_retries_are_bounded(_no_real_sleeping):
    with (
        patch("requests.post", return_value=_Response(503, text="overloaded")) as post,
        pytest.raises(LLMError, match="503"),
    ):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    assert post.call_count == http_module.MAX_ATTEMPTS


def test_backoff_grows_between_attempts(_no_real_sleeping):
    with (
        patch("requests.post", return_value=_Response(429, text="slow down")),
        pytest.raises(LLMError),
    ):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    waits = [call.args[0] for call in _no_real_sleeping.call_args_list]
    assert waits == sorted(waits)
    assert waits[0] < waits[-1]


def test_a_connection_error_is_retried_then_reported(_no_real_sleeping):
    with (
        patch("requests.post", side_effect=requests.ConnectionError("no route")) as post,
        pytest.raises(LLMError, match="request failed"),
    ):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    assert post.call_count == http_module.MAX_ATTEMPTS


def test_a_connection_error_that_clears_succeeds(_no_real_sleeping):
    responses = [requests.ConnectionError("blip"), _Response(200, chat_body())]
    with patch("requests.post", side_effect=responses):
        assert GrokProvider("k", "m", 30).complete_json("s", "u").data == {"ok": True}


# ── Retry-After ──────────────────────────────────────────────────────────────


def test_retry_after_in_seconds_is_honoured(_no_real_sleeping):
    responses = [_Response(429, text="wait", headers={"Retry-After": "4"}),
                 _Response(200, chat_body())]
    with patch("requests.post", side_effect=responses):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    assert _no_real_sleeping.call_args_list[0].args[0] == 4.0


def test_an_excessive_retry_after_is_capped(_no_real_sleeping):
    """A free tier asking for a minute should fall back, not stall the UI."""
    responses = [_Response(429, text="wait", headers={"Retry-After": "600"}),
                 _Response(200, chat_body())]
    with patch("requests.post", side_effect=responses):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    assert _no_real_sleeping.call_args_list[0].args[0] == http_module.MAX_BACKOFF_SECONDS


def test_an_unparseable_retry_after_falls_back_to_backoff(_no_real_sleeping):
    responses = [_Response(503, text="x", headers={"Retry-After": "soon"}),
                 _Response(200, chat_body())]
    with patch("requests.post", side_effect=responses):
        GrokProvider("k", "m", 30).complete_json("s", "u")
    assert _no_real_sleeping.call_args_list[0].args[0] == http_module.INITIAL_BACKOFF_SECONDS


def test_post_with_retry_returns_the_final_response(_no_real_sleeping):
    with patch("requests.post", return_value=_Response(418, text="teapot")):
        response = post_with_retry(
            "http://x", provider="t", json={}, headers={}, timeout=5
        )
    assert response.status_code == 418


# ── OpenRouter ───────────────────────────────────────────────────────────────


def test_openrouter_sends_attribution_headers():
    with patch("requests.post", return_value=_Response(200, chat_body())) as post:
        OpenRouterProvider("k", "some/model:free", 30).complete_json("s", "u")
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert "HTTP-Referer" in headers and "X-Title" in headers


def test_openrouter_records_the_model_that_actually_answered():
    """A routed request can be served by a different model than the one asked for."""
    body = chat_body(model="nvidia/nemotron-3.5-lightning:free")
    with patch("requests.post", return_value=_Response(200, body)):
        result = OpenRouterProvider("k", "openrouter/free", 30).complete_json("s", "u")
    assert result.model == "nvidia/nemotron-3.5-lightning:free"


def test_a_retired_free_model_names_the_setting_to_change():
    with (
        patch("requests.post", return_value=_Response(404, text="no such model")),
        pytest.raises(LLMError, match="OPENROUTER_MODEL"),
    ):
        OpenRouterProvider("k", "gone/model:free", 30).complete_json("s", "u")


def test_a_gateway_error_inside_a_200_is_caught():
    """OpenRouter reports some upstream failures with an HTTP 200 envelope."""
    body = {"error": {"message": "Provider returned error", "code": 429}}
    with (
        patch("requests.post", return_value=_Response(200, body)),
        pytest.raises(LLMError, match="Provider returned error"),
    ):
        OpenRouterProvider("k", "m", 30).complete_json("s", "u")


def test_listing_free_models_filters_and_ranks():
    payload = {
        "data": [
            {"id": "paid/model", "pricing": {"prompt": "0.5"},
             "supported_parameters": ["response_format"], "context_length": 999999},
            {"id": "free/no-json", "pricing": {"prompt": "0"},
             "supported_parameters": [], "context_length": 100000},
            {"id": "free/small", "pricing": {"prompt": "0"},
             "supported_parameters": ["response_format"], "context_length": 8000},
            {"id": "free/structured", "pricing": {"prompt": "0"},
             "supported_parameters": ["response_format", "structured_outputs"],
             "context_length": 32000},
        ]
    }
    with patch("requests.get", return_value=_Response(200, payload)):
        models = list_free_models("k")

    ids = [m["id"] for m in models]
    assert "paid/model" not in ids       # not free
    assert "free/no-json" not in ids     # cannot be asked for JSON
    # Structured-output models first, then by context length.
    assert ids == ["free/structured", "free/small"]


def test_listing_free_models_reports_a_failure():
    with (
        patch("requests.get", side_effect=requests.ConnectionError("down")),
        pytest.raises(LLMError, match="Could not list"),
    ):
        list_free_models("k")


# ── provider registry ────────────────────────────────────────────────────────


def test_openrouter_accepts_either_spelling_of_its_key(monkeypatch):
    """OpenRouter's own dashboard and the wider convention disagree."""
    from placement_ai.llm.registry import get_provider

    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "underscored")
    assert get_provider("openrouter").api_key == "underscored"

    monkeypatch.delenv("OPEN_ROUTER_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "joined")
    assert get_provider("openrouter").api_key == "joined"


def test_auto_prefers_a_direct_vendor_over_the_gateway(monkeypatch):
    from placement_ai.llm.registry import get_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert get_provider("auto").name == "openrouter"

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert get_provider("auto").name == "gemini"


def test_a_missing_dotenv_package_is_reported_not_swallowed(monkeypatch, tmp_path):
    """The failure that looks exactly like having no key at all."""
    import builtins

    from placement_ai.llm import registry

    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=x\n", encoding="utf-8")

    real_import = builtins.__import__

    def no_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("No module named 'dotenv'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    try:
        problem = registry._detect_dotenv_problem()
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)

    assert problem is not None
    assert "python-dotenv" in problem


def test_no_dotenv_and_no_env_file_is_silent(monkeypatch, tmp_path):
    import builtins

    from placement_ai.llm import registry

    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)  # no .env inside
    real_import = builtins.__import__

    def no_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_dotenv)
    try:
        assert registry._detect_dotenv_problem() is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
