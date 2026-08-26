"""
The LLM layer, exercised without any network access.

Nothing here contacts a provider. What is under test is the machinery around
the call — pulling JSON out of a chatty reply, choosing a provider, and the
guarantee that a failing or absent model degrades the run rather than stopping
it. The provider classes themselves are thin HTTP wrappers; their transport is
mocked at the requests boundary.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from placement_ai.llm.base import LLMError, extract_json_object
from placement_ai.llm.gemini import GeminiProvider
from placement_ai.llm.grok import GrokProvider
from placement_ai.llm.registry import get_provider, provider_status, resolve_secret
from placement_ai.planner.narrator import provenance_summary, write_prediction_advice
from placement_ai.planner.planner import Planner
from placement_ai.plans import StageSource
from placement_ai.profiling import canonicalize_columns, classify_target_labels, profile_dataframe

# ── JSON extraction ──────────────────────────────────────────────────────────


def test_bare_json_parses():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_a_fenced_block_parses():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object("```\n{\"a\": 1}\n```") == {"a": 1}


def test_commentary_around_the_object_is_ignored():
    text = 'Here is the plan you asked for:\n{"a": 1}\nHope that helps!'
    assert extract_json_object(text) == {"a": 1}


def test_nested_braces_are_balanced_correctly():
    payload = {"outer": {"inner": [1, 2, {"deep": True}]}, "after": "x"}
    assert extract_json_object(f"prefix {json.dumps(payload)} suffix") == payload


def test_a_brace_inside_a_string_does_not_end_the_object():
    text = '{"note": "use {curly} braces", "n": 2}'
    assert extract_json_object(text) == {"note": "use {curly} braces", "n": 2}


def test_an_escaped_quote_does_not_end_the_string():
    text = r'{"note": "he said \"hi\"", "n": 1}'
    assert extract_json_object(text)["n"] == 1


def test_an_empty_reply_raises():
    with pytest.raises(LLMError, match="empty"):
        extract_json_object("   ")


def test_a_reply_with_no_object_raises():
    with pytest.raises(LLMError, match="No JSON object"):
        extract_json_object("I am afraid I cannot help with that.")


def test_a_bare_array_is_not_accepted():
    """Every stage contract is an object; an array means the model misread it."""
    with pytest.raises(LLMError):
        extract_json_object("[1, 2, 3]")


# ── provider selection ───────────────────────────────────────────────────────


def test_no_key_anywhere_means_no_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("placement_ai.llm.registry._from_streamlit_secrets", lambda name: None)
    assert get_provider("auto") is None


def test_a_key_selects_its_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("placement_ai.llm.registry._from_streamlit_secrets", lambda name: None)
    provider = get_provider("auto")
    assert provider is not None and provider.name == "gemini"


def test_a_provider_can_be_named_explicitly(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("XAI_API_KEY", "x")
    monkeypatch.setattr("placement_ai.llm.registry._from_streamlit_secrets", lambda name: None)
    assert get_provider("grok").name == "grok"


def test_generation_can_be_turned_off_even_with_a_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    for value in ("off", "none", "offline", "heuristic"):
        assert get_provider(value) is None


def test_status_reports_which_keys_are_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("placement_ai.llm.registry._from_streamlit_secrets", lambda name: None)
    status = provider_status()
    assert status["gemini"] is True
    assert status["grok"] is False


def test_a_blank_key_does_not_count(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.setattr("placement_ai.llm.registry._from_streamlit_secrets", lambda name: None)
    assert resolve_secret("GEMINI_API_KEY") == ""


def test_streamlit_secrets_are_consulted_when_the_env_is_empty(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "placement_ai.llm.registry._from_streamlit_secrets",
        lambda name: "from-secrets" if name == "GEMINI_API_KEY" else None,
    )
    assert resolve_secret("GEMINI_API_KEY") == "from-secrets"


# ── transport ────────────────────────────────────────────────────────────────


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


def gemini_body(text, thought=False):
    parts = [{"text": text}]
    if thought:
        parts.insert(0, {"text": "let me think about this", "thought": True})
    return {"candidates": [{"content": {"parts": parts}}], "usageMetadata": {"totalTokenCount": 9}}


def test_gemini_parses_a_normal_reply():
    with patch("requests.post", return_value=_Response(200, gemini_body('{"ok": true}'))):
        result = GeminiProvider("k", "gemini-2.0-flash", 30).complete_json("s", "u")
    assert result.data == {"ok": True}
    assert result.provider == "gemini"
    assert result.latency_ms >= 0


def test_gemini_ignores_reasoning_parts():
    """A thinking model interleaves reasoning; including it would precede the JSON."""
    with patch("requests.post", return_value=_Response(200, gemini_body('{"ok": 1}', thought=True))):
        result = GeminiProvider("k", "m", 30).complete_json("s", "u")
    assert result.data == {"ok": 1}
    assert "think about this" not in result.raw_text


def test_gemini_reports_an_http_error():
    with (
        patch("requests.post", return_value=_Response(400, text="bad request")),
        pytest.raises(LLMError, match="400"),
    ):
        GeminiProvider("k", "m", 30).complete_json("s", "u")


def test_a_retired_gemini_model_names_the_setting_to_change():
    with (
        patch("requests.post", return_value=_Response(404, text="model not found")),
        pytest.raises(LLMError, match="GEMINI_MODEL"),
    ):
        GeminiProvider("k", "gemini-1.0-ancient", 30).complete_json("s", "u")


def test_gemini_reports_truncation_rather_than_bad_json():
    """A reply cut off mid-object is not "no JSON found" — say which it is."""
    body = {
        "candidates": [
            {"content": {"parts": [{"text": '{"headline": "half a sen'}]},
             "finishReason": "MAX_TOKENS"}
        ],
        "usageMetadata": {"thoughtsTokenCount": 1900},
    }
    with (
        patch("requests.post", return_value=_Response(200, body)),
        pytest.raises(LLMError, match="output limit"),
    ):
        GeminiProvider("k", "m", 30).complete_json("s", "u")


def test_gemini_reports_a_blocked_prompt():
    body = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
    with (
        patch("requests.post", return_value=_Response(200, body)),
        pytest.raises(LLMError, match="SAFETY"),
    ):
        GeminiProvider("k", "m", 30).complete_json("s", "u")


def test_gemini_reports_a_truncated_reply():
    body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
    with (
        patch("requests.post", return_value=_Response(200, body)),
        pytest.raises(LLMError, match="MAX_TOKENS"),
    ):
        GeminiProvider("k", "m", 30).complete_json("s", "u")


def test_a_transport_failure_becomes_an_llm_error():
    import requests

    with (
        patch("requests.post", side_effect=requests.ConnectionError("no route")),
        pytest.raises(LLMError, match="request failed"),
    ):
        GeminiProvider("k", "m", 30).complete_json("s", "u")


def test_grok_parses_a_chat_completion():
    body = {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"total_tokens": 12},
    }
    with patch("requests.post", return_value=_Response(200, body)):
        result = GrokProvider("k", "grok-3-mini", 30).complete_json("s", "u")
    assert result.data == {"ok": True}
    assert result.usage["total_tokens"] == 12


def test_grok_reports_empty_content():
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    with (
        patch("requests.post", return_value=_Response(200, body)),
        pytest.raises(LLMError, match="length"),
    ):
        GrokProvider("k", "m", 30).complete_json("s", "u")


def test_a_provider_without_a_key_refuses_to_call():
    with pytest.raises(LLMError, match="No Gemini API key"):
        GeminiProvider("", "m", 30).complete_json("s", "u")


# ── the fallback ladder ──────────────────────────────────────────────────────


class _StubProvider:
    """A provider that answers from a scripted list, then raises."""

    name = "stub"
    model = "stub-1"
    label = "stub:stub-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete_json(self, system, user, max_output_tokens=8192, temperature=0.2):
        from placement_ai.llm.base import LLMResult

        self.calls += 1
        if not self.replies:
            raise LLMError("out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(
            data=reply, raw_text=json.dumps(reply), provider="stub", model="stub-1", latency_ms=1.0
        )


@pytest.fixture
def planning_context(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    return profile_dataframe(frame), classify_target_labels(frame["outcome"])


def test_a_dead_provider_falls_through_to_the_rules(planning_context):
    profile, info = planning_context
    provider = _StubProvider([LLMError("service unavailable")] * 8)
    result = Planner(provider).build_plan(profile, "outcome", info, imbalance_pp=5.0)

    assert all(p.source is StageSource.heuristic for p in result.plan.provenance)
    assert result.plan.schema_plan.feature_columns  # still a usable plan
    assert any("built-in rules" in w for w in result.warnings)


def test_a_transport_error_skips_the_repair_attempt(planning_context):
    """Showing the model its own output cannot fix a rate limit."""
    profile, info = planning_context
    provider = _StubProvider([LLMError("429")] * 8)
    Planner(provider).build_plan(profile, "outcome", info, imbalance_pp=5.0)
    assert provider.calls == 4  # one attempt per stage, not two


def test_a_rejected_reply_triggers_exactly_one_repair(planning_context):
    """The model stage rejects an unknown algorithm outright, unlike the schema
    stage, which is forgiving enough to salvage almost any reply."""
    profile, info = planning_context
    provider = _StubProvider(
        [LLMError("x"), LLMError("x"), LLMError("x")]  # schema, cleaning, features
        + [{"candidates": [{"algorithm": "nope"}]}, {"candidates": [{"algorithm": "still_nope"}]}]
    )
    result = Planner(provider).build_plan(profile, "outcome", info, imbalance_pp=5.0)
    # Three stages give up after one call each; the model stage retries once.
    assert provider.calls == 5
    assert result.plan.model_plan.candidates  # the rules still supplied models


def test_a_successful_repair_is_recorded_as_such(planning_context):
    profile, info = planning_context
    provider = _StubProvider(
        [LLMError("x"), LLMError("x"), LLMError("x")]
        + [
            {"candidates": [{"algorithm": "nope"}]},
            {"candidates": [{"algorithm": "random_forest", "params": {"n_estimators": 50}}]},
        ]
    )
    result = Planner(provider).build_plan(profile, "outcome", info, imbalance_pp=5.0)

    model_stage = next(p for p in result.plan.provenance if p.stage == "model")
    assert model_stage.source is StageSource.llm_repaired
    assert [c.algorithm.value for c in result.plan.model_plan.candidates] == ["random_forest"]


def test_a_forgiving_stage_salvages_a_malformed_reply(planning_context):
    """A wrong-typed `columns` is recovered by rule rather than retried."""
    profile, info = planning_context
    provider = _StubProvider([{"columns": "not a list"}] + [LLMError("x")] * 6)
    result = Planner(provider).build_plan(profile, "outcome", info, imbalance_pp=5.0)
    assert provider.calls == 4  # no repair was needed
    assert result.plan.schema_plan.feature_columns


def test_a_valid_reply_is_recorded_as_llm_authored(planning_context):
    profile, info = planning_context
    schema_reply = {
        "target_column": "outcome",
        "task_type": "binary_classification",
        "positive_class": "Placed",
        "summary": "Student placement records.",
        "columns": [
            {"name": "test_score", "role": "numeric_feature", "display_label": "Test score"},
            {"name": "outcome", "role": "target"},
        ],
    }
    provider = _StubProvider([schema_reply] + [LLMError("x")] * 6)
    result = Planner(provider).build_plan(profile, "outcome", info, imbalance_pp=5.0)

    stages = {p.stage: p.source for p in result.plan.provenance}
    assert stages["schema"] is StageSource.llm
    assert stages["cleaning"] is StageSource.heuristic
    assert result.plan.used_llm
    assert result.plan.llm_authored_stages == ["schema"]
    assert result.plan.schema_plan.spec("test_score").display_label == "Test score"


def test_progress_reports_the_fallback(planning_context):
    profile, info = planning_context
    events: list[tuple[str, str]] = []
    Planner(_StubProvider([LLMError("nope")] * 8)).build_plan(
        profile, "outcome", info, imbalance_pp=5.0,
        progress=lambda stage, status, detail: events.append((stage, status)),
    )
    assert ("schema", "fallback") in events


def test_provenance_summary_reads_plainly(trained):
    assert "built-in rules" in provenance_summary(trained.plan)


# ── narration ────────────────────────────────────────────────────────────────


def test_advice_falls_back_to_a_template_without_a_provider():
    advice = write_prediction_advice(
        None,
        target_column="outcome",
        predicted_label="Placed",
        probability=0.82,
        drivers=[{"label": "Test score", "value": 90, "delta": 0.1}],
        inputs={"test_score": 90},
        positive_class="Placed",
    )
    assert advice["source"] == "template"
    assert "82%" in advice["headline"]
    assert advice["drivers"][0]["direction"] == "helping"


def test_a_failing_provider_still_yields_advice():
    advice = write_prediction_advice(
        _StubProvider([LLMError("down")]),
        target_column="outcome",
        predicted_label="NotPlaced",
        probability=0.2,
        drivers=[{"label": "Attendance", "value": 40, "delta": -0.2}],
        inputs={},
        positive_class="Placed",
    )
    assert advice["source"] == "template"
    assert advice["error"]
    assert advice["drivers"][0]["direction"] == "hurting"
