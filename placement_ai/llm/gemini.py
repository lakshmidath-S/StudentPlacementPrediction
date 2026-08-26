"""
placement_ai/llm/gemini.py
--------------------------
Google Gemini back-end, spoken over plain REST.

Deliberately not the ``google-genai`` SDK: one ``requests`` call has no
dependency to keep in step with the rest of the stack, and the generateContent
body has been stable across model generations. The trade-off is that streaming
and file uploads are unavailable — neither of which a planner needs.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from placement_ai.config import GEMINI_ENDPOINT
from placement_ai.llm.base import LLMError, LLMProvider, LLMResult, extract_json_object


class GeminiProvider(LLMProvider):
    name = "gemini"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMError("No Gemini API key configured.")

        url = GEMINI_ENDPOINT.format(model=self.model)
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                # JSON mode. The prompt still carries the schema, because
                # responseSchema rejects several shapes pydantic emits
                # (notably $defs and nullable unions).
                "responseMimeType": "application/json",
            },
        }

        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.RequestException as exc:
            raise LLMError(f"Gemini request failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            raise LLMError(
                f"Gemini returned HTTP {response.status_code}: "
                f"{response.text.strip()[:300]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"Gemini returned a non-JSON envelope: {exc}") from exc

        text = _first_candidate_text(body)
        return LLMResult(
            data=extract_json_object(text),
            raw_text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            usage=body.get("usageMetadata", {}) or {},
        )


def _first_candidate_text(body: dict[str, Any]) -> str:
    """Concatenate the answer parts of the first candidate.

    Thinking-enabled models interleave reasoning parts flagged ``thought: true``
    in the same list as the answer; including them would put prose in front of
    the JSON. A blocked prompt yields a candidate with no parts at all, which is
    reported with its finishReason rather than as an empty-response error.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback", {})
        blocked = feedback.get("blockReason")
        raise LLMError(
            f"Gemini returned no candidates (blockReason={blocked})"
            if blocked
            else "Gemini returned no candidates."
        )

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    chunks = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and "text" in part and not part.get("thought")
    ]
    if not chunks:
        reason = candidate.get("finishReason", "unknown")
        raise LLMError(
            f"Gemini produced no text (finishReason={reason}). "
            "MAX_TOKENS here usually means the plan needs a larger output budget."
        )
    return "".join(chunks)
