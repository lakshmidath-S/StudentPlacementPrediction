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

from placement_ai.config import GEMINI_ENDPOINT
from placement_ai.llm.base import LLMError, LLMProvider, LLMResult, extract_json_object
from placement_ai.llm.http import post_with_retry


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
        response = post_with_retry(
            url,
            provider="Gemini",
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            detail = response.text.strip()[:400]
            if response.status_code == 404:
                raise LLMError(
                    f"Gemini does not recognise the model {self.model!r}. Google "
                    "retires model versions on a schedule — set GEMINI_MODEL to a "
                    f"current one. Their reply: {detail}"
                )
            raise LLMError(f"Gemini returned HTTP {response.status_code}: {detail}")

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
    reason = candidate.get("finishReason", "")
    if not chunks:
        raise LLMError(
            f"Gemini produced no text (finishReason={reason or 'unknown'}). "
            "MAX_TOKENS here means thinking consumed the whole output budget."
        )
    if reason == "MAX_TOKENS":
        # The text is real but cut off mid-object, so the JSON will not parse.
        # Saying "no JSON object found" would send someone hunting the wrong bug;
        # on Gemini 3.x, reasoning tokens count against maxOutputTokens, so a
        # budget that looks generous can leave nothing for the answer.
        thoughts = (body.get("usageMetadata") or {}).get("thoughtsTokenCount")
        raise LLMError(
            "Gemini hit its output limit before finishing the JSON"
            + (f" (thinking used {thoughts} tokens)" if thoughts else "")
            + ". Raise max_output_tokens for this stage."
        )
    return "".join(chunks)
