"""
placement_ai/llm/openai_compat.py
---------------------------------
Shared transport for every provider that speaks the OpenAI chat-completions
shape — currently xAI and OpenRouter, and most others if one is added later.

Only three things vary between them: the endpoint, any extra headers, and how
much of the response envelope is worth recording. Everything else — the request
body, JSON mode, error handling, extracting the message content — is identical,
so it lives here once rather than being copied per vendor.
"""

from __future__ import annotations

import time
from typing import Any

from placement_ai.llm.base import LLMError, LLMProvider, LLMResult, extract_json_object
from placement_ai.llm.http import post_with_retry


class OpenAICompatibleProvider(LLMProvider):
    """A chat-completions endpoint that returns `choices[0].message.content`."""

    endpoint: str = ""
    # Some gateways want identifying headers; the base sends none.
    extra_headers: dict[str, str] = {}

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _payload(
        self, system_prompt: str, user_prompt: str, max_output_tokens: int, temperature: float
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMError(f"No {self.name} API key configured.")

        started = time.perf_counter()
        response = post_with_retry(
            self.endpoint,
            provider=self.name,
            json=self._payload(system_prompt, user_prompt, max_output_tokens, temperature),
            timeout=self.timeout,
            headers=self._headers(),
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            raise LLMError(
                f"{self.name} returned HTTP {response.status_code}: "
                f"{response.text.strip()[:300]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.name} returned a non-JSON envelope: {exc}") from exc

        # A gateway can report a provider-side failure inside a 200 response.
        if isinstance(body.get("error"), dict):
            error = body["error"]
            raise LLMError(
                f"{self.name} reported an error: {error.get('message', error)}"
            )

        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"{self.name} returned no choices.")

        text = (choices[0].get("message") or {}).get("content") or ""
        if not text.strip():
            reason = choices[0].get("finish_reason", "unknown")
            raise LLMError(f"{self.name} produced no content (finish_reason={reason}).")

        return LLMResult(
            data=extract_json_object(text),
            raw_text=text,
            provider=self.name,
            # Routed gateways can answer with a different model than the one
            # requested, and provenance should record what actually replied.
            model=str(body.get("model") or self.model),
            latency_ms=latency_ms,
            usage=body.get("usage", {}) or {},
        )
