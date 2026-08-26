"""
placement_ai/llm/grok.py
------------------------
xAI Grok back-end.

The xAI API is OpenAI-compatible, so this is a chat-completions call with
``response_format: json_object``. Same reasoning as the Gemini module for using
``requests`` rather than an SDK.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from placement_ai.config import GROK_ENDPOINT
from placement_ai.llm.base import LLMError, LLMProvider, LLMResult, extract_json_object


class GrokProvider(LLMProvider):
    name = "grok"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResult:
        if not self.api_key:
            raise LLMError("No xAI API key configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        started = time.perf_counter()
        try:
            response = requests.post(
                GROK_ENDPOINT,
                json=payload,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except requests.RequestException as exc:
            raise LLMError(f"Grok request failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            raise LLMError(
                f"Grok returned HTTP {response.status_code}: {response.text.strip()[:300]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"Grok returned a non-JSON envelope: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise LLMError("Grok returned no choices.")
        text = (choices[0].get("message") or {}).get("content") or ""
        if not text.strip():
            reason = choices[0].get("finish_reason", "unknown")
            raise LLMError(f"Grok produced no content (finish_reason={reason}).")

        return LLMResult(
            data=extract_json_object(text),
            raw_text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            usage=body.get("usage", {}) or {},
        )
