"""
placement_ai/llm/base.py
------------------------
The provider-agnostic surface every LLM back-end implements.

There is exactly one operation — "answer this with a JSON object" — because
that is all the planner ever asks for. Keeping the interface that narrow is
what makes Gemini, Grok and "no provider at all" interchangeable.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any

# Models wrap JSON in fences more often than not, even when asked for raw JSON
# and even in JSON mode. Stripping them is cheaper than a repair round-trip.
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


class LLMError(RuntimeError):
    """Any failure to obtain a usable JSON object from a provider.

    Deliberately one exception type: the planner's response to a transport
    error, a rate limit, a refusal and unparseable output is identical — record
    it on the stage and fall back to the heuristic.
    """


@dataclass
class LLMResult:
    data: dict[str, Any]
    raw_text: str
    provider: str
    model: str
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Tries, in order: the whole string, a fenced block, then the outermost
    brace-balanced span. The last one matters because models like to append a
    sentence of commentary after the closing brace.
    """
    if not text or not text.strip():
        raise LLMError("The model returned an empty response.")

    candidates: list[str] = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text.strip().replace("\n", " ")[:300]
    raise LLMError(f"No JSON object found in the model response. Got: {preview}")


class LLMProvider(abc.ABC):
    """One hosted model, reachable over HTTP."""

    name: str = "provider"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    @abc.abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Return the model response parsed into a dict, or raise LLMError."""
        ...
