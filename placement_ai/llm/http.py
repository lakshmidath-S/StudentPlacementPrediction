"""
placement_ai/llm/http.py
------------------------
One POST helper, with retries for the failures that are worth retrying.

Free tiers fail transiently far more often than paid ones: 429 when a shared
free model is busy, 503 when a hosted model is briefly overloaded. Both clear in
seconds. Without a retry, a single blip sends that planning stage to the
heuristic planner for the rest of the run — a real quality loss for something
that would have succeeded a second later.

The retry budget is deliberately small. A training run makes five calls, and a
user is watching a progress bar; spending a minute inside one stage to avoid a
fallback is a worse trade than falling back. A provider that asks for a longer
wait than the cap gets it honoured up to the cap and then abandoned.
"""

from __future__ import annotations

import time
from datetime import UTC
from typing import Any

import requests

from placement_ai.llm.base import LLMError

# Codes that mean "try again", as opposed to "you asked for the wrong thing".
# 401/403/404 are absent on purpose: a bad key, an unpaid account or a retired
# model will fail identically forever, and retrying only delays the fallback.
TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

MAX_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.5
# Honour Retry-After only up to this; beyond it, falling back is faster.
MAX_BACKOFF_SECONDS = 8.0


def _retry_after_seconds(response: requests.Response) -> float | None:
    """Parse Retry-After, which may be seconds or an HTTP date."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(raw)
        if target is None:
            return None
        from datetime import datetime

        return max((target - datetime.now(UTC)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def post_with_retry(
    url: str,
    *,
    provider: str,
    json: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    params: dict[str, Any] | None = None,
    attempts: int = MAX_ATTEMPTS,
) -> requests.Response:
    """POST, retrying transient failures. Raises LLMError when out of attempts."""
    backoff = INITIAL_BACKOFF_SECONDS
    last_detail = ""

    for attempt in range(1, max(attempts, 1) + 1):
        try:
            response = requests.post(
                url, json=json, headers=headers, params=params, timeout=timeout
            )
        except requests.RequestException as exc:
            last_detail = f"{type(exc).__name__}: {exc}"
            if attempt >= attempts:
                raise LLMError(f"{provider} request failed: {last_detail}") from exc
            time.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff *= BACKOFF_MULTIPLIER
            continue

        if response.status_code in TRANSIENT_STATUS and attempt < attempts:
            wait = _retry_after_seconds(response) or backoff
            last_detail = f"HTTP {response.status_code}"
            time.sleep(min(wait, MAX_BACKOFF_SECONDS))
            backoff *= BACKOFF_MULTIPLIER
            continue

        return response

    # Unreachable in practice: the loop either returns or raises.
    raise LLMError(f"{provider} request failed after {attempts} attempts ({last_detail}).")
