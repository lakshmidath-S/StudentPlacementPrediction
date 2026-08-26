"""
placement_ai/llm/grok.py
------------------------
xAI Grok. The API is OpenAI-compatible, so this is only an endpoint.
"""

from __future__ import annotations

from placement_ai.config import GROK_ENDPOINT
from placement_ai.llm.openai_compat import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    name = "grok"
    endpoint = GROK_ENDPOINT
