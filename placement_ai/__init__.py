"""
placement_ai
------------
An LLM-orchestrated AutoML service: a user uploads a spreadsheet, a language
model plans how to clean it, what features to derive and which models to weight,
a deterministic executor carries that plan out, and the result is saved as a
single joblib bundle that can be used for predictions until it is retrained.

The package is UI-agnostic. app.py is one caller; nothing here imports Streamlit
outside of optional secret lookup.
"""

__version__ = "2.0.0"
