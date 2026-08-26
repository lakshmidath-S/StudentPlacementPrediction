"""
placement_ai/config.py
----------------------
Every tunable path and environment setting in one place.

Nothing here reads a file or contacts a network at import time — the module is
safe to import from a Streamlit rerun, a test, or a CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# ── Tenancy ──────────────────────────────────────────────────────────────────
# One directory per workspace. Override with PLACEMENT_AI_HOME to point at a
# mounted volume in a cloud deployment, where the repo itself is read-only.
WORKSPACE_ROOT = Path(os.getenv("PLACEMENT_AI_HOME", str(REPO_ROOT / "workspaces")))

# Shipped so a first-time user can click "load the sample dataset" and reach a
# trained model without having to find a CSV of their own first.
SAMPLE_DATASET_PATH = REPO_ROOT / "data" / "raw" / "synthetic_placement_dataset.csv"

# ── LLM providers ────────────────────────────────────────────────────────────
# "auto" picks the first provider holding a usable key, then falls back to the
# deterministic heuristic planner. Force one with LLM_PROVIDER=gemini|grok|off.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

GROK_API_KEY_ENV = "XAI_API_KEY"
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini")
GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
# One retry is a repair pass: the parse error is fed back so the model can fix
# its own JSON. Beyond that the stage falls through to the heuristic planner.
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))

# ── Guardrails ───────────────────────────────────────────────────────────────
# An LLM asked for "useful features" will happily propose a hundred. Each one
# costs a column through the ColumnTransformer and a row in the model card.
MAX_SYNTHESIZED_FEATURES = int(os.getenv("MAX_SYNTHESIZED_FEATURES", "40"))
MAX_TRAINING_ROWS = int(os.getenv("MAX_TRAINING_ROWS", "250000"))
MIN_TRAINING_ROWS = 30
# Below this many rows per class a stratified split stops being meaningful.
MIN_ROWS_PER_CLASS = 5

# Cap on distinct levels a categorical may carry into the one-hot encoder.
MAX_CATEGORY_CARDINALITY = int(os.getenv("MAX_CATEGORY_CARDINALITY", "50"))
# A column whose values are almost all distinct is an identifier, not a feature.
IDENTIFIER_UNIQUE_RATIO = 0.95

RANDOM_SEED = 42
