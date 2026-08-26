"""
placement_ai/utils.py
---------------------
Small shared helpers. Kept dependency-light so every layer can import it.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def utc_now_iso() -> str:
    """Timestamp used in every manifest and history row."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str, fallback: str = "item") -> str:
    """ASCII, lowercase, hyphen-separated. Used for workspace directory names."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")
    return slug or fallback


def safe_column_name(value: str) -> str:
    """Canonical snake_case for a dataframe column coming from an unknown CSV."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    snake = re.sub(r"[^0-9a-zA-Z]+", "_", ascii_only).strip("_").lower()
    snake = re.sub(r"_+", "_", snake)
    if not snake:
        snake = "column"
    if snake[0].isdigit():
        snake = f"col_{snake}"
    return snake


def jsonify(value: Any) -> Any:
    """Coerce numpy/pandas scalars into something json.dump can write.

    Applied to everything that reaches a manifest or the history DB, because a
    numpy.float32 raises TypeError only at serialisation time — long after the
    training run that produced it has finished.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonify(v) for v in value]
    # numpy / pandas scalars expose .item(); pandas NA does not.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonify(item())
        except (ValueError, TypeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return jsonify(tolist())
        except (ValueError, TypeError):
            pass
    try:
        if value != value:  # NaN-like, including pandas.NA  # noqa: PLR0124
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with LF endings.

    newline="\n" is deliberate: manifests are SHA-256'd, and CRLF on Windows
    would change the digest of a file checked out on Linux.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonify(payload), handle, indent=2)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def truncate(value: Any, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"
