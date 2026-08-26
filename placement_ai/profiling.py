"""
placement_ai/profiling.py
-------------------------
Deterministic dataset profiling.

This runs *before* any LLM call and is never skipped. The profile is both the
prompt payload the planner reasons over and the evidence the heuristic planner
falls back on, so it has to be correct on its own — an LLM that never answers
must not stop the pipeline.

Profiling deliberately reports rather than decides. It says "this column is 94%
unique integers named student_id"; deciding that this makes it an identifier to
drop is the planner's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from placement_ai.config import IDENTIFIER_UNIQUE_RATIO, MAX_CATEGORY_CARDINALITY
from placement_ai.utils import jsonify, safe_column_name

# Words that make a column a plausible prediction target. Ordered — the first
# match wins when several columns look target-shaped.
TARGET_NAME_HINTS = (
    "placement_status",
    "placed",
    "placement",
    "target",
    "label",
    "outcome",
    "result",
    "status",
    "churn",
    "converted",
    "approved",
    "selected",
    "hired",
)

# Values that read as "yes" in a two-level target, lowercased and de-spaced.
POSITIVE_TOKENS = {
    "yes", "y", "true", "t", "1", "placed", "pass", "passed", "success",
    "approved", "selected", "hired", "positive", "win", "won", "admitted",
}
NEGATIVE_TOKENS = {
    "no", "n", "false", "f", "0", "notplaced", "not_placed", "unplaced",
    "fail", "failed", "rejected", "negative", "loss", "lost", "denied",
}


@dataclass
class ColumnProfile:
    """Everything measurable about one column, with no interpretation applied."""

    name: str
    dtype: str
    kind: str  # numeric | categorical | boolean | datetime | text | constant | empty
    count: int
    missing: int
    missing_pct: float
    n_unique: int
    unique_ratio: float
    sample_values: list[Any] = field(default_factory=list)
    top_values: list[dict[str, Any]] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    looks_like_identifier: bool = False
    is_numeric_like_text: bool = False
    is_integral: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Compact form handed to the LLM.

        Trimmed hard on purpose — a 60-column dataset must still fit
        comfortably inside one prompt alongside the instructions.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "missing_pct": round(self.missing_pct, 2),
            "n_unique": self.n_unique,
            "unique_ratio": round(self.unique_ratio, 3),
            "examples": [jsonify(v) for v in self.sample_values[:6]],
        }
        if self.kind == "numeric":
            payload["stats"] = {
                "min": jsonify(self.minimum),
                "max": jsonify(self.maximum),
                "mean": None if self.mean is None else round(self.mean, 4),
                "std": None if self.std is None else round(self.std, 4),
            }
        if self.top_values:
            payload["top_values"] = [
                {"value": jsonify(tv["value"]), "count": tv["count"]}
                for tv in self.top_values[:8]
            ]
        if self.looks_like_identifier:
            payload["looks_like_identifier"] = True
        if self.is_numeric_like_text:
            payload["numeric_values_stored_as_text"] = True
        return payload


@dataclass
class DatasetProfile:
    """Whole-dataset view: shape, per-column profiles, and target candidates."""

    n_rows: int
    n_columns: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    target_candidates: list[str] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        for profile in self.columns:
            if profile.name == name:
                return profile
        return None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_payload(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_rows": self.duplicate_rows,
            "target_candidates": self.target_candidates,
            "columns": [c.to_payload() for c in self.columns],
        }


def canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """snake_case every header, returning the frame and the original -> new map.

    Uploaded CSVs arrive with headers like ``Workshops/Certifications`` or
    ``  CGPA  ``. Renaming once here means every downstream layer — plans,
    manifests, the generated input form — speaks one naming convention, while
    the map is kept so user-facing errors can quote the header they typed.
    """
    rename: dict[str, str] = {}
    seen: set[str] = set()
    for original in df.columns:
        candidate = safe_column_name(original)
        # Two different headers can collapse onto the same snake_case name.
        if candidate in seen:
            suffix = 2
            while f"{candidate}_{suffix}" in seen:
                suffix += 1
            candidate = f"{candidate}_{suffix}"
        seen.add(candidate)
        rename[str(original)] = candidate
    return df.rename(columns=rename), rename


def _series_kind(series: pd.Series, n_unique: int, n_rows: int) -> str:
    if series.isna().all():
        return "empty"
    if n_unique <= 1:
        return "constant"
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    if pdt.is_numeric_dtype(series):
        # A numeric column with exactly two levels is a flag, not a measurement.
        return "boolean" if n_unique == 2 else "numeric"
    # Object/string: decide between a categorical and free text.
    if n_unique <= MAX_CATEGORY_CARDINALITY or (n_rows and n_unique / n_rows < 0.5):
        return "categorical"
    return "text"


def _coerces_to_numeric(series: pd.Series) -> bool:
    """True when an object column is really numbers wearing a string dtype.

    Common in exported spreadsheets ("7.7 ", "1,200"). Reported, not fixed —
    the cleaning plan decides whether to coerce it.
    """
    if pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return False
    sample = series.dropna().astype(str).str.strip().str.replace(",", "", regex=False)
    if sample.empty:
        return False
    converted = pd.to_numeric(sample, errors="coerce")
    return bool(converted.notna().mean() > 0.9)


def profile_column(series: pd.Series, name: str, n_rows: int) -> ColumnProfile:
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    missing = int(series.isna().sum())
    kind = _series_kind(series, n_unique, n_rows)

    profile = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        kind=kind,
        count=int(series.shape[0]),
        missing=missing,
        missing_pct=(missing / n_rows * 100.0) if n_rows else 0.0,
        n_unique=n_unique,
        unique_ratio=(n_unique / n_rows) if n_rows else 0.0,
        sample_values=[jsonify(v) for v in non_null.head(6).tolist()],
    )

    if kind in {"numeric", "boolean"}:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            profile.minimum = float(numeric.min())
            profile.maximum = float(numeric.max())
            profile.mean = float(numeric.mean())
            profile.std = float(numeric.std()) if len(numeric) > 1 else 0.0
            profile.median = float(numeric.median())
            # Whether every value is a whole number, which decides count-vs-measure
            # downstream. A float column whose maximum happens to land on 10.0 is
            # not a count, so the check has to look at all the values.
            profile.is_integral = bool(
                numeric.dtype.kind in "iub" or (numeric % 1 == 0).all()
            )

    if kind in {"categorical", "boolean", "constant"} and not non_null.empty:
        counts = non_null.value_counts().head(10)
        profile.top_values = [
            {"value": jsonify(idx), "count": int(cnt)} for idx, cnt in counts.items()
        ]

    profile.is_numeric_like_text = _coerces_to_numeric(series)
    profile.looks_like_identifier = bool(
        (profile.unique_ratio >= IDENTIFIER_UNIQUE_RATIO and n_rows > 20)
        or name.endswith(("_id", "_no", "_number"))
        or name in {"id", "index", "sl_no", "serial", "roll", "roll_no", "uid"}
    )
    return profile


def find_target_candidates(profiles: list[ColumnProfile]) -> list[str]:
    """Rank columns by how much they look like a supervised target.

    Name match dominates, then low cardinality, then position. The LLM receives
    this list as a hint and is free to disagree; the heuristic planner takes the
    top entry as fact.
    """
    scored: list[tuple[float, str]] = []
    for position, profile in enumerate(profiles):
        if profile.kind in {"constant", "empty", "text", "datetime"}:
            continue
        if profile.looks_like_identifier:
            continue
        score = 0.0
        lowered = profile.name.lower()
        for rank, hint in enumerate(TARGET_NAME_HINTS):
            if lowered == hint:
                score += 100 - rank
                break
            if hint in lowered:
                score += 60 - rank
                break
        if profile.n_unique == 2:
            score += 30
        elif 2 < profile.n_unique <= 10:
            score += 10
        # Targets conventionally sit at the end of an exported table.
        score += (position / max(len(profiles) - 1, 1)) * 5
        if score > 0:
            scored.append((score, profile.name))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:5]]


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Profile an already-canonicalized frame."""
    n_rows = int(len(df))
    profiles = [profile_column(df[col], str(col), n_rows) for col in df.columns]
    try:
        duplicates = int(df.duplicated().sum())
    except TypeError:
        # Unhashable cell contents (lists, dicts) make duplicate detection fail;
        # it is a nice-to-have, not a reason to abort profiling.
        duplicates = 0
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        duplicate_rows=duplicates,
        columns=profiles,
        target_candidates=find_target_candidates(profiles),
    )


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def pick_positive_class(levels: list[Any]) -> Any:
    """Choose which of two labels means "the event happened".

    Token match first ("Placed" beats "NotPlaced"), then the label whose
    counterpart is explicitly negative, then the larger number. Getting this
    wrong inverts every probability the product shows, so the chosen class is
    surfaced in the UI before training rather than assumed silently.
    """
    tokens = {_normalize_token(level): level for level in levels}

    positives = [lvl for tok, lvl in tokens.items() if tok in POSITIVE_TOKENS]
    negatives = [lvl for tok, lvl in tokens.items() if tok in NEGATIVE_TOKENS]
    if len(positives) == 1:
        return positives[0]
    if len(negatives) == 1 and len(levels) == 2:
        return next(lvl for lvl in levels if lvl != negatives[0])

    numeric = pd.to_numeric(pd.Series(levels), errors="coerce")
    if numeric.notna().all():
        return levels[int(numeric.astype(float).to_numpy().argmax())]
    return levels[0]


def classify_target_labels(series: pd.Series) -> dict[str, Any]:
    """Describe a candidate target: how many classes, and which reads positive.

    Returned rather than applied, so both planners and the UI can show the user
    "Placed will be treated as the positive class" before training starts.
    """
    values = series.dropna()
    levels = list(values.value_counts().items())
    n_classes = len(levels)
    result: dict[str, Any] = {
        "n_classes": n_classes,
        "levels": [{"value": jsonify(v), "count": int(c)} for v, c in levels[:20]],
        "positive_class": None,
        "task_type": "regression",
    }
    if n_classes == 2:
        result["task_type"] = "binary_classification"
        result["positive_class"] = jsonify(pick_positive_class([v for v, _ in levels]))
    elif 2 < n_classes <= 20 and not pdt.is_float_dtype(values):
        result["task_type"] = "multiclass_classification"
    return result
