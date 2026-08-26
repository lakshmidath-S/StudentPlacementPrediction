"""
placement_ai/pipeline/transformers.py
-------------------------------------
The two custom stages that carry a plan into a fitted scikit-learn pipeline.

Both are ordinary estimators — they fit, they transform, they pickle — which is
what lets the entire cleaning + feature-engineering + encoding + model chain be
saved as one joblib file. The alternative, storing the plan next to the model
and re-running it at prediction time, is how a served model drifts away from the
one that was evaluated.

Everything learned from data lives in a trailing-underscore attribute set during
``fit``: imputation values, the categories worth keeping, the maximum each
``normalize_max`` feature divides by. None of it is ever recomputed at
prediction time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from placement_ai.pipeline.dsl import apply_spec, fit_spec, missing_flag_name
from placement_ai.plans import CleaningPlan, FeaturePlan, FeatureSpec, ImputeStrategy

OTHER_LEVEL = "Other"


class CleaningTransformer(BaseEstimator, TransformerMixin):
    """Apply a CleaningPlan: select, repair, impute, bound, consolidate.

    Row-level operations (dropping duplicates, dropping rows with no label) are
    deliberately *not* here. A transformer that changed the row count would
    desynchronise X from y inside a Pipeline; the runner does those once, up
    front, on the training frame.
    """

    def __init__(
        self,
        plan: CleaningPlan | None = None,
        keep_columns: list[str] | None = None,
    ) -> None:
        self.plan = plan
        self.keep_columns = keep_columns

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: Any = None) -> CleaningTransformer:
        plan = self.plan or CleaningPlan()
        columns = list(self.keep_columns or X.columns)

        self.columns_: list[str] = [c for c in columns if c in X.columns]
        self.fill_values_: dict[str, Any] = {}
        self.kept_levels_: dict[str, list[str]] = {}
        self.numeric_columns_: set[str] = set()

        for column in self.columns_:
            rule = plan.for_column(column)
            series = X[column]
            if rule is not None and rule.coerce_numeric:
                series = _coerce_numeric(series)
                self.numeric_columns_.add(column)
            elif pd.api.types.is_numeric_dtype(series):
                self.numeric_columns_.add(column)

            strategy = rule.impute if rule else ImputeStrategy.median
            self.fill_values_[column] = _learn_fill_value(series, strategy, rule)

            if rule is not None and rule.rare_category_min_frequency:
                frequencies = series.astype("string").str.strip().value_counts(normalize=True)
                keep = frequencies[frequencies >= rule.rare_category_min_frequency]
                # Consolidating everything would leave a constant column, so the
                # rule is skipped when no level clears the bar.
                self.kept_levels_[column] = (
                    [str(level) for level in keep.index] if len(keep) else []
                )

        self.feature_names_out_: list[str] = self.columns_ + [
            missing_flag_name(column) for column in self.columns_
        ]
        return self

    # ── transform ────────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        plan = self.plan or CleaningPlan()
        frame = pd.DataFrame(index=X.index)

        for column in self.columns_:
            rule = plan.for_column(column)
            # A column absent at prediction time becomes all-missing and is then
            # imputed, so a partial upload degrades instead of erroring.
            series = X[column] if column in X.columns else pd.Series(np.nan, index=X.index)

            if rule is not None:
                if rule.coerce_numeric:
                    series = _coerce_numeric(series)
                if rule.strip_whitespace and not pd.api.types.is_numeric_dtype(series):
                    series = series.astype("string").str.strip()
                if rule.lowercase and not pd.api.types.is_numeric_dtype(series):
                    series = series.astype("string").str.lower()

            # Captured before imputation — this is the only point at which the
            # original gaps are still visible.
            frame[missing_flag_name(column)] = series.isna().astype(float)

            fill = self.fill_values_.get(column)
            if fill is not None:
                series = series.fillna(fill)

            if column in self.numeric_columns_:
                series = pd.to_numeric(series, errors="coerce")
                if rule is not None and (rule.clip_min is not None or rule.clip_max is not None):
                    series = series.clip(lower=rule.clip_min, upper=rule.clip_max)
                series = series.astype(float).fillna(0.0)
            else:
                series = series.astype("string").fillna(OTHER_LEVEL)
                keep = self.kept_levels_.get(column)
                if keep:
                    series = series.where(series.isin(keep), OTHER_LEVEL)
                series = series.astype(object)

            frame[column] = series

        return frame[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Rescue numbers stored as text: strip spaces, thousands separators, symbols."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\s*$", "", regex=True)
    )
    return pd.to_numeric(text, errors="coerce")


def _learn_fill_value(series: pd.Series, strategy: ImputeStrategy, rule: Any) -> Any:
    if strategy is ImputeStrategy.none:
        return None
    if strategy is ImputeStrategy.constant:
        return rule.fill_value if rule is not None else 0
    if strategy in (ImputeStrategy.median, ImputeStrategy.mean):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return 0.0
        return float(numeric.median() if strategy is ImputeStrategy.median else numeric.mean())
    non_null = series.dropna()
    if non_null.empty:
        return OTHER_LEVEL
    modes = non_null.mode()
    return modes.iloc[0] if len(modes) else non_null.iloc[0]


class FeatureSynthesizer(BaseEstimator, TransformerMixin):
    """Execute a FeaturePlan, appending one column per surviving spec."""

    def __init__(self, plan: FeaturePlan | None = None) -> None:
        self.plan = plan

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureSynthesizer:
        specs = list((self.plan or FeaturePlan()).features)
        self.states_: dict[str, dict[str, Any]] = {}
        self.specs_: list[FeatureSpec] = []
        self.skipped_: list[tuple[str, str]] = []

        for spec in specs:
            missing = [name for name in spec.inputs if name not in X.columns]
            if missing:
                self.skipped_.append((spec.name, f"input column(s) {missing} are not present"))
                continue
            try:
                state = fit_spec(spec, X)
                probe = apply_spec(spec, X.head(min(len(X), 50)), state)
                if probe.isna().any():
                    raise ValueError("produced missing values on a sample of the training data")
            except Exception as exc:
                self.skipped_.append((spec.name, f"{type(exc).__name__}: {exc}"))
                continue
            self.states_[spec.name] = state
            self.specs_.append(spec)

        self.feature_names_out_: list[str] = list(X.columns) + [s.name for s in self.specs_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for spec in self.specs_:
            frame[spec.name] = apply_spec(spec, frame, self.states_.get(spec.name, {}))
        # Reindex rather than slice: a column that vanished from an inference
        # frame should arrive as 0.0, not raise a KeyError.
        return frame.reindex(columns=self.feature_names_out_, fill_value=0.0)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    @property
    def synthesized_names(self) -> list[str]:
        return [spec.name for spec in self.specs_]


def prune_feature_plan(
    plan: FeaturePlan, cleaned: pd.DataFrame
) -> tuple[FeaturePlan, list[str]]:
    """Drop specs that cannot execute against the cleaned training frame.

    Run before the pipeline is assembled, so the encoder's column list is fixed
    and correct. Without it, a single unrunnable feature would either crash the
    fit or silently leave the ColumnTransformer asking for a column that the
    synthesizer never produced.
    """
    probe = FeatureSynthesizer(plan=plan).fit(cleaned)
    warnings = [f"Dropped feature {name!r}: {reason}." for name, reason in probe.skipped_]
    return FeaturePlan(features=probe.specs_, notes=plan.notes), warnings
