"""
placement_ai/pipeline/builder.py
--------------------------------
Assembling a fitted plan into scikit-learn objects.

The preprocessing chain is built once and shared by every candidate:

    clean -> synthesise features -> scale numerics / one-hot categoricals

Each candidate estimator then trains on that single transformed matrix, and the
winner is glued back onto the same preprocessor to make the deployable pipeline.
Fitting the preprocessor per candidate would be both slower and subtly wrong —
the models would no longer be compared on identical inputs.

Class imbalance is handled with ``sample_weight`` for every algorithm rather
than each one's own ``class_weight``/``scale_pos_weight`` parameter. They are
equivalent, but only ``sample_weight`` is accepted by all of them, which keeps
one code path instead of a per-algorithm branch that breaks whenever an upstream
library changes its signature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from placement_ai.config import RANDOM_SEED
from placement_ai.pipeline.transformers import CleaningTransformer, FeatureSynthesizer
from placement_ai.plans import (
    Algorithm,
    CandidateModel,
    CleaningPlan,
    FeaturePlan,
    SchemaPlan,
)

ALGORITHM_CLASSES: dict[Algorithm, Any] = {
    Algorithm.logistic_regression: LogisticRegression,
    Algorithm.random_forest: RandomForestClassifier,
    Algorithm.extra_trees: ExtraTreesClassifier,
    Algorithm.gradient_boosting: GradientBoostingClassifier,
    Algorithm.hist_gradient_boosting: HistGradientBoostingClassifier,
}

DISPLAY_NAMES: dict[str, str] = {
    Algorithm.logistic_regression.value: "Logistic Regression",
    Algorithm.random_forest.value: "Random Forest",
    Algorithm.extra_trees.value: "Extra Trees",
    Algorithm.gradient_boosting.value: "Gradient Boosting",
    Algorithm.hist_gradient_boosting.value: "Histogram Gradient Boosting",
    Algorithm.xgboost.value: "XGBoost",
    "ensemble": "Weighted Ensemble",
}


def display_name(algorithm: str) -> str:
    return DISPLAY_NAMES.get(algorithm, algorithm.replace("_", " ").title())


def _resolve_class(algorithm: Algorithm) -> Any:
    if algorithm is Algorithm.xgboost:
        from xgboost import XGBClassifier

        return XGBClassifier
    return ALGORITHM_CLASSES[algorithm]


def _coerce_params(estimator_cls: Any, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only parameters the estimator accepts, cast to the right type.

    A generated plan will occasionally propose a hyperparameter that belongs to
    a different algorithm, or send ``"300"`` where an int is wanted. Both are
    recoverable, and recovering beats failing a ten-minute training run on a
    typo.
    """
    try:
        defaults = estimator_cls().get_params()
    except Exception:
        return dict(params), []

    accepted: dict[str, Any] = {}
    rejected: list[str] = []
    for key, value in params.items():
        if key not in defaults:
            rejected.append(key)
            continue
        default = defaults[key]
        accepted[key] = _cast_like(value, default)
    return accepted, rejected


def _cast_like(value: Any, default: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(value, int):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if isinstance(default, float) and not isinstance(value, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if isinstance(value, str) and default is None:
        # e.g. max_depth: "none" -> None
        return None if value.strip().lower() in {"none", "null", ""} else value
    return value


def build_estimator(candidate: CandidateModel, n_classes: int) -> tuple[Any, list[str]]:
    """Instantiate one candidate, returning it alongside any ignored params."""
    estimator_cls = _resolve_class(candidate.algorithm)
    params, rejected = _coerce_params(estimator_cls, candidate.params)

    defaults: dict[str, Any] = {}
    if candidate.algorithm is Algorithm.logistic_regression:
        defaults = {"max_iter": 2000, "random_state": RANDOM_SEED}
    elif candidate.algorithm in (Algorithm.random_forest, Algorithm.extra_trees):
        defaults = {"n_estimators": 300, "n_jobs": -1, "random_state": RANDOM_SEED}
    elif candidate.algorithm is Algorithm.xgboost:
        defaults = {
            "n_estimators": 300,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "tree_method": "hist",
        }
    else:
        defaults = {"random_state": RANDOM_SEED}

    merged = {**defaults, **params}
    # Only pass what this class really accepts — the defaults above include
    # keys (n_jobs, random_state) that not every estimator declares.
    merged, _ = _coerce_params(estimator_cls, merged)
    return estimator_cls(**merged), rejected


def build_preprocessor(
    schema: SchemaPlan,
    cleaning: CleaningPlan,
    features: FeaturePlan,
) -> tuple[Pipeline, list[str], list[str]]:
    """Build the shared clean -> synthesise -> encode chain.

    Also returns the numeric and categorical column lists the encoder selects,
    which the model card needs and which no fitted object exposes in a readable
    form afterwards.
    """
    numeric_columns = list(schema.numeric_features) + [f.name for f in features.features]
    categorical_columns = list(schema.categorical_features)

    numeric_branch = Pipeline(
        steps=[
            # The cleaner already imputes; this is a backstop for a derived
            # column that turns out non-finite on unseen data.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_branch, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_branch, categorical_columns))

    encoder = ColumnTransformer(transformers=transformers, remainder="drop")

    preprocessor = Pipeline(
        steps=[
            (
                "clean",
                CleaningTransformer(plan=cleaning, keep_columns=schema.feature_columns),
            ),
            ("features", FeatureSynthesizer(plan=features)),
            ("encode", encoder),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def encoded_feature_names(preprocessor: Pipeline) -> list[str]:
    """Readable names for the columns the estimator actually sees."""
    encoder: ColumnTransformer = preprocessor.named_steps["encode"]
    try:
        raw = encoder.get_feature_names_out()
    except Exception:
        return []
    return [str(name).split("__", 1)[-1] for name in raw]


def compute_sample_weights(y: pd.Series, strategy: str, custom: dict[str, float] | None) -> np.ndarray | None:
    """Per-row weights that offset class imbalance.

    Returns None when weighting is off, so the caller can skip passing
    ``sample_weight`` entirely rather than passing a vector of ones.
    """
    if strategy == "none":
        return None

    labels = pd.Series(y).astype(object)
    counts = labels.value_counts()

    if strategy == "custom" and custom:
        weights = {key: float(value) for key, value in custom.items()}
        mapped = labels.map(lambda value: weights.get(str(value), 1.0))
        return mapped.to_numpy(dtype=float)

    # "balanced": n_samples / (n_classes * count), matching sklearn's own rule.
    total = float(len(labels))
    n_classes = float(len(counts))
    per_class = {value: total / (n_classes * float(count)) for value, count in counts.items()}
    return labels.map(per_class).to_numpy(dtype=float)


def assemble_final_pipeline(preprocessor: Pipeline, estimator: Any) -> Pipeline:
    """Glue the fitted preprocessor to the winning estimator.

    The steps are already fitted, so this object is ready to predict; it is
    never re-fitted, which is what keeps the deployed model byte-identical to
    the one that was evaluated.
    """
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
