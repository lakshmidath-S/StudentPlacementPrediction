"""
placement_ai/planner/heuristic.py
---------------------------------
The rule-based planner.

This is not a stub. It is the floor the whole product stands on: it authors a
complete, trainable plan from the profile alone, with no network and no key. It
runs in CI, it runs on a fresh clone, and it takes over any individual stage the
LLM fails to deliver.

Holding it to that standard has a design consequence worth stating plainly: the
LLM has to *beat* a competent baseline rather than merely exist. What it adds is
domain reading — knowing that ``backlogs`` is a negative signal, that ``cgpa``
and ``ssc_percentage`` measure the same underlying thing on different scales,
and that a ratio of technical to soft skills is worth looking at. The rules
below can only see shapes and ranges.
"""

from __future__ import annotations

from typing import Any

from placement_ai.config import (
    MAX_CATEGORY_CARDINALITY,
    MAX_SYNTHESIZED_FEATURES,
    RANDOM_SEED,
)
from placement_ai.plans import (
    Algorithm,
    CandidateModel,
    CleaningPlan,
    ColumnCleaning,
    ColumnRole,
    ColumnSpec,
    FeatureOp,
    FeaturePlan,
    FeatureSpec,
    ImputeStrategy,
    Metric,
    ModelPlan,
    SchemaPlan,
    TaskType,
    ThresholdStrategy,
)
from placement_ai.profiling import ColumnProfile, DatasetProfile


def xgboost_available() -> bool:
    """XGBoost is an optional accelerator, not a requirement."""
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


# ── Stage 1: schema ──────────────────────────────────────────────────────────


def _role_for(profile: ColumnProfile) -> tuple[ColumnRole, str]:
    if profile.looks_like_identifier:
        return ColumnRole.identifier, "Nearly unique per row — an identifier, not a signal."
    if profile.kind in {"constant", "empty"}:
        return ColumnRole.drop, "No variation, so it cannot separate the classes."
    if profile.kind == "text":
        return ColumnRole.drop, "Free text; this build does not vectorise text columns."
    if profile.kind == "datetime":
        return ColumnRole.drop, "Raw timestamps are not used as features."
    if profile.missing_pct > 60:
        return ColumnRole.drop, f"{profile.missing_pct:.0f}% missing — too sparse to impute."
    if profile.kind == "numeric":
        return ColumnRole.numeric_feature, ""
    if profile.kind == "boolean":
        # A 0/1 numeric column is already model-ready; a Yes/No string is not.
        if profile.dtype.startswith(("int", "float", "bool", "uint")):
            return ColumnRole.numeric_feature, ""
        return ColumnRole.categorical_feature, ""
    if profile.n_unique > MAX_CATEGORY_CARDINALITY:
        return ColumnRole.drop, (
            f"{profile.n_unique} distinct levels would explode the one-hot encoding."
        )
    return ColumnRole.categorical_feature, ""


def heuristic_schema_plan(
    profile: DatasetProfile,
    target_column: str,
    target_info: dict[str, Any],
) -> SchemaPlan:
    columns: list[ColumnSpec] = []
    for column in profile.columns:
        if column.name == target_column:
            columns.append(
                ColumnSpec(
                    name=column.name,
                    role=ColumnRole.target,
                    description="The outcome being predicted.",
                )
            )
            continue
        role, reason = _role_for(column)
        columns.append(ColumnSpec(name=column.name, role=role, reason=reason))

    task_type = (
        TaskType.multiclass_classification
        if target_info.get("task_type") == "multiclass_classification"
        else TaskType.binary_classification
    )
    positive = target_info.get("positive_class")

    return SchemaPlan(
        target_column=target_column,
        task_type=task_type,
        positive_class=None if task_type is TaskType.multiclass_classification else (
            None if positive is None else str(positive)
        ),
        columns=columns,
        summary=(
            f"{profile.n_rows:,} rows x {profile.n_columns} columns. "
            f"Predicting {target_column} from "
            f"{len([c for c in columns if c.role is ColumnRole.numeric_feature])} numeric and "
            f"{len([c for c in columns if c.role is ColumnRole.categorical_feature])} categorical "
            "columns, selected by column shape."
        ),
    )


# ── Stage 2: cleaning ────────────────────────────────────────────────────────


def heuristic_cleaning_plan(profile: DatasetProfile, schema: SchemaPlan) -> CleaningPlan:
    steps: list[ColumnCleaning] = []
    for name in schema.feature_columns:
        column = profile.column(name)
        if column is None:
            continue
        if name in schema.numeric_features:
            steps.append(
                ColumnCleaning(
                    column=name,
                    coerce_numeric=column.is_numeric_like_text,
                    impute=ImputeStrategy.median,
                    reason=(
                        "Median fill resists the outliers a mean would chase."
                        if column.missing
                        else "No gaps; the rule is a no-op kept for reproducibility."
                    ),
                )
            )
        else:
            steps.append(
                ColumnCleaning(
                    column=name,
                    strip_whitespace=True,
                    impute=ImputeStrategy.most_frequent,
                    # 1% of rows is roughly where a one-hot column stops
                    # carrying enough examples to learn anything from.
                    rare_category_min_frequency=0.01,
                    reason="Trim stray whitespace and fold very rare levels into Other.",
                )
            )

    return CleaningPlan(
        drop_columns=schema.dropped_columns,
        drop_duplicate_rows=profile.duplicate_rows > 0,
        drop_rows_missing_target=True,
        columns=steps,
        notes="Rule-based cleaning derived from column shape and missingness.",
    )


# ── Stage 3: features ────────────────────────────────────────────────────────

# Numeric columns are grouped by the scale they appear to live on, so a
# composite averages things that are actually comparable. Averaging a 0-100
# percentage with a 0-5 rating produces a number that means nothing.
_SCALE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("percentage", 0.0, 100.0),
    ("rating", 0.0, 10.0),
    ("count", 0.0, 50.0),
)


def _scale_band(column: ColumnProfile) -> str | None:
    if column.minimum is None or column.maximum is None:
        return None
    low, high = float(column.minimum), float(column.maximum)
    if low < 0:
        return None
    if 20.0 <= high <= 100.0:
        return "percentage"
    if high <= 10.0 and not column.is_integral:
        return "rating"
    if column.is_integral and high <= 50.0:
        return "count"
    return None


def heuristic_feature_plan(profile: DatasetProfile, schema: SchemaPlan) -> FeaturePlan:
    """Synthesise features from shape alone.

    Three moves, in priority order: composite scores over same-scale groups,
    totals over count-like columns, and explicit flags for binary categoricals
    and for missingness. Nothing here needs to know what the columns mean.
    """
    features: list[FeatureSpec] = []
    numeric_profiles = [
        p for p in (profile.column(n) for n in schema.numeric_features) if p is not None
    ]

    bands: dict[str, list[str]] = {}
    for column in numeric_profiles:
        band = _scale_band(column)
        if band:
            bands.setdefault(band, []).append(column.name)

    for band, members in bands.items():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        features.append(
            FeatureSpec(
                name=f"{band}_composite",
                op=FeatureOp.mean,
                inputs=ordered,
                rationale=(
                    f"Average of the {len(ordered)} columns sharing the {band} scale — "
                    "a single summary is more stable than any one of them."
                ),
            )
        )
        if len(ordered) >= 3:
            features.append(
                FeatureSpec(
                    name=f"{band}_spread",
                    op=FeatureOp.spread,
                    inputs=ordered,
                    rationale=(
                        f"Gap between the best and worst {band} column, as a "
                        "consistency signal."
                    ),
                )
            )
            features.append(
                FeatureSpec(
                    name=f"{band}_weakest",
                    op=FeatureOp.rowwise_min,
                    inputs=ordered,
                    rationale=f"The weakest {band} value, which often drives the outcome.",
                )
            )

    count_columns = sorted(bands.get("count", []))
    if len(count_columns) >= 2:
        features.append(
            FeatureSpec(
                name="count_total",
                op=FeatureOp.sum,
                inputs=count_columns,
                rationale="Total across the count-like columns as a volume signal.",
            )
        )
        for name in count_columns:
            features.append(
                FeatureSpec(
                    name=f"{name}_normalized",
                    op=FeatureOp.normalize_max,
                    inputs=[name],
                    rationale=(
                        "Scaled by the maximum seen in training, so a single row and "
                        "the full cohort land on the same scale."
                    ),
                )
            )

    for name in schema.categorical_features:
        column = profile.column(name)
        if column is None or column.n_unique != 2 or not column.top_values:
            continue
        level = column.top_values[0]["value"]
        features.append(
            FeatureSpec(
                name=f"{name}_is_{str(level).lower().replace(' ', '_')[:24]}",
                op=FeatureOp.binarize_equals,
                inputs=[name],
                params={"value": level},
                rationale=f"Explicit 0/1 flag for {name} == {level}.",
            )
        )

    for name in schema.feature_columns:
        column = profile.column(name)
        # Below ~2% missing the flag is nearly constant and carries no signal.
        if column is not None and column.missing_pct > 2.0:
            features.append(
                FeatureSpec(
                    name=f"{name}_was_missing",
                    op=FeatureOp.is_missing,
                    inputs=[name],
                    rationale=(
                        f"{column.missing_pct:.0f}% of rows lack {name}; whether a value "
                        "was recorded can itself be predictive."
                    ),
                )
            )

    return FeaturePlan(
        features=features[:MAX_SYNTHESIZED_FEATURES],
        notes="Shape-derived features: same-scale composites, totals and explicit flags.",
    )


# ── Stage 4: models ──────────────────────────────────────────────────────────


def heuristic_model_plan(profile: DatasetProfile, schema: SchemaPlan, imbalance: float) -> ModelPlan:
    """Pick learners sized to the dataset.

    ``imbalance`` is the gap in percentage points between the largest and
    smallest class; past ~10pp the class weighting is turned on and the primary
    metric moves off plain accuracy, which a majority-class guesser would win.
    """
    n_rows = profile.n_rows
    binary = schema.task_type is TaskType.binary_classification

    candidates = [
        CandidateModel(
            algorithm=Algorithm.logistic_regression,
            params={"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
            ensemble_weight=0.8,
            rationale="Linear baseline — fast, calibrated, and hard to overfit.",
        ),
        CandidateModel(
            algorithm=Algorithm.random_forest,
            params={
                "n_estimators": 300 if n_rows <= 50_000 else 150,
                "max_depth": 16,
                "min_samples_leaf": 2,
                "n_jobs": -1,
                "random_state": RANDOM_SEED,
            },
            ensemble_weight=1.0,
            rationale="Captures interactions the linear model cannot, with little tuning.",
        ),
    ]

    if xgboost_available() and n_rows >= 200:
        candidates.append(
            CandidateModel(
                algorithm=Algorithm.xgboost,
                params={
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.08,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "random_state": RANDOM_SEED,
                },
                ensemble_weight=1.2,
                rationale="Gradient boosting usually leads on tabular data of this size.",
            )
        )
    else:
        candidates.append(
            CandidateModel(
                algorithm=Algorithm.hist_gradient_boosting,
                params={"max_iter": 300, "learning_rate": 0.08, "random_state": RANDOM_SEED},
                ensemble_weight=1.1,
                rationale="Boosted trees via scikit-learn, with no extra dependency.",
            )
        )

    return ModelPlan(
        candidates=candidates,
        class_weight_strategy="balanced" if imbalance >= 10 else "none",
        primary_metric=Metric.roc_auc if binary else Metric.balanced_accuracy,
        test_size=0.2 if n_rows >= 500 else 0.25,
        cv_folds=5 if n_rows >= 500 else 3,
        build_ensemble=True,
        threshold_strategy=ThresholdStrategy.best_f1 if binary else ThresholdStrategy.default,
        notes=(
            f"Baseline sweep sized for {n_rows:,} rows; class weighting "
            f"{'on' if imbalance >= 10 else 'off'} at {imbalance:.1f}pp imbalance."
        ),
    )
