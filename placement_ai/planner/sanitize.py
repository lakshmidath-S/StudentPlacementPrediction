"""
placement_ai/planner/sanitize.py
--------------------------------
Turning a raw LLM reply into a plan the executor can trust.

Pydantic checks that a plan is *well-formed*. These functions check that it is
*true about this dataset* — that every referenced column exists, that a numeric
operation was not pointed at a text column, that a weighted sum has one weight
per input. Pydantic cannot do this because it never sees the profile.

The split between filtering and raising is deliberate. A plan with two bad
features out of twenty is a good plan with two bad features: drop them, record a
warning, keep going. A plan where most of the content is wrong means the model
misread the task, so it raises — which triggers the repair pass, and failing
that, the heuristic planner.
"""

from __future__ import annotations

from typing import Any

from placement_ai.config import MAX_SYNTHESIZED_FEATURES
from placement_ai.planner.heuristic import _role_for, xgboost_available
from placement_ai.plans import (
    CATEGORICAL_OPS,
    CleaningPlan,
    ColumnRole,
    ColumnSpec,
    FeatureOp,
    FeaturePlan,
    FeatureSpec,
    ModelPlan,
    SchemaPlan,
    TaskType,
)
from placement_ai.profiling import DatasetProfile
from placement_ai.utils import safe_column_name

# Past this share of unusable features, the reply is treated as a misread of the
# task rather than a few slips worth filtering out.
_FEATURE_REJECT_LIMIT = 0.6

# Params each op must carry to be executable at all.
_REQUIRED_PARAMS: dict[FeatureOp, tuple[str, ...]] = {
    FeatureOp.scale: ("factor",),
    FeatureOp.offset: ("offset",),
    FeatureOp.count_above: ("threshold",),
    FeatureOp.binarize_threshold: ("threshold",),
    FeatureOp.binarize_equals: ("value",),
    FeatureOp.category_map: ("mapping",),
}


class PlanRejected(ValueError):
    """The reply was well-formed JSON but wrong about this dataset."""


# ── Stage 1: schema ──────────────────────────────────────────────────────────


def coerce_schema_plan(
    raw: dict[str, Any],
    profile: DatasetProfile,
    target_column: str,
    target_info: dict[str, Any],
) -> tuple[SchemaPlan, list[str]]:
    warnings: list[str] = []
    known = {c.name: c for c in profile.columns}

    specs: list[ColumnSpec] = []
    seen: set[str] = set()
    for entry in raw.get("columns") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if name not in known:
            # Try the snake_case form before giving up — models sometimes echo
            # the original header casing back.
            alt = safe_column_name(name)
            if alt in known:
                name = alt
            else:
                warnings.append(f"Ignored a rule for unknown column {name!r}.")
                continue
        if name in seen:
            continue
        seen.add(name)

        try:
            role = ColumnRole(str(entry.get("role", "")).strip())
        except ValueError:
            role, reason = _role_for(known[name])
            warnings.append(f"Column {name!r} had an unrecognised role; used {role.value}.")
            entry = {**entry, "reason": entry.get("reason") or reason}

        specs.append(
            ColumnSpec(
                name=name,
                role=role,
                display_label=str(entry.get("display_label") or ""),
                description=str(entry.get("description") or ""),
                leakage_risk=bool(entry.get("leakage_risk", False)),
                reason=str(entry.get("reason") or ""),
            )
        )

    # Any column the model forgot is filled in from the rules, so the plan
    # always covers the whole table.
    for name, column in known.items():
        if name in seen:
            continue
        if name == target_column:
            specs.append(ColumnSpec(name=name, role=ColumnRole.target))
        else:
            role, reason = _role_for(column)
            specs.append(ColumnSpec(name=name, role=role, reason=reason))
        warnings.append(f"Column {name!r} was missing from the plan; assigned by rule.")

    # A column flagged as leaky is not a feature, whatever role it was given.
    for spec in specs:
        if spec.leakage_risk and spec.role in (
            ColumnRole.numeric_feature,
            ColumnRole.categorical_feature,
        ):
            spec.role = ColumnRole.drop
            warnings.append(f"Column {spec.name!r} was dropped as a leakage risk.")

    declared_task = str(raw.get("task_type") or target_info.get("task_type") or "")
    task_type = (
        TaskType.multiclass_classification
        if declared_task == "multiclass_classification"
        or target_info.get("task_type") == "multiclass_classification"
        else TaskType.binary_classification
    )

    positive = raw.get("positive_class")
    observed = [str(level["value"]) for level in target_info.get("levels", [])]
    if task_type is TaskType.multiclass_classification:
        positive = None
    elif positive is None or str(positive) not in observed:
        fallback = target_info.get("positive_class")
        if positive is not None:
            warnings.append(
                f"Positive class {positive!r} is not one of the observed labels; "
                f"using {fallback!r}."
            )
        positive = fallback

    plan = SchemaPlan(
        target_column=target_column,
        task_type=task_type,
        positive_class=None if positive is None else str(positive),
        columns=specs,
        summary=str(raw.get("summary") or ""),
    )
    return plan, warnings


# ── Stage 2: cleaning ────────────────────────────────────────────────────────


def coerce_cleaning_plan(
    raw: dict[str, Any],
    profile: DatasetProfile,
    schema: SchemaPlan,
) -> tuple[CleaningPlan, list[str]]:
    warnings: list[str] = []
    features = set(schema.feature_columns)

    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw.get("columns") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("column", "")).strip()
        if name not in features:
            name = safe_column_name(name)
        if name not in features or name in seen:
            continue
        seen.add(name)
        steps.append({**entry, "column": name})

    plan = CleaningPlan.model_validate(
        {
            # The schema stage decides what gets dropped; a cleaning plan is not
            # allowed to quietly remove a column the schema kept as a feature.
            "drop_columns": schema.dropped_columns,
            "drop_duplicate_rows": bool(raw.get("drop_duplicate_rows", True)),
            "drop_rows_missing_target": bool(raw.get("drop_rows_missing_target", True)),
            "columns": steps,
            "notes": str(raw.get("notes") or ""),
        }
    )

    # Fill in anything the model skipped, so every feature has a defined rule.
    from placement_ai.planner.heuristic import heuristic_cleaning_plan

    defaults = heuristic_cleaning_plan(profile, schema)
    for default in defaults.columns:
        if default.column not in seen:
            plan.columns.append(default)
            warnings.append(f"Column {default.column!r} had no cleaning rule; used the default.")

    return plan, warnings


# ── Stage 3: features ────────────────────────────────────────────────────────


def _feature_is_usable(
    spec: FeatureSpec,
    numeric: set[str],
    categorical: set[str],
    taken: set[str],
) -> str | None:
    """Return None when the spec is executable, else the reason it is not."""
    allowed = numeric | categorical
    for column in spec.inputs:
        if column not in allowed:
            return f"input column {column!r} is not an available feature"

    if spec.op in CATEGORICAL_OPS:
        pass  # These read either kind.
    elif any(column in categorical for column in spec.inputs):
        return f"{spec.op.value} needs numeric inputs but was given a categorical column"

    if spec.name in taken:
        return "the name collides with an existing column or feature"

    missing = [key for key in _REQUIRED_PARAMS.get(spec.op, ()) if key not in spec.params]
    if missing:
        return f"missing required param(s) {missing}"

    if spec.op is FeatureOp.weighted_sum:
        weights = spec.params.get("weights")
        if not isinstance(weights, list) or len(weights) != len(spec.inputs):
            return "weights must be a list with one entry per input column"
        if not all(isinstance(w, (int, float)) for w in weights):
            return "weights must all be numbers"

    if spec.op is FeatureOp.clip and not {"min", "max"} & set(spec.params):
        return "clip needs at least one of min/max"

    if spec.op is FeatureOp.category_map and not isinstance(spec.params.get("mapping"), dict):
        return "mapping must be an object of level -> number"

    return None


def coerce_feature_plan(
    raw: dict[str, Any],
    profile: DatasetProfile,
    schema: SchemaPlan,
) -> tuple[FeaturePlan, list[str]]:
    warnings: list[str] = []
    numeric = set(schema.numeric_features)
    categorical = set(schema.categorical_features)
    taken = set(profile.column_names)

    proposed = raw.get("features") or []
    kept: list[FeatureSpec] = []
    rejected = 0

    for entry in proposed:
        if not isinstance(entry, dict):
            rejected += 1
            continue
        entry = dict(entry)
        entry["name"] = safe_column_name(str(entry.get("name", "")))
        entry["inputs"] = [
            column if column in taken else safe_column_name(str(column))
            for column in (entry.get("inputs") or [])
        ]
        if not isinstance(entry.get("params"), dict):
            entry["params"] = {}

        try:
            spec = FeatureSpec.model_validate(entry)
        except Exception as exc:
            rejected += 1
            warnings.append(f"Dropped feature {entry.get('name')!r}: {_first_line(exc)}")
            continue

        problem = _feature_is_usable(spec, numeric, categorical, taken)
        if problem:
            rejected += 1
            warnings.append(f"Dropped feature {spec.name!r}: {problem}.")
            continue

        taken.add(spec.name)
        kept.append(spec)
        if len(kept) >= MAX_SYNTHESIZED_FEATURES:
            warnings.append(
                f"Kept the first {MAX_SYNTHESIZED_FEATURES} features and ignored the rest."
            )
            break

    if proposed and rejected / max(len(proposed), 1) > _FEATURE_REJECT_LIMIT:
        raise PlanRejected(
            f"{rejected} of {len(proposed)} proposed features referenced columns or "
            "operations that do not exist for this dataset."
        )

    return FeaturePlan(features=kept, notes=str(raw.get("notes") or "")), warnings


def _first_line(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return text[1].strip() if len(text) > 1 else (text[0] if text else type(exc).__name__)


# ── Stage 4: models ──────────────────────────────────────────────────────────


def coerce_model_plan(raw: dict[str, Any], schema: SchemaPlan) -> tuple[ModelPlan, list[str]]:
    warnings: list[str] = []
    has_xgboost = xgboost_available()

    candidates: list[dict[str, Any]] = []
    for entry in raw.get("candidates") or []:
        if not isinstance(entry, dict):
            continue
        algorithm = str(entry.get("algorithm", "")).strip()
        if algorithm == "xgboost" and not has_xgboost:
            algorithm = "hist_gradient_boosting"
            warnings.append("XGBoost is not installed; used scikit-learn boosting instead.")
        params = entry.get("params")
        candidates.append(
            {
                "algorithm": algorithm,
                "params": params if isinstance(params, dict) else {},
                "ensemble_weight": entry.get("ensemble_weight", 1.0),
                "rationale": str(entry.get("rationale") or ""),
            }
        )

    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            from placement_ai.plans import CandidateModel

            CandidateModel.model_validate(candidate)
        except Exception as exc:
            warnings.append(
                f"Dropped candidate {candidate.get('algorithm')!r}: {_first_line(exc)}"
            )
            continue
        valid.append(candidate)

    if not valid:
        raise PlanRejected("No proposed model was a recognised algorithm.")

    payload = {
        "candidates": valid,
        "class_weight_strategy": raw.get("class_weight_strategy", "balanced"),
        "custom_class_weights": raw.get("custom_class_weights"),
        "primary_metric": raw.get("primary_metric", "roc_auc"),
        "test_size": raw.get("test_size", 0.2),
        "cv_folds": raw.get("cv_folds", 5),
        "build_ensemble": bool(raw.get("build_ensemble", True)),
        "threshold_strategy": raw.get("threshold_strategy", "best_f1"),
        "notes": str(raw.get("notes") or ""),
    }
    if payload["class_weight_strategy"] not in {"balanced", "none", "custom"}:
        payload["class_weight_strategy"] = "balanced"

    try:
        plan = ModelPlan.model_validate(payload)
    except Exception as exc:
        raise PlanRejected(f"Model plan failed validation: {_first_line(exc)}") from exc

    if schema.task_type is TaskType.multiclass_classification:
        # ROC-AUC and F1 need an averaging choice for multiclass; sidestep it.
        from placement_ai.plans import Metric, ThresholdStrategy

        if plan.primary_metric in (Metric.roc_auc, Metric.f1, Metric.average_precision):
            warnings.append(
                f"{plan.primary_metric.value} is ambiguous for a multiclass target; "
                "scored on balanced accuracy instead."
            )
            plan.primary_metric = Metric.balanced_accuracy
        plan.threshold_strategy = ThresholdStrategy.default

    return plan, warnings
