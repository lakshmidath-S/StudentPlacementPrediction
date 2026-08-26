"""
placement_ai/inference/explain.py
---------------------------------
Why did this record get this prediction, and what would change it.

The attribution here is ablation, not SHAP. For each field, the value is
swapped for the training-typical one and the change in probability is measured.
That is a weaker theoretical guarantee than Shapley values, and it is the right
trade for this product for three reasons:

  - it works on *any* fitted pipeline, which matters when the model was chosen
    by a planner and could be linear, a forest, or a weighted ensemble;
  - it costs one prediction per field on a single row, so it runs inline in the
    UI rather than behind a spinner;
  - it answers the question a user actually asks. "Your attendance is costing
    you 9 points versus a typical student" is directly actionable in a way that
    a Shapley value in log-odds space is not.

Its real limitation is worth stating: measuring one field at a time misses
interactions, so the deltas will not sum exactly to the gap between this
prediction and the baseline. They are a ranking, not a decomposition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from placement_ai.inference.predictor import WorkspacePredictor

# Enough points to draw a smooth curve without making the UI wait.
CURVE_POINTS = 25


def _score(predictor: WorkspacePredictor, record: dict[str, Any]) -> float:
    return predictor.predict_one(record).probability


def attribute(
    predictor: WorkspacePredictor,
    inputs: dict[str, Any],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rank fields by how much this record's value moved the prediction.

    A positive ``delta`` means the supplied value raised the probability
    relative to a typical value; negative means it lowered it.
    """
    record = predictor.build_record(inputs)
    actual = _score(predictor, record)
    typical = predictor.defaults

    rows: list[dict[str, Any]] = []
    for spec in predictor.input_schema:
        name = str(spec["name"])
        value = record.get(name)
        reference = typical.get(name)
        # An identical value has nothing to explain, and scoring it would just
        # spend a prediction to return zero.
        if value == reference or reference is None:
            continue

        counterfactual = {**record, name: reference}
        rows.append(
            {
                "column": name,
                "label": str(spec.get("label", name)),
                "kind": spec.get("kind", "numeric"),
                "value": value,
                "typical": reference,
                "delta": round(actual - _score(predictor, counterfactual), 4),
            }
        )

    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return rows[:top_n] if top_n else rows


def what_if_curve(
    predictor: WorkspacePredictor,
    inputs: dict[str, Any],
    column: str,
    points: int = CURVE_POINTS,
) -> pd.DataFrame:
    """Sweep one field across its training range, holding everything else fixed.

    Values outside the training range are not offered: the model has no evidence
    there, and a curve drawn through that region invites a decision it cannot
    support.
    """
    spec = predictor.field(column)
    if spec is None:
        return pd.DataFrame(columns=["value", "probability"])

    record = predictor.build_record(inputs)

    if spec.get("kind") == "categorical":
        values: list[Any] = list(spec.get("levels") or [])
    else:
        low, high = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
        if high <= low:
            return pd.DataFrame(columns=["value", "probability"])
        grid = np.linspace(low, high, points)
        values = [round(float(v)) for v in grid] if spec.get("integer") else [
            round(float(v), 4) for v in grid
        ]
        values = list(dict.fromkeys(values))

    return pd.DataFrame(
        {
            "value": values,
            "probability": [_score(predictor, {**record, column: value}) for value in values],
        }
    )


def improvement_levers(
    predictor: WorkspacePredictor,
    inputs: dict[str, Any],
    max_levers: int = 4,
) -> list[dict[str, Any]]:
    """For a record below the threshold, the smallest single change that clears it.

    Only fields the model saw improve monotonically are reported, and only where
    the required value stays inside the training range — advice that requires
    extrapolating past anything in the data is not advice.
    """
    record = predictor.build_record(inputs)
    current = _score(predictor, record)
    threshold = predictor.threshold
    if current >= threshold:
        return []

    levers: list[dict[str, Any]] = []
    for spec in predictor.input_schema:
        name = str(spec["name"])
        if spec.get("kind") == "categorical":
            best = _best_category(predictor, record, spec, current)
            if best:
                levers.append(best)
            continue

        low, high = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
        try:
            value_now = float(record.get(name, low))
        except (TypeError, ValueError):
            continue
        if high <= value_now:
            continue

        grid = np.linspace(value_now, high, CURVE_POINTS)[1:]
        for candidate in grid:
            candidate_value = round(float(candidate)) if spec.get("integer") else float(candidate)
            score = _score(predictor, {**record, name: candidate_value})
            if score >= threshold:
                levers.append(
                    {
                        "column": name,
                        "label": str(spec.get("label", name)),
                        "from": value_now,
                        "to": candidate_value,
                        "gain": round(score - current, 4),
                        "reaches_threshold": True,
                    }
                )
                break

    levers.sort(key=lambda row: abs(float(row["to"]) - float(row["from"])) if isinstance(row.get("to"), (int, float)) else 1e9)
    return levers[:max_levers]


def _best_category(
    predictor: WorkspacePredictor,
    record: dict[str, Any],
    spec: dict[str, Any],
    current: float,
) -> dict[str, Any] | None:
    name = str(spec["name"])
    levels = list(spec.get("levels") or [])
    now = record.get(name)
    best_level, best_score = None, current
    for level in levels:
        if level == now:
            continue
        score = _score(predictor, {**record, name: level})
        if score > best_score:
            best_level, best_score = level, score
    if best_level is None or best_score < predictor.threshold:
        return None
    return {
        "column": name,
        "label": str(spec.get("label", name)),
        "from": now,
        "to": best_level,
        "gain": round(best_score - current, 4),
        "reaches_threshold": True,
    }


def cohort_summary(scored: pd.DataFrame, predictor: WorkspacePredictor) -> dict[str, Any]:
    """Headline numbers for a scored batch."""
    from placement_ai.inference.predictor import (
        CONFIDENCE_COLUMN,
        PREDICTION_COLUMN,
        PROBABILITY_COLUMN,
    )

    if scored.empty:
        return {"total": 0}

    counts = scored[PREDICTION_COLUMN].value_counts()
    positive = predictor.positive_class
    return {
        "total": int(len(scored)),
        "by_label": {str(k): int(v) for k, v in counts.items()},
        "positive_rate": (
            round(float((scored[PREDICTION_COLUMN] == positive).mean()), 4) if positive else None
        ),
        "mean_probability": round(float(scored[PROBABILITY_COLUMN].mean()), 4),
        "borderline": int((scored[CONFIDENCE_COLUMN] == "Borderline").sum()),
    }
