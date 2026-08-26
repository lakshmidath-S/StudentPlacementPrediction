"""
placement_ai/training/evaluation.py
-----------------------------------
Scoring, threshold selection and importance.

Everything here is computed on the held-out split and stored inside the model
bundle, so the numbers a user reads months later are the ones the model actually
earned rather than a re-run against whatever data is lying around.

Two choices worth flagging:

*The decision threshold is a fitted parameter.* On an imbalanced target a 0.5
cut-off quietly predicts the majority class for nearly everyone while still
scoring a respectable ROC-AUC. The threshold is chosen on the holdout and
travels in the bundle.

*Importance is measured over the columns a person recognises.* Permuting a
derived feature like ``percentage_composite`` tells a placement officer nothing.
Permuting ``attendance_percentage`` — the thing they can actually act on — runs
through the whole pipeline, derived columns included, and answers the question
they are really asking.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from placement_ai.plans import Metric, ThresholdStrategy

# Curves are stored in the bundle manifest, so they are thinned to something a
# JSON file and a browser chart can both carry comfortably.
MAX_CURVE_POINTS = 200


def _thin(values: np.ndarray, limit: int = MAX_CURVE_POINTS) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    indices = np.linspace(0, values.shape[0] - 1, limit).astype(int)
    return values[indices]


def _renormalize(probabilities: np.ndarray) -> np.ndarray:
    """Force each row to sum to exactly 1.

    XGBoost returns float32 and the ensemble averages several matrices, so rows
    routinely land on 0.99999994. scikit-learn's log_loss checks that sum with a
    tight tolerance and warns on every call otherwise — a warning that says
    nothing about the model and buries the ones that do.
    """
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2:
        return array
    totals = array.sum(axis=1, keepdims=True)
    # A row summing to zero can only come from a class-alignment miss; spread it
    # evenly rather than dividing by zero.
    safe = np.where(totals > 0, totals, 1.0)
    normalized = array / safe
    return np.where(totals > 0, normalized, 1.0 / array.shape[1])


def choose_threshold(
    y_true_binary: np.ndarray,
    probabilities: np.ndarray,
    strategy: ThresholdStrategy,
) -> float:
    """Pick the probability cut-off that turns scores into decisions."""
    if strategy is ThresholdStrategy.default:
        return 0.5

    unique = np.unique(y_true_binary)
    if unique.shape[0] < 2:
        return 0.5

    if strategy is ThresholdStrategy.best_youden:
        false_positive, true_positive, thresholds = roc_curve(y_true_binary, probabilities)
        youden = true_positive - false_positive
        best = int(np.argmax(youden))
        candidate = float(thresholds[best])
        # roc_curve prepends an infinite threshold for the "predict nothing" point.
        return 0.5 if not np.isfinite(candidate) else candidate

    precision, recall, thresholds = precision_recall_curve(y_true_binary, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(denominator > 0, 2 * precision[:-1] * recall[:-1] / denominator, 0.0)
    return float(thresholds[int(np.argmax(f1_scores))])


def evaluate_classifier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    positive_label: Any = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score one fitted model on the holdout split.

    ``probabilities`` is the full (n_samples, n_classes) matrix; the binary
    metrics are derived from the positive-class column so the threshold is
    applied consistently everywhere.
    """
    classes = np.asarray(classes)
    binary = classes.shape[0] == 2 and positive_label is not None
    probabilities = _renormalize(probabilities)

    report: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "classes": [str(c) for c in classes],
        "threshold": float(threshold),
        "is_binary": bool(binary),
    }

    if binary:
        positive_index = int(np.where(classes == positive_label)[0][0])
        positive_scores = probabilities[:, positive_index]
        y_binary = (np.asarray(y_true) == positive_label).astype(int)
        y_hat = (positive_scores >= threshold).astype(int)

        report.update(
            {
                "positive_class": str(positive_label),
                "accuracy": float(accuracy_score(y_binary, y_hat)),
                "balanced_accuracy": float(balanced_accuracy_score(y_binary, y_hat)),
                "precision": float(precision_score(y_binary, y_hat, zero_division=0)),
                "recall": float(recall_score(y_binary, y_hat, zero_division=0)),
                "f1": float(f1_score(y_binary, y_hat, zero_division=0)),
                "positive_rate": float(y_hat.mean()),
                "base_rate": float(y_binary.mean()),
            }
        )

        if np.unique(y_binary).shape[0] == 2:
            report["roc_auc"] = float(roc_auc_score(y_binary, positive_scores))
            report["average_precision"] = float(average_precision_score(y_binary, positive_scores))
            report["brier_score"] = float(brier_score_loss(y_binary, positive_scores))

            false_positive, true_positive, _ = roc_curve(y_binary, positive_scores)
            precision, recall, _ = precision_recall_curve(y_binary, positive_scores)
            report["roc_curve"] = {
                "fpr": _thin(false_positive).round(5).tolist(),
                "tpr": _thin(true_positive).round(5).tolist(),
            }
            report["pr_curve"] = {
                "precision": _thin(precision).round(5).tolist(),
                "recall": _thin(recall).round(5).tolist(),
            }

        matrix = confusion_matrix(y_binary, y_hat, labels=[0, 1])
        negative_label = str(next(c for c in classes if c != positive_label))
        report["confusion_matrix"] = {
            "labels": [negative_label, str(positive_label)],
            "matrix": matrix.astype(int).tolist(),
        }
    else:
        y_hat = classes[np.argmax(probabilities, axis=1)]
        report.update(
            {
                "accuracy": float(accuracy_score(y_true, y_hat)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
                "f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
                "precision": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
            }
        )
        if len(classes) > 2 and np.unique(y_true).shape[0] == len(classes):
            # Undefined when a class is absent from the holdout; the other
            # metrics still stand, so an omitted key beats a failed report.
            with contextlib.suppress(ValueError):
                report["roc_auc"] = float(
                    roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
                )
        matrix = confusion_matrix(y_true, y_hat, labels=classes)
        report["confusion_matrix"] = {
            "labels": [str(c) for c in classes],
            "matrix": matrix.astype(int).tolist(),
        }

    with contextlib.suppress(ValueError):
        report["log_loss"] = float(log_loss(y_true, probabilities, labels=list(classes)))

    return report


def primary_score(report: dict[str, Any], metric: Metric) -> float:
    """Read the chosen metric out of a report, degrading to a sensible neighbour.

    ROC-AUC is absent whenever the holdout happened to contain a single class,
    so falling back keeps model selection working instead of scoring every
    candidate as zero and picking the first.
    """
    order = [
        metric.value,
        Metric.roc_auc.value,
        Metric.average_precision.value,
        Metric.f1.value,
        Metric.balanced_accuracy.value,
        Metric.accuracy.value,
    ]
    for key in order:
        value = report.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return 0.0


def permutation_importance_raw(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    classes: np.ndarray,
    positive_label: Any,
    metric: Metric,
    threshold: float,
    n_repeats: int = 3,
    max_rows: int = 2000,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Shuffle each raw input column and measure what the score loses.

    Sampled and repeated only a few times on purpose: this runs inside an
    interactive training job, and the ranking stabilises long before the
    estimate does.
    """
    rng = np.random.default_rng(random_state)
    if len(X) > max_rows:
        sample_index = rng.choice(len(X), size=max_rows, replace=False)
        X = X.iloc[sample_index]
        y = y.iloc[sample_index]

    def score_of(frame: pd.DataFrame) -> float:
        probabilities = pipeline.predict_proba(frame)
        report = evaluate_classifier(
            np.asarray(y), probabilities, classes, positive_label, threshold
        )
        return primary_score(report, metric)

    baseline = score_of(X)
    results: list[dict[str, Any]] = []

    for column in X.columns:
        drops: list[float] = []
        original = X[column].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(original)
            drops.append(baseline - score_of(shuffled))
        results.append(
            {
                "column": str(column),
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
            }
        )

    results.sort(key=lambda row: row["importance"], reverse=True)
    return results


def class_balance(y: pd.Series) -> tuple[list[dict[str, Any]], float]:
    """Class counts plus the gap in percentage points between largest and smallest."""
    counts = pd.Series(y).astype(object).value_counts()
    total = float(counts.sum()) or 1.0
    levels = [
        {"value": str(value), "count": int(count), "share": round(float(count) / total * 100, 2)}
        for value, count in counts.items()
    ]
    gap = (float(counts.max()) - float(counts.min())) / total * 100 if len(counts) > 1 else 0.0
    return levels, gap
