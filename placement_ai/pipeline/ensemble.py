"""
placement_ai/pipeline/ensemble.py
---------------------------------
A soft-voting ensemble over estimators that are already fitted.

scikit-learn's VotingClassifier refits every member when you fit it, which here
would mean training each candidate twice: once to score it, once inside the
ensemble. On a large upload that is minutes of duplicated work for an identical
result, so this holds the fitted estimators as-is and only averages their
probabilities.

The weights come from the planner — its judgement about which learner suits this
data. They are not a guarantee: the runner scores the ensemble against the best
single model and keeps whichever actually wins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class WeightedSoftVoter(BaseEstimator, ClassifierMixin):
    """Weighted average of ``predict_proba`` across pre-fitted estimators."""

    def __init__(
        self,
        estimators: list[tuple[str, Any]] | None = None,
        weights: list[float] | None = None,
        classes: np.ndarray | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.classes = classes

    def fit(self, X: Any = None, y: Any = None) -> WeightedSoftVoter:
        """Members arrive fitted; this only settles the class order.

        Present so the object satisfies the estimator protocol and can sit as
        the final step of a Pipeline.
        """
        if self.classes is not None:
            self.classes_ = np.asarray(self.classes)
        elif y is not None:
            self.classes_ = np.unique(y)
        else:
            _, first = (self.estimators or [(None, None)])[0]
            self.classes_ = np.asarray(getattr(first, "classes_", [0, 1]))
        return self

    @property
    def _normalized_weights(self) -> np.ndarray:
        members = self.estimators or []
        raw = np.asarray(
            self.weights if self.weights else [1.0] * len(members), dtype=float
        )
        if raw.shape[0] != len(members):
            raw = np.ones(len(members), dtype=float)
        total = raw.sum()
        # All-zero weights would divide by zero; fall back to a plain mean.
        return raw / total if total > 0 else np.ones(len(members)) / max(len(members), 1)

    def predict_proba(self, X: Any) -> np.ndarray:
        members = self.estimators or []
        if not members:
            raise RuntimeError("The ensemble holds no estimators.")
        weights = self._normalized_weights
        classes = np.asarray(getattr(self, "classes_", members[0][1].classes_))

        total: np.ndarray | None = None
        for (_, estimator), weight in zip(members, weights):
            probabilities = _aligned_proba(estimator, X, classes)
            total = probabilities * weight if total is None else total + probabilities * weight
        assert total is not None
        return total

    def predict(self, X: Any) -> np.ndarray:
        classes = np.asarray(getattr(self, "classes_", [0, 1]))
        return classes[np.argmax(self.predict_proba(X), axis=1)]


def _aligned_proba(estimator: Any, X: Any, classes: np.ndarray) -> np.ndarray:
    """Reorder one member's probabilities onto the ensemble's class order.

    Members can disagree on column order — and an estimator that never saw a
    class returns a narrower matrix — so averaging raw outputs would silently
    add the wrong columns together.
    """
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    member_classes = np.asarray(getattr(estimator, "classes_", classes))
    if member_classes.shape == classes.shape and np.array_equal(member_classes, classes):
        return probabilities

    aligned = np.zeros((probabilities.shape[0], len(classes)), dtype=float)
    lookup = {value: index for index, value in enumerate(classes)}
    for position, value in enumerate(member_classes):
        target = lookup.get(value)
        if target is not None:
            aligned[:, target] = probabilities[:, position]
    return aligned
