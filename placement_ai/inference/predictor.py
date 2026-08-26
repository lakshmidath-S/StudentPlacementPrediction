"""
placement_ai/inference/predictor.py
-----------------------------------
Serving a saved bundle.

The pipeline inside a bundle already contains every fitted step, so prediction
is genuinely just "load it and call it". What this layer adds is the part that
cannot live inside a scikit-learn object: knowing which columns the model wants,
reconciling that against whatever the user actually supplied, and turning a
probability vector back into the labels they uploaded.

Missing columns are tolerated rather than rejected. The cleaning step imputes an
absent column to its training-time fill value, so a partial upload degrades in
accuracy instead of failing outright — but the caller is told exactly which
columns were absent, because a silent imputation of the most important field is
not something anyone should discover later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from placement_ai.profiling import canonicalize_columns
from placement_ai.registry.model_store import ModelBundle

PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"
CONFIDENCE_COLUMN = "confidence"

# How far a probability must sit from the decision threshold before the result
# is reported as anything more than borderline.
BORDERLINE_MARGIN = 0.10
CONFIDENT_MARGIN = 0.30


@dataclass
class Prediction:
    label: str
    probabilities: dict[str, float]
    threshold: float
    inputs: dict[str, Any] = field(default_factory=dict)
    positive_class: str | None = None

    @property
    def probability(self) -> float:
        """Confidence in the positive class, or in the predicted label."""
        if self.positive_class and self.positive_class in self.probabilities:
            return self.probabilities[self.positive_class]
        return self.probabilities.get(self.label, 0.0)

    @property
    def margin(self) -> float:
        return abs(self.probability - self.threshold)

    @property
    def confidence_band(self) -> str:
        if self.margin < BORDERLINE_MARGIN:
            return "Borderline"
        return "High confidence" if self.margin >= CONFIDENT_MARGIN else "Moderate confidence"


@dataclass
class FrameCheck:
    """What a supplied frame has, lacks, and carries that the model ignores."""

    expected: list[str]
    present: list[str]
    missing: list[str]
    extra: list[str]
    renamed: dict[str, str]

    @property
    def usable(self) -> bool:
        # One recognised column is enough to run; the rest are imputed. Zero
        # means the user has almost certainly uploaded the wrong file.
        return bool(self.present)


class WorkspacePredictor:
    """Wraps one ModelBundle for single and batch prediction."""

    def __init__(self, bundle: ModelBundle, verify: bool = True) -> None:
        self.bundle = bundle
        self.pipeline = bundle.load_pipeline(verify=verify)
        self.class_labels = bundle.class_labels
        self.positive_class = bundle.positive_class
        self.threshold = bundle.threshold
        self.target_column = bundle.target_column
        self.input_schema = bundle.input_schema
        self.expected_columns = [str(f["name"]) for f in self.input_schema]

    # ── introspection ────────────────────────────────────────────────────
    @property
    def version(self) -> str:
        return self.bundle.version

    @property
    def defaults(self) -> dict[str, Any]:
        """A typical record, built from the training data's central values.

        Used to fill an unsupplied field and as the reference point every
        attribution is measured against.
        """
        return {str(f["name"]): f.get("default") for f in self.input_schema}

    def field(self, name: str) -> dict[str, Any] | None:
        return next((f for f in self.input_schema if str(f["name"]) == name), None)

    # ── frame handling ───────────────────────────────────────────────────
    def check_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FrameCheck]:
        """Canonicalise headers and report the fit against what the model wants."""
        frame, rename_map = canonicalize_columns(df)
        present = [c for c in self.expected_columns if c in frame.columns]
        missing = [c for c in self.expected_columns if c not in frame.columns]
        extra = [str(c) for c in frame.columns if c not in self.expected_columns]
        check = FrameCheck(
            expected=list(self.expected_columns),
            present=present,
            missing=missing,
            extra=extra,
            renamed={k: v for k, v in rename_map.items() if k != v},
        )
        return frame, check

    def _model_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The exact column set the pipeline was fitted on, in order.

        Absent columns are materialised as NaN so the cleaning step imputes
        them, which is what makes a partial upload work at all.
        """
        data = {
            column: (df[column] if column in df.columns else pd.Series(np.nan, index=df.index))
            for column in self.expected_columns
        }
        return pd.DataFrame(data, index=df.index)

    # ── prediction ───────────────────────────────────────────────────────
    def _probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.pipeline.predict_proba(self._model_frame(frame)), dtype=float)
        totals = raw.sum(axis=1, keepdims=True)
        return np.divide(raw, np.where(totals > 0, totals, 1.0))

    def _label_from(self, row: np.ndarray) -> str:
        """Apply the fitted threshold for binary, argmax otherwise.

        The threshold matters: it was chosen during training to suit an
        imbalanced target, and ignoring it here in favour of a plain argmax
        would serve different decisions than the ones that were evaluated.
        """
        if len(self.class_labels) == 2 and self.positive_class in self.class_labels:
            positive_index = self.class_labels.index(self.positive_class)
            negative = next(c for c in self.class_labels if c != self.positive_class)
            return self.positive_class if row[positive_index] >= self.threshold else negative
        return self.class_labels[int(np.argmax(row))]

    def predict_one(self, inputs: dict[str, Any]) -> Prediction:
        frame = pd.DataFrame([inputs])
        probabilities = self._probabilities(frame)[0]
        return Prediction(
            label=self._label_from(probabilities),
            probabilities={
                label: float(probabilities[index])
                for index, label in enumerate(self.class_labels)
            },
            threshold=self.threshold,
            inputs=dict(inputs),
            positive_class=self.positive_class,
        )

    def predict_proba_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Raw probabilities for an already-canonicalised frame."""
        return self._probabilities(df)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch, returning the input frame plus prediction columns."""
        probabilities = self._probabilities(df)
        result = df.copy()
        result[PREDICTION_COLUMN] = [self._label_from(row) for row in probabilities]

        for index, label in enumerate(self.class_labels):
            result[f"p_{label}"] = probabilities[:, index].round(4)

        if self.positive_class in self.class_labels:
            positive_index = self.class_labels.index(self.positive_class)
            scores = probabilities[:, positive_index]
        else:
            scores = probabilities.max(axis=1)
        result[PROBABILITY_COLUMN] = scores.round(4)

        margins = np.abs(scores - self.threshold)
        result[CONFIDENCE_COLUMN] = np.where(
            margins < BORDERLINE_MARGIN,
            "Borderline",
            np.where(margins >= CONFIDENT_MARGIN, "High confidence", "Moderate confidence"),
        )
        return result

    def build_record(self, values: dict[str, Any]) -> dict[str, Any]:
        """Fill any unsupplied field with its training-typical value."""
        record = self.defaults
        record.update({k: v for k, v in values.items() if k in record})
        return record
