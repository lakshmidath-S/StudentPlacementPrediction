"""
placement_ai/inference/drift.py
-------------------------------
Has the population being scored drifted away from the one the model learned on?

Population Stability Index, computed per input column between the training
distribution (recorded in the bundle at training time) and the records this
workspace has actually been scoring.

The convention is the standard one from credit risk:

    PSI < 0.10   stable
    0.10 - 0.25  worth watching
    > 0.25       the population has moved; retrain

The important detail is where the baseline comes from. It is captured during
training and stored in the manifest, so drift is measured against what the model
truly saw. Recomputing a baseline from recent data — the tempting shortcut —
compares the present against itself and reports calm regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Small mass added to empty buckets so a category absent from one side gives a
# large-but-finite contribution rather than infinity.
EPSILON = 1e-6
DEFAULT_BUCKETS = 10
# Below this, PSI is dominated by sampling noise and reports drift that is not there.
MIN_SAMPLES = 30

WATCH_THRESHOLD = 0.10
ALERT_THRESHOLD = 0.25


@dataclass
class ColumnDrift:
    column: str
    label: str
    psi: float
    status: str
    detail: str = ""


@dataclass
class DriftReport:
    status: str
    n_recent: int
    columns: list[ColumnDrift] = field(default_factory=list)
    message: str = ""

    @property
    def worst(self) -> ColumnDrift | None:
        return max(self.columns, key=lambda c: c.psi) if self.columns else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_recent": self.n_recent,
            "message": self.message,
            "columns": [
                {"column": c.column, "label": c.label, "psi": c.psi, "status": c.status}
                for c in self.columns
            ],
        }


def _status_for(psi: float) -> str:
    if psi >= ALERT_THRESHOLD:
        return "alert"
    return "watch" if psi >= WATCH_THRESHOLD else "ok"


def psi_numeric(baseline: pd.Series, recent: pd.Series, buckets: int = DEFAULT_BUCKETS) -> float:
    """PSI over quantile buckets cut on the baseline.

    Cutting on the baseline, not the combined data, is what makes the number
    comparable over time: the buckets stay fixed while the population moves
    through them.
    """
    base = pd.to_numeric(baseline, errors="coerce").dropna()
    new = pd.to_numeric(recent, errors="coerce").dropna()
    if base.empty or new.empty:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(base.quantile(quantiles).to_numpy())
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    base_share = np.histogram(base.to_numpy(), bins=edges)[0] / len(base)
    new_share = np.histogram(new.to_numpy(), bins=edges)[0] / len(new)

    base_share = np.clip(base_share, EPSILON, None)
    new_share = np.clip(new_share, EPSILON, None)
    return float(np.sum((new_share - base_share) * np.log(new_share / base_share)))


def psi_categorical(baseline: pd.Series, recent: pd.Series) -> float:
    """PSI over the union of observed levels."""
    base_counts = baseline.astype(str).value_counts(normalize=True)
    if base_counts.empty:
        return 0.0
    return psi_from_shares(base_counts.to_dict(), recent)


def psi_from_shares(base_shares: dict[str, float], recent: pd.Series) -> float:
    """PSI against stored level shares, without rebuilding a baseline sample."""
    new_counts = recent.astype(str).value_counts(normalize=True)
    if not base_shares or new_counts.empty:
        return 0.0

    levels = sorted(set(base_shares) | set(new_counts.index))
    base_share = np.clip(
        np.array([float(base_shares.get(level, 0.0)) for level in levels]), EPSILON, None
    )
    new_share = np.clip(
        np.array([float(new_counts.get(level, 0.0)) for level in levels]), EPSILON, None
    )
    return float(np.sum((new_share - base_share) * np.log(new_share / base_share)))


def psi_from_edges(edges: list[float], shares: list[float], recent: pd.Series) -> float:
    """PSI against stored bin edges and the true training mass in each bin.

    The shares have to be stored rather than assumed uniform. Quantile edges
    look like they imply 1/n per bucket, but a discrete column defeats that:
    ``backlogs`` taking values 0-8 produces repeated edges, deduplicating to a
    handful of bins holding wildly unequal shares. Assuming uniformity there
    reports a large PSI against the training data itself — drift where none
    exists, which is worse than no drift detection at all.
    """
    values = pd.to_numeric(recent, errors="coerce").dropna()
    bins = np.asarray(edges, dtype=float)
    base = np.asarray(shares, dtype=float)
    if values.empty or bins.size < 2 or base.size != bins.size - 1:
        return 0.0

    bins = bins.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    new_share = np.histogram(values.to_numpy(), bins=bins)[0] / len(values)

    base_share = np.clip(base, EPSILON, None)
    new_share = np.clip(new_share, EPSILON, None)
    return float(np.sum((new_share - base_share) * np.log(new_share / base_share)))


def training_baseline(training_frame: pd.DataFrame, input_schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the training distribution for storage in the manifest.

    Quantiles for numerics, level shares for categoricals — enough to compute
    PSI later without keeping a copy of anyone's training data in the bundle.
    """
    baseline: dict[str, Any] = {}
    for spec in input_schema:
        name = str(spec["name"])
        if name not in training_frame.columns:
            continue
        series = training_frame[name].dropna()
        if series.empty:
            continue
        if spec.get("kind") == "categorical":
            shares = series.astype(str).value_counts(normalize=True)
            baseline[name] = {
                "kind": "categorical",
                "shares": {str(k): round(float(v), 6) for k, v in shares.head(50).items()},
            }
        else:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                continue
            quantiles = np.linspace(0, 1, DEFAULT_BUCKETS + 1)
            edges = np.unique(numeric.quantile(quantiles).to_numpy())
            if edges.size < 2:
                continue
            # Measure the real mass in each bin rather than assuming quantiles
            # split it evenly — they do not once values repeat.
            open_edges = edges.copy()
            open_edges[0], open_edges[-1] = -np.inf, np.inf
            counts = np.histogram(numeric.to_numpy(), bins=open_edges)[0]
            baseline[name] = {
                "kind": "numeric",
                "edges": [round(float(v), 6) for v in edges.tolist()],
                "shares": [round(float(c) / len(numeric), 6) for c in counts],
            }
    return baseline


def check_drift(
    baseline: dict[str, Any],
    recent: pd.DataFrame,
    input_schema: list[dict[str, Any]],
) -> DriftReport:
    """Compare recent scored records against the stored training baseline."""
    if not baseline:
        return DriftReport(
            status="unavailable",
            n_recent=int(len(recent)),
            message=(
                "This model was saved without a training baseline, so drift cannot "
                "be measured. Retrain to capture one."
            ),
        )
    if len(recent) < MIN_SAMPLES:
        return DriftReport(
            status="insufficient_data",
            n_recent=int(len(recent)),
            message=(
                f"{len(recent)} prediction(s) recorded. At least {MIN_SAMPLES} are needed "
                "before a drift reading means anything."
            ),
        )

    labels = {str(spec["name"]): str(spec.get("label", spec["name"])) for spec in input_schema}
    results: list[ColumnDrift] = []

    for column, reference in baseline.items():
        if column not in recent.columns:
            continue
        series = recent[column].dropna()
        if series.empty:
            continue

        if reference.get("kind") == "categorical":
            shares = reference.get("shares") or {}
            if not shares:
                continue
            psi = psi_from_shares(shares, series)
        else:
            edges = reference.get("edges") or []
            shares = reference.get("shares") or []
            if len(edges) < 2 or len(shares) != len(edges) - 1:
                continue
            psi = psi_from_edges(edges, shares, series)

        results.append(
            ColumnDrift(
                column=column,
                label=labels.get(column, column),
                psi=round(float(psi), 4),
                status=_status_for(psi),
            )
        )

    results.sort(key=lambda row: row.psi, reverse=True)
    if not results:
        return DriftReport(
            status="unavailable",
            n_recent=int(len(recent)),
            message="No column could be compared against the training baseline.",
        )

    worst = results[0]
    overall = worst.status
    message = {
        "ok": "The records being scored still look like the training data.",
        "watch": f"{worst.label} has started to shift. Keep an eye on it.",
        "alert": f"{worst.label} has moved well away from training. Retrain when you can.",
    }[overall]

    return DriftReport(
        status=overall, n_recent=int(len(recent)), columns=results, message=message
    )
