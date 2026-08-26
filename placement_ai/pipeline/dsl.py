"""
placement_ai/pipeline/dsl.py
----------------------------
The feature operations, and the only place a generated plan turns into numbers.

Every op is a closed function of named columns. There is no ``eval``, no
expression parser, and no way for a plan to name an operation that is not
implemented here — an unknown op is a KeyError at validation time, long before
anything runs.

Two conventions run through all of it:

*Degenerate maths yields zero, never NaN or infinity.* Dividing by a zero
denominator, scaling by a zero maximum, or standardising a constant column all
produce 0.0. A NaN would survive the transformer and blow up inside the
estimator with a message that points nowhere near the plan that caused it.

*Stateful ops learn only at fit time.* ``normalize_max`` divides by the maximum
seen in training, stored in the fitted transformer and reused verbatim
afterwards. Recomputing it per batch is the classic silent corruption here: a
single-row prediction would divide the value by itself and hand the model a 1.0
for everybody.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from placement_ai.plans import FeatureOp, FeatureSpec

# Name of the helper column the cleaner leaves behind for is_missing to read.
MISSING_FLAG_PREFIX = "__missing__"


def missing_flag_name(column: str) -> str:
    return f"{MISSING_FLAG_PREFIX}{column}"


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """A float view of one column, with unparseable entries as 0.0."""
    series = pd.to_numeric(df[column], errors="coerce")
    return series.astype(float).fillna(0.0)


def _frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({name: _numeric(df, name) for name in columns})


def _finite(values: np.ndarray | pd.Series, index: pd.Index) -> pd.Series:
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(array, index=index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.divide(
            numerator.to_numpy(dtype=float), denominator.to_numpy(dtype=float)
        )
    return _finite(values, numerator.index)


# ── Fitting the stateful ops ─────────────────────────────────────────────────


def fit_spec(spec: FeatureSpec, df: pd.DataFrame) -> dict[str, Any]:
    """Learn whatever constant this op needs. Stateless ops return {}."""
    if spec.op is FeatureOp.normalize_max:
        maximum = float(_numeric(df, spec.inputs[0]).max())
        return {"max": maximum if maximum not in (0.0, np.inf, -np.inf) else 0.0}

    if spec.op is FeatureOp.min_max_scale:
        column = _numeric(df, spec.inputs[0])
        return {"min": float(column.min()), "max": float(column.max())}

    if spec.op is FeatureOp.zscore:
        column = _numeric(df, spec.inputs[0])
        std = float(column.std())
        return {"mean": float(column.mean()), "std": std if std > 0 else 0.0}

    return {}


# ── Applying an op ───────────────────────────────────────────────────────────


def apply_spec(spec: FeatureSpec, df: pd.DataFrame, state: dict[str, Any]) -> pd.Series:
    """Compute one derived column. `state` comes from fit_spec."""
    op = spec.op
    params = spec.params
    index = df.index

    # ── multi-column combinations ────────────────────────────────────────
    if op is FeatureOp.sum:
        return _finite(_frame(df, spec.inputs).sum(axis=1), index)

    if op is FeatureOp.mean:
        return _finite(_frame(df, spec.inputs).mean(axis=1), index)

    if op is FeatureOp.weighted_sum:
        weights = [float(w) for w in params.get("weights", [])]
        frame = _frame(df, spec.inputs)
        total = sum(frame[name] * weight for name, weight in zip(spec.inputs, weights))
        return _finite(total, index)

    if op is FeatureOp.product:
        frame = _frame(df, spec.inputs)
        return _finite(frame.prod(axis=1), index)

    if op is FeatureOp.difference:
        left, right = spec.inputs
        return _finite(_numeric(df, left) - _numeric(df, right), index)

    if op is FeatureOp.abs_difference:
        left, right = spec.inputs
        return _finite((_numeric(df, left) - _numeric(df, right)).abs(), index)

    if op is FeatureOp.ratio:
        left, right = spec.inputs
        return _safe_divide(_numeric(df, left), _numeric(df, right))

    if op is FeatureOp.per_unit:
        left, right = spec.inputs
        return _safe_divide(_numeric(df, left), _numeric(df, right) + 1.0)

    if op is FeatureOp.spread:
        frame = _frame(df, spec.inputs)
        return _finite(frame.max(axis=1) - frame.min(axis=1), index)

    if op is FeatureOp.rowwise_max:
        return _finite(_frame(df, spec.inputs).max(axis=1), index)

    if op is FeatureOp.rowwise_min:
        return _finite(_frame(df, spec.inputs).min(axis=1), index)

    if op is FeatureOp.count_above:
        threshold = float(params.get("threshold", 0.0))
        frame = _frame(df, spec.inputs)
        return _finite((frame >= threshold).sum(axis=1), index)

    # ── single-column transforms ─────────────────────────────────────────
    if op is FeatureOp.scale:
        return _finite(_numeric(df, spec.inputs[0]) * float(params.get("factor", 1.0)), index)

    if op is FeatureOp.offset:
        return _finite(_numeric(df, spec.inputs[0]) + float(params.get("offset", 0.0)), index)

    if op is FeatureOp.clip:
        low = params.get("min")
        high = params.get("max")
        clipped = _numeric(df, spec.inputs[0]).clip(
            lower=None if low is None else float(low),
            upper=None if high is None else float(high),
        )
        return _finite(clipped, index)

    if op is FeatureOp.log1p:
        # Negative inputs would give NaN; the floor keeps the op total.
        column = _numeric(df, spec.inputs[0]).clip(lower=-0.999999)
        return _finite(np.log1p(column.to_numpy(dtype=float)), index)

    if op is FeatureOp.sqrt:
        column = _numeric(df, spec.inputs[0]).clip(lower=0.0)
        return _finite(np.sqrt(column.to_numpy(dtype=float)), index)

    if op is FeatureOp.binarize_threshold:
        threshold = float(params.get("threshold", 0.0))
        return (_numeric(df, spec.inputs[0]) >= threshold).astype(float)

    if op is FeatureOp.binarize_equals:
        # Compared as text so a JSON "1" still matches an integer 1.
        target = str(params.get("value"))
        column = df[spec.inputs[0]].astype("string").fillna("")
        return (column.str.strip() == target.strip()).astype(float)

    if op is FeatureOp.category_map:
        mapping = {str(k): float(v) for k, v in (params.get("mapping") or {}).items()}
        default = float(params.get("default", 0.0))
        column = df[spec.inputs[0]].astype("string").fillna("")
        mapped = column.str.strip().map(mapping)
        return _finite(mapped.astype(float).fillna(default), index)

    if op is FeatureOp.is_missing:
        source = spec.inputs[0]
        flag = missing_flag_name(source)
        # The cleaner records the mask before imputing; without that helper the
        # column has already been filled and .isna() would always be False.
        if flag in df.columns:
            return _numeric(df, flag)
        return df[source].isna().astype(float)

    # ── stateful ─────────────────────────────────────────────────────────
    if op is FeatureOp.normalize_max:
        maximum = float(state.get("max", 0.0))
        if maximum == 0.0:
            return pd.Series(0.0, index=index)
        return _finite(_numeric(df, spec.inputs[0]) / maximum, index)

    if op is FeatureOp.min_max_scale:
        low = float(state.get("min", 0.0))
        high = float(state.get("max", 0.0))
        if high == low:
            return pd.Series(0.0, index=index)
        return _finite((_numeric(df, spec.inputs[0]) - low) / (high - low), index)

    if op is FeatureOp.zscore:
        std = float(state.get("std", 0.0))
        if std == 0.0:
            return pd.Series(0.0, index=index)
        return _finite((_numeric(df, spec.inputs[0]) - float(state.get("mean", 0.0))) / std, index)

    raise KeyError(f"Unimplemented feature operation: {op}")


def describe_spec(spec: FeatureSpec) -> str:
    """One-line formula for the model card, e.g. ``a - b`` or ``mean(x, y, z)``."""
    inputs = spec.inputs
    op = spec.op
    if op is FeatureOp.difference:
        return f"{inputs[0]} - {inputs[1]}"
    if op is FeatureOp.abs_difference:
        return f"|{inputs[0]} - {inputs[1]}|"
    if op is FeatureOp.ratio:
        return f"{inputs[0]} / {inputs[1]}"
    if op is FeatureOp.per_unit:
        return f"{inputs[0]} / ({inputs[1]} + 1)"
    if op is FeatureOp.scale:
        return f"{inputs[0]} x {spec.params.get('factor')}"
    if op is FeatureOp.offset:
        return f"{inputs[0]} + {spec.params.get('offset')}"
    if op is FeatureOp.binarize_equals:
        return f"1 when {inputs[0]} = {spec.params.get('value')}"
    if op is FeatureOp.binarize_threshold:
        return f"1 when {inputs[0]} >= {spec.params.get('threshold')}"
    if op is FeatureOp.count_above:
        return f"how many of ({', '.join(inputs)}) >= {spec.params.get('threshold')}"
    if op is FeatureOp.is_missing:
        return f"1 when {inputs[0]} was blank"
    if op is FeatureOp.weighted_sum:
        weights = spec.params.get("weights", [])
        terms = [f"{w}x{name}" for name, w in zip(inputs, weights)]
        return " + ".join(terms)
    return f"{op.value}({', '.join(inputs)})"
