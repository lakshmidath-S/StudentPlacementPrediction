"""
The feature DSL.

This is the layer that executes generated plans, so it gets the closest
scrutiny in the suite. Two properties matter beyond "the arithmetic is right":

  - degenerate input yields a finite number, never NaN or infinity, because a
    NaN here surfaces as an incomprehensible error deep inside an estimator;
  - stateful ops use the constant learned at fit time, never one recomputed
    from the batch in front of them. Getting that wrong is silent: every
    prediction still returns a number, just the wrong one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from placement_ai.pipeline.dsl import apply_spec, describe_spec, fit_spec, missing_flag_name
from placement_ai.plans import FeatureOp, FeatureSpec


def run(op, inputs, frame, params=None, state=None):
    spec = FeatureSpec(name="f", op=op, inputs=inputs, params=params or {})
    return apply_spec(spec, frame, state if state is not None else fit_spec(spec, frame))


# ── combinations ─────────────────────────────────────────────────────────────


def test_sum_and_mean(clean_frame):
    assert run(FeatureOp.sum, ["a", "b"], clean_frame).tolist() == [11, 22, 33, 44]
    assert run(FeatureOp.mean, ["a", "b"], clean_frame).tolist() == [5.5, 11, 16.5, 22]


def test_difference_and_absolute(clean_frame):
    assert run(FeatureOp.difference, ["a", "b"], clean_frame).tolist() == [-9, -18, -27, -36]
    assert run(FeatureOp.abs_difference, ["a", "b"], clean_frame).tolist() == [9, 18, 27, 36]


def test_weighted_sum_uses_one_weight_per_input(clean_frame):
    result = run(FeatureOp.weighted_sum, ["a", "b"], clean_frame, {"weights": [2.0, 0.5]})
    assert result.tolist() == [7, 14, 21, 28]


def test_spread_and_rowwise_extremes(clean_frame):
    assert run(FeatureOp.spread, ["a", "b"], clean_frame).tolist() == [9, 18, 27, 36]
    assert run(FeatureOp.rowwise_max, ["a", "b"], clean_frame).tolist() == [10, 20, 30, 40]
    assert run(FeatureOp.rowwise_min, ["a", "b"], clean_frame).tolist() == [1, 2, 3, 4]


def test_count_above_counts_inputs_clearing_the_threshold(clean_frame):
    result = run(FeatureOp.count_above, ["a", "b"], clean_frame, {"threshold": 3.0})
    assert result.tolist() == [1, 1, 2, 2]


def test_product(clean_frame):
    assert run(FeatureOp.product, ["a", "b"], clean_frame).tolist() == [10, 40, 90, 160]


# ── guarded arithmetic ───────────────────────────────────────────────────────


def test_ratio_by_zero_is_zero_not_infinity(clean_frame):
    result = run(FeatureOp.ratio, ["a", "c"], clean_frame)
    assert result.tolist() == [0.0, 2.0, 0.0, 2.0]
    assert np.isfinite(result).all()


def test_per_unit_offsets_the_denominator(clean_frame):
    result = run(FeatureOp.per_unit, ["a", "c"], clean_frame)
    assert result.tolist() == [1.0, 1.0, 3.0, 4 / 3]


def test_log1p_survives_a_negative_input():
    frame = pd.DataFrame({"a": [-5.0, 0.0, 3.0]})
    result = run(FeatureOp.log1p, ["a"], frame)
    assert np.isfinite(result).all()
    assert result.iloc[1] == 0.0


def test_sqrt_floors_at_zero():
    frame = pd.DataFrame({"a": [-4.0, 0.0, 9.0]})
    assert run(FeatureOp.sqrt, ["a"], frame).tolist() == [0.0, 0.0, 3.0]


def test_non_numeric_text_becomes_zero_rather_than_nan():
    frame = pd.DataFrame({"a": ["1", "oops", "3"], "b": [1.0, 1.0, 1.0]})
    result = run(FeatureOp.sum, ["a", "b"], frame)
    assert result.tolist() == [2.0, 1.0, 4.0]


# ── single-column transforms ─────────────────────────────────────────────────


def test_scale_offset_and_clip(clean_frame):
    assert run(FeatureOp.scale, ["a"], clean_frame, {"factor": 20}).tolist() == [20, 40, 60, 80]
    assert run(FeatureOp.offset, ["a"], clean_frame, {"offset": 5}).tolist() == [6, 7, 8, 9]
    assert run(FeatureOp.clip, ["a"], clean_frame, {"min": 2, "max": 3}).tolist() == [2, 2, 3, 3]


def test_binarize_threshold(clean_frame):
    assert run(FeatureOp.binarize_threshold, ["a"], clean_frame, {"threshold": 3}).tolist() == [
        0.0, 0.0, 1.0, 1.0
    ]


def test_binarize_equals_compares_as_text(clean_frame):
    """A JSON "1" from a plan must still match an integer 1 in the data."""
    assert run(FeatureOp.binarize_equals, ["grade"], clean_frame, {"value": "Low"}).tolist() == [
        1.0, 0.0, 1.0, 0.0
    ]
    numeric = pd.DataFrame({"flag": [1, 0, 1]})
    assert run(FeatureOp.binarize_equals, ["flag"], numeric, {"value": "1"}).tolist() == [
        1.0, 0.0, 1.0
    ]


def test_category_map_applies_the_default_for_unseen_levels(clean_frame):
    result = run(
        FeatureOp.category_map,
        ["grade"],
        clean_frame,
        {"mapping": {"Low": 0, "High": 2}, "default": -1},
    )
    assert result.tolist() == [0.0, 2.0, 0.0, -1.0]


def test_is_missing_prefers_the_flag_the_cleaner_left_behind():
    """After imputation the original gaps are only visible via the helper column."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], missing_flag_name("a"): [0.0, 1.0, 0.0]})
    assert run(FeatureOp.is_missing, ["a"], frame).tolist() == [0.0, 1.0, 0.0]


def test_is_missing_falls_back_to_the_column_itself():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert run(FeatureOp.is_missing, ["a"], frame).tolist() == [0.0, 1.0, 0.0]


# ── stateful ops ─────────────────────────────────────────────────────────────


def test_normalize_max_reuses_the_fitted_maximum():
    """The failure this guards against: a single row divided by itself is 1.0."""
    train = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
    spec = FeatureSpec(name="f", op=FeatureOp.normalize_max, inputs=["a"])
    state = fit_spec(spec, train)
    assert state == {"max": 10.0}

    one_row = pd.DataFrame({"a": [4.0]})
    assert apply_spec(spec, one_row, state).tolist() == [0.4]
    # Refitting on the single row would return 1.0 — the corruption being avoided.
    assert apply_spec(spec, one_row, fit_spec(spec, one_row)).tolist() == [1.0]


def test_normalize_max_of_an_all_zero_column_is_zero():
    frame = pd.DataFrame({"a": [0.0, 0.0]})
    spec = FeatureSpec(name="f", op=FeatureOp.normalize_max, inputs=["a"])
    result = apply_spec(spec, frame, fit_spec(spec, frame))
    assert result.tolist() == [0.0, 0.0]


def test_min_max_scale_uses_the_training_range():
    train = pd.DataFrame({"a": [10.0, 20.0]})
    spec = FeatureSpec(name="f", op=FeatureOp.min_max_scale, inputs=["a"])
    state = fit_spec(spec, train)
    assert apply_spec(spec, pd.DataFrame({"a": [15.0]}), state).tolist() == [0.5]


def test_min_max_scale_of_a_constant_column_is_zero():
    frame = pd.DataFrame({"a": [3.0, 3.0]})
    spec = FeatureSpec(name="f", op=FeatureOp.min_max_scale, inputs=["a"])
    assert apply_spec(spec, frame, fit_spec(spec, frame)).tolist() == [0.0, 0.0]


def test_zscore_uses_training_mean_and_spread():
    train = pd.DataFrame({"a": [0.0, 10.0, 20.0]})
    spec = FeatureSpec(name="f", op=FeatureOp.zscore, inputs=["a"])
    state = fit_spec(spec, train)
    assert state["mean"] == 10.0
    assert apply_spec(spec, pd.DataFrame({"a": [10.0]}), state).tolist() == [0.0]


def test_zscore_of_a_constant_column_is_zero():
    frame = pd.DataFrame({"a": [4.0, 4.0]})
    spec = FeatureSpec(name="f", op=FeatureOp.zscore, inputs=["a"])
    assert apply_spec(spec, frame, fit_spec(spec, frame)).tolist() == [0.0, 0.0]


# ── coverage & description ───────────────────────────────────────────────────


@pytest.mark.parametrize("op", list(FeatureOp))
def test_every_declared_op_is_implemented(op, clean_frame):
    """A new enum member without an implementation must fail here, not in a run."""
    from placement_ai.plans import CATEGORICAL_OPS, OP_ARITY

    minimum, _ = OP_ARITY[op]
    column = "grade" if op in CATEGORICAL_OPS and op is not FeatureOp.is_missing else "a"
    inputs = [column] if minimum == 1 else ["a", "b"][:max(minimum, 2)]
    params = {
        FeatureOp.scale: {"factor": 2},
        FeatureOp.offset: {"offset": 1},
        FeatureOp.clip: {"min": 0, "max": 5},
        FeatureOp.count_above: {"threshold": 1},
        FeatureOp.binarize_threshold: {"threshold": 1},
        FeatureOp.binarize_equals: {"value": "Low"},
        FeatureOp.category_map: {"mapping": {"Low": 1}, "default": 0},
        FeatureOp.weighted_sum: {"weights": [1.0, 1.0]},
    }.get(op, {})

    result = run(op, inputs, clean_frame, params)
    assert len(result) == len(clean_frame)
    assert np.isfinite(result).all(), f"{op.value} produced a non-finite value"


def test_describe_spec_reads_as_a_formula():
    spec = FeatureSpec(name="f", op=FeatureOp.difference, inputs=["a", "b"])
    assert describe_spec(spec) == "a - b"
    spec = FeatureSpec(name="f", op=FeatureOp.mean, inputs=["a", "b", "c"])
    assert describe_spec(spec) == "mean(a, b, c)"
