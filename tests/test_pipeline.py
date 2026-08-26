"""
The transformers and the assembled pipeline.

The recurring theme: everything learned from data is learned once, at fit time,
and reused verbatim afterwards. A transformer that recomputes anything during
transform will pass a round-trip test on the training set and be wrong on every
single-row prediction — so most of these tests fit on one frame and assert
against a different one.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from placement_ai.pipeline.builder import (
    build_estimator,
    build_preprocessor,
    compute_sample_weights,
    display_name,
)
from placement_ai.pipeline.dsl import missing_flag_name
from placement_ai.pipeline.ensemble import WeightedSoftVoter
from placement_ai.pipeline.transformers import (
    OTHER_LEVEL,
    CleaningTransformer,
    FeatureSynthesizer,
    prune_feature_plan,
)
from placement_ai.plans import (
    Algorithm,
    CandidateModel,
    CleaningPlan,
    ColumnCleaning,
    FeatureOp,
    FeaturePlan,
    FeatureSpec,
    ImputeStrategy,
)

# ── CleaningTransformer ──────────────────────────────────────────────────────


def test_median_fill_uses_the_training_median():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    plan = CleaningPlan(columns=[ColumnCleaning(column="a", impute=ImputeStrategy.median)])
    cleaner = CleaningTransformer(plan=plan, keep_columns=["a"]).fit(train)

    # 2.0 is the training median; a batch of NaN must not derive its own.
    result = cleaner.transform(pd.DataFrame({"a": [np.nan, np.nan]}))
    assert result["a"].tolist() == [2.0, 2.0]


def test_constant_fill_uses_the_declared_value():
    plan = CleaningPlan(
        columns=[ColumnCleaning(column="a", impute=ImputeStrategy.constant, fill_value=0)]
    )
    cleaner = CleaningTransformer(plan=plan, keep_columns=["a"]).fit(
        pd.DataFrame({"a": [5.0, np.nan]})
    )
    assert cleaner.transform(pd.DataFrame({"a": [np.nan]}))["a"].tolist() == [0.0]


def test_numbers_stored_as_text_are_coerced():
    plan = CleaningPlan(columns=[ColumnCleaning(column="a", coerce_numeric=True)])
    cleaner = CleaningTransformer(plan=plan, keep_columns=["a"]).fit(
        pd.DataFrame({"a": ["1,200", "3,400", "900"]})
    )
    result = cleaner.transform(pd.DataFrame({"a": ["2,500"]}))
    assert result["a"].tolist() == [2500.0]


def test_the_missing_mask_is_captured_before_imputation():
    """is_missing is only meaningful if the flag predates the fill."""
    plan = CleaningPlan(columns=[ColumnCleaning(column="a", impute=ImputeStrategy.median)])
    cleaner = CleaningTransformer(plan=plan, keep_columns=["a"]).fit(
        pd.DataFrame({"a": [1.0, 2.0, np.nan]})
    )
    result = cleaner.transform(pd.DataFrame({"a": [5.0, np.nan]}))
    assert result[missing_flag_name("a")].tolist() == [0.0, 1.0]
    assert not result["a"].isna().any()


def test_rare_categories_fold_into_other():
    train = pd.DataFrame({"g": ["A"] * 90 + ["B"] * 9 + ["C"]})
    plan = CleaningPlan(
        columns=[
            ColumnCleaning(
                column="g", impute=ImputeStrategy.most_frequent, rare_category_min_frequency=0.05
            )
        ]
    )
    cleaner = CleaningTransformer(plan=plan, keep_columns=["g"]).fit(train)
    result = cleaner.transform(pd.DataFrame({"g": ["A", "B", "C"]}))
    assert result["g"].tolist() == ["A", "B", OTHER_LEVEL]


def test_a_level_never_seen_in_training_becomes_other():
    train = pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50})
    plan = CleaningPlan(
        columns=[ColumnCleaning(column="g", impute=ImputeStrategy.most_frequent,
                                rare_category_min_frequency=0.01)]
    )
    cleaner = CleaningTransformer(plan=plan, keep_columns=["g"]).fit(train)
    assert cleaner.transform(pd.DataFrame({"g": ["Z"]}))["g"].tolist() == [OTHER_LEVEL]


def test_a_column_absent_at_prediction_time_is_imputed_not_fatal():
    """A partial upload should degrade in accuracy, not raise."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [9.0, 9.0, 9.0]})
    plan = CleaningPlan(
        columns=[ColumnCleaning(column="a"), ColumnCleaning(column="b")]
    )
    cleaner = CleaningTransformer(plan=plan, keep_columns=["a", "b"]).fit(train)
    result = cleaner.transform(pd.DataFrame({"a": [5.0]}))
    assert result["b"].tolist() == [9.0]
    assert result[missing_flag_name("b")].tolist() == [1.0]


def test_clipping_bounds_impossible_values():
    plan = CleaningPlan(columns=[ColumnCleaning(column="p", clip_min=0, clip_max=100)])
    cleaner = CleaningTransformer(plan=plan, keep_columns=["p"]).fit(
        pd.DataFrame({"p": [50.0, 70.0]})
    )
    assert cleaner.transform(pd.DataFrame({"p": [-20.0, 250.0]}))["p"].tolist() == [0.0, 100.0]


# ── FeatureSynthesizer ───────────────────────────────────────────────────────


def test_synthesized_columns_are_appended():
    plan = FeaturePlan(
        features=[FeatureSpec(name="total", op=FeatureOp.sum, inputs=["a", "b"])]
    )
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = FeatureSynthesizer(plan=plan).fit(frame).transform(frame)
    assert result["total"].tolist() == [4.0, 6.0]
    assert list(result.columns) == ["a", "b", "total"]


def test_a_spec_naming_a_missing_column_is_skipped_at_fit():
    plan = FeaturePlan(features=[FeatureSpec(name="x", op=FeatureOp.sum, inputs=["a", "nope"])])
    synth = FeatureSynthesizer(plan=plan).fit(pd.DataFrame({"a": [1.0]}))
    assert synth.specs_ == []
    assert synth.skipped_ and "not present" in synth.skipped_[0][1]


def test_prune_reports_what_it_dropped():
    plan = FeaturePlan(
        features=[
            FeatureSpec(name="good", op=FeatureOp.sum, inputs=["a", "b"]),
            FeatureSpec(name="bad", op=FeatureOp.sum, inputs=["a", "ghost"]),
        ]
    )
    pruned, warnings = prune_feature_plan(plan, pd.DataFrame({"a": [1.0], "b": [2.0]}))
    assert [f.name for f in pruned.features] == ["good"]
    assert len(warnings) == 1


def test_transform_tolerates_a_column_vanishing_after_fit():
    plan = FeaturePlan(features=[FeatureSpec(name="total", op=FeatureOp.sum, inputs=["a", "b"])])
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    synth = FeatureSynthesizer(plan=plan).fit(frame)
    result = synth.transform(pd.DataFrame({"a": [1.0], "b": [2.0], "extra": [9.0]}))
    assert list(result.columns) == ["a", "b", "total"]


# ── sample weights ───────────────────────────────────────────────────────────


def test_balanced_weights_equalise_class_mass():
    y = pd.Series([0] * 90 + [1] * 10)
    weights = compute_sample_weights(y, "balanced", None)
    assert weights is not None
    # Each class ends up carrying the same total weight.
    assert np.isclose(weights[y == 0].sum(), weights[y == 1].sum())


def test_no_weighting_returns_none():
    assert compute_sample_weights(pd.Series([0, 1]), "none", None) is None


def test_custom_weights_are_applied_by_label():
    y = pd.Series(["a", "b", "a"])
    weights = compute_sample_weights(y, "custom", {"a": 2.0, "b": 5.0})
    assert weights.tolist() == [2.0, 5.0, 2.0]


# ── estimator construction ───────────────────────────────────────────────────


def test_an_unsupported_hyperparameter_is_ignored_not_fatal():
    candidate = CandidateModel(
        algorithm=Algorithm.logistic_regression,
        params={"C": 0.5, "n_estimators": 500, "made_up": True},
    )
    estimator, ignored = build_estimator(candidate, n_classes=2)
    assert estimator.C == 0.5
    assert set(ignored) == {"n_estimators", "made_up"}


def test_a_string_hyperparameter_is_cast_to_the_right_type():
    candidate = CandidateModel(
        algorithm=Algorithm.random_forest, params={"n_estimators": "50", "max_depth": "none"}
    )
    estimator, _ = build_estimator(candidate, n_classes=2)
    assert estimator.n_estimators == 50
    assert estimator.max_depth is None


def test_display_names_are_human_readable():
    assert display_name("random_forest") == "Random Forest"
    assert display_name("ensemble") == "Weighted Ensemble"


# ── ensemble ─────────────────────────────────────────────────────────────────


class _Stub:
    """A fixed-output classifier, so weighting can be asserted exactly."""

    def __init__(self, probabilities, classes):
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return np.repeat(self._probabilities[None, :], len(X), axis=0)


def test_soft_voting_respects_the_weights():
    voter = WeightedSoftVoter(
        estimators=[("a", _Stub([0.0, 1.0], [0, 1])), ("b", _Stub([1.0, 0.0], [0, 1]))],
        weights=[3.0, 1.0],
        classes=np.array([0, 1]),
    ).fit()
    probabilities = voter.predict_proba(np.zeros((1, 1)))
    assert np.allclose(probabilities[0], [0.25, 0.75])


def test_members_with_a_different_class_order_are_realigned():
    """Averaging raw columns would add the wrong classes together.

    Member A says P(class 1) = 0.8. Member B lists its classes as [1, 0], so
    its [0.2, 0.8] means P(class 1) = 0.2 — the opposite call. Correctly
    aligned they cancel to 0.5; averaged position-by-position they would agree
    on 0.5/0.5 by coincidence here, so the second assertion pins the alignment
    itself rather than only the average.
    """
    member_b = _Stub([0.2, 0.8], [1, 0])
    voter = WeightedSoftVoter(
        estimators=[("a", _Stub([0.2, 0.8], [0, 1])), ("b", member_b)],
        weights=[1.0, 1.0],
        classes=np.array([0, 1]),
    ).fit()
    assert np.allclose(voter.predict_proba(np.zeros((1, 1)))[0], [0.5, 0.5])

    from placement_ai.pipeline.ensemble import _aligned_proba

    aligned = _aligned_proba(member_b, np.zeros((1, 1)), np.array([0, 1]))
    assert np.allclose(aligned[0], [0.8, 0.2])


def test_a_member_missing_a_class_contributes_zero_for_it():
    """A fold that never saw a class returns a narrower matrix."""
    from placement_ai.pipeline.ensemble import _aligned_proba

    narrow = _Stub([1.0], [0])
    aligned = _aligned_proba(narrow, np.zeros((1, 1)), np.array([0, 1]))
    assert np.allclose(aligned[0], [1.0, 0.0])


def test_all_zero_weights_fall_back_to_a_plain_mean():
    voter = WeightedSoftVoter(
        estimators=[("a", _Stub([0.0, 1.0], [0, 1])), ("b", _Stub([1.0, 0.0], [0, 1]))],
        weights=[0.0, 0.0],
        classes=np.array([0, 1]),
    ).fit()
    assert np.allclose(voter.predict_proba(np.zeros((1, 1)))[0], [0.5, 0.5])


# ── the assembled pipeline ───────────────────────────────────────────────────


def test_the_whole_pipeline_pickles_and_reloads(tmp_path, trained):
    """A bundle is one joblib; if any step is unpicklable the product breaks."""
    path = tmp_path / "pipeline.joblib"
    joblib.dump(trained.pipeline, path)
    reloaded = joblib.load(path)

    sample = pd.DataFrame([{f["name"]: f["default"] for f in trained.input_schema}])
    before = trained.pipeline.predict_proba(sample)
    after = reloaded.predict_proba(sample)
    assert np.allclose(before, after)


def test_the_encoder_selects_exactly_the_planned_columns(messy_frame):
    from placement_ai.planner.heuristic import heuristic_cleaning_plan, heuristic_schema_plan
    from placement_ai.profiling import (
        canonicalize_columns,
        classify_target_labels,
        profile_dataframe,
    )

    frame, _ = canonicalize_columns(messy_frame)
    profile = profile_dataframe(frame)
    schema = heuristic_schema_plan(profile, "outcome", classify_target_labels(frame["outcome"]))
    cleaning = heuristic_cleaning_plan(profile, schema)
    features = FeaturePlan(
        features=[FeatureSpec(name="combo", op=FeatureOp.sum, inputs=["test_score", "projects"])]
    )

    _, numeric, categorical = build_preprocessor(schema, cleaning, features)
    assert "combo" in numeric
    assert set(categorical) == set(schema.categorical_features)
    # The helper missing-flag columns exist in the frame but are never encoded.
    assert not any(name.startswith("__missing__") for name in numeric + categorical)


@pytest.mark.parametrize("n_rows", [1, 5])
def test_predicting_on_a_tiny_batch_matches_the_full_batch(trained, messy_frame, n_rows):
    """Guards the classic per-batch-statistic bug from the other direction."""
    from placement_ai.profiling import canonicalize_columns

    frame, _ = canonicalize_columns(messy_frame)
    columns = [f["name"] for f in trained.input_schema]
    full = trained.pipeline.predict_proba(frame[columns])
    partial = trained.pipeline.predict_proba(frame[columns].head(n_rows))
    assert np.allclose(full[:n_rows], partial)
