"""
Serving a saved model: prediction, attribution, and drift.

The thread running through these is that inference must reproduce training
exactly. The fitted threshold is applied, not a plain argmax; a partial upload
is imputed rather than rejected; and drift is measured against the baseline
captured at training time, not against recent traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from placement_ai.inference.drift import (
    ALERT_THRESHOLD,
    MIN_SAMPLES,
    check_drift,
    psi_from_edges,
    psi_from_shares,
    psi_numeric,
    training_baseline,
)
from placement_ai.inference.explain import (
    attribute,
    cohort_summary,
    improvement_levers,
    what_if_curve,
)
from placement_ai.inference.predictor import WorkspacePredictor
from placement_ai.profiling import canonicalize_columns
from placement_ai.registry.model_store import ModelStore


@pytest.fixture
def bundle(workspace, trained):
    return ModelStore(workspace).save(trained)


@pytest.fixture
def predictor(bundle):
    return WorkspacePredictor(bundle)


@pytest.fixture
def typical(predictor):
    return predictor.defaults


# ── single prediction ────────────────────────────────────────────────────────


def test_a_prediction_returns_a_label_and_calibrated_probabilities(predictor, typical):
    prediction = predictor.predict_one(typical)
    assert prediction.label in predictor.class_labels
    assert set(prediction.probabilities) == set(predictor.class_labels)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_the_fitted_threshold_decides_the_label_not_argmax(predictor, typical):
    """A 0.45 probability with a 0.40 threshold is still the positive class."""
    positive = predictor.positive_class
    negative = next(c for c in predictor.class_labels if c != positive)

    prediction = predictor.predict_one(typical)
    expected = positive if prediction.probability >= predictor.threshold else negative
    assert prediction.label == expected


def test_the_confidence_band_reflects_distance_from_the_threshold(predictor, typical):
    prediction = predictor.predict_one(typical)
    if prediction.margin < 0.10:
        assert prediction.confidence_band == "Borderline"
    elif prediction.margin >= 0.30:
        assert prediction.confidence_band == "High confidence"
    else:
        assert prediction.confidence_band == "Moderate confidence"


def test_a_stronger_record_scores_higher(predictor, typical):
    weak = {**typical, "test_score": 30.0, "projects": 0, "attendance": 40.0}
    strong = {**typical, "test_score": 95.0, "projects": 5, "attendance": 98.0}
    assert predictor.predict_one(strong).probability > predictor.predict_one(weak).probability


def test_unsupplied_fields_fall_back_to_typical_values(predictor):
    record = predictor.build_record({"test_score": 80.0})
    assert set(record) == set(predictor.expected_columns)
    assert record["test_score"] == 80.0


def test_an_unknown_field_is_ignored(predictor):
    record = predictor.build_record({"not_a_column": 1})
    assert "not_a_column" not in record


# ── batch prediction ─────────────────────────────────────────────────────────


def test_a_frame_check_reports_what_is_present_missing_and_extra(predictor, messy_frame):
    frame, check = predictor.check_frame(messy_frame)
    assert check.usable
    assert not check.missing
    # student_id/notes/campus were excluded from the model.
    assert "student_id" in check.extra
    assert check.renamed


def test_a_missing_column_is_reported_but_still_scores(predictor, messy_frame):
    reduced = messy_frame.drop(columns=["Attendance %"])
    frame, check = predictor.check_frame(reduced)
    assert "attendance" in check.missing
    scored = predictor.predict_frame(frame)
    assert len(scored) == len(reduced)
    assert scored["probability"].notna().all()


def test_a_completely_unrelated_file_is_flagged_as_unusable(predictor):
    _, check = predictor.check_frame(pd.DataFrame({"totally": [1], "different": [2]}))
    assert not check.usable


def test_batch_scoring_adds_one_column_per_class(predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    scored = predictor.predict_frame(frame)
    for label in predictor.class_labels:
        assert f"p_{label}" in scored.columns
    assert {"prediction", "probability", "confidence"} <= set(scored.columns)


def test_batch_and_single_prediction_agree(predictor, messy_frame):
    """The classic per-batch-statistic bug would make these diverge."""
    frame, _ = canonicalize_columns(messy_frame)
    frame = frame.head(5)
    scored = predictor.predict_frame(frame)
    for position in range(len(frame)):
        record = predictor.build_record(
            {c: frame.iloc[position][c] for c in predictor.expected_columns if c in frame.columns}
        )
        single = predictor.predict_one(record)
        assert single.probability == pytest.approx(scored["probability"].iloc[position], abs=1e-4)


def test_a_cohort_summary_counts_the_batch(predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    summary = cohort_summary(predictor.predict_frame(frame), predictor)
    assert summary["total"] == len(messy_frame)
    assert 0.0 <= summary["positive_rate"] <= 1.0
    assert sum(summary["by_label"].values()) == summary["total"]


# ── attribution ──────────────────────────────────────────────────────────────


def test_attribution_names_the_fields_that_differ_from_typical(predictor, typical):
    record = {**typical, "test_score": 95.0, "projects": 5}
    drivers = attribute(predictor, record)
    columns = {row["column"] for row in drivers}
    assert {"test_score", "projects"} <= columns
    # A field left at its typical value has nothing to explain.
    assert all(row["value"] != row["typical"] for row in drivers)


def test_a_helpful_value_gets_a_positive_delta(predictor, typical):
    record = {**typical, "test_score": 98.0}
    driver = next(r for r in attribute(predictor, record) if r["column"] == "test_score")
    assert driver["delta"] > 0


def test_a_harmful_value_gets_a_negative_delta(predictor, typical):
    record = {**typical, "test_score": 20.0}
    driver = next(r for r in attribute(predictor, record) if r["column"] == "test_score")
    assert driver["delta"] < 0


def test_attribution_is_ranked_by_magnitude(predictor, typical):
    record = {**typical, "test_score": 95.0, "projects": 5, "attendance": 95.0}
    deltas = [abs(row["delta"]) for row in attribute(predictor, record)]
    assert deltas == sorted(deltas, reverse=True)


def test_attribution_can_be_capped(predictor, typical):
    record = {**typical, "test_score": 95.0, "projects": 5, "attendance": 95.0}
    assert len(attribute(predictor, record, top_n=2)) == 2


def test_a_typical_record_has_nothing_to_attribute(predictor, typical):
    assert attribute(predictor, typical) == []


# ── what-if and levers ───────────────────────────────────────────────────────


def test_a_what_if_curve_stays_inside_the_training_range(predictor, typical):
    curve = what_if_curve(predictor, typical, "test_score")
    field = predictor.field("test_score")
    assert not curve.empty
    assert curve["value"].min() >= field["min"]
    assert curve["value"].max() <= field["max"]
    assert curve["probability"].between(0, 1).all()


def test_a_what_if_curve_over_a_categorical_covers_its_levels(predictor, typical):
    curve = what_if_curve(predictor, typical, "department")
    assert set(curve["value"]) == set(predictor.field("department")["levels"])


def test_an_unknown_field_yields_an_empty_curve(predictor, typical):
    assert what_if_curve(predictor, typical, "not_a_field").empty


def test_levers_are_offered_only_below_the_threshold(predictor, typical):
    strong = {**typical, "test_score": 99.0, "projects": 5, "attendance": 99.0}
    assert improvement_levers(predictor, strong) == []


def test_a_lever_actually_clears_the_threshold(predictor, typical):
    weak = {**typical, "test_score": 45.0, "projects": 0, "attendance": 55.0}
    if predictor.predict_one(weak).probability >= predictor.threshold:
        pytest.skip("This model already places the weak record above the threshold.")

    for lever in improvement_levers(predictor, weak):
        improved = predictor.predict_one({**weak, lever["column"]: lever["to"]})
        assert improved.probability >= predictor.threshold
        assert lever["gain"] > 0


# ── PSI ──────────────────────────────────────────────────────────────────────


def test_identical_distributions_have_no_drift():
    values = pd.Series(np.random.default_rng(0).normal(0, 1, 2000))
    assert psi_numeric(values, values) < 0.01


def test_a_shifted_distribution_registers_drift():
    generator = np.random.default_rng(0)
    base = pd.Series(generator.normal(0, 1, 2000))
    shifted = pd.Series(generator.normal(3, 1, 2000))
    assert psi_numeric(base, shifted) > ALERT_THRESHOLD


def test_psi_is_never_negative():
    generator = np.random.default_rng(1)
    for shift in (0.0, 0.5, 2.0):
        base = pd.Series(generator.normal(0, 1, 500))
        other = pd.Series(generator.normal(shift, 1, 500))
        assert psi_numeric(base, other) >= 0


def test_categorical_psi_reacts_to_a_changed_mix():
    shares = {"A": 0.5, "B": 0.5}
    same = pd.Series(["A"] * 500 + ["B"] * 500)
    different = pd.Series(["A"] * 950 + ["B"] * 50)
    assert psi_from_shares(shares, same) < 0.01
    assert psi_from_shares(shares, different) > ALERT_THRESHOLD


def test_an_unseen_category_is_finite_not_infinite():
    """Zero baseline mass must not produce an infinite PSI."""
    value = psi_from_shares({"A": 1.0}, pd.Series(["B"] * 100))
    assert np.isfinite(value)
    assert value > ALERT_THRESHOLD


def test_bucket_shares_are_stored_rather_than_assumed_uniform():
    """A discrete column collapses quantile edges; assuming 1/n reports false drift."""
    discrete = pd.Series([0] * 700 + [1] * 200 + [2] * 100)
    baseline = training_baseline(
        pd.DataFrame({"backlogs": discrete}),
        [{"name": "backlogs", "kind": "numeric"}],
    )["backlogs"]

    assert len(baseline["shares"]) == len(baseline["edges"]) - 1
    # Scoring the very data it was fitted on must read as no drift.
    assert psi_from_edges(baseline["edges"], baseline["shares"], discrete) < 0.05


# ── drift reports ────────────────────────────────────────────────────────────


def test_scoring_the_training_population_reads_as_stable(bundle, predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    report = check_drift(bundle.drift_baseline, frame, bundle.input_schema)
    assert report.status == "ok"
    assert all(column.psi < 0.10 for column in report.columns)


def test_a_shifted_population_raises_an_alert(bundle, predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    shifted = frame.copy()
    shifted["test_score"] = shifted["test_score"] * 0.4
    report = check_drift(bundle.drift_baseline, shifted, bundle.input_schema)
    assert report.status == "alert"
    assert report.worst.column == "test_score"
    assert "test_score" in report.worst.label.lower().replace(" ", "_")


def test_too_few_records_reports_insufficient_data(bundle, predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    report = check_drift(bundle.drift_baseline, frame.head(5), bundle.input_schema)
    assert report.status == "insufficient_data"
    assert str(MIN_SAMPLES) in report.message


def test_a_model_without_a_baseline_says_so(predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    report = check_drift({}, frame, predictor.input_schema)
    assert report.status == "unavailable"
    assert "retrain" in report.message.lower()


def test_a_drift_report_serialises(bundle, predictor, messy_frame):
    frame, _ = predictor.check_frame(messy_frame)
    payload = check_drift(bundle.drift_baseline, frame, bundle.input_schema).to_dict()
    assert payload["status"] == "ok"
    assert payload["columns"] and "psi" in payload["columns"][0]
