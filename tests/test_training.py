"""
End-to-end training, and the guards that stop a run producing something
misleading.

Every test here runs with ``provider=None`` — the no-key path. That is
deliberate: it is what CI executes, what a fresh clone executes, and what every
LLM stage falls back to. If the rule-based path ever stops producing a usable
model, the whole product loses its floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from placement_ai.plans import StageSource
from placement_ai.training.runner import STAGES, TrainingError, TrainingRunner

# ── a successful run ─────────────────────────────────────────────────────────


def test_a_model_is_produced_without_any_llm(trained):
    assert trained.champion
    assert trained.pipeline is not None
    assert trained.class_labels == ["NotPlaced", "Placed"]
    assert trained.positive_class == "Placed"


def test_the_model_actually_learns_the_signal(trained):
    """The fixture carries a real relationship; a broken pipeline would not find it."""
    assert trained.metrics["roc_auc"] > 0.75
    assert trained.metrics["n_samples"] > 0


def test_every_stage_reports_progress(messy_frame):
    seen: list[tuple[str, str]] = []
    TrainingRunner(provider=None, progress=lambda e: seen.append((e.stage, e.status))).run(
        messy_frame, target_column="outcome"
    )
    completed = {stage for stage, status in seen if status in {"ok", "fallback"}}
    assert completed == set(STAGES)


def test_progress_fraction_climbs_to_one(messy_frame):
    fractions: list[float] = []
    TrainingRunner(provider=None, progress=lambda e: fractions.append(e.fraction)).run(
        messy_frame, target_column="outcome"
    )
    assert fractions[-1] == pytest.approx(1.0)
    assert fractions == sorted(fractions)


def test_provenance_records_the_rule_based_fallback(trained):
    sources = {p.stage: p.source for p in trained.plan.provenance}
    assert set(sources) == {"schema", "cleaning", "features", "model"}
    assert all(source is StageSource.heuristic for source in sources.values())
    assert trained.plan.used_llm is False


# ── what the plan decided ────────────────────────────────────────────────────


def test_the_identifier_and_junk_columns_are_excluded(trained):
    features = trained.plan.schema_plan.feature_columns
    assert "student_id" not in features
    assert "campus" not in features  # constant
    assert "notes" not in features  # free text
    assert "test_score" in features


def test_derived_features_are_built_and_explained(trained):
    assert trained.engineered_features
    for feature in trained.engineered_features:
        assert feature["name"] and feature["op"]
        assert feature["rationale"], f"{feature['name']} has no stated reason"


def test_a_missingness_flag_is_built_for_the_gappy_column(trained):
    names = {f["name"] for f in trained.engineered_features}
    assert "attendance_was_missing" in names


# ── threshold, weighting and selection ───────────────────────────────────────


def test_the_decision_threshold_is_fitted_not_assumed(trained):
    assert 0.0 < trained.threshold < 1.0
    assert trained.metrics["threshold"] == trained.threshold


def test_the_winner_is_the_best_scoring_candidate(trained):
    best = max(trained.candidates, key=lambda c: c.score)
    assert best.algorithm == trained.champion


def test_every_candidate_is_scored(trained):
    assert len(trained.candidates) >= 2
    for candidate in trained.candidates:
        assert candidate.metrics
        assert 0.0 <= candidate.score <= 1.0


def test_the_ensemble_is_only_kept_when_it_wins(trained):
    ensemble = next((c for c in trained.candidates if c.algorithm == "ensemble"), None)
    if trained.champion == "ensemble":
        assert ensemble is not None
        others = [c.score for c in trained.candidates if c.algorithm != "ensemble"]
        assert ensemble.score >= max(others)


# ── what the run hands back ──────────────────────────────────────────────────


def test_the_input_schema_describes_every_feature(trained):
    names = {f["name"] for f in trained.input_schema}
    assert names == set(trained.plan.schema_plan.feature_columns)
    for field in trained.input_schema:
        assert field["label"]
        if field["kind"] == "numeric":
            assert field["min"] <= field["default"] <= field["max"]
        else:
            assert field["levels"]
            assert field["default"] in field["levels"]


def test_integer_columns_are_marked_for_a_whole_number_control(trained):
    projects = next(f for f in trained.input_schema if f["name"] == "projects")
    assert projects["integer"] is True
    assert projects["step"] == 1.0
    soft_skills = next(f for f in trained.input_schema if f["name"] == "soft_skills")
    assert soft_skills["integer"] is False


def test_importance_covers_the_raw_columns_a_user_recognises(trained):
    columns = {row["column"] for row in trained.importance}
    assert columns == set(trained.plan.schema_plan.feature_columns)
    assert all(row["label"] for row in trained.importance)


def test_a_drift_baseline_is_captured(trained):
    assert trained.drift_baseline
    for reference in trained.drift_baseline.values():
        if reference["kind"] == "numeric":
            assert len(reference["shares"]) == len(reference["edges"]) - 1
            assert sum(reference["shares"]) == pytest.approx(1.0, abs=1e-3)
        else:
            assert reference["shares"]


def test_a_narrative_is_always_produced(trained):
    assert trained.narrative["headline"]
    assert trained.narrative["summary"]
    # No provider was configured, so it is the template rather than generated.
    assert trained.narrative["source"] == "template"


def test_the_dataset_summary_reports_what_happened(trained):
    summary = trained.dataset_summary
    assert summary["rows_used"] <= summary["rows_supplied"]
    assert summary["train_rows"] + summary["test_rows"] == summary["rows_used"]
    assert summary["target_column"] == "outcome"
    # Headers were renamed from "Test Score" etc. on the way in.
    assert summary["renamed_headers"]


# ── guards ───────────────────────────────────────────────────────────────────


def test_too_few_rows_is_refused():
    frame = pd.DataFrame({"a": range(10), "y": ["x", "y"] * 5})
    with pytest.raises(TrainingError, match="At least"):
        TrainingRunner(provider=None).run(frame, target_column="y")


def test_a_single_outcome_is_refused():
    frame = pd.DataFrame({"a": np.arange(100.0), "y": ["Placed"] * 100})
    with pytest.raises(TrainingError, match="same value"):
        TrainingRunner(provider=None).run(frame, target_column="y")


def test_an_outcome_with_too_few_examples_is_refused():
    frame = pd.DataFrame(
        {"a": np.arange(100.0), "y": ["A"] * 97 + ["B", "B", "B"]}
    )
    with pytest.raises(TrainingError, match="too few times"):
        TrainingRunner(provider=None).run(frame, target_column="y")


def test_a_continuous_target_is_refused_with_an_explanation():
    frame = pd.DataFrame({"a": np.arange(200.0), "salary": np.linspace(1e4, 9e4, 200)})
    with pytest.raises(TrainingError, match="predicts categories"):
        TrainingRunner(provider=None).run(frame, target_column="salary")


def test_an_unknown_target_column_is_refused():
    frame = pd.DataFrame({"a": np.arange(100.0), "y": ["A", "B"] * 50})
    with pytest.raises(TrainingError, match="not in the uploaded file"):
        TrainingRunner(provider=None).run(frame, target_column="ghost")


def test_the_target_is_detected_when_not_supplied(messy_frame):
    outcome = TrainingRunner(provider=None).run(messy_frame)
    assert outcome.plan.schema_plan.target_column == "outcome"


# ── other shapes ─────────────────────────────────────────────────────────────


def test_a_multiclass_target_trains_and_skips_the_threshold():
    generator = np.random.default_rng(3)
    size = 300
    feature = generator.normal(0, 1, size)
    grade = np.where(feature < -0.5, "C", np.where(feature < 0.5, "B", "A"))
    frame = pd.DataFrame({"feature": feature, "noise": generator.normal(0, 1, size), "grade": grade})

    outcome = TrainingRunner(provider=None).run(frame, target_column="grade")
    assert set(outcome.class_labels) == {"A", "B", "C"}
    assert outcome.positive_class is None
    assert outcome.threshold == 0.5
    assert outcome.metrics["accuracy"] > 0.6


def test_a_numeric_zero_one_target_works():
    generator = np.random.default_rng(11)
    size = 250
    feature = generator.normal(0, 1, size)
    frame = pd.DataFrame({"x": feature, "placed": (feature > 0).astype(int)})
    outcome = TrainingRunner(provider=None).run(frame, target_column="placed")
    assert outcome.class_labels == ["0", "1"]
    assert outcome.positive_class == "1"


def test_duplicate_rows_are_removed_and_reported():
    generator = np.random.default_rng(5)
    base = pd.DataFrame(
        {
            "x": generator.normal(0, 1, 200).round(2),
            "y": generator.choice(["A", "B"], 200),
        }
    )
    frame = pd.concat([base, base.head(40)], ignore_index=True)
    outcome = TrainingRunner(provider=None).run(frame, target_column="y")
    assert outcome.dataset_summary["rows_dropped"] >= 40
    assert any("duplicate" in w for w in outcome.warnings)
