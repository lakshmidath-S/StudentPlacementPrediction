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


# ── a run the LLM actually plans ─────────────────────────────────────────────


class _ScriptedProvider:
    """Replays realistic replies in stage order, in the shape the prompts ask for.

    This is the only place the *success* path of the LLM integration is
    exercised end to end. Everything else in the suite runs with no provider, so
    without this a change that broke the prompt-to-executor contract would ship
    green.
    """

    name = "scripted"
    model = "scripted-1"
    label = "scripted:scripted-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete_json(self, system, user, max_output_tokens=8192, temperature=0.2):
        import json as _json

        from placement_ai.llm.base import LLMError, LLMResult

        self.calls += 1
        if not self.replies:
            raise LLMError("no scripted reply left")
        reply = self.replies.pop(0)
        return LLMResult(
            data=reply,
            raw_text=_json.dumps(reply),
            provider=self.name,
            model=self.model,
            latency_ms=1.0,
        )


SCRIPTED_SCHEMA = {
    "target_column": "outcome",
    "task_type": "binary_classification",
    "positive_class": "Placed",
    "summary": "Student records with a placement outcome.",
    "columns": [
        {"name": "student_id", "role": "identifier", "reason": "A roll number."},
        {"name": "test_score", "role": "numeric_feature", "display_label": "Test score"},
        {"name": "attendance", "role": "numeric_feature", "display_label": "Attendance"},
        {"name": "projects", "role": "numeric_feature", "display_label": "Projects"},
        {"name": "soft_skills", "role": "numeric_feature", "display_label": "Soft skills"},
        {"name": "stipend", "role": "numeric_feature", "leakage_risk": True,
         "reason": "Only known after an offer is made."},
        {"name": "department", "role": "categorical_feature"},
        {"name": "training", "role": "categorical_feature"},
        {"name": "campus", "role": "drop", "reason": "Same for every row."},
        {"name": "notes", "role": "drop", "reason": "Free text."},
        {"name": "outcome", "role": "target"},
    ],
}

SCRIPTED_CLEANING = {
    "drop_duplicate_rows": True,
    "drop_rows_missing_target": True,
    "notes": "Mostly clean; attendance has gaps.",
    "columns": [
        {"column": "test_score", "impute": "median", "reason": "Resists outliers."},
        {"column": "attendance", "impute": "median", "clip_min": 0, "clip_max": 100,
         "reason": "A percentage cannot exceed 100."},
        {"column": "projects", "impute": "constant", "fill_value": 0,
         "reason": "A blank means none were completed."},
        {"column": "soft_skills", "impute": "mean", "reason": "Symmetric rating."},
        {"column": "department", "impute": "most_frequent", "strip_whitespace": True,
         "rare_category_min_frequency": 0.02, "reason": "Fold thin departments together."},
        {"column": "training", "impute": "most_frequent", "reason": "Most common value."},
    ],
}

SCRIPTED_FEATURES = {
    "notes": "Composites and gaps a placement officer would compute by hand.",
    "features": [
        {"name": "soft_skills_scaled", "op": "scale", "inputs": ["soft_skills"],
         "params": {"factor": 20}, "rationale": "Puts the 1-5 rating on a 0-100 scale."},
        {"name": "readiness", "op": "weighted_sum", "inputs": ["test_score", "attendance"],
         "params": {"weights": [0.7, 0.3]},
         "rationale": "Weighs exam performance above attendance."},
        {"name": "score_attendance_gap", "op": "difference",
         "inputs": ["test_score", "attendance"],
         "rationale": "A student who shows up but underperforms looks different."},
        {"name": "projects_normalized", "op": "normalize_max", "inputs": ["projects"],
         "rationale": "Scales an open-ended count against the training maximum."},
        {"name": "attendance_missing", "op": "is_missing", "inputs": ["attendance"],
         "rationale": "Whether attendance was recorded can itself be a signal."},
        {"name": "has_training", "op": "binarize_equals", "inputs": ["training"],
         "params": {"value": "Yes"}, "rationale": "Explicit flag for placement training."},
        {"name": "leaky_feature", "op": "scale", "inputs": ["stipend"],
         "params": {"factor": 1}, "rationale": "Should be rejected — stipend was dropped."},
    ],
}

SCRIPTED_MODEL = {
    "primary_metric": "roc_auc",
    "class_weight_strategy": "balanced",
    "test_size": 0.25,
    "cv_folds": 4,
    "build_ensemble": True,
    "threshold_strategy": "best_f1",
    "notes": "A linear baseline against a forest.",
    "candidates": [
        {"algorithm": "logistic_regression", "params": {"C": 0.8}, "ensemble_weight": 1.0,
         "rationale": "Fast, calibrated baseline."},
        {"algorithm": "random_forest", "params": {"n_estimators": 60, "max_depth": 8},
         "ensemble_weight": 1.4, "rationale": "Captures interactions."},
    ],
}


SCRIPTED_NARRATIVE = {
    "headline": "This model separates placed from unplaced students well.",
    "summary": "It reads exam performance and attendance to estimate placement.",
    "strengths": ["Catches most students who do get placed."],
    "cautions": ["Trained on a small sample."],
    "next_steps": ["Re-check against real outcomes next term."],
}


@pytest.fixture
def llm_run(messy_frame):
    # Five calls, not four: the four planning stages plus the narration pass.
    provider = _ScriptedProvider(
        [SCRIPTED_SCHEMA, SCRIPTED_CLEANING, SCRIPTED_FEATURES, SCRIPTED_MODEL,
         SCRIPTED_NARRATIVE]
    )
    outcome = TrainingRunner(provider=provider).run(messy_frame, target_column="outcome")
    return outcome, provider


def test_a_scripted_run_is_planned_entirely_by_the_model(llm_run):
    outcome, provider = llm_run
    assert provider.calls == 5  # four planning stages + narration
    assert {p.source for p in outcome.plan.provenance} == {StageSource.llm}
    assert outcome.plan.llm_authored_stages == ["schema", "cleaning", "features", "model"]


def test_the_models_column_roles_are_honoured(llm_run):
    outcome, _ = llm_run
    schema = outcome.plan.schema_plan
    assert "student_id" not in schema.feature_columns
    assert "campus" not in schema.feature_columns
    # Flagged as leakage by the model, so excluded whatever role it was given.
    assert "stipend" not in schema.feature_columns
    assert schema.spec("test_score").display_label == "Test score"


def test_the_models_features_are_built_and_the_bad_one_is_not(llm_run):
    outcome, _ = llm_run
    names = {f["name"] for f in outcome.engineered_features}
    assert {"soft_skills_scaled", "readiness", "score_attendance_gap",
            "projects_normalized", "has_training"} <= names
    # It referenced a column the schema stage dropped for leakage.
    assert "leaky_feature" not in names


def test_the_models_settings_are_applied(llm_run):
    outcome, _ = llm_run
    plan = outcome.plan.model_plan
    assert plan.test_size == 0.25
    assert plan.cv_folds == 4
    assert [c.algorithm.value for c in plan.candidates] == [
        "logistic_regression", "random_forest"
    ]
    assert outcome.dataset_summary["test_rows"] == pytest.approx(
        outcome.dataset_summary["rows_used"] * 0.25, rel=0.05
    )


def test_an_llm_planned_model_still_learns(llm_run):
    outcome, _ = llm_run
    assert outcome.metrics["roc_auc"] > 0.75
    assert outcome.pipeline is not None


def test_the_narrative_comes_from_the_model_when_it_answers(llm_run):
    outcome, _ = llm_run
    assert outcome.narrative["source"] == "llm"
    assert outcome.narrative["headline"] == SCRIPTED_NARRATIVE["headline"]
    assert outcome.narrative["cautions"] == SCRIPTED_NARRATIVE["cautions"]
