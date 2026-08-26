"""
The trust boundary: what happens to a plan the LLM just produced.

These tests are written as if the model is adversarial rather than merely
imperfect, because from the executor's point of view the two are the same. The
rule being enforced throughout is that a bad plan degrades into a good one —
never into an exception the user sees, and never into an execution the
vocabulary does not allow.
"""

from __future__ import annotations

import pytest

from placement_ai.planner.sanitize import (
    PlanRejected,
    coerce_cleaning_plan,
    coerce_feature_plan,
    coerce_model_plan,
    coerce_schema_plan,
)
from placement_ai.plans import ColumnRole, FeatureOp, FeatureSpec, SchemaPlan, TaskType
from placement_ai.profiling import canonicalize_columns, classify_target_labels, profile_dataframe


@pytest.fixture
def context(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    profile = profile_dataframe(frame)
    info = classify_target_labels(frame["outcome"])
    return frame, profile, info


@pytest.fixture
def schema(context):
    _, profile, info = context
    plan, _ = coerce_schema_plan(
        {
            "target_column": "outcome",
            "task_type": "binary_classification",
            "positive_class": "Placed",
            "columns": [
                {"name": "test_score", "role": "numeric_feature"},
                {"name": "attendance", "role": "numeric_feature"},
                {"name": "projects", "role": "numeric_feature"},
                {"name": "soft_skills", "role": "numeric_feature"},
                {"name": "department", "role": "categorical_feature"},
                {"name": "training", "role": "categorical_feature"},
                {"name": "student_id", "role": "identifier"},
                {"name": "outcome", "role": "target"},
            ],
        },
        profile,
        "outcome",
        info,
    )
    return plan


# ── schema stage ─────────────────────────────────────────────────────────────


def test_columns_the_model_forgot_are_filled_in_by_rule(schema, context):
    """The plan must always cover the whole table, however partial the reply."""
    _, profile, _ = context
    assert set(schema.feature_columns) | set(schema.dropped_columns) | {"outcome"} == set(
        profile.column_names
    )
    # notes/campus/stipend were absent from the reply above.
    assert "notes" in schema.dropped_columns
    assert "campus" in schema.dropped_columns


def test_an_unknown_column_is_ignored(context):
    _, profile, info = context
    plan, warnings = coerce_schema_plan(
        {"columns": [{"name": "invented_column", "role": "numeric_feature"}]},
        profile, "outcome", info,
    )
    assert "invented_column" not in plan.feature_columns
    assert any("invented_column" in w for w in warnings)


def test_an_original_cased_header_is_matched_to_its_snake_case_form(context):
    _, profile, info = context
    plan, _ = coerce_schema_plan(
        {"columns": [{"name": "Test Score", "role": "numeric_feature"}]},
        profile, "outcome", info,
    )
    assert "test_score" in plan.numeric_features


def test_a_leakage_flag_overrides_the_assigned_role(context):
    _, profile, info = context
    plan, warnings = coerce_schema_plan(
        {
            "columns": [
                {"name": "stipend", "role": "numeric_feature", "leakage_risk": True},
                {"name": "test_score", "role": "numeric_feature"},
            ]
        },
        profile, "outcome", info,
    )
    assert "stipend" not in plan.feature_columns
    assert any("leakage" in w for w in warnings)


def test_an_unrecognised_role_falls_back_to_the_rule(context):
    _, profile, info = context
    plan, warnings = coerce_schema_plan(
        {"columns": [{"name": "test_score", "role": "magic"}]}, profile, "outcome", info
    )
    assert "test_score" in plan.feature_columns
    assert any("role" in w for w in warnings)


def test_an_invented_positive_class_is_replaced_with_an_observed_one(context):
    _, profile, info = context
    plan, warnings = coerce_schema_plan(
        {"positive_class": "Employed", "columns": [{"name": "test_score", "role": "numeric_feature"}]},
        profile, "outcome", info,
    )
    assert plan.positive_class == "Placed"
    assert any("Employed" in w for w in warnings)


def test_a_second_target_is_demoted(context):
    _, profile, info = context
    plan, _ = coerce_schema_plan(
        {
            "columns": [
                {"name": "outcome", "role": "target"},
                {"name": "training", "role": "target"},
                {"name": "test_score", "role": "numeric_feature"},
            ]
        },
        profile, "outcome", info,
    )
    assert plan.target_column == "outcome"
    assert plan.spec("training").role is ColumnRole.drop


def test_a_plan_leaving_no_features_is_rejected(context):
    _, profile, info = context
    with pytest.raises(ValueError, match="no usable feature"):
        SchemaPlan(
            target_column="outcome",
            columns=[{"name": "outcome", "role": "target"}],
        )


# ── cleaning stage ───────────────────────────────────────────────────────────


def test_every_feature_ends_up_with_a_cleaning_rule(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_cleaning_plan(
        {"columns": [{"column": "test_score", "impute": "mean", "reason": "symmetric"}]},
        profile, schema,
    )
    covered = {rule.column for rule in plan.columns}
    assert set(schema.feature_columns).issubset(covered)
    assert plan.for_column("test_score").impute.value == "mean"
    assert any("attendance" in w for w in warnings)


def test_a_cleaning_plan_cannot_drop_a_column_the_schema_kept(schema, context):
    """Otherwise the encoder would ask for a column that no longer exists."""
    _, profile, _ = context
    plan, _ = coerce_cleaning_plan(
        {"drop_columns": ["test_score", "attendance"], "columns": []}, profile, schema
    )
    assert "test_score" not in plan.drop_columns
    assert plan.drop_columns == schema.dropped_columns


def test_a_wild_rare_category_threshold_is_clamped(schema, context):
    _, profile, _ = context
    plan, _ = coerce_cleaning_plan(
        {"columns": [{"column": "department", "rare_category_min_frequency": 0.9}]},
        profile, schema,
    )
    assert plan.for_column("department").rare_category_min_frequency <= 0.2


# ── feature stage ────────────────────────────────────────────────────────────


def test_a_valid_feature_survives(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {
            "features": [
                {
                    "name": "score_per_project",
                    "op": "per_unit",
                    "inputs": ["test_score", "projects"],
                    "rationale": "Output relative to effort.",
                }
            ]
        },
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["score_per_project"]
    assert not warnings


def test_a_feature_naming_a_dropped_column_is_discarded(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {"features": [{"name": "leak", "op": "scale", "inputs": ["notes"], "params": {"factor": 2}},
                      {"name": "keeper", "op": "sum", "inputs": ["test_score", "attendance"]}]},
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["keeper"]
    assert any("not an available feature" in w for w in warnings)


def test_a_numeric_op_pointed_at_a_categorical_is_discarded(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {"features": [{"name": "bad", "op": "sum", "inputs": ["department", "test_score"]},
                      {"name": "keeper", "op": "sum", "inputs": ["test_score", "attendance"]}]},
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["keeper"]
    assert any("categorical" in w for w in warnings)


def test_a_categorical_op_on_a_categorical_is_allowed(schema, context):
    _, profile, _ = context
    plan, _ = coerce_feature_plan(
        {
            "features": [
                {
                    "name": "is_cs",
                    "op": "binarize_equals",
                    "inputs": ["department"],
                    "params": {"value": "CS"},
                }
            ]
        },
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["is_cs"]


def test_a_feature_colliding_with_an_existing_column_is_discarded(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {"features": [{"name": "test_score", "op": "scale", "inputs": ["test_score"],
                       "params": {"factor": 2}},
                      {"name": "keeper", "op": "sum", "inputs": ["test_score", "attendance"]}]},
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["keeper"]
    assert any("collides" in w for w in warnings)


def test_mismatched_weights_are_discarded(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {
            "features": [
                {
                    "name": "combo",
                    "op": "weighted_sum",
                    "inputs": ["test_score", "attendance"],
                    "params": {"weights": [1.0]},
                },
                {"name": "keeper", "op": "sum", "inputs": ["test_score", "attendance"]},
            ]
        },
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["keeper"]
    assert any("one entry per input" in w for w in warnings)


def test_a_missing_required_param_is_discarded(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {"features": [{"name": "scaled", "op": "scale", "inputs": ["test_score"]},
                      {"name": "keeper", "op": "sum", "inputs": ["test_score", "attendance"]}]},
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["keeper"]
    assert any("required param" in w for w in warnings)


def test_wrong_arity_is_discarded(schema, context):
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {"features": [{"name": "d", "op": "difference",
                       "inputs": ["test_score", "attendance", "projects"]},
                      {"name": "keeper", "op": "sum", "inputs": ["test_score", "attendance"]}]},
        profile, schema,
    )
    assert [f.name for f in plan.features] == ["keeper"]
    assert warnings


def test_mostly_broken_features_raise_so_the_repair_pass_can_run(schema, context):
    """A few bad features get filtered; a mostly-wrong reply means a misread."""
    _, profile, _ = context
    with pytest.raises(PlanRejected):
        coerce_feature_plan(
            {
                "features": [
                    {"name": f"bad_{i}", "op": "sum", "inputs": ["nope_a", "nope_b"]}
                    for i in range(8)
                ]
                + [{"name": "ok", "op": "sum", "inputs": ["test_score", "attendance"]}]
            },
            profile, schema,
        )


def test_the_feature_count_is_capped(schema, context, monkeypatch):
    monkeypatch.setattr("placement_ai.planner.sanitize.MAX_SYNTHESIZED_FEATURES", 3)
    _, profile, _ = context
    plan, warnings = coerce_feature_plan(
        {
            "features": [
                {"name": f"f{i}", "op": "scale", "inputs": ["test_score"], "params": {"factor": i + 1}}
                for i in range(10)
            ]
        },
        profile, schema,
    )
    assert len(plan.features) == 3
    assert any("ignored the rest" in w for w in warnings)


def test_feature_arity_is_enforced_by_the_model_itself():
    with pytest.raises(ValueError, match="at least 2"):
        FeatureSpec(name="f", op=FeatureOp.sum, inputs=["only_one"])
    with pytest.raises(ValueError, match="at most 2"):
        FeatureSpec(name="f", op=FeatureOp.ratio, inputs=["a", "b", "c"])


# ── model stage ──────────────────────────────────────────────────────────────


def test_an_unknown_algorithm_is_dropped(schema):
    plan, warnings = coerce_model_plan(
        {
            "candidates": [
                {"algorithm": "neural_quantum_forest"},
                {"algorithm": "random_forest", "params": {"n_estimators": 50}},
            ]
        },
        schema,
    )
    assert [c.algorithm.value for c in plan.candidates] == ["random_forest"]
    assert any("neural_quantum_forest" in w for w in warnings)


def test_no_recognised_algorithm_raises(schema):
    with pytest.raises(PlanRejected):
        coerce_model_plan({"candidates": [{"algorithm": "made_up"}]}, schema)


def test_out_of_range_settings_are_clamped(schema):
    plan, _ = coerce_model_plan(
        {
            "candidates": [{"algorithm": "logistic_regression"}],
            "test_size": 0.95,
            "cv_folds": 99,
        },
        schema,
    )
    assert plan.test_size == 0.4
    assert plan.cv_folds == 10


def test_an_ambiguous_metric_is_swapped_for_multiclass(context):
    _, profile, info = context
    multiclass = SchemaPlan(
        target_column="outcome",
        task_type=TaskType.multiclass_classification,
        columns=[
            {"name": "outcome", "role": "target"},
            {"name": "test_score", "role": "numeric_feature"},
        ],
    )
    plan, warnings = coerce_model_plan(
        {"candidates": [{"algorithm": "random_forest"}], "primary_metric": "roc_auc"}, multiclass
    )
    assert plan.primary_metric.value == "balanced_accuracy"
    assert plan.threshold_strategy.value == "default"
    assert any("multiclass" in w for w in warnings)


def test_a_bad_class_weight_strategy_defaults_to_balanced(schema):
    plan, _ = coerce_model_plan(
        {"candidates": [{"algorithm": "logistic_regression"}], "class_weight_strategy": "vibes"},
        schema,
    )
    assert plan.class_weight_strategy == "balanced"


def test_a_negative_ensemble_weight_is_floored_at_zero(schema):
    plan, _ = coerce_model_plan(
        {"candidates": [{"algorithm": "random_forest", "ensemble_weight": -3}]}, schema
    )
    assert plan.candidates[0].ensemble_weight == 0.0
