"""
Profiling: the deterministic reading of a dataset that everything downstream
depends on. When the LLM is unavailable this is the *only* evidence the planner
has, so an error here is invisible rather than loud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from placement_ai.profiling import (
    canonicalize_columns,
    classify_target_labels,
    find_target_candidates,
    pick_positive_class,
    profile_dataframe,
)

# ── header canonicalisation ──────────────────────────────────────────────────


def test_headers_become_snake_case():
    frame = pd.DataFrame({"Test Score": [1], "Attendance %": [2], "Workshops/Certs": [3]})
    renamed, mapping = canonicalize_columns(frame)
    assert list(renamed.columns) == ["test_score", "attendance", "workshops_certs"]
    assert mapping["Attendance %"] == "attendance"


def test_headers_colliding_on_the_same_name_are_kept_distinct():
    frame = pd.DataFrame({"CGPA": [1], "cgpa": [2], "c.g.p.a": [3]})
    renamed, _ = canonicalize_columns(frame)
    assert len(set(renamed.columns)) == 3
    assert renamed.columns[0] == "cgpa"


def test_a_leading_digit_gets_a_prefix():
    renamed, _ = canonicalize_columns(pd.DataFrame({"2024 score": [1]}))
    assert renamed.columns[0] == "col_2024_score"


def test_canonicalisation_is_idempotent():
    once, _ = canonicalize_columns(pd.DataFrame({"Test Score": [1]}))
    twice, _ = canonicalize_columns(once)
    assert list(once.columns) == list(twice.columns)


# ── column kinds ─────────────────────────────────────────────────────────────


def test_column_kinds_are_read_from_shape(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    kinds = {c.name: c.kind for c in profile_dataframe(frame).columns}

    assert kinds["test_score"] == "numeric"
    assert kinds["department"] == "categorical"
    assert kinds["campus"] == "constant"
    assert kinds["notes"] == "text"
    # A two-level *string* column is categorical, not boolean: it still needs
    # one-hot encoding. Only a numeric or bool column with two levels is a flag.
    assert kinds["training"] == "categorical"
    assert profile_dataframe(frame).column("training").n_unique == 2


def test_identifier_columns_are_flagged(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    profile = profile_dataframe(frame)
    assert profile.column("student_id").looks_like_identifier
    assert not profile.column("test_score").looks_like_identifier


def test_numbers_stored_as_text_are_reported_not_fixed(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    stipend = profile_dataframe(frame).column("stipend")
    assert stipend.is_numeric_like_text
    # Reported only — the raw column is left as text for the cleaning plan to
    # decide about. (pandas 3 infers a str dtype here, pandas 2 used object.)
    assert not pd.api.types.is_numeric_dtype(frame["stipend"])


def test_integrality_is_measured_over_all_values_not_the_maximum():
    """A float column whose max lands on 10.0 is not a count column."""
    frame = pd.DataFrame({"cgpa": [6.79, 8.2, 10.0] * 10, "backlogs": [0, 3, 8] * 10})
    profile = profile_dataframe(frame)
    assert profile.column("cgpa").is_integral is False
    assert profile.column("backlogs").is_integral is True


def test_missingness_is_measured(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    attendance = profile_dataframe(frame).column("attendance")
    assert attendance.missing > 0
    assert 5 < attendance.missing_pct < 15


def test_profile_payload_stays_compact(messy_frame):
    """The payload goes into a prompt, so it must not carry whole columns."""
    frame, _ = canonicalize_columns(messy_frame)
    payload = profile_dataframe(frame).to_payload()
    assert payload["n_rows"] == len(frame)
    for column in payload["columns"]:
        assert len(column["examples"]) <= 6
        assert len(column.get("top_values", [])) <= 8


# ── target detection ─────────────────────────────────────────────────────────


def test_target_candidates_rank_the_outcome_column_first(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    assert profile_dataframe(frame).target_candidates[0] == "outcome"


def test_a_name_match_beats_position():
    frame = pd.DataFrame(
        {"placement_status": ["Placed", "NotPlaced"] * 20, "trailing": list(range(40))}
    )
    assert find_target_candidates(profile_dataframe(frame).columns)[0] == "placement_status"


def test_identifiers_are_never_target_candidates():
    frame = pd.DataFrame({"student_id": range(50), "result": ["a", "b"] * 25})
    assert "student_id" not in profile_dataframe(frame).target_candidates


# ── positive class ───────────────────────────────────────────────────────────


def test_positive_class_prefers_the_affirmative_token():
    assert pick_positive_class(["NotPlaced", "Placed"]) == "Placed"
    assert pick_positive_class(["No", "Yes"]) == "Yes"
    assert pick_positive_class(["Rejected", "Selected"]) == "Selected"


def test_positive_class_falls_back_to_the_non_negative_label():
    """Only one side is a recognised token; the other is whatever remains."""
    assert pick_positive_class(["Failed", "Distinction"]) == "Distinction"


def test_positive_class_of_two_numbers_is_the_larger():
    assert pick_positive_class([0, 1]) == 1


def test_binary_target_is_classified(messy_frame):
    frame, _ = canonicalize_columns(messy_frame)
    info = classify_target_labels(frame["outcome"])
    assert info["task_type"] == "binary_classification"
    assert info["positive_class"] == "Placed"
    assert info["n_classes"] == 2


def test_a_continuous_column_is_reported_as_regression():
    salary = pd.Series(np.linspace(20000, 90000, 500))
    assert classify_target_labels(salary)["task_type"] == "regression"


def test_a_handful_of_classes_is_multiclass():
    grades = pd.Series(["A", "B", "C", "D"] * 30)
    info = classify_target_labels(grades)
    assert info["task_type"] == "multiclass_classification"
    assert info["positive_class"] is None


def test_duplicate_rows_are_counted():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert profile_dataframe(frame).duplicate_rows == 1
