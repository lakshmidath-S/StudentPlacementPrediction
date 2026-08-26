"""
Workspaces, saved models, and prediction history.

The property under test throughout is tenancy: one organisation's data must not
be reachable, listable, or overwritable from another's, and a saved model must
either load exactly as it was written or refuse to load at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from placement_ai.registry.history import HistoryEntry, PredictionHistory
from placement_ai.registry.model_store import (
    MANIFEST_FILE,
    PIPELINE_FILE,
    ModelStore,
    ModelStoreError,
)
from placement_ai.registry.workspace import WorkspaceError, WorkspaceStore

# ── workspaces ───────────────────────────────────────────────────────────────


def test_creating_a_workspace_returns_its_code_once(workspace_root):
    store = WorkspaceStore(workspace_root)
    created, code = store.create("St Xavier College", "Placement cell")
    assert created.name == "St Xavier College"
    assert created.id.startswith("st-xavier-college-")
    assert len(code) == 8
    # Only the hash is persisted, so the code cannot be read back off disk.
    assert code not in created.config_path.read_text(encoding="utf-8")


def test_the_right_code_opens_and_a_wrong_one_does_not(workspace_root):
    store = WorkspaceStore(workspace_root)
    created, code = store.create("Acme")
    assert store.open(created.id, code).id == created.id
    with pytest.raises(WorkspaceError, match="access code"):
        store.open(created.id, "WRONGXXX")


def test_an_access_code_is_case_insensitive(workspace_root):
    store = WorkspaceStore(workspace_root)
    created, code = store.create("Acme")
    assert store.open(created.id, code.lower()).id == created.id


def test_a_nameless_workspace_is_refused(workspace_root):
    with pytest.raises(WorkspaceError, match="needs a name"):
        WorkspaceStore(workspace_root).create("   ")


def test_two_workspaces_with_the_same_name_stay_separate(workspace_root):
    store = WorkspaceStore(workspace_root)
    first, _ = store.create("City College")
    second, _ = store.create("City College")
    assert first.id != second.id
    assert len(store.list()) == 2


def test_listing_scans_the_directory_rather_than_an_index(workspace_root):
    """No shared index file means concurrent creates cannot corrupt each other."""
    store = WorkspaceStore(workspace_root)
    store.create("One")
    store.create("Two")
    assert not (workspace_root / "registry.json").exists()
    assert {w.name for w in store.list()} == {"One", "Two"}


def test_resetting_the_code_invalidates_the_old_one(workspace_root):
    store = WorkspaceStore(workspace_root)
    created, original = store.create("Acme")
    replacement = store.reset_code(created.id)
    assert replacement != original
    with pytest.raises(WorkspaceError):
        store.open(created.id, original)
    assert store.open(created.id, replacement).id == created.id


def test_deleting_requires_the_code(workspace_root):
    store = WorkspaceStore(workspace_root)
    created, code = store.create("Acme")
    with pytest.raises(WorkspaceError):
        store.delete(created.id, "NOPE")
    store.delete(created.id, code)
    assert store.get(created.id) is None


def test_a_directory_without_a_config_is_not_a_workspace(workspace_root):
    (workspace_root / "stray-folder").mkdir()
    assert WorkspaceStore(workspace_root).list() == []


# ── model store ──────────────────────────────────────────────────────────────


def test_saving_writes_a_pipeline_and_a_manifest(workspace, trained):
    store = ModelStore(workspace)
    bundle = store.save(trained, dataset_name="students.csv", notes="first run")

    assert (bundle.path / PIPELINE_FILE).exists()
    assert (bundle.path / MANIFEST_FILE).exists()
    assert bundle.dataset_name == "students.csv"
    assert bundle.target_column == "outcome"
    assert bundle.headline_score > 0.7


def test_the_manifest_records_the_whole_plan(workspace, trained):
    bundle = ModelStore(workspace).save(trained)
    plan = bundle.manifest["plan"]
    assert {"schema_plan", "cleaning_plan", "feature_plan", "model_plan"} <= set(plan)
    assert [p["stage"] for p in plan["provenance"]] == [
        "schema", "cleaning", "features", "model"
    ]
    assert bundle.manifest["library_versions"]["python"]


def test_a_saved_model_predicts_identically_to_the_one_that_was_trained(workspace, trained):
    import numpy as np

    bundle = ModelStore(workspace).save(trained)
    sample = pd.DataFrame([{f["name"]: f["default"] for f in trained.input_schema}])
    assert np.allclose(
        trained.pipeline.predict_proba(sample), bundle.load_pipeline().predict_proba(sample)
    )


def test_a_tampered_model_file_refuses_to_load(workspace, trained):
    bundle = ModelStore(workspace).save(trained)
    with open(bundle.pipeline_path, "ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ModelStoreError, match="checksum"):
        bundle.load_pipeline()


def test_a_missing_model_file_reports_clearly(workspace, trained):
    bundle = ModelStore(workspace).save(trained)
    bundle.pipeline_path.unlink()
    with pytest.raises(ModelStoreError, match="missing"):
        bundle.load_pipeline()


def test_versions_are_listed_newest_first(workspace, trained):
    store = ModelStore(workspace)
    first = store.save(trained, dataset_name="a.csv")
    second = store.save(trained, dataset_name="b.csv")
    # Distinct directories even when saved within the same minute.
    assert first.version != second.version
    assert len(store.list()) == 2


def test_the_newest_save_becomes_active(workspace, trained):
    store = ModelStore(workspace)
    store.save(trained)
    second = store.save(trained)
    assert store.active().version == second.version


def test_an_older_version_can_be_reactivated(workspace, trained):
    store = ModelStore(workspace)
    first = store.save(trained)
    store.save(trained)
    store.set_active(first.version)
    assert ModelStore(WorkspaceStore(workspace.root.parent).get(workspace.id)).active().version == (
        first.version
    )


def test_activating_an_unknown_version_is_refused(workspace, trained):
    with pytest.raises(ModelStoreError, match="No saved model"):
        ModelStore(workspace).set_active("2020-01-01-0000-dead")


def test_deleting_the_active_version_falls_back_to_another(workspace, trained):
    store = ModelStore(workspace)
    first = store.save(trained)
    second = store.save(trained)
    store.delete(second.version)
    assert store.active().version == first.version


def test_deleting_the_last_version_leaves_no_active_model(workspace, trained):
    store = ModelStore(workspace)
    only = store.save(trained)
    store.delete(only.version)
    assert store.active() is None


def test_a_failed_save_leaves_no_half_written_version(workspace, trained, monkeypatch):
    """`list()` must never surface a directory that has no usable model in it."""
    store = ModelStore(workspace)
    monkeypatch.setattr(
        "placement_ai.registry.model_store.write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError):
        store.save(trained)
    assert store.list() == []


def test_one_workspace_cannot_see_another_workspaces_models(workspace_root, trained):
    store = WorkspaceStore(workspace_root)
    first, _ = store.create("First")
    second, _ = store.create("Second")
    ModelStore(first).save(trained)
    assert len(ModelStore(first).list()) == 1
    assert ModelStore(second).list() == []


# ── history ──────────────────────────────────────────────────────────────────


def entry(**overrides) -> HistoryEntry:
    payload = {
        "model_version": "v1",
        "target_column": "outcome",
        "inputs": {"score": 70, "department": "CS"},
        "predicted_label": "Placed",
        "probability": 0.82,
        "probabilities": {"Placed": 0.82, "NotPlaced": 0.18},
    }
    payload.update(overrides)
    return HistoryEntry(**payload)


def test_predictions_are_recorded_and_counted(workspace):
    history = PredictionHistory(workspace.history_path)
    history.log(entry())
    history.log_many([entry(), entry(predicted_label="NotPlaced", probability=0.2)])
    assert history.count() == 3
    assert history.count(model_version="v1") == 3
    assert history.count(model_version="other") == 0


def test_the_summary_groups_by_version_and_label(workspace):
    history = PredictionHistory(workspace.history_path)
    history.log_many([entry(), entry(predicted_label="NotPlaced"), entry(model_version="v2")])
    summary = history.summary()
    assert summary["total"] == 3
    assert summary["by_version"] == {"v1": 2, "v2": 1}
    assert summary["by_label"]["Placed"] == 2
    assert summary["last"]


def test_inputs_expand_into_columns_on_read(workspace):
    history = PredictionHistory(workspace.history_path)
    history.log(entry())
    frame = history.expanded()
    assert "score" in frame.columns
    assert "department" in frame.columns
    assert frame["score"].iloc[0] == 70


def test_recording_an_outcome_makes_it_available_as_training_data(workspace):
    history = PredictionHistory(workspace.history_path)
    history.log(entry())
    row_id = int(history.recent()["id"].iloc[0])
    assert history.record_actual(row_id, "NotPlaced")

    labelled = history.labelled()
    assert len(labelled) == 1
    # Named after the target, so it concatenates onto the original training file.
    assert labelled["outcome"].iloc[0] == "NotPlaced"
    assert "score" in labelled.columns
    assert history.summary()["labelled"] == 1


def test_a_batch_shares_one_identifier(workspace):
    history = PredictionHistory(workspace.history_path)
    history.log_many([entry(source="batch", batch_id="b1") for _ in range(5)])
    frame = history.recent()
    assert set(frame["batch_id"]) == {"b1"}
    assert set(frame["source"]) == {"batch"}


def test_history_is_per_workspace(workspace_root):
    store = WorkspaceStore(workspace_root)
    first, _ = store.create("First")
    second, _ = store.create("Second")
    PredictionHistory(first.history_path).log(entry())
    assert PredictionHistory(first.history_path).count() == 1
    assert PredictionHistory(second.history_path).count() == 0


def test_a_logging_failure_reports_instead_of_raising(workspace, monkeypatch):
    """A prediction must never be lost because the log could not be written."""
    history = PredictionHistory(workspace.history_path)
    import sqlite3

    monkeypatch.setattr(
        history, "_connect", lambda: (_ for _ in ()).throw(sqlite3.Error("locked"))
    )
    assert history.log(entry()) is None
    assert history.last_error and "history database" in history.last_error


def test_reading_an_empty_history_returns_an_empty_frame(workspace):
    history = PredictionHistory(workspace.history_path)
    assert history.recent().empty
    assert history.expanded().empty
    assert history.labelled().empty
    assert history.summary()["total"] == 0
