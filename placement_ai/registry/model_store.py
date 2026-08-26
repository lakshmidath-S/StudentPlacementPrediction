"""
placement_ai/registry/model_store.py
------------------------------------
Versioned model bundles on disk.

    models/2026-08-26-1431-7c1e/
      pipeline.joblib     the whole fitted chain, clean -> features -> encode -> model
      manifest.json       the model card: plan, scores, importances, provenance

One joblib per version, holding the entire pipeline rather than a bare
estimator. That is what makes a saved model self-sufficient: the cleaning rules,
the fitted feature constants, the encoder categories and the classifier all
travel together, so nothing has to be reconstructed — or guessed at — when the
model is loaded back months later.

The manifest carries a SHA-256 of the joblib, checked on load. A corrupted or
half-written file then fails loudly at load time instead of producing quietly
wrong predictions.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from placement_ai.registry.workspace import Workspace
from placement_ai.utils import read_json, utc_now_iso, write_json

PIPELINE_FILE = "pipeline.joblib"
MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_VERSION = 3


class ModelStoreError(RuntimeError):
    """A bundle is missing, unreadable, or does not match its manifest."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions() -> dict[str, str]:
    """Record what built this bundle.

    A pipeline pickled by one scikit-learn and unpickled by another is the
    single most common way a saved model breaks, and the error it throws rarely
    names the version gap. Writing it down makes the diagnosis a five-second
    read.
    """
    versions = {"python": sys.version.split()[0]}
    for module in ("sklearn", "pandas", "numpy", "xgboost"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            continue
    return versions


@dataclass
class ModelBundle:
    """One saved model version, loaded lazily."""

    version: str
    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def pipeline_path(self) -> Path:
        return self.path / PIPELINE_FILE

    @property
    def created_at(self) -> str:
        return str(self.manifest.get("created_at_utc", ""))

    @property
    def label(self) -> str:
        champion = self.manifest.get("champion", {})
        return str(champion.get("label", "model"))

    @property
    def target_column(self) -> str:
        return str(self.manifest.get("target_column", ""))

    @property
    def primary_metric(self) -> str:
        return str(self.manifest.get("primary_metric", "roc_auc"))

    @property
    def headline_score(self) -> float | None:
        value = (self.manifest.get("metrics") or {}).get(self.primary_metric)
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def dataset_name(self) -> str:
        return str(self.manifest.get("dataset_name", "uploaded data"))

    @property
    def input_schema(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("input_schema") or [])

    @property
    def class_labels(self) -> list[str]:
        return [str(c) for c in (self.manifest.get("class_labels") or [])]

    @property
    def positive_class(self) -> str | None:
        value = self.manifest.get("positive_class")
        return str(value) if value else None

    @property
    def drift_baseline(self) -> dict[str, Any]:
        return dict(self.manifest.get("drift_baseline") or {})

    @property
    def threshold(self) -> float:
        try:
            return float(self.manifest.get("threshold", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def load_pipeline(self, verify: bool = True) -> Any:
        """Deserialise the fitted pipeline, checking its checksum first."""
        if not self.pipeline_path.exists():
            raise ModelStoreError(f"The model file is missing: {self.pipeline_path}")

        expected = (self.manifest.get("artifacts", {}).get(PIPELINE_FILE, {}) or {}).get("sha256")
        if verify and expected:
            actual = sha256_of(self.pipeline_path)
            if actual != expected:
                raise ModelStoreError(
                    f"Model {self.version} does not match its manifest checksum. "
                    "The file has been modified or was not written completely; "
                    "retrain to replace it."
                )
        try:
            return joblib.load(self.pipeline_path)
        except Exception as exc:
            built_with = self.manifest.get("library_versions", {})
            raise ModelStoreError(
                f"Could not load model {self.version}: {type(exc).__name__}: {exc}\n"
                f"It was built with {built_with}. A different scikit-learn version "
                "is the usual cause; retraining in this environment will fix it."
            ) from exc


class ModelStore:
    """The set of trained versions belonging to one workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.models_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # ── reading ──────────────────────────────────────────────────────────
    def _load(self, directory: Path) -> ModelBundle | None:
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            return None
        try:
            manifest = read_json(manifest_path)
        except (ValueError, OSError):
            return None
        return ModelBundle(version=directory.name, path=directory, manifest=manifest)

    def list(self) -> list[ModelBundle]:
        """Every saved version, newest first."""
        if not self.root.exists():
            return []
        bundles = [
            bundle
            for directory in self.root.iterdir()
            if directory.is_dir() and (bundle := self._load(directory)) is not None
        ]
        return sorted(bundles, key=lambda b: b.created_at, reverse=True)

    def get(self, version: str) -> ModelBundle | None:
        return self._load(self.root / version)

    def active(self) -> ModelBundle | None:
        """The version predictions use, defaulting to the newest.

        Falling back rather than returning None means a workspace whose active
        pointer was left dangling by a deletion still serves predictions from
        its most recent model.
        """
        if self.workspace.active_model:
            bundle = self.get(self.workspace.active_model)
            if bundle is not None:
                return bundle
        existing = self.list()
        return existing[0] if existing else None

    def set_active(self, version: str) -> ModelBundle:
        bundle = self.get(version)
        if bundle is None:
            raise ModelStoreError(f"No saved model with version {version!r}.")
        self.workspace.set_active_model(version)
        return bundle

    # ── writing ──────────────────────────────────────────────────────────
    def new_version(self) -> str:
        """A sortable, human-readable version string.

        Timestamp first so directory order is chronological order; a short
        random tail so two runs finishing in the same minute cannot collide.
        """
        import secrets

        stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        return f"{stamp}-{secrets.token_hex(2)}"

    def save(
        self,
        outcome: Any,
        dataset_name: str = "uploaded data",
        notes: str = "",
        make_active: bool = True,
    ) -> ModelBundle:
        """Persist a TrainingOutcome as a new version and return its bundle."""
        version = self.new_version()
        directory = self.root / version
        directory.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(outcome.pipeline, directory / PIPELINE_FILE, compress=3)

            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "version": version,
                "workspace_id": self.workspace.id,
                "workspace_name": self.workspace.name,
                "created_at_utc": utc_now_iso(),
                "dataset_name": dataset_name,
                "notes": notes,
                "target_column": outcome.plan.schema_plan.target_column,
                "task_type": outcome.plan.schema_plan.task_type.value,
                "class_labels": outcome.class_labels,
                "positive_class": outcome.positive_class,
                "threshold": outcome.threshold,
                "primary_metric": outcome.plan.model_plan.primary_metric.value,
                "champion": {"algorithm": outcome.champion, "label": outcome.champion_label},
                "metrics": outcome.metrics,
                "candidates": [
                    {
                        "algorithm": c.algorithm,
                        "label": c.label,
                        "score": c.score,
                        "cv_mean": c.cv_mean,
                        "cv_std": c.cv_std,
                        "train_seconds": round(c.train_seconds, 2),
                        "ensemble_weight": c.ensemble_weight,
                        "rationale": c.rationale,
                        "metrics": c.metrics,
                    }
                    for c in outcome.candidates
                ],
                "importance": outcome.importance,
                "input_schema": outcome.input_schema,
                "engineered_features": outcome.engineered_features,
                "encoded_feature_count": outcome.encoded_feature_count,
                "dataset_summary": outcome.dataset_summary,
                "drift_baseline": outcome.drift_baseline,
                "narrative": outcome.narrative,
                "warnings": outcome.warnings,
                "duration_seconds": round(outcome.duration_seconds, 2),
                # The full plan, stored verbatim. This is what makes a run
                # auditable long after the fact.
                "plan": outcome.plan.model_dump(mode="json"),
                "library_versions": _library_versions(),
                "artifacts": {
                    PIPELINE_FILE: {"sha256": sha256_of(directory / PIPELINE_FILE)}
                },
            }
            write_json(directory / MANIFEST_FILE, manifest)
        except Exception:
            # Never leave a half-written version behind for `list()` to find.
            shutil.rmtree(directory, ignore_errors=True)
            raise

        bundle = ModelBundle(version=version, path=directory, manifest=manifest)
        if make_active:
            self.workspace.set_active_model(version)
        return bundle

    def delete(self, version: str) -> None:
        bundle = self.get(version)
        if bundle is None:
            return
        shutil.rmtree(bundle.path, ignore_errors=True)
        if self.workspace.active_model == version:
            remaining = self.list()
            self.workspace.set_active_model(remaining[0].version if remaining else None)
