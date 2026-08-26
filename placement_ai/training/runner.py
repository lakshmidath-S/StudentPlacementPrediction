"""
placement_ai/training/runner.py
-------------------------------
The training job: raw upload in, evaluated model out.

Eleven stages, each reported through a progress callback so the UI can show
what is happening during a run that may take minutes:

    ingest -> profile -> understand -> clean -> engineer -> select
    -> train -> evaluate -> choose -> explain -> package

Stages 3 to 6 are the LLM's; the rest are deterministic. That boundary is the
design: a language model decides *what* should happen to the data, and ordinary
Python decides whether it worked. Nothing the model emits is trusted enough to
skip validation, and nothing it fails at is fatal.

Labels are encoded to integers for the whole run. XGBoost refuses string
targets, and encoding once here beats a per-algorithm branch; the original
labels travel in the bundle and every user-facing number maps back to them.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from placement_ai.config import (
    MAX_TRAINING_ROWS,
    MIN_ROWS_PER_CLASS,
    MIN_TRAINING_ROWS,
    RANDOM_SEED,
)
from placement_ai.inference.drift import training_baseline
from placement_ai.llm.base import LLMProvider
from placement_ai.pipeline.builder import (
    assemble_final_pipeline,
    build_estimator,
    build_preprocessor,
    compute_sample_weights,
    display_name,
    encoded_feature_names,
)
from placement_ai.pipeline.ensemble import WeightedSoftVoter, _aligned_proba
from placement_ai.pipeline.transformers import CleaningTransformer, prune_feature_plan
from placement_ai.planner.narrator import write_training_report
from placement_ai.planner.planner import Planner
from placement_ai.plans import TaskType, ThresholdStrategy, TrainingPlan
from placement_ai.profiling import (
    canonicalize_columns,
    classify_target_labels,
    profile_dataframe,
)
from placement_ai.training.evaluation import (
    choose_threshold,
    class_balance,
    evaluate_classifier,
    permutation_importance_raw,
    primary_score,
)

STAGES = (
    "ingest",
    "profile",
    "understand",
    "clean",
    "engineer",
    "select",
    "train",
    "evaluate",
    "choose",
    "explain",
    "package",
)

# The planner names its own stages; this maps them onto the run's stage list so
# the progress bar advances once per stage instead of reporting four "understand".
PLANNER_STAGE_MAP = {
    "schema": "understand",
    "cleaning": "clean",
    "features": "engineer",
    "model": "select",
}

STAGE_TITLES = {
    "ingest": "Reading the file",
    "profile": "Profiling the columns",
    "understand": "Understanding the schema",
    "clean": "Planning the clean-up",
    "engineer": "Designing features",
    "select": "Choosing models",
    "train": "Training",
    "evaluate": "Scoring on held-out data",
    "choose": "Selecting the winner",
    "explain": "Writing the summary",
    "package": "Packaging the model",
}

ProgressFn = Callable[["TrainingProgress"], None]


class TrainingError(RuntimeError):
    """The upload cannot be trained on, with a reason a non-expert can act on."""


@dataclass
class TrainingProgress:
    stage: str
    status: str  # start | ok | fallback | info
    detail: str = ""
    index: int = 0
    total: int = len(STAGES)
    elapsed: float = 0.0

    @property
    def fraction(self) -> float:
        return min(max(self.index / max(self.total, 1), 0.0), 1.0)


@dataclass
class CandidateResult:
    algorithm: str
    label: str
    metrics: dict[str, Any]
    cv_mean: float
    cv_std: float
    train_seconds: float
    ensemble_weight: float
    rationale: str = ""
    ignored_params: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class TrainingOutcome:
    plan: TrainingPlan
    pipeline: Any
    champion: str
    champion_label: str
    candidates: list[CandidateResult]
    metrics: dict[str, Any]
    threshold: float
    class_labels: list[str]
    positive_class: str | None
    importance: list[dict[str, Any]]
    input_schema: list[dict[str, Any]]
    dataset_summary: dict[str, Any]
    engineered_features: list[dict[str, Any]]
    encoded_feature_count: int
    drift_baseline: dict[str, Any]
    narrative: dict[str, Any]
    warnings: list[str]
    duration_seconds: float


class TrainingRunner:
    """Runs one end-to-end training job."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.provider = provider
        self.planner = Planner(provider)
        self._progress = progress
        self._started = 0.0
        self._stage_index = 0

    # ── progress plumbing ────────────────────────────────────────────────
    def _emit(self, stage: str, status: str, detail: str = "") -> None:
        if status in {"ok", "fallback"}:
            # Monotonic: stages do not complete in strict list order. The planner
            # finishes "select" before the runner finishes its own "engineer"
            # pass over the proposed features, and a progress bar that jumps
            # backwards reads as a failure.
            self._stage_index = max(self._stage_index, STAGES.index(stage) + 1)
        if self._progress:
            self._progress(
                TrainingProgress(
                    stage=stage,
                    status=status,
                    detail=detail,
                    index=self._stage_index,
                    elapsed=time.perf_counter() - self._started,
                )
            )

    def _planner_progress(self, stage: str, status: str, detail: str) -> None:
        self._emit(PLANNER_STAGE_MAP.get(stage, "understand"), status, detail)

    # ── the run ──────────────────────────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        objective: str = "",
    ) -> TrainingOutcome:
        self._started = time.perf_counter()
        warnings: list[str] = []

        # ── 1. ingest ────────────────────────────────────────────────────
        self._emit("ingest", "start", "Reading the upload")
        frame, rename_map = canonicalize_columns(df)
        renamed = {k: v for k, v in rename_map.items() if k != v}
        if renamed:
            warnings.append(f"Renamed {len(renamed)} column header(s) to a consistent format.")

        if len(frame) > MAX_TRAINING_ROWS:
            frame = frame.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_SEED)
            warnings.append(
                f"Sampled {MAX_TRAINING_ROWS:,} rows at random from a larger upload "
                "to keep training within a reasonable time."
            )
        if len(frame) < MIN_TRAINING_ROWS:
            raise TrainingError(
                f"Only {len(frame)} rows were supplied. At least {MIN_TRAINING_ROWS} are "
                "needed before a model can be evaluated honestly."
            )
        self._emit("ingest", "ok", f"{len(frame):,} rows x {frame.shape[1]} columns")

        # ── 2. profile ───────────────────────────────────────────────────
        self._emit("profile", "start", "Measuring every column")
        profile = profile_dataframe(frame)
        target_column = self._resolve_target(profile, target_column, frame)

        # Judge the target on its labelled rows only — a mostly-blank column
        # would otherwise look like it has a single usable class. The guards run
        # before planning so an untrainable upload fails in a second with an
        # accurate message, rather than after four LLM calls with a vague one.
        labelled = frame[frame[target_column].notna()]
        target_info = classify_target_labels(labelled[target_column])
        self._emit(
            "profile",
            "ok",
            f"Target {target_column!r} with {target_info['n_classes']} distinct outcomes",
        )
        self._guard_target(labelled[target_column], target_column, target_info)

        # ── 3-6. planning ────────────────────────────────────────────────
        levels, imbalance_pp = class_balance(frame[target_column])
        planning = self.planner.build_plan(
            profile=profile,
            target_column=target_column,
            target_info=target_info,
            imbalance_pp=imbalance_pp,
            objective=objective,
            progress=self._planner_progress,
        )
        plan = planning.plan
        warnings.extend(planning.warnings)
        schema = plan.schema_plan

        # ── row-level cleaning ───────────────────────────────────────────
        # These change the row count, so they happen here rather than inside a
        # transformer, where X and y would fall out of step.
        rows_before = len(frame)
        if plan.cleaning_plan.drop_rows_missing_target:
            frame = frame[frame[target_column].notna()]
        if plan.cleaning_plan.drop_duplicate_rows:
            # Unhashable cell contents make this raise; skipping dedupe is a
            # far smaller loss than failing the run over it.
            with contextlib.suppress(TypeError):
                frame = frame.drop_duplicates()
        dropped_rows = rows_before - len(frame)
        if dropped_rows:
            warnings.append(f"Removed {dropped_rows:,} duplicate or unlabelled row(s).")

        # Re-checked because de-duplication can push a thin class below the
        # minimum even though the raw upload cleared it.
        self._guard_target(frame[target_column], target_column, target_info)

        # ── label encoding ───────────────────────────────────────────────
        label_encoder = LabelEncoder().fit(frame[target_column].astype(str))
        class_labels = [str(c) for c in label_encoder.classes_]
        y = pd.Series(
            label_encoder.transform(frame[target_column].astype(str)), index=frame.index
        )
        classes = np.arange(len(class_labels))

        positive_class = schema.positive_class
        positive_index: int | None = None
        if schema.task_type is TaskType.binary_classification:
            if positive_class is None or positive_class not in class_labels:
                positive_class = class_labels[-1]
            positive_index = class_labels.index(positive_class)
        else:
            positive_class = None

        X = frame[schema.feature_columns]

        # ── split ────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=plan.model_plan.test_size,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        # ── feature pruning + preprocessing ──────────────────────────────
        self._emit("engineer", "start", "Checking the proposed features against the data")
        probe_cleaner = CleaningTransformer(
            plan=plan.cleaning_plan, keep_columns=schema.feature_columns
        )
        cleaned_train = probe_cleaner.fit_transform(X_train)
        pruned, prune_warnings = prune_feature_plan(plan.feature_plan, cleaned_train)
        warnings.extend(prune_warnings)
        plan.feature_plan = pruned

        preprocessor, numeric_columns, categorical_columns = build_preprocessor(
            schema, plan.cleaning_plan, pruned
        )
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        encoded_names = encoded_feature_names(preprocessor)
        self._emit(
            "engineer",
            "ok",
            f"{len(pruned.features)} derived feature(s); {X_train_t.shape[1]} model inputs",
        )

        # ── 7. train candidates ──────────────────────────────────────────
        sample_weights = compute_sample_weights(
            y_train,
            plan.model_plan.class_weight_strategy,
            plan.model_plan.custom_class_weights,
        )
        folds = self._fold_count(plan.model_plan.cv_folds, y_train)

        self._emit(
            "train",
            "start",
            f"Training {len(plan.model_plan.candidates)} model(s) with {folds}-fold validation",
        )
        fitted: list[tuple[str, Any, float]] = []
        results: list[CandidateResult] = []
        oof_by_algorithm: dict[str, np.ndarray] = {}

        for candidate in plan.model_plan.candidates:
            name = candidate.algorithm.value
            self._emit("train", "info", f"Fitting {display_name(name)}")
            try:
                estimator, ignored = build_estimator(candidate, len(class_labels))
            except Exception as exc:
                warnings.append(f"Skipped {display_name(name)}: {type(exc).__name__}: {exc}")
                continue
            if ignored:
                warnings.append(
                    f"{display_name(name)}: ignored unsupported setting(s) {ignored}."
                )

            started = time.perf_counter()
            try:
                oof, fold_scores = self._cross_validate(
                    estimator, X_train_t, y_train.to_numpy(), sample_weights, classes, folds,
                    plan, positive_index,
                )
                fitted_estimator = clone(estimator)
                _fit(fitted_estimator, X_train_t, y_train.to_numpy(), sample_weights)
            except Exception as exc:
                warnings.append(f"{display_name(name)} failed to train: {type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - started

            oof_by_algorithm[name] = oof
            fitted.append((name, fitted_estimator, candidate.ensemble_weight))
            results.append(
                CandidateResult(
                    algorithm=name,
                    label=display_name(name),
                    metrics={},
                    cv_mean=float(np.mean(fold_scores)) if fold_scores else 0.0,
                    cv_std=float(np.std(fold_scores)) if fold_scores else 0.0,
                    train_seconds=elapsed,
                    ensemble_weight=candidate.ensemble_weight,
                    rationale=candidate.rationale,
                    ignored_params=ignored,
                )
            )

        if not fitted:
            raise TrainingError(
                "No model finished training on this data. The warnings above list "
                "what each attempt failed on."
            )
        self._emit("train", "ok", f"{len(fitted)} model(s) trained")

        # ── 8-9. evaluate & choose ───────────────────────────────────────
        self._emit("evaluate", "start", "Scoring on the held-out split")
        metric = plan.model_plan.primary_metric
        threshold_strategy = (
            plan.model_plan.threshold_strategy
            if positive_index is not None
            else ThresholdStrategy.default
        )

        contenders: list[tuple[str, Any, dict[str, Any], float, float]] = []
        for name, estimator, _weight in fitted:
            threshold = self._threshold_from_oof(
                oof_by_algorithm[name], y_train.to_numpy(), positive_index, threshold_strategy
            )
            probabilities = _aligned_proba(estimator, X_test_t, classes)
            report = evaluate_classifier(
                y_test.to_numpy(), probabilities, classes,
                positive_index if positive_index is not None else None, threshold,
            )
            report = _relabel(report, class_labels, positive_class)
            contenders.append((name, estimator, report, threshold, primary_score(report, metric)))

        if plan.model_plan.build_ensemble and len(fitted) > 1:
            voter = WeightedSoftVoter(
                estimators=[(name, est) for name, est, _ in fitted],
                weights=[weight for _, _, weight in fitted],
                classes=classes,
            ).fit(None, y_train.to_numpy())
            oof_stack = sum(
                oof_by_algorithm[name] * weight for name, _, weight in fitted
            ) / max(sum(weight for _, _, weight in fitted), 1e-9)
            threshold = self._threshold_from_oof(
                oof_stack, y_train.to_numpy(), positive_index, threshold_strategy
            )
            probabilities = voter.predict_proba(X_test_t)
            report = evaluate_classifier(
                y_test.to_numpy(), probabilities, classes,
                positive_index if positive_index is not None else None, threshold,
            )
            report = _relabel(report, class_labels, positive_class)
            contenders.append(("ensemble", voter, report, threshold, primary_score(report, metric)))
            results.append(
                CandidateResult(
                    algorithm="ensemble",
                    label=display_name("ensemble"),
                    metrics={},
                    # The ensemble reuses estimators already cross-validated
                    # individually, so it has no CV score of its own. NaN keeps
                    # that visible instead of implying a genuine 0.000.
                    cv_mean=float("nan"),
                    cv_std=float("nan"),
                    train_seconds=0.0,
                    ensemble_weight=sum(weight for _, _, weight in fitted),
                    rationale=(
                        "Weighted average of the trained models, using the weights the "
                        "planner assigned. Kept only if it beats every single model."
                    ),
                )
            )

        for result in results:
            match = next((c for c in contenders if c[0] == result.algorithm), None)
            if match:
                result.metrics = match[2]
                result.score = match[4]
        results = [r for r in results if r.metrics]

        champion_name, champion_estimator, champion_report, champion_threshold, _ = max(
            contenders, key=lambda row: row[4]
        )
        self._emit(
            "evaluate",
            "ok",
            f"Best: {display_name(champion_name)} "
            f"({metric.value} {primary_score(champion_report, metric):.4f})",
        )

        self._emit("choose", "start", "Measuring which columns carry the signal")
        final_pipeline = assemble_final_pipeline(preprocessor, champion_estimator)
        importance = permutation_importance_raw(
            final_pipeline,
            X_test,
            y_test,
            classes,
            positive_index if positive_index is not None else None,
            metric,
            champion_threshold,
        )
        importance = [
            {**row, "label": _label_for(schema, row["column"])} for row in importance
        ]
        self._emit("choose", "ok", f"{display_name(champion_name)} selected")

        # ── 10. narrate ──────────────────────────────────────────────────
        self._emit("explain", "start", "Writing the plain-language summary")
        dataset_summary = {
            "rows_supplied": int(rows_before),
            "rows_used": int(len(frame)),
            "rows_dropped": int(dropped_rows),
            "columns_supplied": int(profile.n_columns),
            "target_column": target_column,
            "class_balance": levels,
            "imbalance_pp": round(float(imbalance_pp), 2),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "renamed_headers": renamed,
        }
        narrative = write_training_report(
            self.provider,
            plan=plan,
            dataset_summary=dataset_summary,
            champion=display_name(champion_name),
            metrics=champion_report,
            importance=importance[:8],
            metric=metric.value,
        )
        self._emit("explain", "ok", narrative.get("headline", "Summary ready"))

        # ── 11. package ──────────────────────────────────────────────────
        self._emit("package", "start", "Preparing the model for saving")
        input_schema = build_input_schema(X, schema, probe_cleaner)
        outcome = TrainingOutcome(
            plan=plan,
            pipeline=final_pipeline,
            champion=champion_name,
            champion_label=display_name(champion_name),
            candidates=sorted(results, key=lambda r: r.score, reverse=True),
            metrics=champion_report,
            threshold=float(champion_threshold),
            class_labels=class_labels,
            positive_class=positive_class,
            importance=importance,
            input_schema=input_schema,
            dataset_summary=dataset_summary,
            engineered_features=[
                {
                    "name": spec.name,
                    "op": spec.op.value,
                    "inputs": spec.inputs,
                    "rationale": spec.rationale,
                }
                for spec in pruned.features
            ],
            encoded_feature_count=len(encoded_names) or int(X_train_t.shape[1]),
            # Captured now, from the data the model actually learned on. A
            # baseline recomputed later from recent traffic would compare the
            # present against itself and never report drift.
            drift_baseline=training_baseline(X_train, input_schema),
            narrative=narrative,
            warnings=warnings,
            duration_seconds=time.perf_counter() - self._started,
        )
        self._emit("package", "ok", "Ready")
        return outcome

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_target(profile: Any, requested: str | None, frame: pd.DataFrame) -> str:
        if requested:
            if requested in frame.columns:
                return requested
            raise TrainingError(f"Column {requested!r} is not in the uploaded file.")
        if profile.target_candidates:
            return profile.target_candidates[0]
        raise TrainingError(
            "No column in this file looks like an outcome to predict. Choose one "
            "explicitly — it should hold a small set of repeated answers, such as "
            "Placed / Not placed."
        )

    @staticmethod
    def _guard_target(target: pd.Series, name: str, target_info: dict[str, Any]) -> None:
        """Refuse an untrainable target, in the order that gives the best message.

        A constant column and a continuous one both have "not two usable
        classes" as their problem, but they need completely different advice, so
        the constant case is caught before the regression check rather than
        being described as 1 distinct continuous value.
        """
        counts = target.astype(str).value_counts()
        if len(counts) < 2:
            raise TrainingError(
                f"Every row has the same value in {name!r}. A model needs at least two "
                "different outcomes to learn the difference between them."
            )

        if target_info.get("task_type") == "regression":
            raise TrainingError(
                f"Column {name!r} holds {target_info['n_classes']:,} distinct values, "
                "which is a quantity to estimate rather than an outcome to classify. "
                "This build predicts categories — pick a column with a small set of "
                "repeated answers, such as Placed / Not placed."
            )

        thin = counts[counts < MIN_ROWS_PER_CLASS]
        if len(thin):
            listed = ", ".join(f"{value} ({count})" for value, count in thin.items())
            raise TrainingError(
                f"These outcomes appear too few times to learn or test on: {listed}. "
                f"At least {MIN_ROWS_PER_CLASS} rows of each are needed."
            )

    @staticmethod
    def _fold_count(requested: int, y: pd.Series) -> int:
        smallest = int(pd.Series(y).value_counts().min())
        return max(2, min(requested, smallest, 10))

    def _cross_validate(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None,
        classes: np.ndarray,
        folds: int,
        plan: TrainingPlan,
        positive_index: int | None,
    ) -> tuple[np.ndarray, list[float]]:
        """Hand-rolled stratified CV producing out-of-fold probabilities.

        Written out rather than using cross_val_predict because the folds have
        to carry ``sample_weight``, which otherwise requires turning on
        scikit-learn metadata routing globally — a process-wide switch that a
        library module has no business flipping.

        The out-of-fold matrix does double duty: it scores the candidate, and it
        is where the decision threshold is chosen. Picking the threshold on the
        test split instead would quietly tune against the data used to report
        the final numbers.
        """
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
        oof = np.zeros((len(y), len(classes)), dtype=float)
        scores: list[float] = []

        for train_index, valid_index in splitter.split(X, y):
            fold_estimator = clone(estimator)
            _fit(
                fold_estimator,
                X[train_index],
                y[train_index],
                None if weights is None else weights[train_index],
            )
            probabilities = _aligned_proba(fold_estimator, X[valid_index], classes)
            oof[valid_index] = probabilities
            report = evaluate_classifier(
                y[valid_index], probabilities, classes, positive_index, 0.5
            )
            scores.append(primary_score(report, plan.model_plan.primary_metric))

        return oof, scores

    @staticmethod
    def _threshold_from_oof(
        oof: np.ndarray,
        y: np.ndarray,
        positive_index: int | None,
        strategy: ThresholdStrategy,
    ) -> float:
        if positive_index is None:
            return 0.5
        return choose_threshold(
            (y == positive_index).astype(int), oof[:, positive_index], strategy
        )


def _fit(estimator: Any, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> None:
    """Fit with sample weights when the estimator accepts them.

    Not every third-party classifier does, and losing class balancing is far
    better than losing the candidate.
    """
    if weights is None:
        estimator.fit(X, y)
        return
    try:
        estimator.fit(X, y, sample_weight=weights)
    except TypeError:
        estimator.fit(X, y)


def _relabel(report: dict[str, Any], class_labels: list[str], positive: str | None) -> dict[str, Any]:
    """Swap encoded class indices back to the labels the user uploaded."""
    report = dict(report)
    report["classes"] = list(class_labels)
    if positive is not None:
        report["positive_class"] = positive
    matrix = report.get("confusion_matrix")
    if matrix:
        if report.get("is_binary") and positive is not None:
            negative = next((c for c in class_labels if c != positive), "Other")
            matrix["labels"] = [negative, positive]
        else:
            matrix["labels"] = list(class_labels)
    return report


def _label_for(schema: Any, column: str) -> str:
    spec = schema.spec(column)
    return spec.display_label if spec and spec.display_label else column.replace("_", " ").title()


def build_input_schema(
    X: pd.DataFrame,
    schema: Any,
    cleaner: CleaningTransformer,
) -> list[dict[str, Any]]:
    """Describe each input so a form can be generated for it at prediction time.

    This is why the product can take an arbitrary CSV and still offer a typed,
    bounded input form afterwards: the bundle carries its own description of
    what it expects, learned from the data it was trained on.
    """
    fields: list[dict[str, Any]] = []
    for column in schema.feature_columns:
        if column not in X.columns:
            continue
        spec = schema.spec(column)
        series = X[column]
        entry: dict[str, Any] = {
            "name": column,
            "label": spec.display_label if spec and spec.display_label else column.replace("_", " ").title(),
            "description": (spec.description if spec else "") or "",
            "kind": "numeric" if column in schema.numeric_features else "categorical",
        }

        if entry["kind"] == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                entry.update({"min": 0.0, "max": 1.0, "default": 0.0, "step": 0.1, "integer": False})
            else:
                low, high = float(numeric.min()), float(numeric.max())
                # Whole numbers get an integer slider with a step of 1; anything
                # else gets a hundredth-of-range step.
                integral = bool(numeric.dtype.kind in "iub" or (numeric % 1 == 0).all())
                entry.update(
                    {
                        "min": low,
                        "max": high,
                        "default": float(numeric.median()),
                        "step": 1.0 if integral else round(max((high - low) / 100, 0.01), 4),
                        "integer": integral,
                    }
                )
        else:
            kept = cleaner.kept_levels_.get(column) if hasattr(cleaner, "kept_levels_") else None
            levels = kept or [str(v) for v in series.dropna().astype(str).unique()[:50]]
            mode = series.dropna().astype(str).mode()
            entry.update(
                {
                    "levels": sorted(dict.fromkeys(str(level) for level in levels)),
                    "default": str(mode.iloc[0]) if len(mode) else (levels[0] if levels else ""),
                }
            )
        fields.append(entry)
    return fields
