"""
placement_ai/plans.py
---------------------
The contract between the LLM and the executor.

An LLM never touches a dataframe and never emits code. It emits a *plan* — a
JSON document validated against the models below — and a deterministic executor
carries it out. Everything the model is allowed to ask for is enumerated here,
so the blast radius of a bad (or adversarial) generation is a rejected plan,
never arbitrary execution.

That split is also what makes a run auditable: the plan is stored verbatim
inside the model bundle, so months later you can read exactly which cleaning
rule and which derived feature the system chose, and why it said it chose it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Plan(BaseModel):
    """Shared config for every plan model.

    ``protected_namespaces=()`` is required because several fields legitimately
    start with ``model_`` (``model_plan``, ``model_version``), which pydantic
    otherwise warns about as a clash with its own ``model_`` API.
    ``extra="ignore"`` keeps a chatty LLM that adds an unrequested key from
    failing an otherwise valid plan.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="ignore")


# ── Stage 1: schema understanding ───────────────────────────────────────────


class TaskType(str, Enum):
    binary_classification = "binary_classification"
    multiclass_classification = "multiclass_classification"


class ColumnRole(str, Enum):
    """What a column is *for*, which is not the same as what dtype it holds."""

    numeric_feature = "numeric_feature"
    categorical_feature = "categorical_feature"
    target = "target"
    identifier = "identifier"  # unique per row, no predictive content
    drop = "drop"  # constant, empty, free text, or leaky


class ColumnSpec(_Plan):
    name: str
    role: ColumnRole
    display_label: str = ""
    description: str = ""
    # A column that could only be known *after* the outcome. Kept as its own
    # flag rather than folded into `drop`, so the model card can say why.
    leakage_risk: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def _default_label(self) -> ColumnSpec:
        if not self.display_label:
            self.display_label = self.name.replace("_", " ").strip().title()
        return self


class SchemaPlan(_Plan):
    target_column: str
    task_type: TaskType = TaskType.binary_classification
    # Which label counts as "the event happened". None for multiclass.
    positive_class: str | None = None
    columns: list[ColumnSpec] = Field(default_factory=list)
    summary: str = ""

    @model_validator(mode="after")
    def _target_is_consistent(self) -> SchemaPlan:
        by_name = {c.name: c for c in self.columns}
        if self.target_column not in by_name:
            raise ValueError(
                f"target_column {self.target_column!r} is not among the profiled columns"
            )
        # Whatever role the LLM assigned it, the target is the target.
        by_name[self.target_column].role = ColumnRole.target
        for spec in self.columns:
            if spec.role is ColumnRole.target and spec.name != self.target_column:
                spec.role = ColumnRole.drop
                spec.reason = spec.reason or "A second column marked as target."
        if not self.feature_columns:
            raise ValueError("The plan leaves no usable feature columns.")
        return self

    @property
    def feature_columns(self) -> list[str]:
        return [
            c.name
            for c in self.columns
            if c.role in (ColumnRole.numeric_feature, ColumnRole.categorical_feature)
        ]

    @property
    def numeric_features(self) -> list[str]:
        return [c.name for c in self.columns if c.role is ColumnRole.numeric_feature]

    @property
    def categorical_features(self) -> list[str]:
        return [c.name for c in self.columns if c.role is ColumnRole.categorical_feature]

    @property
    def dropped_columns(self) -> list[str]:
        return [
            c.name for c in self.columns if c.role in (ColumnRole.identifier, ColumnRole.drop)
        ]

    def spec(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)


# ── Stage 2: cleaning ────────────────────────────────────────────────────────


class ImputeStrategy(str, Enum):
    median = "median"
    mean = "mean"
    most_frequent = "most_frequent"
    constant = "constant"
    none = "none"


class ColumnCleaning(_Plan):
    column: str
    # Rescue numbers stored as text ("7.7 ", "1,200") before imputing.
    coerce_numeric: bool = False
    strip_whitespace: bool = False
    lowercase: bool = False
    impute: ImputeStrategy = ImputeStrategy.median
    fill_value: float | str | None = None
    clip_min: float | None = None
    clip_max: float | None = None
    # Categories rarer than this share of rows collapse into "Other", so a
    # one-hot encoder does not sprout a column seen four times in training.
    rare_category_min_frequency: float | None = None
    reason: str = ""

    @field_validator("rare_category_min_frequency")
    @classmethod
    def _sane_frequency(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return min(max(float(value), 0.0), 0.2)


class CleaningPlan(_Plan):
    drop_columns: list[str] = Field(default_factory=list)
    drop_duplicate_rows: bool = True
    drop_rows_missing_target: bool = True
    columns: list[ColumnCleaning] = Field(default_factory=list)
    notes: str = ""

    def for_column(self, name: str) -> ColumnCleaning | None:
        return next((c for c in self.columns if c.column == name), None)


# ── Stage 3: feature synthesis ───────────────────────────────────────────────


class FeatureOp(str, Enum):
    """The complete vocabulary a generated feature may use.

    Every op is a closed-form transform of named columns. There is no
    passthrough for an expression string, which is the point: a plan cannot
    smuggle in code, and an op that does not exist is a validation error rather
    than a runtime surprise.
    """

    # Combinations across several columns
    sum = "sum"
    mean = "mean"
    weighted_sum = "weighted_sum"
    product = "product"
    difference = "difference"  # a - b
    abs_difference = "abs_difference"
    ratio = "ratio"  # a / b, guarded
    per_unit = "per_unit"  # a / (b + 1), for count denominators
    spread = "spread"  # rowwise max - min
    rowwise_max = "rowwise_max"
    rowwise_min = "rowwise_min"
    count_above = "count_above"  # how many inputs clear a threshold

    # Single-column transforms
    scale = "scale"
    offset = "offset"
    clip = "clip"
    log1p = "log1p"
    sqrt = "sqrt"
    binarize_threshold = "binarize_threshold"
    binarize_equals = "binarize_equals"
    category_map = "category_map"
    is_missing = "is_missing"

    # Stateful — fitted on the training split, reused verbatim at inference.
    normalize_max = "normalize_max"
    min_max_scale = "min_max_scale"
    zscore = "zscore"


# Ops that learn a constant during fit. Their fitted state travels inside the
# bundle, so one student, a filtered cohort and the full dataset all land on
# the same scale — the failure mode that a per-batch max silently causes.
STATEFUL_OPS = {FeatureOp.normalize_max, FeatureOp.min_max_scale, FeatureOp.zscore}

# Minimum input columns each op needs. Ops taking exactly two are checked for
# exactly two, because `difference` over three columns is meaningless.
OP_ARITY: dict[FeatureOp, tuple[int, int | None]] = {
    FeatureOp.sum: (2, None),
    FeatureOp.mean: (2, None),
    FeatureOp.weighted_sum: (2, None),
    FeatureOp.product: (2, None),
    FeatureOp.difference: (2, 2),
    FeatureOp.abs_difference: (2, 2),
    FeatureOp.ratio: (2, 2),
    FeatureOp.per_unit: (2, 2),
    FeatureOp.spread: (2, None),
    FeatureOp.rowwise_max: (2, None),
    FeatureOp.rowwise_min: (2, None),
    FeatureOp.count_above: (1, None),
    FeatureOp.scale: (1, 1),
    FeatureOp.offset: (1, 1),
    FeatureOp.clip: (1, 1),
    FeatureOp.log1p: (1, 1),
    FeatureOp.sqrt: (1, 1),
    FeatureOp.binarize_threshold: (1, 1),
    FeatureOp.binarize_equals: (1, 1),
    FeatureOp.category_map: (1, 1),
    FeatureOp.is_missing: (1, 1),
    FeatureOp.normalize_max: (1, 1),
    FeatureOp.min_max_scale: (1, 1),
    FeatureOp.zscore: (1, 1),
}

# Ops that read a categorical column; everything else requires numerics.
CATEGORICAL_OPS = {FeatureOp.binarize_equals, FeatureOp.category_map, FeatureOp.is_missing}


class FeatureSpec(_Plan):
    name: str
    op: FeatureOp
    inputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="after")
    def _arity_matches_op(self) -> FeatureSpec:
        minimum, maximum = OP_ARITY[self.op]
        if len(self.inputs) < minimum:
            raise ValueError(
                f"feature {self.name!r} ({self.op.value}) needs at least {minimum} "
                f"input column(s), got {len(self.inputs)}"
            )
        if maximum is not None and len(self.inputs) > maximum:
            raise ValueError(
                f"feature {self.name!r} ({self.op.value}) accepts at most {maximum} "
                f"input column(s), got {len(self.inputs)}"
            )
        return self


class FeaturePlan(_Plan):
    features: list[FeatureSpec] = Field(default_factory=list)
    notes: str = ""


# ── Stage 4: model selection & weighting ─────────────────────────────────────


class Algorithm(str, Enum):
    logistic_regression = "logistic_regression"
    random_forest = "random_forest"
    extra_trees = "extra_trees"
    gradient_boosting = "gradient_boosting"
    hist_gradient_boosting = "hist_gradient_boosting"
    xgboost = "xgboost"


class Metric(str, Enum):
    roc_auc = "roc_auc"
    average_precision = "average_precision"
    f1 = "f1"
    balanced_accuracy = "balanced_accuracy"
    accuracy = "accuracy"


class ThresholdStrategy(str, Enum):
    """How the 0/1 cut-off is chosen once probabilities exist.

    `default` keeps 0.5. The others are picked on the validation split, which
    matters on an imbalanced target where 0.5 predicts the majority class for
    almost everyone.
    """

    default = "default"
    best_f1 = "best_f1"
    best_youden = "best_youden"


class CandidateModel(_Plan):
    algorithm: Algorithm
    params: dict[str, Any] = Field(default_factory=dict)
    # Relative say in the soft-voting ensemble. This is the "assign weights to
    # the model" step: the planner expresses how much it trusts each learner
    # given the data profile, and the ensemble is then judged against the best
    # single model rather than assumed better.
    ensemble_weight: float = 1.0
    rationale: str = ""

    @field_validator("ensemble_weight")
    @classmethod
    def _positive_weight(cls, value: float) -> float:
        return max(float(value), 0.0)


class ModelPlan(_Plan):
    candidates: list[CandidateModel] = Field(default_factory=list)
    class_weight_strategy: Literal["balanced", "none", "custom"] = "balanced"
    custom_class_weights: dict[str, float] | None = None
    primary_metric: Metric = Metric.roc_auc
    test_size: float = 0.2
    cv_folds: int = 5
    build_ensemble: bool = True
    threshold_strategy: ThresholdStrategy = ThresholdStrategy.best_f1
    notes: str = ""

    @field_validator("test_size")
    @classmethod
    def _sane_test_size(cls, value: float) -> float:
        return min(max(float(value), 0.1), 0.4)

    @field_validator("cv_folds")
    @classmethod
    def _sane_folds(cls, value: int) -> int:
        return min(max(int(value), 2), 10)

    @model_validator(mode="after")
    def _at_least_one_candidate(self) -> ModelPlan:
        if not self.candidates:
            raise ValueError("A model plan must propose at least one candidate.")
        return self


# ── Provenance ───────────────────────────────────────────────────────────────


class StageSource(str, Enum):
    llm = "llm"
    llm_repaired = "llm_repaired"  # first generation failed validation, retry passed
    heuristic = "heuristic"  # no provider, or the LLM never produced a valid plan


class StageProvenance(_Plan):
    """Who authored one stage of the plan, and what it cost.

    Recorded per stage rather than per run, because a run routinely mixes both:
    the LLM writes the schema and feature plans, then times out on the model
    plan and the heuristic finishes the job. The model card shows the mix
    instead of claiming the whole pipeline was AI-designed.
    """

    stage: str
    source: StageSource
    provider: str | None = None
    llm_model: str | None = None
    latency_ms: float | None = None
    error: str | None = None


class TrainingPlan(_Plan):
    """The four stage plans plus the record of who wrote each one."""

    schema_plan: SchemaPlan
    cleaning_plan: CleaningPlan
    feature_plan: FeaturePlan
    model_plan: ModelPlan
    provenance: list[StageProvenance] = Field(default_factory=list)

    @property
    def llm_authored_stages(self) -> list[str]:
        return [
            p.stage
            for p in self.provenance
            if p.source in (StageSource.llm, StageSource.llm_repaired)
        ]

    @property
    def used_llm(self) -> bool:
        return bool(self.llm_authored_stages)
