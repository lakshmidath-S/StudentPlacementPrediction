"""
placement_ai/planner/planner.py
-------------------------------
Stage orchestration: ask the model, validate, repair once, else fall back.

Every stage runs the same three-step ladder, and the ladder is the whole
reliability story:

    1. the provider answers and the reply validates      -> source "llm"
    2. it fails, is shown its own error, and retries      -> source "llm_repaired"
    3. it fails again, or there is no provider at all     -> source "heuristic"

Because step 3 always produces a valid plan, no LLM failure can stop a training
run. It only degrades how much reasoning went into it, and the degradation is
recorded per stage rather than hidden.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from placement_ai.config import LLM_MAX_ATTEMPTS
from placement_ai.llm.base import LLMError, LLMProvider
from placement_ai.planner import prompts
from placement_ai.planner.heuristic import (
    heuristic_cleaning_plan,
    heuristic_feature_plan,
    heuristic_model_plan,
    heuristic_schema_plan,
    xgboost_available,
)
from placement_ai.planner.sanitize import (
    coerce_cleaning_plan,
    coerce_feature_plan,
    coerce_model_plan,
    coerce_schema_plan,
)
from placement_ai.plans import (
    CleaningPlan,
    FeaturePlan,
    ModelPlan,
    SchemaPlan,
    StageProvenance,
    StageSource,
    TrainingPlan,
)
from placement_ai.profiling import DatasetProfile

# stage name, status ("start" | "ok" | "fallback"), human detail
ProgressFn = Callable[[str, str, str], None]


@dataclass
class StageOutcome:
    plan: Any
    provenance: StageProvenance
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlanningResult:
    plan: TrainingPlan
    warnings: list[str] = field(default_factory=list)


class Planner:
    """Builds a complete TrainingPlan, with or without a provider."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    @property
    def provider_label(self) -> str:
        return self.provider.label if self.provider else "heuristic (no LLM configured)"

    # ── the ladder ───────────────────────────────────────────────────────────

    def _run_stage(
        self,
        stage: str,
        prompt: str,
        coerce: Callable[[dict[str, Any]], tuple[Any, list[str]]],
        fallback: Callable[[], Any],
        progress: ProgressFn | None = None,
        max_output_tokens: int = 8192,
    ) -> StageOutcome:
        if progress:
            progress(stage, "start", f"Planning with {self.provider_label}")

        if self.provider is None:
            plan = fallback()
            # Still report completion. Without this the caller's progress bar
            # sits at the pre-planning stage through all four planning steps and
            # then jumps, which reads as a hang on a slow dataset.
            if progress:
                progress(stage, "ok", "Planned with the built-in rules")
            return StageOutcome(
                plan=plan,
                provenance=StageProvenance(stage=stage, source=StageSource.heuristic),
            )

        attempt_prompt = prompt
        last_error = ""
        last_raw = ""
        total_latency = 0.0

        for attempt in range(1, max(LLM_MAX_ATTEMPTS, 1) + 1):
            try:
                result = self.provider.complete_json(
                    prompts.SYSTEM_PROMPT, attempt_prompt, max_output_tokens=max_output_tokens
                )
                total_latency += result.latency_ms
                last_raw = result.raw_text
                plan, warnings = coerce(result.data)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, LLMError):
                    # A transport or quota failure will not be fixed by showing
                    # the model its own output; go straight to the fallback.
                    break
                if attempt < LLM_MAX_ATTEMPTS:
                    attempt_prompt = prompts.repair_prompt(prompt, last_raw, str(exc))
                    if progress:
                        progress(stage, "start", "The first plan did not validate; retrying")
                    continue
                break
            else:
                source = StageSource.llm if attempt == 1 else StageSource.llm_repaired
                if progress:
                    progress(stage, "ok", f"{self.provider_label} in {total_latency / 1000:.1f}s")
                return StageOutcome(
                    plan=plan,
                    provenance=StageProvenance(
                        stage=stage,
                        source=source,
                        provider=self.provider.name,
                        llm_model=self.provider.model,
                        latency_ms=round(total_latency, 1),
                    ),
                    warnings=warnings,
                )

        if progress:
            progress(stage, "fallback", f"Using built-in rules ({_short(last_error)})")
        return StageOutcome(
            plan=fallback(),
            provenance=StageProvenance(
                stage=stage,
                source=StageSource.heuristic,
                provider=self.provider.name,
                llm_model=self.provider.model,
                latency_ms=round(total_latency, 1) or None,
                error=last_error or None,
            ),
            warnings=[f"The {stage} stage fell back to built-in rules. {last_error}"],
        )

    # ── the four stages ──────────────────────────────────────────────────────

    def build_plan(
        self,
        profile: DatasetProfile,
        target_column: str,
        target_info: dict[str, Any],
        imbalance_pp: float,
        objective: str = "",
        progress: ProgressFn | None = None,
    ) -> PlanningResult:
        warnings: list[str] = []
        provenance: list[StageProvenance] = []

        schema_outcome = self._run_stage(
            stage="schema",
            prompt=prompts.schema_prompt(profile, target_column, target_info, objective),
            coerce=lambda raw: coerce_schema_plan(raw, profile, target_column, target_info),
            fallback=lambda: heuristic_schema_plan(profile, target_column, target_info),
            progress=progress,
        )
        schema: SchemaPlan = schema_outcome.plan
        warnings.extend(schema_outcome.warnings)
        provenance.append(schema_outcome.provenance)

        schema_payload = _schema_payload(schema)

        cleaning_outcome = self._run_stage(
            stage="cleaning",
            prompt=prompts.cleaning_prompt(profile, schema_payload),
            coerce=lambda raw: coerce_cleaning_plan(raw, profile, schema),
            fallback=lambda: heuristic_cleaning_plan(profile, schema),
            progress=progress,
        )
        cleaning: CleaningPlan = cleaning_outcome.plan
        warnings.extend(cleaning_outcome.warnings)
        provenance.append(cleaning_outcome.provenance)

        feature_outcome = self._run_stage(
            stage="features",
            prompt=prompts.feature_prompt(profile, schema_payload),
            coerce=lambda raw: coerce_feature_plan(raw, profile, schema),
            fallback=lambda: heuristic_feature_plan(profile, schema),
            progress=progress,
        )
        features: FeaturePlan = feature_outcome.plan
        warnings.extend(feature_outcome.warnings)
        provenance.append(feature_outcome.provenance)

        n_features = len(schema.feature_columns) + len(features.features)
        model_outcome = self._run_stage(
            stage="model",
            prompt=prompts.model_prompt(
                profile,
                schema_payload,
                n_features,
                target_info.get("levels", []),
                imbalance_pp,
                xgboost_available(),
            ),
            coerce=lambda raw: coerce_model_plan(raw, schema),
            fallback=lambda: heuristic_model_plan(profile, schema, imbalance_pp),
            progress=progress,
            max_output_tokens=4096,
        )
        model: ModelPlan = model_outcome.plan
        warnings.extend(model_outcome.warnings)
        provenance.append(model_outcome.provenance)

        return PlanningResult(
            plan=TrainingPlan(
                schema_plan=schema,
                cleaning_plan=cleaning,
                feature_plan=features,
                model_plan=model,
                provenance=provenance,
            ),
            warnings=warnings,
        )


def _schema_payload(schema: SchemaPlan) -> dict[str, Any]:
    """The compact schema view later prompts are given.

    Only what a later stage needs to reason: which columns survived, of which
    kind. Resending the full column-by-column plan wastes budget the feature
    stage needs for its own output.
    """
    return {
        "target_column": schema.target_column,
        "task_type": schema.task_type.value,
        "positive_class": schema.positive_class,
        "numeric_features": schema.numeric_features,
        "categorical_features": schema.categorical_features,
        "excluded": schema.dropped_columns,
    }


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn and report elapsed seconds alongside its result."""
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started
