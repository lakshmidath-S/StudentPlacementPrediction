"""
placement_ai/planner/narrator.py
--------------------------------
Plain-language write-ups: one for a finished training run, one for a single
prediction.

This is the stage where the LLM is talking to the *user* rather than to the
executor, so the failure mode is different from planning. A wrong plan breaks
the pipeline and gets caught; wrong prose is read and believed. So the numbers
are computed here and passed in already finished — the model is asked to phrase
them, never to work them out, and is told explicitly not to invent any.

Both functions always return a usable document. Without a provider they fall
back to a template built from the same numbers, which reads plainly rather than
pretending an explanation was generated.
"""

from __future__ import annotations

import json
from typing import Any

from placement_ai.config import LLM_NARRATION_TOKENS
from placement_ai.llm.base import LLMProvider
from placement_ai.plans import StageSource, TrainingPlan

NARRATOR_SYSTEM = """You explain machine-learning results to people who do not \
work in machine learning — a placement officer, an HR coordinator, a college \
administrator.

Hard rules:
- Use only the numbers you are given. Never invent, round misleadingly, or infer \
a figure that is not in the input.
- No jargon without a plain gloss. "ROC-AUC 0.87" becomes "it ranks students \
correctly about 87% of the time when comparing a placed student against an \
unplaced one".
- Be honest about weakness. If a score is mediocre or a class is rare, say so.
- Never promise an outcome. These are estimates from past patterns, not fate.
- Reply with a single JSON object and nothing else."""


def _round(value: Any, places: int = 4) -> Any:
    """Trim a float to something a person would read aloud."""
    return round(value, places) if isinstance(value, float) else value


def _rounded(values: dict[str, Any], places: int = 4) -> dict[str, Any]:
    """The numeric entries of a metrics dict, rounded for quoting."""
    return {
        key: _round(value, places)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _call(provider: LLMProvider | None, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """One narration attempt; the template stands in on any failure."""
    if provider is None:
        return {**fallback, "source": "template"}
    try:
        result = provider.complete_json(
            NARRATOR_SYSTEM, prompt, max_output_tokens=LLM_NARRATION_TOKENS, temperature=0.4
        )
    except Exception as exc:
        return {**fallback, "source": "template", "error": f"{type(exc).__name__}: {exc}"}

    merged = {**fallback, **{k: v for k, v in result.data.items() if v}}
    merged["source"] = "llm"
    merged["provider"] = provider.label
    return merged


# ── Training report ──────────────────────────────────────────────────────────


def write_training_report(
    provider: LLMProvider | None,
    plan: TrainingPlan,
    dataset_summary: dict[str, Any],
    champion: str,
    metrics: dict[str, Any],
    importance: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Summarise a finished training run for the person who started it."""
    headline_score = metrics.get(metric)
    fallback = _training_template(champion, metrics, importance, metric, dataset_summary, plan)

    # Round before the numbers reach the prompt. The model quotes what it is
    # given, and a raw float lands in user-facing copy as "a recall score of
    # 0.7740863787375415" — accurate, and unreadable.
    scores = _rounded(metrics)

    prompt = f"""Summarise this completed training run.

WHAT WAS TRAINED
- winning model: {champion}
- headline metric: {metric} = {_round(headline_score)}
- decision threshold: {_round(metrics.get("threshold"))}
- all scores: {json.dumps(scores, ensure_ascii=False)}
- confusion matrix: {json.dumps(metrics.get("confusion_matrix", {}), ensure_ascii=False)}

THE DATA
{json.dumps(dataset_summary, ensure_ascii=False)}

WHAT THE MODEL LEANS ON MOST (permutation importance, larger = more influential)
{json.dumps(importance, ensure_ascii=False)}

DERIVED FEATURES THAT WERE BUILT
{json.dumps([{"name": f.name, "why": f.rationale} for f in plan.feature_plan.features][:12], ensure_ascii=False)}

Return JSON exactly like:
{{
  "headline": "<one sentence, under 15 words, stating how well this model works>",
  "summary": "<2-4 sentences: what it predicts, how accurate it is in practical terms, what that means day to day>",
  "strengths": ["<2-4 specific things this model does well, each tied to a number>"],
  "cautions": ["<2-4 honest limitations: weak classes, small samples, columns doing suspiciously heavy lifting>"],
  "next_steps": ["<2-3 concrete actions to improve it, e.g. data to collect>"]
}}"""

    return _call(provider, prompt, fallback)


def _training_template(
    champion: str,
    metrics: dict[str, Any],
    importance: list[dict[str, Any]],
    metric: str,
    dataset_summary: dict[str, Any],
    plan: TrainingPlan,
) -> dict[str, Any]:
    score = metrics.get(metric)
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "not available"
    top = [row["label"] for row in importance[:3]] or ["no single dominant column"]

    strengths = [f"{champion} scored {metric.replace('_', ' ')} of {score_text} on data it never saw."]
    if isinstance(metrics.get("recall"), (int, float)):
        strengths.append(
            f"It catches {metrics['recall'] * 100:.0f}% of the "
            f"{metrics.get('positive_class', 'positive')} cases in the test split."
        )
    if plan.feature_plan.features:
        strengths.append(
            f"{len(plan.feature_plan.features)} derived feature(s) were built on top of the "
            "columns you supplied."
        )

    cautions = []
    if isinstance(score, (int, float)) and score < 0.75:
        cautions.append(
            f"A {metric.replace('_', ' ')} of {score_text} is modest — treat these as a "
            "prompt to look closer, not a verdict."
        )
    if dataset_summary.get("imbalance_pp", 0) >= 20:
        cautions.append(
            f"The outcomes are unevenly split ({dataset_summary['imbalance_pp']:.0f} "
            "percentage points apart), so the rarer one is harder to predict."
        )
    if dataset_summary.get("rows_used", 0) < 500:
        cautions.append(
            f"Only {dataset_summary.get('rows_used', 0)} rows were available; more data "
            "would make these numbers steadier."
        )
    if not cautions:
        cautions.append(
            "Every number here comes from one held-out split. Re-check it against real "
            "outcomes before relying on it for decisions."
        )

    return {
        "headline": f"{champion} reached {metric.replace('_', ' ')} of {score_text}.",
        "summary": (
            f"Trained on {dataset_summary.get('rows_used', 0):,} rows to predict "
            f"{dataset_summary.get('target_column')}. {champion} performed best of the "
            f"{len(plan.model_plan.candidates)} model(s) tried. The strongest signals came "
            f"from {', '.join(top)}."
        ),
        "strengths": strengths,
        "cautions": cautions,
        "next_steps": [
            "Run a batch of current records through the Predict tab and sanity-check the results.",
            "Retrain once you have new confirmed outcomes to add.",
        ],
    }


# ── Per-prediction advice ────────────────────────────────────────────────────


def write_prediction_advice(
    provider: LLMProvider | None,
    target_column: str,
    predicted_label: str,
    probability: float,
    drivers: list[dict[str, Any]],
    inputs: dict[str, Any],
    positive_class: str | None,
) -> dict[str, Any]:
    """Turn one prediction and its attributions into something actionable."""
    fallback = _advice_template(predicted_label, probability, drivers, positive_class)

    prompt = f"""Explain one prediction to the person it is about.

PREDICTION
- outcome column: {target_column}
- predicted: {predicted_label}
- confidence in "{positive_class or predicted_label}": {probability:.3f}

WHAT MOVED THIS PREDICTION
Each entry gives a field, this person's value, and the change in probability \
caused by that value compared with a typical value. Positive helps, negative hurts.
{json.dumps([{**d, 'delta': _round(d.get('delta')), 'value': _round(d.get('value'))} for d in drivers], ensure_ascii=False, default=str)}

THEIR FULL RECORD
{json.dumps(inputs, ensure_ascii=False, default=str)}

Return JSON exactly like:
{{
  "headline": "<one sentence stating the prediction and how confident it is>",
  "reading": "<2-3 sentences explaining what drove it, naming the specific fields and values>",
  "drivers": [
    {{"factor": "<field, in plain words>", "direction": "helping" | "hurting", "note": "<one line on why>"}}
  ],
  "actions": ["<2-4 concrete, achievable steps that would move this outcome; skip anything the person cannot change, like age or gender>"]
}}"""

    return _call(provider, prompt, fallback)


def _advice_template(
    predicted_label: str,
    probability: float,
    drivers: list[dict[str, Any]],
    positive_class: str | None,
) -> dict[str, Any]:
    helping = [d for d in drivers if d.get("delta", 0) > 0][:3]
    hurting = [d for d in drivers if d.get("delta", 0) < 0][:3]

    return {
        "headline": (
            f"Predicted {predicted_label} with {probability * 100:.0f}% confidence"
            + (f" in {positive_class}" if positive_class else "")
            + "."
        ),
        "reading": (
            "Strongest supporting factors: "
            + (", ".join(d["label"] for d in helping) if helping else "none stood out")
            + ". Working against it: "
            + (", ".join(d["label"] for d in hurting) if hurting else "nothing stood out")
            + "."
        ),
        "drivers": [
            {
                "factor": d["label"],
                "direction": "helping" if d.get("delta", 0) > 0 else "hurting",
                "note": f"Value {d.get('value')} shifts the probability by {d.get('delta', 0):+.3f}.",
            }
            for d in (helping + hurting)
        ],
        "actions": [
            f"Focus on {d['label']} — it is currently the biggest drag on this prediction."
            for d in hurting[:2]
        ]
        or ["No single field stands out as holding this record back."],
    }


def provenance_summary(plan: TrainingPlan) -> str:
    """One line naming which stages the LLM authored — used in the model card."""
    llm_stages = plan.llm_authored_stages
    if not llm_stages:
        return "Planned entirely by the built-in rules (no LLM was configured)."
    provider = next(
        (p.provider for p in plan.provenance if p.source is not StageSource.heuristic), None
    )
    fallen_back = [p.stage for p in plan.provenance if p.source is StageSource.heuristic]
    text = f"{', '.join(llm_stages)} planned by {provider or 'the model'}"
    if fallen_back:
        text += f"; {', '.join(fallen_back)} fell back to built-in rules"
    return text + "."
