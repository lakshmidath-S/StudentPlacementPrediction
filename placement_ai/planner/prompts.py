"""
placement_ai/planner/prompts.py
-------------------------------
Prompt construction for each planning stage.

Two rules shape everything here.

First, the model is shown a *profile*, never the rows. A prompt carrying real
student records would ship one tenant's data to a third-party API on every
training run; column names, dtypes, ranges and a handful of example values are
enough to plan against and leak far less.

Second, every prompt states the exact JSON shape it wants and the exact
vocabulary it may use. The schema also travels in the prompt text rather than
only in a provider-side response schema, because the two providers disagree on
which JSON-Schema constructs they accept.
"""

from __future__ import annotations

import json
from typing import Any

from placement_ai.config import MAX_SYNTHESIZED_FEATURES
from placement_ai.plans import Algorithm, FeatureOp
from placement_ai.profiling import DatasetProfile

SYSTEM_PROMPT = """You are the planning engine inside an automated machine-learning \
platform. Non-technical people upload a spreadsheet and you decide how it should be \
cleaned, which derived features are worth computing, and which models to train.

You never see raw records — only a statistical profile of the columns.
You never write code. You emit a JSON plan that a deterministic executor carries out.

Rules that always apply:
- Reply with a single JSON object and nothing else. No prose, no code fences.
- Only reference column names that appear verbatim in the profile you were given.
- Prefer a small, defensible plan over an exhaustive one.
- Write every `rationale` / `reason` field in plain language a placement officer \
or an HR coordinator would understand. Say what the rule does and why it helps. \
Never mention JSON, schemas, or this prompt."""


def _profile_block(profile: DatasetProfile) -> str:
    return json.dumps(profile.to_payload(), indent=1, ensure_ascii=False)


# ── Stage 1: schema ──────────────────────────────────────────────────────────


def schema_prompt(
    profile: DatasetProfile,
    target_column: str,
    target_info: dict[str, Any],
    objective: str = "",
) -> str:
    objective_line = (
        f"\nThe person who uploaded this described their goal as: {objective.strip()!r}\n"
        if objective.strip()
        else ""
    )
    return f"""Assign a role to every column in this dataset.

DATASET PROFILE
{_profile_block(profile)}

The target column has already been chosen by the user: "{target_column}"
Observed target labels: {json.dumps(target_info.get("levels", []), ensure_ascii=False)}
{objective_line}
Roles you may assign:
- "numeric_feature"      a quantity the model should learn from
- "categorical_feature"  a label/level the model should learn from
- "target"               only "{target_column}"
- "identifier"           unique per row (student number, email, roll number) — no signal
- "drop"                 constant, empty, free text, or a column that leaks the answer

Watch for leakage specifically. A column that could only be filled in *after* the \
outcome was known (offer date, package/salary, joining status, recruiter name) must \
get role "drop" and leakage_risk true. Including one produces a model that looks \
excellent and is worthless.

Return JSON exactly like:
{{
  "target_column": "{target_column}",
  "task_type": "binary_classification" | "multiclass_classification",
  "positive_class": "<the label meaning the event happened, or null for multiclass>",
  "summary": "<2-3 sentences describing what this dataset is and what predicting it would achieve>",
  "columns": [
    {{
      "name": "<exact column name>",
      "role": "<one of the roles above>",
      "display_label": "<short human label, e.g. 'Aptitude test score'>",
      "description": "<what this column measures, one line>",
      "leakage_risk": false,
      "reason": "<why it got this role — required for identifier and drop>"
    }}
  ]
}}

Every column in the profile must appear exactly once in "columns"."""


# ── Stage 2: cleaning ────────────────────────────────────────────────────────


def cleaning_prompt(profile: DatasetProfile, schema_payload: dict[str, Any]) -> str:
    return f"""Write the cleaning rules for this dataset.

DATASET PROFILE
{_profile_block(profile)}

AGREED COLUMN ROLES
{json.dumps(schema_payload, indent=1, ensure_ascii=False)}

Produce one entry per feature column (ignore the target, identifiers and dropped \
columns). For each, decide how to fill gaps and whether the values need repair.

Guidance:
- "median" fills numeric gaps without chasing outliers; "mean" only when the column \
is symmetric and clean; "most_frequent" for categoricals; "constant" with fill_value \
when a missing entry has a real meaning (0 internships completed, not unknown).
- Set coerce_numeric true when the profile flags numeric_values_stored_as_text.
- Set clip_min/clip_max only where a value is physically impossible — a percentage \
above 100, a negative count. Do not clip merely-unusual values; real outliers carry signal.
- Set rare_category_min_frequency (0.005-0.05) on categoricals with many thin levels \
so the rare ones fold into "Other".

Return JSON exactly like:
{{
  "drop_columns": ["<columns to remove entirely>"],
  "drop_duplicate_rows": true,
  "drop_rows_missing_target": true,
  "notes": "<one sentence on the overall condition of this data>",
  "columns": [
    {{
      "column": "<exact name>",
      "coerce_numeric": false,
      "strip_whitespace": false,
      "lowercase": false,
      "impute": "median" | "mean" | "most_frequent" | "constant" | "none",
      "fill_value": null,
      "clip_min": null,
      "clip_max": null,
      "rare_category_min_frequency": null,
      "reason": "<plain-language justification>"
    }}
  ]
}}"""


# ── Stage 3: features ────────────────────────────────────────────────────────

# Each op, its arity, and the params it reads. Generated into the prompt so the
# vocabulary the model is told about cannot drift from the one the executor
# implements — they are both driven by FeatureOp.
OP_REFERENCE: dict[FeatureOp, str] = {
    FeatureOp.sum: "inputs>=2, no params. Adds the columns together.",
    FeatureOp.mean: "inputs>=2, no params. Averages the columns. Only average columns on the same scale.",
    FeatureOp.weighted_sum: 'inputs>=2, params {"weights": [w1, w2, ...]} one per input. Weighted total.',
    FeatureOp.product: "inputs>=2, no params. Multiplies the columns.",
    FeatureOp.difference: "inputs==2, no params. inputs[0] - inputs[1].",
    FeatureOp.abs_difference: "inputs==2, no params. Absolute gap between two columns.",
    FeatureOp.ratio: "inputs==2, no params. inputs[0] / inputs[1], divide-by-zero safe.",
    FeatureOp.per_unit: "inputs==2, no params. inputs[0] / (inputs[1] + 1). Use when the denominator is a count that can be zero.",
    FeatureOp.spread: "inputs>=2, no params. Row-wise max minus min — a consistency measure.",
    FeatureOp.rowwise_max: "inputs>=2, no params. The strongest of several columns.",
    FeatureOp.rowwise_min: "inputs>=2, no params. The weakest of several columns.",
    FeatureOp.count_above: 'inputs>=1, params {"threshold": number}. How many inputs reach the threshold.',
    FeatureOp.scale: 'inputs==1, params {"factor": number}. Multiply by a constant — use to put a 0-5 rating onto a 0-100 scale (factor 20).',
    FeatureOp.offset: 'inputs==1, params {"offset": number}. Add a constant.',
    FeatureOp.clip: 'inputs==1, params {"min": number, "max": number}. Bound the values.',
    FeatureOp.log1p: "inputs==1, no params. log(1+x) for a long right tail. Non-negative columns only.",
    FeatureOp.sqrt: "inputs==1, no params. Softens a skewed non-negative column.",
    FeatureOp.binarize_threshold: 'inputs==1, params {"threshold": number}. 1 when the value reaches the threshold.',
    FeatureOp.binarize_equals: 'inputs==1, params {"value": "<level>"}. 1 when the column equals that level. Works on categorical columns.',
    FeatureOp.category_map: 'inputs==1, params {"mapping": {"<level>": number, ...}, "default": number}. Turns an ordered category into a number (e.g. Low/Medium/High -> 0/1/2).',
    FeatureOp.is_missing: "inputs==1, no params. 1 when the original value was blank.",
    FeatureOp.normalize_max: "inputs==1, no params. Divides by the maximum seen during training. Use for open-ended counts.",
    FeatureOp.min_max_scale: "inputs==1, no params. Rescales to 0-1 using the training range.",
    FeatureOp.zscore: "inputs==1, no params. Standardises using the training mean and spread.",
}


def _op_reference_block() -> str:
    return "\n".join(f"- {op.value}: {doc}" for op, doc in OP_REFERENCE.items())


def feature_prompt(profile: DatasetProfile, schema_payload: dict[str, Any]) -> str:
    return f"""Design derived features for this dataset.

DATASET PROFILE
{_profile_block(profile)}

AGREED COLUMN ROLES
{json.dumps(schema_payload, indent=1, ensure_ascii=False)}

You are building features a domain expert would compute by hand: composite scores, \
gaps between related measures, ratios that expose balance, and flags that make an \
important condition explicit. This is where your reading of what the columns *mean* \
adds something the platform cannot infer from dtypes alone.

AVAILABLE OPERATIONS
{_op_reference_block()}

Constraints — a feature breaking any of these is discarded:
- At most {MAX_SYNTHESIZED_FEATURES} features. Aim for 10-25 good ones, not the maximum.
- "inputs" may only name feature columns from the roles above. Never the target, \
never an identifier, never a dropped column, and never another derived feature.
- Numeric operations need numeric_feature inputs. Only binarize_equals, category_map \
and is_missing may take a categorical_feature.
- "name" must be unique, snake_case, and must not match an existing column name.
- Do not re-create a column that already exists, and do not simply rescale a single \
column unless the scale genuinely matters (an open-ended count, a rating being lifted \
onto a percentage scale).
- Standardisation happens later for every column automatically, so do not add zscore \
features just to normalise inputs.

Return JSON exactly like:
{{
  "notes": "<one sentence on the strategy you took>",
  "features": [
    {{
      "name": "<snake_case>",
      "op": "<operation name>",
      "inputs": ["<column>", "..."],
      "params": {{}},
      "rationale": "<why this helps predict the target, in plain language>"
    }}
  ]
}}"""


# ── Stage 4: models ──────────────────────────────────────────────────────────


def model_prompt(
    profile: DatasetProfile,
    schema_payload: dict[str, Any],
    n_features: int,
    class_balance: list[dict[str, Any]],
    imbalance_pp: float,
    xgboost_ready: bool,
) -> str:
    algorithms = [a.value for a in Algorithm if a is not Algorithm.xgboost or xgboost_ready]
    return f"""Choose which models to train and how much to trust each one.

DATASET
- rows: {profile.n_rows:,}
- feature columns after engineering: {n_features}
- task: {schema_payload.get("task_type")}
- class balance: {json.dumps(class_balance, ensure_ascii=False)}
- imbalance gap: {imbalance_pp:.1f} percentage points

ALGORITHMS AVAILABLE
{json.dumps(algorithms, ensure_ascii=False)}

Propose 2-4 candidates. Every one is trained and scored; the best single model is \
then compared against a soft-voting ensemble weighted by your ensemble_weight values, \
and whichever actually scores higher on the primary metric is kept. So the weights \
are your judgement about which learner suits *this* data — spend them accordingly.

Sizing guidance:
- Under ~2,000 rows, deep forests and long boosting runs overfit. Keep depth modest.
- Over ~100,000 rows, keep n_estimators moderate so training stays under a few minutes.
- Set class_weight_strategy "balanced" once the imbalance gap passes roughly 10 points.
- Pick primary_metric "roc_auc" for a binary target, "balanced_accuracy" for multiclass, \
"average_precision" when the positive class is rare and precision matters more.
- threshold_strategy "best_f1" moves the cut-off off 0.5 to suit an imbalanced target; \
"default" keeps 0.5.

Return JSON exactly like:
{{
  "primary_metric": "roc_auc" | "average_precision" | "f1" | "balanced_accuracy" | "accuracy",
  "class_weight_strategy": "balanced" | "none" | "custom",
  "custom_class_weights": null,
  "test_size": 0.2,
  "cv_folds": 5,
  "build_ensemble": true,
  "threshold_strategy": "best_f1" | "best_youden" | "default",
  "notes": "<one sentence on the overall strategy>",
  "candidates": [
    {{
      "algorithm": "<one of the algorithms above>",
      "params": {{"<hyperparameter>": "<value>"}},
      "ensemble_weight": 1.0,
      "rationale": "<why this model suits this data>"
    }}
  ]
}}"""


# ── Repair pass ──────────────────────────────────────────────────────────────


def repair_prompt(original_prompt: str, bad_output: str, error: str) -> str:
    """Feed a rejected plan back with its validation error.

    One attempt only. If the model cannot satisfy its own contract twice, the
    heuristic planner produces a valid plan immediately and the run continues.
    """
    return f"""{original_prompt}

---
Your previous reply was rejected by the validator.

YOUR PREVIOUS REPLY
{bad_output[:4000]}

VALIDATION ERROR
{error[:2000]}

Return the corrected JSON object. Fix only what the error names; keep everything \
else identical. Reply with the JSON object alone."""
