# Architecture

How the system is put together, and — more usefully — which alternatives were
considered at each decision and why they were dropped.

---

## 1. The core problem

An AutoML service that a non-technical person drives has one hard constraint:
**it must never produce a confidently wrong answer**. A wrong answer that
announces itself is recoverable. A model that quietly trained on a leaked column,
or scaled a feature differently at prediction time than at training time, hands
a placement officer a number that looks exactly like a good one.

Every significant decision below is downstream of that.

---

## 2. Where the LLM sits

The obvious design is to let a language model write pandas and scikit-learn code
and run it. That is what most "AI data scientist" demos do. It was rejected.

| Approach | Why not |
|---|---|
| LLM generates Python, executed directly | Arbitrary code execution on a multi-tenant service. A prompt injected through a CSV column name becomes a shell. Sandboxing this properly is a larger project than the product. |
| LLM generates Python, executed in a sandbox | Solves the security problem, not the correctness one. Generated code still silently recomputes a scaler at prediction time, and there is nothing to validate against. |
| LLM picks from a fixed menu of preset pipelines | Safe, but adds nothing over a rule engine. It cannot notice that `backlogs` is a negative signal or that two columns measure the same thing on different scales. |
| **LLM emits a validated plan; a deterministic executor runs it** | **Chosen.** The model reasons about the domain; it cannot execute. Every plan is checked against both a schema and the actual dataset before anything runs. |

The plan vocabulary lives in `placement_ai/plans.py`. A feature is a record:

```json
{
  "name": "technical_soft_gap",
  "op": "difference",
  "inputs": ["technical_skill_score", "soft_skills_scaled"],
  "rationale": "A student strong on paper but weak in interviews shows up here."
}
```

`op` must be a member of `FeatureOp`. `inputs` must name columns the schema stage
kept as features, of the right kind. Arity, required params, name collisions and
weight-vector lengths are all checked. An op that does not exist is a validation
error at parse time, not a `KeyError` forty minutes into a training run.

### What the model actually sees

A profile, never rows:

```json
{"name": "cgpa", "kind": "numeric", "dtype": "float64", "missing_pct": 0.0,
 "n_unique": 341, "examples": [6.79, 5.92, 8.4],
 "stats": {"min": 5.0, "max": 10.0, "mean": 7.29, "std": 0.94}}
```

This is a privacy decision as much as a token-budget one. On a multi-tenant
service, sending student records to a third-party API on every training run is
not a defensible default.

---

## 3. The fallback ladder

Each planning stage runs:

```
        ┌──────────────┐   valid    ┌───────────────┐
        │  generate    ├───────────►│  source: llm  │
        └──────┬───────┘            └───────────────┘
               │ invalid
               ▼
        ┌──────────────┐   valid    ┌────────────────────────┐
        │ repair once  ├───────────►│ source: llm_repaired   │
        └──────┬───────┘            └────────────────────────┘
               │ still invalid, or a transport error
               ▼
        ┌────────────────────────────────────────────┐
        │ rule-based planner  ·  source: heuristic   │
        └────────────────────────────────────────────┘
```

The repair pass feeds the model its own output alongside the validation error.
It is attempted exactly once, and skipped entirely for transport failures — a
rate limit is not fixed by showing the model its own JSON.

**The bottom rung is load-bearing.** `planner/heuristic.py` writes a complete
plan from the profile alone: it groups numeric columns by the scale they appear
to live on and builds composites within each group, totals count-like columns,
normalises open-ended counts against a fitted maximum, and adds explicit flags
for two-level categoricals and for missingness. On the bundled sample that
reaches ROC-AUC 0.8813 with no network access.

Holding the fallback to that standard has a deliberate consequence: **the LLM
has to beat a competent baseline rather than merely exist.** What it adds is
domain reading — knowing that `backlogs` counts against a student, that `cgpa`
and `ssc_percentage` measure the same thing on different scales, that a ratio of
technical to soft skills is a real signal. The rules can only see shapes.

Provenance is recorded per stage, not per run, because a real run routinely
mixes both. The model card says which stages were generated and which fell back.

---

## 4. One bundle, one file

A saved model is a single joblib holding the entire fitted chain:

```
CleaningTransformer → FeatureSynthesizer → ColumnTransformer → estimator
```

| Approach | Why not |
|---|---|
| Save the estimator; re-run the plan at prediction time | The deployed model drifts from the evaluated one the moment the code changes. This is the bug that motivated the whole design. |
| Save the estimator + preprocessor as separate files | Two files that must stay mutually consistent. v1 of this repo did exactly this, plus a third file of frozen normalization statistics, and keeping them in sync was a recurring source of breakage. |
| **One pipeline, one file** | **Chosen.** Nothing to keep in sync. Cleaning rules, fitted imputation values, learned feature constants, encoder categories and the classifier all travel together. |

Everything learned from data lives in a fitted attribute inside that file. The
`normalize_max` operation divides by the maximum seen in training; that constant
is inside the transformer. This is the single most important invariant in the
system, because getting it wrong is silent — recomputing a maximum on a one-row
prediction divides the value by itself and hands the model `1.0` for everybody,
and every prediction still returns a plausible number.

`tests/test_dsl.py::test_normalize_max_reuses_the_fitted_maximum` pins it from
both directions.

The manifest beside it carries a SHA-256 of the joblib, verified on load, plus
the full plan, every candidate's score, the permutation importances, the input
schema, the drift baseline, and the library versions that built it.

---

## 5. The input schema, and why the form is generated

Each bundle records what it expects:

```json
{"name": "cgpa", "label": "Cgpa", "kind": "numeric",
 "min": 5.0, "max": 10.0, "default": 7.29, "step": 0.05, "integer": false}
```

This is what makes an arbitrary-CSV product possible at all. The Predict tab
builds its form from this record — bounds, step size, categorical levels,
sensible defaults, all learned from the training data. A model trained on a
completely different spreadsheet renders a completely different form, with no
code change.

It also gives every other feature its reference point. Attribution measures each
field against `default`. What-if curves sweep between `min` and `max` and refuse
to go beyond, because outside the training range the model has no evidence.

---

## 6. Explanations

| Approach | Why not |
|---|---|
| SHAP | Correct attribution, but `TreeExplainer` does not cover a linear model or a weighted ensemble, `KernelExplainer` is far too slow to run inline, and the output is in log-odds space, which is not what a placement officer is asking. |
| Global feature importance only | Answers "what matters in general", not "why this student". |
| **Ablation against a typical record** | **Chosen.** Works on any pipeline the planner assembles. One prediction per field on a single row, so it runs inline. And the output is directly actionable: "your attendance is costing you 9 points against a typical student". |

The honest limitation, stated in the UI: measuring one field at a time misses
interactions, so the deltas rank influence rather than summing to the total.

Global importance uses permutation over the **raw** columns, through the whole
pipeline. Permuting a derived feature like `percentage_composite` tells nobody
anything; permuting `attendance_percentage` answers the question actually being
asked.

---

## 7. Thresholds and imbalance

Two decisions that a naive AutoML gets wrong on an imbalanced intake:

**The decision threshold is fitted.** At 0.5, a model on a 75/25 split predicts
the majority class for nearly everyone and still scores a respectable ROC-AUC.
The threshold is chosen on out-of-fold predictions from the training split —
never on the test split, which would tune against the data used to report the
final numbers — and travels in the bundle. Serving ignores it at its peril, so
`WorkspacePredictor` applies it rather than taking an argmax.

**Class weighting is `sample_weight`, uniformly.** Every algorithm has its own
mechanism: `class_weight` for the sklearn ensembles, `scale_pos_weight` for
XGBoost, nothing at all for `GradientBoostingClassifier`. They are equivalent,
but only `sample_weight` is accepted by all of them, which keeps one code path
instead of a per-algorithm branch that breaks when an upstream library changes a
signature.

---

## 8. The ensemble

Candidates are trained once on a shared, once-fitted preprocessing matrix. The
ensemble then reuses those fitted estimators and averages their probabilities
with the planner's weights.

scikit-learn's `VotingClassifier` refits every member, which would mean training
each candidate twice for an identical result. `WeightedSoftVoter` avoids that,
and realigns members that disagree on class order — averaging raw probability
columns would otherwise add the wrong classes together.

The weights are the planner's judgement, not a guarantee. The ensemble is scored
against the best single model on the held-out split and kept only if it actually
wins. On the bundled sample it usually does not.

---

## 9. Tenancy

```
workspaces/<org-slug>-<random>/
  workspace.json     name, description, SHA-256 of the access code
  models/<version>/  pipeline.joblib + manifest.json
  history.db         this workspace's predictions
  datasets/          uploaded snapshots
```

There is deliberately **no central index file**. Listing scans directories, so
two people creating a workspace at the same moment cannot corrupt a shared
registry — a real risk in Streamlit, where every browser session runs the same
script.

The access code is separation, not authentication, and the code says so. It
keeps tenants out of each other's workspaces in the UI; it does not protect
anything from someone with filesystem access. Real deployment needs a real
identity provider in front.

---

## 10. Drift

PSI per column, between the records being scored and the training population.

The baseline is captured **at training time** and stored in the manifest.
Recomputing it from recent data — the tempting shortcut — compares the present
against itself and reports calm regardless.

One subtlety worth recording, because it produced a real bug: quantile edges
look like they imply an even `1/n` of the data per bucket, but a discrete column
defeats that. `backlogs` taking values 0–8 produces repeated edges that
deduplicate to a handful of bins holding wildly unequal shares, and assuming
uniformity reported PSI 0.32 against the training data itself. The true bucket
shares are now stored alongside the edges.

---

## 11. Why no API

v1 shipped a FastAPI service alongside the Streamlit dashboard. The dashboard
did not call it — it loaded the same artifacts and ran inference in-process — so
the API was a second implementation of the model layer for an audience that, in
this product, does not exist. A placement officer does not POST JSON.

`placement_ai/` is UI-agnostic and imports Streamlit nowhere except an optional
secrets lookup. If a programmatic interface is wanted later it is a thin wrapper
over `TrainingRunner` and `WorkspacePredictor`, not a rewrite.

---

## 12. Failure modes and what happens

| Failure | Behaviour |
|---|---|
| No API key anywhere | Rule-based planner; product fully functional; sidebar says so |
| Provider returns invalid JSON | One repair attempt, then the rules; recorded in the model card |
| Provider rate-limits or times out | Straight to the rules, no repair attempt |
| A generated feature names a column that does not exist | That feature is dropped with a warning; the rest of the plan runs |
| Most generated features are unusable | Plan rejected, repair attempted, then the rules |
| A candidate algorithm fails to fit | Skipped with a warning; the run continues on the others |
| Target is continuous | Refused before planning, with an explanation |
| Target has one class, or a class under 5 rows | Refused before planning, with the counts |
| Uploaded file is missing columns at prediction time | Imputed to training values; the missing list is shown |
| A model file is corrupted | Checksum mismatch, refuses to load, tells the user to retrain |
| Prediction log write fails | Prediction is still returned; the failure is surfaced, not raised |
| A save fails halfway | The partial version directory is removed, so `list()` never shows it |
