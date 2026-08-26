# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Overview

An LLM-orchestrated AutoML service. A non-technical user uploads a spreadsheet;
a language model plans the pipeline for it; a deterministic executor runs the
plan; the result is one joblib the user owns and predicts with until they choose
to retrain.

There is **one consumer**: `app.py`, a Streamlit app. There is no API, no
worker, and no pre-trained model in the repository. `placement_ai/` is
UI-agnostic and imports Streamlit nowhere except an optional secrets lookup.

## Commands

```bash
# Run
streamlit run app.py

# Test / lint — no API key needed for any of it
pytest tests/ -q
pytest tests/test_dsl.py -q                              # single file
pytest tests/test_dsl.py::test_ratio_by_zero_is_zero_not_infinity -q   # single test
ruff check .                                             # run before finishing any change
pyright --project pyrightconfig.json                     # advisory in CI

# The CI gate: creates a workspace, trains on the bundled sample, saves,
# reloads from disk, predicts, explains, batch-scores, checks drift.
python scripts/smoke_test.py

# Force the rule-based planner even with a key present
LLM_PROVIDER=off streamlit run app.py
```

CI runs ruff → pyright (advisory) → pytest → `smoke_test.py`, with
`LLM_PROVIDER=off`. **The rule-based path is what CI verifies**, deliberately.

## Architecture

### The LLM emits plans, never code

`placement_ai/plans.py` is the contract. Every planning stage returns JSON
validated against a pydantic model, and a deterministic executor carries it out.
There is no `eval`, no expression parser, and no passthrough for a code string.

A derived feature is `{name, op, inputs, params}` where `op` must be a member of
`FeatureOp`. **Adding an operation means editing three places, and all three are
enforced by tests:**

1. `FeatureOp` + `OP_ARITY` (+ `CATEGORICAL_OPS` / `STATEFUL_OPS` if relevant) in `plans.py`
2. the implementation in `pipeline/dsl.py` — `apply_spec`, and `fit_spec` if stateful
3. `OP_REFERENCE` in `planner/prompts.py`, so the model is told it exists

`tests/test_dsl.py::test_every_declared_op_is_implemented` parametrises over the
whole enum and will fail on a missing implementation.

The LLM never sees a data row — only `DatasetProfile.to_payload()`. Keep it that
way; this is a multi-tenant service and rows are other people's records.

### The fallback ladder

Each stage: generate → validate → repair once from the validation error → fall
back to `planner/heuristic.py`. Transport errors (`LLMError`) skip the repair —
showing the model its own JSON does not fix a rate limit.

**The heuristic planner is not a stub.** It writes a complete trainable plan
from the profile alone and reaches ROC-AUC 0.8813 on the bundled sample with no
network. It runs in CI, on a fresh clone, and for any stage the LLM fails.
Breaking it breaks the product's floor. If you change it, `pytest
tests/test_training.py` is the check that matters.

Provenance is per stage, not per run — a real run routinely mixes LLM and
heuristic stages, and the model card says which was which.

### `sanitize.py` is the trust boundary

Pydantic checks a plan is *well-formed*. `planner/sanitize.py` checks it is
*true about this dataset* — every referenced column exists, numeric ops are not
pointed at text, weight vectors match input counts. Pydantic cannot do this
because it never sees the profile.

The filter/raise split is deliberate: a few bad features get dropped with a
warning, but past `_FEATURE_REJECT_LIMIT` (60%) the reply is treated as a misread
and raises `PlanRejected`, which triggers the repair pass.

### One bundle, one file

`workspaces/<org>/models/<version>/pipeline.joblib` holds the entire fitted
chain: `CleaningTransformer → FeatureSynthesizer → ColumnTransformer →
estimator`. Nothing is reconstructed at prediction time. `manifest.json` beside
it carries a SHA-256 verified on load, plus the full plan, all candidate scores,
importances, the input schema and the drift baseline.

**Never split a fitted parameter out of the bundle.** The previous version of
this repo kept `normalization_stats.json` next to the model and it was a
recurring source of breakage.

### Everything fitted lives in a trailing-underscore attribute

`normalize_max` divides by the maximum seen in training, stored in
`FeatureSynthesizer.states_`. Recomputing it during `transform` is the classic
silent corruption here: a one-row prediction divides the value by itself and
hands the model `1.0` for everybody, and every prediction still returns a
plausible number. The same applies to imputation values, kept category levels,
and the decision threshold.

If you add a transformer, fit on one frame and assert against a *different* one.
A round-trip on the training set will pass either way.

## Gotchas

- **The decision threshold is a fitted parameter, chosen on out-of-fold
  predictions.** Never on the test split — that tunes against the data used to
  report the final numbers. `WorkspacePredictor._label_from` applies it rather
  than taking an argmax; an argmax would serve different decisions than the ones
  that were evaluated.
- **Class weighting uses `sample_weight` for every algorithm**, not each one's
  own `class_weight` / `scale_pos_weight`. They are equivalent, but only
  `sample_weight` is accepted by all of them. Do not reintroduce the
  per-algorithm branch.
- **Cross-validation is hand-rolled in `runner._cross_validate`.** `cross_val_predict`
  cannot carry `sample_weight` without enabling scikit-learn metadata routing
  globally, which a library module has no business doing. The out-of-fold matrix
  does double duty: it scores the candidate and it is where the threshold is chosen.
- **Row-count changes happen in the runner, not in a transformer.** A transformer
  that dropped rows would desynchronise X from y inside a Pipeline.
- **`_emit` keeps the stage index monotonic.** Stages do not complete in list
  order — the planner finishes `select` before the runner finishes its own
  `engineer` pass — and a progress bar running backwards reads as a failure.
- **Identifier detection requires whole numbers or strings.** A continuous float
  column is normally unique per row and is exactly the kind of column a model
  wants most; treating high cardinality alone as an identifier left some datasets
  with no features at all.
- **The drift baseline stores true bucket shares, not just quantile edges.**
  Edges look like they imply `1/n` per bucket, but a discrete column
  (`backlogs`, 0–8) deduplicates to a few bins holding unequal shares, and
  assuming uniformity reported PSI 0.32 against the training data itself.
- **The baseline is captured at training time.** Recomputing it from recent
  traffic compares the present against itself and reports calm regardless.
- **Streamlit reruns `app.py` top to bottom on every widget change.**
  - Nothing lives in a local variable across a rerun. State is either in
    `st.session_state` or on disk.
  - A button press survives exactly one rerun. `use_sample` is parked in session
    state for this reason.
  - Training results are re-rendered from the *saved bundle*, not held in memory,
    so they survive any later interaction.
  - `load_predictor` is `@st.cache_resource` keyed on `(workspace_id, version,
    checksum)`. The checksum is in the key so a corrupted or replaced file is a
    cache miss rather than a stale object.
  - Use `width="stretch"`, not `use_container_width=True` — the latter is past
    its removal date and warns on every call.
- **Explanations are ablation, not SHAP.** One prediction per field against the
  training-typical value. It works on any pipeline the planner assembles and runs
  inline. The limitation — one field at a time misses interactions, so deltas
  rank rather than decompose — is stated in the UI and should stay stated.
- **Global importance is permutation over the raw columns**, through the whole
  pipeline. Permuting a derived feature tells a placement officer nothing.
- **Classification only.** A continuous target is refused before planning with an
  explanation. Do not silently bucket it.
- **`workspaces/` is tenant data.** Gitignored, and it must stay that way.

## Conventions

- ruff at the repo root: line length 100, rules `E,W,F,I,UP,B,C4,SIM`, target
  py311, first-party `placement_ai`.
- `app.py` sections use a `# ===` divider with a numbered title. Keep the
  numbering contiguous when adding or removing one.
- Streamlit UI copy: material icons in labels (`:material/bar_chart:`) and
  sentence casing.
- Every user-facing message says what to do about it. Compare
  `TrainingError("invalid target")` with the messages in `runner._guard_target`.
- Every generated `rationale` / `reason` field is shown to a non-technical user.
  The prompts say so; keep them saying so.
- Comments explain *why*, especially where the obvious implementation is wrong.
  Most comments in `dsl.py`, `transformers.py` and `drift.py` mark a real bug
  that was hit — do not delete them as noise.
