# Adaptive Placement Intelligence

A self-service machine-learning service. A placement officer uploads their own
spreadsheet, a language model plans the entire pipeline around it, and they walk
away with a trained model that belongs to them — no data scientist, no code, no
retraining until they ask for it.

[![CI](https://github.com/lakshmidath-S/StudentPlacementPrediction/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshmidath-S/StudentPlacementPrediction/actions/workflows/ci.yml)

---

## The goal

Placement prediction is normally shipped as *one model, trained once, on one
college's data*. That model is useless to the college next door: their columns
are different, their grading scale is different, and their intake is different.

This system inverts that. The product is not a model — it is the **thing that
builds models**.

Every organisation gets a workspace. They upload whatever spreadsheet they
already keep. An LLM reads the *shape* of that data and writes the plan: which
columns matter, which leak the answer, how to repair the messy ones, what
derived features are worth computing, which algorithms to try and how much to
weight each. A deterministic executor carries the plan out, scores the result
honestly, and saves it as a single file they own.

Two actions, and only two:

| | What the user does | What they get |
|---|---|---|
| **Train** | Upload a spreadsheet, name the outcome column | A scored, explained, versioned model saved to their workspace |
| **Predict** | Fill in a form, or upload a file of records | Predictions, the reasons behind them, and what would change them |

They repeat step 2 for as long as they like. Step 1 happens again only when they
choose — or when the drift monitor tells them their intake has moved away from
what the model learned.

### What changed from v1

| | Before | Now |
|---|---|---|
| Training | Offline scripts, run by a developer | In the app, by the user, on their data |
| Schema | 17 hardcoded placement columns | Whatever the user uploads |
| Features | 21 formulas written by hand in `feature_engineering.py` | Designed per dataset by the LLM, from a closed vocabulary |
| Models | Three, fixed, pre-trained and committed | Chosen and weighted per dataset; nothing pre-trained |
| Tenancy | One organisation | Many, each isolated |
| Surface | FastAPI service + Streamlit dashboard | Streamlit only |

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

That is the whole setup. **No API key is required** — without one the planner
uses its rule-based fallback and the product works end to end, just without
generated reasoning.

In the app: create a workspace in the sidebar → **Train** tab → *Use sample
data* → *Start training*. About twenty seconds later you have a model, and the
**Predict** tab has built a form for it.

### Turning the AI planner on

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey)
(Gemini), [OpenRouter](https://openrouter.ai/keys) (a gateway to several hundred
models, some free), or [xAI](https://console.x.ai/) (Grok), then either:

```bash
# a .env file at the repo root
echo 'GEMINI_API_KEY=your-key-here' > .env
```

or set `GEMINI_API_KEY` / `OPENROUTER_API_KEY` / `XAI_API_KEY` as an environment
variable, or put it in `.streamlit/secrets.toml` for Streamlit Community Cloud.
The sidebar shows which providers it can see.

Reading `.env` needs `python-dotenv`. It is in `requirements.txt`, and if it is
missing while a `.env` exists the sidebar says so rather than silently ignoring
the file. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full settings list.

---

## How a training run works

Eleven stages. The LLM owns four of them; everything else is ordinary Python.

```
  ingest ─► profile ─►┌ understand ─► clean ─► engineer ─► select ┐─► train
                      └──────────── the LLM plans ───────────────┘     │
                                                                       ▼
                package ◄─ explain ◄─ choose ◄─ evaluate ◄─────────────┘
```

| Stage | Who | What happens |
|---|---|---|
| ingest | code | Read the file, snake_case the headers, cap the row count |
| profile | code | Measure every column: dtype, range, gaps, cardinality, examples |
| **understand** | **LLM** | Assign each column a role; find the target; flag leakage |
| **clean** | **LLM** | Per-column repair and imputation rules |
| **engineer** | **LLM** | Derived features, from a fixed operation vocabulary |
| **select** | **LLM** | Candidate algorithms, hyperparameters, ensemble weights |
| train | code | Fit each candidate with stratified cross-validation |
| evaluate | code | Score on a held-out split it never saw |
| choose | code | Best single model vs. the weighted ensemble; permutation importance |
| explain | LLM* | Plain-language summary of what was built |
| package | code | Save one joblib + a manifest recording the whole plan |

\* falls back to a template built from the same numbers.

### The three rules that make this safe

**The LLM emits plans, never code.** Every derived feature is a `{name, op,
inputs, params}` record drawn from an enumerated vocabulary — `ratio`, `spread`,
`normalize_max`, `binarize_equals`, and about twenty others. There is no `eval`,
no expression parser, and no way to name an operation that does not exist. A bad
generation is a rejected plan, not arbitrary execution.

**The LLM never sees a data row.** It plans against a statistical profile —
column names, dtypes, ranges, missingness, a handful of example values. One
tenant's student records never reach a third-party API.

**Every stage can fail without stopping the run.** Each one runs a ladder:
generate → validate → repair once from the validation error → fall back to a
rule-based planner that produces a complete, trainable plan on its own. The
fallback is not a stub; it is what CI runs, and it reaches ROC-AUC 0.8813 on the
bundled sample. Which stages the LLM actually authored is recorded per stage in
the model card, so nothing claims to be AI-designed when it was not.

---

## What you get back

- **A model card** — the full plan, verbatim: every column role and why, every
  cleaning rule and why, every derived feature and its formula, every algorithm
  tried with its score and cross-validated spread.
- **Honest scores** — computed once on a held-out split, stored in the bundle.
  The decision threshold is a *fitted parameter*, chosen on out-of-fold
  predictions rather than left at 0.5, which matters on an imbalanced intake.
- **Per-prediction explanations** — for each field, the change in probability
  caused by this record's value versus a typical one, plus the smallest single
  change that would flip the outcome.
- **Drift monitoring** — PSI between the records you are scoring now and the
  training population, measured against a baseline captured at training time.
- **A feedback loop** — record what actually happened to past predictions and
  export them as training data for the next run.

---

## Repository layout

```
app.py                       the entire UI: workspace, train, predict, history, model card
placement_ai/
  config.py                  paths, provider settings, guardrails
  profiling.py               deterministic dataset profiling
  plans.py                   the LLM/executor contract — every plan is validated against this
  llm/                       Gemini, Grok and OpenRouter over REST; retries, key resolution
  planner/
    prompts.py               one prompt per stage
    heuristic.py             the rule-based planner — the floor the product stands on
    sanitize.py              turning a raw reply into a plan that is true about this dataset
    planner.py               generate -> validate -> repair -> fall back
    narrator.py              plain-language write-ups
  pipeline/
    dsl.py                   the feature operations; the only place a plan becomes numbers
    transformers.py          cleaning + feature synthesis as fitted sklearn estimators
    ensemble.py              weighted soft voting over already-fitted models
    builder.py               assembling plans into a scikit-learn pipeline
  training/
    runner.py                the eleven-stage job
    evaluation.py            metrics, threshold selection, permutation importance
  registry/
    workspace.py             one directory per organisation
    model_store.py           versioned joblib bundles with checksummed manifests
    history.py               per-workspace prediction log (SQLite)
  inference/
    predictor.py             serving a saved bundle
    explain.py               attribution, what-if curves, improvement levers
    drift.py                 PSI against the training baseline
docs/
  ARCHITECTURE.md            the design, and the alternatives that were rejected
  DEPLOYMENT.md              running it locally and in the cloud
scripts/smoke_test.py        CI gate: trains a real model with no key
tests/                       248 tests, all running without a provider
data/raw/                    the bundled synthetic sample
workspaces/                  tenant data — gitignored, never committed
```

---

## The bundled sample

`data/raw/synthetic_placement_dataset.csv` — 10,000 synthetic student records,
19 columns, no real people. It exists so a first-time user can reach a trained
model without hunting for a CSV.

Trained on it with **no LLM at all**, the rule-based planner produces:

| | |
|---|---|
| Champion | Logistic Regression (beat Random Forest and XGBoost) |
| ROC-AUC | **0.8813** on 2,000 held-out rows |
| Decision threshold | 0.403, fitted — not 0.5 |
| Derived features | 15, built from column shape alone |
| Training time | ~20 s |

For reference, the previous hand-engineered pipeline scored 0.8780 on the same
data. The point is not that 0.8813 is better — the difference is noise. The
point is that a system with *no knowledge of placement data* matched a
hand-tuned pipeline, which is what makes it plausible on a dataset nobody has
tuned for.

---

## Limitations

Stated plainly, because a system that plans its own pipeline invites more trust
than it has earned:

- **Classification only.** Binary and multiclass. A continuous target is
  refused with an explanation rather than silently bucketed.
- **A workspace is separation, not authentication.** There is no login: anyone
  reaching the app can open any workspace, and anyone with filesystem access can
  read everything. Putting this in front of real students means putting a real
  identity provider in front of it first.
- **Attribution is ablation, not SHAP.** One field at a time, so the numbers
  rank influence rather than decomposing it exactly, and interactions are
  missed. The trade buys explanations that work on any pipeline the planner
  assembles and run inline in the UI.
- **Free tiers are rate-limited and their models churn.** A training run makes
  four planning calls plus one narration call. Transient 429/503 responses are
  retried with backoff; past that the stage falls back to the rules, visibly, in
  the model card. Model IDs also get retired — both providers' errors name the
  setting to change.
- **Protected attributes are not filtered.** If a spreadsheet contains gender or
  caste, those columns become features like any other. The model card shows
  exactly what was used; deciding what *should* be used is the institution's
  call, and it is a real one.
- **No text or datetime features.** Free-text and timestamp columns are
  reported and dropped, not vectorised.

---

## Development

```bash
pytest tests/ -q                 # 248 tests, no API key needed
ruff check .                     # run before finishing any change
python scripts/smoke_test.py     # the full loop end to end, no key
streamlit run app.py
```

CI runs ruff → pyright (advisory) → pytest → the smoke test, with
`LLM_PROVIDER=off` so the rule-based path is what gets verified.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design decisions and
the alternatives that were considered and rejected.
