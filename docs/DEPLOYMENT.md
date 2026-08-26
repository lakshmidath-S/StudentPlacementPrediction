# Deployment

One Streamlit app, one process. There is no API, no worker, and no database
server — a workspace is a directory and its history is a SQLite file inside it.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or newer. The app opens on <http://localhost:8501>.

No API key is needed to run it. Without one the planner uses its rule-based
fallback; the sidebar shows which mode you are in.

---

## Configuration

Everything is an environment variable. All are optional.

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables the Gemini planner. Free key from [AI Studio](https://aistudio.google.com/apikey). |
| `XAI_API_KEY` | — | Enables the Grok planner. Key from [console.x.ai](https://console.x.ai/). |
| `LLM_PROVIDER` | `auto` | `auto`, `gemini`, `grok`, or `off`. `off` forces the rule-based planner even when a key is present. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Which Gemini model to call. |
| `GROK_MODEL` | `grok-3-mini` | Which Grok model to call. |
| `LLM_TIMEOUT_SECONDS` | `90` | Per-request timeout. A stage that times out falls back to the rules. |
| `LLM_MAX_ATTEMPTS` | `2` | Generation attempts per stage. `2` means one repair pass. |
| `PLACEMENT_AI_HOME` | `./workspaces` | Where tenant data lives. **Point this at a persistent volume in any cloud deployment.** |
| `MAX_TRAINING_ROWS` | `250000` | Larger uploads are randomly sampled down, and the user is told. |
| `MAX_SYNTHESIZED_FEATURES` | `40` | Cap on derived features per model. |
| `MAX_CATEGORY_CARDINALITY` | `50` | Above this many distinct levels, a categorical column is dropped. |

### Where keys are read from

In order: environment variable → `.env` at the repo root → `st.secrets`. The
last one exists because Streamlit Community Cloud has no way to set environment
variables.

```bash
# local development
cat > .env <<'EOF'
GEMINI_API_KEY=your-key-here
EOF
```

```toml
# .streamlit/secrets.toml — for Streamlit Community Cloud
GEMINI_API_KEY = "your-key-here"
```

Both are gitignored. Keep them that way.

---

## Streamlit Community Cloud

1. Push to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at the
   repo with `app.py` as the entry point.
3. Add `GEMINI_API_KEY` under **Settings → Secrets** if you want the AI planner.

**The important caveat: the Community Cloud filesystem is ephemeral.** It is
wiped on every redeploy and on the container recycling that follows a period of
inactivity. Trained models and prediction history *will* disappear. That is fine
for a demo and unacceptable for real use.

For anything beyond a demo, run it somewhere with a persistent disk and set
`PLACEMENT_AI_HOME` to a path on it.

---

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Tenant data lives on a mounted volume, never in the image layer.
ENV PLACEMENT_AI_HOME=/data
VOLUME /data

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t placement-ai .
docker run -p 8501:8501 -v placement-data:/data -e GEMINI_API_KEY=... placement-ai
```

---

## Sizing

A training run is CPU-bound and holds the whole dataset in memory.

| Rows | Time (rule-based) | Peak memory |
|---|---|---|
| ~300 | ~2 s | < 200 MB |
| ~10,000 | ~20 s | ~400 MB |
| ~100,000 | 3–8 min | 1–2 GB |

The four planning calls add 5–30 s depending on the provider and dataset width,
and happen before any model is fitted.

Streamlit blocks its script thread while a run is in progress, so a single
container serves one training run at a time. Concurrent *predictions* are fine —
they are milliseconds and the loaded model is shared via `@st.cache_resource`.

Trained bundles are small: a few KB for a linear model, low single-digit MB for
a large forest.

---

## Backup

Everything a tenant owns is under `PLACEMENT_AI_HOME`:

```bash
tar czf backup-$(date +%F).tar.gz -C "$PLACEMENT_AI_HOME" .
```

Restore by unpacking it back. There is no schema migration to run — a manifest
records its own `schema_version`, and bundles built by a different library
version report the mismatch at load time rather than failing obscurely.

---

## Before real students go in

The gaps that matter, in order:

1. **Authentication.** The workspace access code separates tenants in the UI. It
   is not authentication and does not pretend to be. Put an identity provider in
   front and map users to workspaces.
2. **Encryption at rest.** Uploaded spreadsheets and prediction history sit in
   plain files. Use an encrypted volume.
3. **Retention.** Nothing is ever deleted automatically. Decide how long
   `history.db` and `datasets/` should live, and enforce it.
4. **An LLM data-processing agreement.** Only column *profiles* leave the
   machine — never rows — but that still needs to be written down and agreed to
   before anyone else's data is involved.
5. **A fairness review.** If an uploaded spreadsheet contains gender, caste or
   similar, those columns become features like any other. The model card shows
   exactly what was used. Deciding what *should* be used is an institutional
   decision, and a real one.

---

## Troubleshooting

**The sidebar says "Running on built-in rules" but I set a key.**
Streamlit reads `.env` and `st.secrets` at process start. Restart the app. Check
the variable name — `GEMINI_API_KEY`, not `GOOGLE_API_KEY`.

**Training says a stage "fell back to built-in rules".**
Expected and safe. Open the model card → *The plan* → *Stage-by-stage
provenance* for the specific error. On a free tier the usual cause is a rate
limit; wait a minute and retrain.

**"Model … does not match its manifest checksum".**
The joblib was modified or truncated after it was written. Retrain to replace
it; the manifest is doing its job.

**"Could not load model … built with {sklearn: …}".**
The bundle was pickled by a different scikit-learn than the one loading it.
Retraining in the current environment fixes it. The manifest names both versions.

**My models vanished.**
Almost certainly an ephemeral filesystem. Set `PLACEMENT_AI_HOME` to a
persistent volume.

**Training is refused with "predicts categories".**
The chosen outcome column holds too many distinct values to be a category. Pick
the column with the small set of repeated answers, not a score or a salary.
