# Adaptive Placement Intelligence — Setup Guide

Complete step-by-step setup and installation instructions for running the **Adaptive Placement Intelligence** system locally, in Docker, or deploying to cloud environments.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quickstart (Zero-Configuration)](#quickstart-zero-configuration)
- [Detailed Local Installation](#detailed-local-installation)
  - [1. Clone / Open Repository](#1-clone--open-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
- [Environment Configuration (Optional AI Features)](#environment-configuration-optional-ai-features)
  - [Setting up API Keys](#setting-up-api-keys)
  - [Configuration Reference](#configuration-reference)
- [Running the Application](#running-the-application)
- [Verifying the Installation](#verifying-the-installation)
  - [Smoke Test](#smoke-test)
  - [Running Unit Tests](#running-unit-tests)
  - [Code Quality & Linting](#code-quality--linting)
- [First-Time User Walkthrough](#first-time-user-walkthrough)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## System Requirements

- **Python**: Version `3.11` or higher (`3.11`, `3.12` recommended)
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu/Debian/Fedora/Arch)
- **Memory (RAM)**: Minimum 2 GB RAM (400 MB for standard datasets up to 10k rows)
- **Disk Space**: ~500 MB for dependencies and virtual environment
- **Network**: Internet access for initial package downloads (and LLM API calls if enabled)

> [!NOTE]
> **No GPU is required.** All training pipelines (Logistic Regression, Random Forest, XGBoost, HistGradientBoosting) are optimized to run quickly on standard CPU cores.

---

## Quickstart (Zero-Configuration)

If you have Python 3.11+ installed, run:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Streamlit application
streamlit run app.py
```

The application will launch on `http://localhost:8501`.

> [!TIP]
> **No API key is required to use the system!** If no API key is provided, the platform automatically utilizes its rule-based heuristic planner, achieving over **0.88 ROC-AUC** on the bundled synthetic dataset.

---

## Detailed Local Installation

### 1. Clone / Open Repository

Open a terminal (PowerShell, Command Prompt, or Bash) in your project directory:

```bash
git clone https://github.com/lakshmidath-S/StudentPlacementPrediction.git
cd StudentPlacementPrediction
```

### 2. Create a Virtual Environment

It is recommended to isolate dependencies inside a dedicated Python virtual environment.

```bash
# Create a virtual environment named 'venv'
python -m venv venv
```

### 3. Activate the Virtual Environment

#### On Windows:

- **PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

- **Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

- **Git Bash:**
  ```bash
  source venv/Scripts/activate
  ```

#### On macOS / Linux:

```bash
source venv/bin/activate
```

*(Once activated, your terminal prompt will be prefixed with `(venv)`).*

---

### 4. Install Dependencies

Upgrade `pip` and install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### What this installs:
- **Data & Modeling**: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`, `xgboost`
- **Application & Visualization**: `streamlit`, `plotly`
- **File Handling & Validation**: `openpyxl`, `pydantic`
- **Networking & Config**: `requests`, `python-dotenv`
- **Testing & Quality**: `pytest`, `ruff`

---

## Environment Configuration (Optional AI Features)

The application supports multiple LLM providers (**Google Gemini**, **xAI Grok**, **OpenRouter**) to automatically reason over dataset profiles, clean features, and synthesize derived variables.

### Setting up API Keys

1. Copy the example `.env.example` file to `.env`:

   - **Windows (PowerShell):**
     ```powershell
     Copy-Item .env.example .env
     ```
   - **Linux / macOS:**
     ```bash
     cp .env.example .env
     ```

2. Open `.env` in any text editor and fill in your API key:

   ```ini
   # Google Gemini — Free key from https://aistudio.google.com/apikey
   GEMINI_API_KEY=your_gemini_api_key_here

   # xAI Grok — https://console.x.ai/
   XAI_API_KEY=

   # OpenRouter — https://openrouter.ai/keys
   OPENROUTER_API_KEY=

   # auto | gemini | grok | openrouter | off
   LLM_PROVIDER=auto
   ```

### Configuration Reference

All settings can be configured via `.env` or system environment variables:

| Environment Variable | Default Value | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(None)* | Google Gemini API key (Free tier available). |
| `OPENROUTER_API_KEY` | *(None)* | OpenRouter API key for open-source and proprietary models. |
| `XAI_API_KEY` | *(None)* | xAI Grok API key. |
| `LLM_PROVIDER` | `auto` | Provider selection: `auto`, `gemini`, `grok`, `openrouter`, or `off` (forces heuristic rules). |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model ID to use. |
| `OPENROUTER_MODEL` | `minimax/minimax-m3:free` | Model ID for OpenRouter. |
| `GROK_MODEL` | `grok-3-mini` | Model ID for Grok. |
| `PLACEMENT_AI_HOME` | `./workspaces` | Directory path where tenant data, models, and SQLite logs are stored. |
| `MAX_TRAINING_ROWS` | `250000` | Safety limit: datasets exceeding this are sampled with notification. |
| `MAX_SYNTHESIZED_FEATURES` | `40` | Maximum number of derived features synthesized during pipeline design. |
| `LLM_TIMEOUT_SECONDS` | `90` | Per-stage REST call timeout before falling back to heuristics. |

---

## Running the Application

### 1. Standard Launch

With your virtual environment activated:

```bash
streamlit run app.py
```

Open your browser at: **`http://localhost:8501`**

### 2. Custom Port or Network Binding

```bash
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

### 3. Force Rule-Based Offline Mode

If you want to ensure no LLM calls are made:

- **Windows (PowerShell):**
  ```powershell
  $env:LLM_PROVIDER="off"; streamlit run app.py
  ```
- **macOS / Linux:**
  ```bash
  LLM_PROVIDER=off streamlit run app.py
  ```

---

## Verifying the Installation

### Smoke Test

Run the automated end-to-end smoke test. This tests workspace creation, synthetic dataset ingestion, pipeline synthesis, model evaluation, persistence, single/batch inference, and drift calculation:

```bash
python scripts/smoke_test.py
```

*Expected output: `✓ Smoke test passed: ...`*

### Running Unit Tests

Run the full test suite (285 unit and integration tests, all offline without API keys):

```bash
pytest tests/ -q
```

To run a specific test file:
```bash
pytest tests/test_pipeline.py -v
```

### Code Quality & Linting

Run Ruff linter:

```bash
ruff check .
```

Run Pyright static type checker:

```bash
pyright --project pyrightconfig.json
```

---

## First-Time User Walkthrough

1. **Create or Open a Workspace**:
   - In the left sidebar, open the **Create** tab, enter an **Organisation name** (e.g., `Engineering Placement Cell`) and an optional description.
   - Click **Create workspace**. There is no access code or login — pick an existing workspace under **Open** to return to one.

2. **Train a Model**:
   - Navigate to the **Train** tab.
   - Click **Use sample data** (loads the bundled `10,000`-row synthetic placement dataset) or upload your custom `.csv` / `.xlsx`.
   - Select the target column representing placement outcome — `placement_status` in the bundled sample.
   - Click **Start training**.
   - Monitor the 11-stage pipeline in real time.

3. **Inspect the Model Card**:
   - View the champion model comparison. The planner picks its candidates from Logistic
     Regression, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting and XGBoost.
   - Check ROC-AUC, Precision, Recall, F1, confusion matrix, and feature importances.
   - Read the generated plain-language plan rationale and stage-by-stage provenance.

4. **Make Predictions & What-If Simulations**:
   - Switch to the **Predict** tab.
   - Fill out the generated student profile form or upload a batch CSV.
   - View predicted placement probabilities, key influencing factors (ablation attribution), and actionable improvement levers.

5. **History & Drift Tracking**:
   - Open the **History** tab for this workspace's past predictions.
   - The same tab reports Population Stability Index (PSI) drift of an uploaded batch
     against the baseline captured at training time.

---

## Docker Deployment

**The repository ships no Dockerfile.** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#docker) carries the
one to write — copy it into `Dockerfile` at the project root, then build and run:

```bash
docker build -t placement-ai .
docker run -p 8501:8501 -v placement-data:/data -e GEMINI_API_KEY="your_api_key_here" placement-ai
```

Access the service at `http://localhost:8501`.

> [!IMPORTANT]
> That Dockerfile sets `PLACEMENT_AI_HOME=/data` and declares `/data` a volume. Workspaces,
> trained models and prediction history all live there — mount it, or every model a user
> trains dies with the container.

---

## Troubleshooting & FAQs

### Q1: `Activate.ps1 cannot be loaded because running scripts is disabled` (Windows)
**Fix:** Run PowerShell as Administrator or execute in your current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Q2: Sidebar says `"Running on built-in rules"` even though `.env` is created
**Fix:**
1. Ensure `python-dotenv` is installed (`pip install python-dotenv`).
2. Verify the variable name is `GEMINI_API_KEY` (not `GOOGLE_API_KEY`).
3. Ensure `.env` is located in the root project folder next to `app.py`.
4. Restart the Streamlit server (`Ctrl+C` and `streamlit run app.py`).

### Q3: `Port 8501 is already in use`
**Fix:** Specify an alternative port:
```bash
streamlit run app.py --server.port 8502
```

### Q4: Model fails with `"predicts categories"` error
**Fix:** The chosen outcome column has too many distinct values or continuous floating-point numbers. Select a categorical target column (e.g. `0`/`1`, `Placed`/`Not Placed`).

### Q5: Where are models and datasets saved?
**Fix:** Everything is stored under `workspaces/` (or the directory specified by `PLACEMENT_AI_HOME`). Each workspace has its own isolated subfolder with datasets, manifests, `pipeline.joblib`, and a SQLite history database.

---

*For architecture details and design choices, refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*  
*For cloud hosting guidelines, refer to [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).*
