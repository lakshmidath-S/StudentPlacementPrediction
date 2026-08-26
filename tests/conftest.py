"""
Shared fixtures.

The synthetic frame here is deliberately awkward: it carries an identifier, a
constant column, free text, numbers stored as strings, a column with real gaps,
and a rare categorical level. Those are the shapes that break a pipeline built
against a clean dataset, so every fixture-driven test exercises them by default.

It is also small and strongly separable, so an end-to-end training run finishes
in a couple of seconds and can live in CI rather than in a nightly job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RANDOM_SEED = 7
N_ROWS = 320


PROVIDER_KEY_VARS = (
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "LLM_PROVIDER",
)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """Make the suite hermetic with respect to API keys.

    Without this, a developer with a working .env gets different results from
    CI: tests asserting "no provider is configured" pass on the build machine
    and fail on the laptop that actually has keys. Worse, a test could make a
    real billed call by accident.

    Both the environment and the .env loader are neutralised. A test that wants
    a key sets one itself with monkeypatch.setenv.
    """
    for name in PROVIDER_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("placement_ai.llm.registry._dotenv_problem", lambda: None)
    monkeypatch.setattr(
        "placement_ai.llm.registry._from_streamlit_secrets", lambda name: None
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def messy_frame() -> pd.DataFrame:
    """A small, deliberately imperfect dataset with a learnable signal."""
    generator = np.random.default_rng(RANDOM_SEED)

    score = generator.normal(65, 15, N_ROWS).clip(0, 100)
    attendance = generator.normal(80, 12, N_ROWS).clip(30, 100)
    projects = generator.integers(0, 6, N_ROWS)
    rating = generator.uniform(1, 5, N_ROWS).round(1)

    # A signal strong enough that any competent model clears 0.75 ROC-AUC,
    # so a test asserting on quality is testing the pipeline, not luck.
    logit = 0.06 * (score - 65) + 0.04 * (attendance - 80) + 0.5 * projects - 1.5
    probability = 1 / (1 + np.exp(-logit))
    placed = generator.random(N_ROWS) < probability

    frame = pd.DataFrame(
        {
            "Student ID": np.arange(1000, 1000 + N_ROWS),
            "Test Score": score.round(1),
            "Attendance %": attendance.round(1),
            "Projects": projects,
            "Soft Skills": rating,
            # Numbers wearing a string dtype, with thousands separators.
            "Stipend": [f"{int(v):,}" for v in generator.integers(0, 40000, N_ROWS)],
            "Department": generator.choice(
                ["CS", "EE", "ME", "ChemE"], N_ROWS, p=[0.5, 0.25, 0.2, 0.05]
            ),
            "Training": generator.choice(["Yes", "No"], N_ROWS),
            "Campus": "Main",  # constant
            "Notes": [f"free text row {i}" for i in range(N_ROWS)],
            "Outcome": np.where(placed, "Placed", "NotPlaced"),
        }
    )

    # Real gaps in one numeric column, to exercise imputation and is_missing.
    gaps = generator.choice(N_ROWS, size=N_ROWS // 10, replace=False)
    frame.loc[gaps, "Attendance %"] = np.nan
    return frame


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """Minimal, already-canonical frame for unit-level tests."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "c": [0.0, 1.0, 0.0, 2.0],
            "grade": ["Low", "High", "Low", "Medium"],
        }
    )


@pytest.fixture
def workspace_root(tmp_path, monkeypatch):
    """An isolated workspace root, so tests never touch the real one."""
    root = tmp_path / "workspaces"
    root.mkdir()
    monkeypatch.setenv("PLACEMENT_AI_HOME", str(root))
    return root


@pytest.fixture
def workspace(workspace_root):
    from placement_ai.registry.workspace import WorkspaceStore

    return WorkspaceStore(workspace_root).create("Test Org", "fixture workspace")


@pytest.fixture(scope="session")
def trained(messy_frame):
    """One offline training run, shared across the tests that need a model.

    Session-scoped because it is the single slowest thing in the suite and
    nothing that consumes it mutates the result.
    """
    from placement_ai.training.runner import TrainingRunner

    return TrainingRunner(provider=None).run(messy_frame, target_column="outcome")
