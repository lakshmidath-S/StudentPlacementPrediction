"""
scripts/smoke_test.py
---------------------
The hard gate: prove the whole loop works with no API key and no committed model.

The previous version of this repo smoke-tested three pre-trained bundles, which
only worked because the .joblib files were committed. Nothing is pre-trained now,
so the equivalent check is to *do the thing the product does* — train on the
bundled sample, save it, reload it from disk, and predict — end to end, in CI.

That makes the guarantee stronger than it was: it is no longer "the committed
artifacts still deserialise", it is "a first-time user with no key can reach a
working model".

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set before importing the package: config reads it at import time, and the run
# must never touch a real workspace directory.
_TEMP_HOME = tempfile.mkdtemp(prefix="placement-ai-smoke-")
os.environ["PLACEMENT_AI_HOME"] = _TEMP_HOME
os.environ["LLM_PROVIDER"] = "off"

import pandas as pd  # noqa: E402

from placement_ai.config import SAMPLE_DATASET_PATH  # noqa: E402
from placement_ai.inference.drift import check_drift  # noqa: E402
from placement_ai.inference.explain import attribute  # noqa: E402
from placement_ai.inference.predictor import WorkspacePredictor  # noqa: E402
from placement_ai.registry.history import HistoryEntry, PredictionHistory  # noqa: E402
from placement_ai.registry.model_store import ModelStore  # noqa: E402
from placement_ai.registry.workspace import WorkspaceStore  # noqa: E402
from placement_ai.training.runner import TrainingRunner  # noqa: E402

MIN_ROC_AUC = 0.70


def step(number: int, message: str) -> None:
    print(f"[{number}/7] {message}")


def main() -> int:
    started = time.perf_counter()
    print("=" * 68)
    print("Smoke test - full train/predict loop, no LLM provider")
    print("=" * 68)

    if not SAMPLE_DATASET_PATH.exists():
        print(f"[FAIL] Sample dataset missing: {SAMPLE_DATASET_PATH}")
        return 1

    step(1, "Creating a workspace")
    store = WorkspaceStore(Path(_TEMP_HOME))
    workspace = store.create("Smoke Test College")
    assert store.open(workspace.id).id == workspace.id
    print(f"      {workspace.id}")

    step(2, "Loading the bundled sample")
    frame = pd.read_csv(SAMPLE_DATASET_PATH)
    print(f"      {len(frame):,} rows x {frame.shape[1]} columns")

    step(3, "Training with the rule-based planner")
    outcome = TrainingRunner(provider=None).run(frame, target_column="placement_status")
    score = outcome.metrics.get("roc_auc")
    print(f"      champion: {outcome.champion_label}")
    print(f"      roc_auc : {score:.4f}  (threshold {outcome.threshold:.3f})")
    print(f"      features: {len(outcome.engineered_features)} derived, "
          f"{outcome.encoded_feature_count} model inputs")
    if score is None or score < MIN_ROC_AUC:
        print(f"[FAIL] ROC-AUC {score} is below the {MIN_ROC_AUC} floor.")
        return 1

    step(4, "Saving the bundle")
    bundle = ModelStore(workspace).save(outcome, dataset_name=SAMPLE_DATASET_PATH.name)
    size_kb = bundle.pipeline_path.stat().st_size / 1024
    print(f"      {bundle.version} ({size_kb:.0f} KB)")

    step(5, "Reloading from disk and predicting")
    # A fresh ModelStore, so this genuinely reads the manifest and verifies the
    # checksum rather than reusing the in-memory object.
    reloaded = ModelStore(store.get(workspace.id)).active()
    predictor = WorkspacePredictor(reloaded)
    prediction = predictor.predict_one(predictor.defaults)
    print(f"      typical record -> {prediction.label} "
          f"({prediction.probability:.3f}, {prediction.confidence_band})")
    if prediction.label not in predictor.class_labels:
        print("[FAIL] Prediction returned a label the model does not know.")
        return 1

    step(6, "Explaining and batch scoring")
    strong = {**predictor.defaults, "cgpa": 9.5, "aptitude_test_score": 95}
    drivers = attribute(predictor, strong, top_n=3)
    for driver in drivers:
        print(f"      {driver['label']}: {driver['delta']:+.4f}")
    if not drivers:
        print("[FAIL] Attribution produced nothing for an atypical record.")
        return 1

    scored_frame, check = predictor.check_frame(frame.head(200))
    scored = predictor.predict_frame(scored_frame)
    print(f"      scored {len(scored):,} records, {len(check.missing)} column(s) imputed")

    step(7, "History and drift")
    history = PredictionHistory(workspace.history_path)
    written = history.log_many(
        [
            HistoryEntry(
                model_version=bundle.version,
                target_column=bundle.target_column,
                inputs=row,
                predicted_label=str(label),
                probability=float(probability),
                source="batch",
                batch_id="smoke",
            )
            for row, label, probability in zip(  # noqa: B905
                scored[predictor.expected_columns].to_dict("records"),
                scored["prediction"].tolist(),
                scored["probability"].tolist(),
            )
        ]
    )
    report = check_drift(bundle.drift_baseline, history.expanded(), bundle.input_schema)
    print(f"      logged {written:,} predictions; drift status: {report.status}")
    if report.status not in {"ok", "watch"}:
        # Scoring a sample of the training data must not look like drift.
        print(f"[FAIL] Unexpected drift status on training-like data: {report.status}")
        return 1

    print("=" * 68)
    print(f"[OK] Full loop passed in {time.perf_counter() - started:.1f}s")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        import shutil

        shutil.rmtree(_TEMP_HOME, ignore_errors=True)
    sys.exit(code)
