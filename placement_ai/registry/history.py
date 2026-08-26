"""
placement_ai/registry/history.py
--------------------------------
Per-workspace prediction history, in SQLite.

Every prediction the workspace makes is written here with the inputs that
produced it and the model version that made it. That record does three jobs:

  - it is the audit trail ("why was this student flagged in March?"),
  - it is how a workspace watches its own model drift away from the data it
    was trained on,
  - and once real outcomes are filled in, it is the training data for the next
    retrain.

Writes are best-effort. A logging failure must never lose a user their
prediction, so every write is wrapped and the error is surfaced as a flag rather
than an exception.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from placement_ai.utils import jsonify, utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    model_version     TEXT    NOT NULL,
    target_column     TEXT    NOT NULL DEFAULT '',
    source            TEXT    NOT NULL DEFAULT 'single',
    batch_id          TEXT,
    reference         TEXT,
    inputs            TEXT    NOT NULL,
    predicted_label   TEXT    NOT NULL,
    probability       REAL,
    probabilities     TEXT,
    actual_label      TEXT,
    actual_recorded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created  ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_version  ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_batch    ON predictions(batch_id);
"""


@dataclass
class HistoryEntry:
    model_version: str
    target_column: str
    inputs: dict[str, Any]
    predicted_label: str
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    source: str = "single"
    batch_id: str | None = None
    reference: str | None = None


class PredictionHistory:
    """SQLite-backed history for one workspace."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_error: str | None = None
        self._ensure_schema()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(SCHEMA)
        except sqlite3.Error as exc:
            self.last_error = f"Could not open the history database: {exc}"

    # ── writing ──────────────────────────────────────────────────────────
    def log(self, entry: HistoryEntry) -> int | None:
        return self.log_many([entry])

    def log_many(self, entries: list[HistoryEntry]) -> int | None:
        """Insert a batch. Returns the row count written, or None on failure."""
        if not entries:
            return 0
        now = utc_now_iso()
        rows = [
            (
                now,
                entry.model_version,
                entry.target_column,
                entry.source,
                entry.batch_id,
                entry.reference,
                json.dumps(jsonify(entry.inputs), ensure_ascii=False),
                entry.predicted_label,
                entry.probability,
                json.dumps(jsonify(entry.probabilities or {}), ensure_ascii=False),
            )
            for entry in entries
        ]
        try:
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO predictions
                        (created_at, model_version, target_column, source, batch_id,
                         reference, inputs, predicted_label, probability, probabilities)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.Error as exc:
            self.last_error = f"Could not write to the history database: {exc}"
            return None
        self.last_error = None
        return len(rows)

    def record_actual(self, row_id: int, actual_label: str) -> bool:
        """Attach the real outcome to a past prediction — the retraining fuel."""
        try:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE predictions SET actual_label = ?, actual_recorded_at = ? WHERE id = ?",
                    (str(actual_label), utc_now_iso(), int(row_id)),
                )
        except sqlite3.Error as exc:
            self.last_error = f"Could not record the outcome: {exc}"
            return False
        return True

    # ── reading ──────────────────────────────────────────────────────────
    def count(self, model_version: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM predictions"
        params: tuple = ()
        if model_version:
            query += " WHERE model_version = ?"
            params = (model_version,)
        try:
            with self._connect() as connection:
                return int(connection.execute(query, params).fetchone()[0])
        except sqlite3.Error:
            return 0

    def recent(self, limit: int = 200, model_version: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM predictions"
        params: list[Any] = []
        if model_version:
            query += " WHERE model_version = ?"
            params.append(model_version)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        try:
            with self._connect() as connection:
                rows = [dict(row) for row in connection.execute(query, params).fetchall()]
        except sqlite3.Error as exc:
            self.last_error = f"Could not read the history database: {exc}"
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["inputs"] = frame["inputs"].apply(_safe_json)
        frame["probabilities"] = frame["probabilities"].apply(_safe_json)
        return frame

    def expanded(self, limit: int = 2000, model_version: str | None = None) -> pd.DataFrame:
        """History with each input field as its own column.

        This is the shape the drift check and the CSV export both want; keeping
        the raw JSON in the table and flattening on read means a model retrained
        on different columns does not require a schema migration.
        """
        frame = self.recent(limit=limit, model_version=model_version)
        if frame.empty:
            return frame
        inputs = pd.json_normalize(frame["inputs"]).set_index(frame.index)
        meta = frame.drop(columns=["inputs", "probabilities"])
        return pd.concat([meta, inputs], axis=1)

    def labelled(self, limit: int = 100_000) -> pd.DataFrame:
        """Past predictions whose real outcome has since been recorded.

        Returned as inputs plus a column named after the target, so it can be
        concatenated straight onto the original training file for a retrain.
        """
        try:
            with self._connect() as connection:
                rows = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM predictions WHERE actual_label IS NOT NULL "
                        "ORDER BY id DESC LIMIT ?",
                        (int(limit),),
                    ).fetchall()
                ]
        except sqlite3.Error:
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        inputs = pd.json_normalize(frame["inputs"].apply(_safe_json))
        target_column = str(frame["target_column"].iloc[0] or "actual_label")
        inputs[target_column] = frame["actual_label"].to_numpy()
        return inputs

    def summary(self) -> dict[str, Any]:
        try:
            with self._connect() as connection:
                total = int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
                labelled = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM predictions WHERE actual_label IS NOT NULL"
                    ).fetchone()[0]
                )
                by_version = {
                    row["model_version"]: int(row["n"])
                    for row in connection.execute(
                        "SELECT model_version, COUNT(*) AS n FROM predictions "
                        "GROUP BY model_version ORDER BY n DESC"
                    ).fetchall()
                }
                by_label = {
                    str(row["predicted_label"]): int(row["n"])
                    for row in connection.execute(
                        "SELECT predicted_label, COUNT(*) AS n FROM predictions "
                        "GROUP BY predicted_label ORDER BY n DESC"
                    ).fetchall()
                }
                last = connection.execute(
                    "SELECT created_at FROM predictions ORDER BY id DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error as exc:
            self.last_error = f"Could not summarise the history: {exc}"
            return {"total": 0, "labelled": 0, "by_version": {}, "by_label": {}, "last": None}

        return {
            "total": total,
            "labelled": labelled,
            "by_version": by_version,
            "by_label": by_label,
            "last": last["created_at"] if last else None,
        }

    def clear(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM predictions")
        except sqlite3.Error as exc:
            self.last_error = f"Could not clear the history: {exc}"


def _safe_json(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
