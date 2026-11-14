"""Append-only SQLite store for analyst labels (one row per labelling event)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id      TEXT NOT NULL,
    label       TEXT NOT NULL,
    analyst_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_txn ON labels (txn_id);
"""


class LabelStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, txn_id: str, label: str, analyst_id: str) -> dict:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO labels (txn_id, label, analyst_id, created_at) VALUES (?, ?, ?, ?)",
                (txn_id, label, analyst_id, created_at))
        return {"id": cur.lastrowid, "txn_id": txn_id, "label": label,
                "analyst_id": analyst_id, "created_at": created_at}

    def labelled_txn_ids(self) -> set[str]:
        with closing(self._connect()) as conn:
            return {row[0] for row in conn.execute("SELECT DISTINCT txn_id FROM labels")}
