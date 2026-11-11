"""Small SQLite history store for completed Satyameter runs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteStore:
    """Persist run summaries and claim-level evidence locally."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    rewritten_query TEXT,
                    mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    faithfulness REAL,
                    contradiction_rate REAL,
                    coverage REAL,
                    latency_s REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    claim TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entailment REAL,
                    contradiction REAL,
                    neutral REAL,
                    evidence_title TEXT,
                    evidence_url TEXT,
                    evidence_text TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_claims_run_id ON claims(run_id);
                """
            )

    def save_run(self, answer, verification, mode: str) -> int:
        """Save a completed run and return its generated run ID."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            cursor = db.execute(
                """INSERT INTO runs (
                    created_at, question, rewritten_query, mode, model, answer,
                    faithfulness, contradiction_rate, coverage, latency_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    created_at,
                    answer.question,
                    getattr(answer, "rewritten_query", None),
                    mode,
                    answer.model,
                    answer.answer,
                    verification.faithfulness,
                    verification.contradiction_rate,
                    verification.coverage,
                    verification.latency_s,
                ),
            )
            run_id = int(cursor.lastrowid)
            for verdict in verification.verdicts:
                best = verdict.best
                scores = best.scores if best else None
                chunk = best.chunk if best else None
                db.execute(
                    """INSERT INTO claims (
                        run_id, claim, status, entailment, contradiction, neutral,
                        evidence_title, evidence_url, evidence_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        verdict.claim,
                        verdict.status.value,
                        scores.entailment if scores else None,
                        scores.contradiction if scores else None,
                        scores.neutral if scores else None,
                        f"{chunk.doc_title} — {chunk.section}" if chunk else None,
                        chunk.url if chunk else None,
                        chunk.text if chunk else None,
                    ),
                )
        return run_id

    def recent_runs(self, limit: int = 10) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def claims_for_run(self, run_id: int) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM claims WHERE run_id = ? ORDER BY claim_id", (int(run_id),)
            ).fetchall()
        return [dict(row) for row in rows]
