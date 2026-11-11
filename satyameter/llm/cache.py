"""SQLite response cache. Saves free-tier quota and makes experiments repeatable.

Thread-safe: every operation opens its own short-lived connection, because Streamlit
runs each interaction on a different thread and SQLite connections cannot be shared.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from satyameter.llm.base import LLMClient


class CachedLLM(LLMClient):
    def __init__(self, inner: LLMClient, db_path: Path):
        self.inner = inner
        self.name = inner.name
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, model TEXT, response TEXT, created_at TEXT)"
            )
        self.hits = 0
        self.misses = 0

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30)

    @staticmethod
    def _key(name: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = json.dumps([name, system, user, temperature, max_tokens])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        key = self._key(self.inner.name, system, user, temperature, max_tokens)
        with closing(self._connect()) as db:
            row = db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        response = self.inner.generate(
            system, user, temperature=temperature, max_tokens=max_tokens
        )
        if response:  # never cache empty replies
            with closing(self._connect()) as db, db:
                db.execute(
                    "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
                    (key, self.inner.name, response, datetime.now(timezone.utc).isoformat()),
                )
        return response
