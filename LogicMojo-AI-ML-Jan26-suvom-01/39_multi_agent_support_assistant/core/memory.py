"""
Long-term memory (the Store pattern), persisted to disk.

Short-term memory = the message history of one conversation (handled by
LangGraph's checkpointer). Long-term memory = durable FACTS about a customer
that must survive a brand-new conversation tomorrow. We keep those in a small
JSON file keyed by customer id, so they persist across app restarts — closer to
the Redis/Postgres-backed store you'd use in production.
"""

from __future__ import annotations

import json
import os

from config import MEMORY_PATH


class MemoryStore:
    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self._data: dict[str, list[str]] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        # Write to a temporary file first, then atomically replace the real file.
        # This avoids half-written JSON if the app is stopped during a save.
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def get_facts(self, customer_id: str) -> list[str]:
        return list(self._data.get(customer_id, []))

    def add_fact(self, customer_id: str, fact: str) -> bool:
        """Add a fact unless we already know something very similar. Returns True if added."""
        fact = fact.strip()
        if not fact:
            return False
        existing = self._data.setdefault(customer_id, [])
        low = fact.lower()
        for e in existing:
            # A simple substring check is enough for this classroom demo. In a
            # production system, you might use embeddings or normalized facts.
            if low in e.lower() or e.lower() in low:  # cheap dedupe
                return False
        existing.append(fact)
        self._save()
        return True

    def clear(self, customer_id: str) -> None:
        self._data.pop(customer_id, None)
        self._save()
