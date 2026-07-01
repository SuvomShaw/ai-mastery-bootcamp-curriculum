from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from core import tools
from core.backend import MAX_SEATS, SupportBackend
from core.knowledge_base import KnowledgeBase
from core.memory import MemoryStore

try:
    from core import agents
except ModuleNotFoundError as exc:
    agents = None
    AGENTS_IMPORT_ERROR = exc
else:
    AGENTS_IMPORT_ERROR = None


class SupportBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = SupportBackend()

    def test_refund_uses_today_and_rejects_invalid_amounts(self) -> None:
        refund = self.backend.apply_refund("CUST-1001", 30, "duplicate charge")

        self.assertEqual(refund["date"], date.today().isoformat())
        self.assertEqual(refund["amount"], -30)
        self.assertEqual(self.backend.list_invoices("CUST-1001")[0]["id"], refund["id"])

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.backend.apply_refund("CUST-1001", 0, "bad amount")

    def test_plan_names_are_normalized_and_free_plan_limits_seats(self) -> None:
        upgraded = self.backend.change_plan("CUST-1001", "business plan")
        self.assertEqual(upgraded["plan"], "Business")
        self.assertEqual(upgraded["mrr"], 400)

        downgraded = self.backend.change_plan("CUST-1001", "FREE PLAN")
        self.assertEqual(downgraded["plan"], "Free")
        self.assertEqual(downgraded["seats"], 1)
        self.assertEqual(self.backend.get_subscription("CUST-1001")["seats"], 1)

    def test_update_seats_validates_bounds_and_free_plan(self) -> None:
        self.assertEqual(self.backend.update_seats("CUST-1001", "8")["seats"], 8)

        with self.assertRaisesRegex(ValueError, "between 1"):
            self.backend.update_seats("CUST-1001", 0)
        with self.assertRaisesRegex(ValueError, "between 1"):
            self.backend.update_seats("CUST-1001", MAX_SEATS + 1)
        with self.assertRaisesRegex(ValueError, "Free plan"):
            self.backend.update_seats("CUST-1003", 2)

    def test_ticket_priority_is_normalized(self) -> None:
        before = self.backend.get_account("CUST-1001")["open_tickets"]
        ticket = self.backend.create_ticket("CUST-1001", "  Sync still failing  ", "severe")

        self.assertEqual(ticket["summary"], "Sync still failing")
        self.assertEqual(ticket["priority"], "normal")
        self.assertEqual(self.backend.get_account("CUST-1001")["open_tickets"], before + 1)


class KnowledgeBaseTest(unittest.TestCase):
    def test_keyword_fallback_ranks_relevant_articles_without_embeddings(self) -> None:
        kb = KnowledgeBase()
        kb._ensure_embeddings = lambda: False

        hits = kb.search("websocket board frozen sync", k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["title"], "Why my board is not syncing")

    def test_keyword_fallback_ignores_empty_stopword_queries(self) -> None:
        kb = KnowledgeBase()
        kb._ensure_embeddings = lambda: False

        self.assertEqual(kb.search("what is the and to", k=2), [])


class MemoryStoreTest(unittest.TestCase):
    def test_memory_persists_and_dedupes_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "memory.json"
            store = MemoryStore(str(path))

            self.assertTrue(store.add_fact("CUST-1", "Prefers email updates."))
            self.assertFalse(store.add_fact("CUST-1", "Prefers email updates."))

            reloaded = MemoryStore(str(path))
            self.assertEqual(reloaded.get_facts("CUST-1"), ["Prefers email updates."])


class AgentUtilityTest(unittest.TestCase):
    @unittest.skipIf(agents is None, f"agents import unavailable: {AGENTS_IMPORT_ERROR}")
    def test_refund_amount_coercion_accepts_money_strings(self) -> None:
        self.assertEqual(agents._coerce_positive_amount("$1,200.50"), 1200.5)

        with self.assertRaisesRegex(ValueError, "number"):
            agents._coerce_positive_amount("thirty dollars")
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            agents._coerce_positive_amount("-5")


class ToolLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = SupportBackend()
        tools.init(self.backend, KnowledgeBase())
        self.token = tools.current_customer.set("CUST-1003")

    def tearDown(self) -> None:
        tools.current_customer.reset(self.token)

    def test_tool_returns_friendly_validation_errors(self) -> None:
        message = tools.update_seats.invoke({"seats": 2})
        self.assertIn("Could not update seats", message)
        self.assertIn("Free plan supports only 1 seat", message)

    def test_tool_parses_string_booleans(self) -> None:
        tools.current_customer.reset(self.token)
        self.token = tools.current_customer.set("CUST-1002")

        message = tools.set_two_factor.invoke({"enabled": "false"})

        self.assertIn("OFF", message)
        self.assertFalse(self.backend.get_account("CUST-1002")["two_factor"])


if __name__ == "__main__":
    unittest.main()
