"""
Mock business backend (the "company systems" the agents act on).

In production these methods would call your real billing API, auth service and
ticketing system. Here they read a JSON seed file and mutate an in-memory copy,
so the demo behaves like a live system (refunds applied, plans changed, seats
updated) without touching anything real. Restarting the app resets the world.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from copy import deepcopy
from datetime import date

from config import CUSTOMERS_PATH

PLAN_PRICES = {"Free": 0, "Pro": 75, "Business": 400}
PLAN_LOOKUP = {plan.lower(): plan for plan in PLAN_PRICES}
VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
MAX_SEATS = 500


class SupportBackend:
    def __init__(self, path: str = CUSTOMERS_PATH):
        with open(path, encoding="utf-8") as f:
            seed = json.load(f)
        # Work on a copy so students can freely click around without modifying
        # the source data in data/customers.json.
        self._customers: dict = deepcopy(seed)
        self._tickets: list[dict] = []

    # ── Read helpers ──────────────────────────────────────────────────────────
    def _account(self, cid: str) -> dict:
        if cid not in self._customers:
            raise ValueError(f"Unknown customer id '{cid}'")
        return self._customers[cid]

    def list_customers(self) -> list[dict]:
        return [self._customers[cid] for cid in sorted(self._customers)]

    def get_account(self, cid: str) -> dict:
        return self._account(cid)

    def get_subscription(self, cid: str) -> dict:
        c = self._account(cid)
        return {"plan": c["plan"], "seats": c["seats"], "mrr": c["mrr"],
                "payment_method": c["payment_method"], "since": c["since"]}

    def list_invoices(self, cid: str) -> list[dict]:
        return self._account(cid)["invoices"]

    # ── Write actions (these change state) ────────────────────────────────────
    def apply_refund(self, cid: str, amount: float, reason: str) -> dict:
        # Backend methods validate inputs because tools are model-facing. Never
        # trust model-generated arguments just because they came through a schema.
        amount = float(amount)
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Refund amount must be greater than zero.")
        reason = str(reason).strip() or "No reason provided"
        inv = {"id": f"REF-{uuid.uuid4().hex[:4].upper()}",
               "date": date.today().isoformat(), "amount": -amount,
               "status": "refunded", "reason": reason}
        self._account(cid)["invoices"].insert(0, inv)
        return inv

    def change_plan(self, cid: str, new_plan: str) -> dict:
        # Accept natural phrases like "business plan", but normalize them before
        # touching account state.
        requested = re.sub(r"\s+plan$", "", str(new_plan).strip(), flags=re.I).lower()
        plan = PLAN_LOOKUP.get(requested)
        if plan is None:
            raise ValueError(f"Unknown plan '{new_plan}'")
        account = self._account(cid)
        if plan == "Free" and account["seats"] > 1:
            account["seats"] = 1
        account["plan"] = plan
        account["mrr"] = PLAN_PRICES[plan]
        return {"plan": plan, "mrr": PLAN_PRICES[plan], "seats": account["seats"]}

    def update_seats(self, cid: str, seats: int) -> dict:
        account = self._account(cid)
        try:
            seats = int(seats)
        except (TypeError, ValueError):
            raise ValueError("Seat count must be a whole number.") from None
        if seats < 1 or seats > MAX_SEATS:
            raise ValueError(f"Seat count must be between 1 and {MAX_SEATS}.")
        if account["plan"] == "Free" and seats > 1:
            raise ValueError("Free plan supports only 1 seat; upgrade before adding seats.")
        account["seats"] = seats
        return {"seats": seats}

    def reset_password(self, cid: str) -> dict:
        self._account(cid)
        return {"reset_link": f"https://app.cloudnova.io/reset/{uuid.uuid4().hex[:12]}",
                "expires_in": "30 minutes"}

    def set_two_factor(self, cid: str, enabled: bool) -> dict:
        account = self._account(cid)
        account["two_factor"] = bool(enabled)
        return {"two_factor": account["two_factor"]}

    def service_status(self) -> str:
        # A simulated status page. "Realtime Sync" is degraded on purpose so the
        # technical agent has a realistic, grounded incident to reference.
        return ("CloudNova status — "
                "Web app: operational | API: operational | "
                "Realtime Sync: DEGRADED (incident INC-204, fix in progress) | "
                "Billing: operational")

    def create_ticket(self, cid: str, summary: str, priority: str = "normal") -> dict:
        # Ticket creation changes both the ticket list and the customer's open
        # ticket counter, giving students an easy state mutation to inspect.
        self._account(cid)
        summary = str(summary).strip()
        if not summary:
            raise ValueError("Ticket summary cannot be empty.")
        priority = str(priority).strip().lower()
        if priority not in VALID_PRIORITIES:
            priority = "normal"
        ticket = {"id": f"TK-{uuid.uuid4().hex[:5].upper()}", "customer": cid,
                  "summary": summary, "priority": priority, "status": "open"}
        self._tickets.append(ticket)
        self._account(cid)["open_tickets"] += 1
        return ticket
