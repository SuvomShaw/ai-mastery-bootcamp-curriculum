"""
Tools the specialist agents can call (the "Act" in ReAct).

Design choice worth explaining: a tool should only receive *intent* arguments
(an amount, a new plan, a search query) — never the customer id. We scope the
"who am I helping" through a ContextVar set once per turn, so the model can't
target the wrong account by hallucinating an id. Each tool reads `current_customer`
and calls the backend.

`request_refund` is special: it does NOT move money. It only records the intent.
The refund POLICY (auto-approve small amounts, send large ones to a human) lives
in the graph, in one place — so the agent can never bypass it.
"""

from __future__ import annotations

import contextvars

from langchain_core.tools import tool

# Set per request by the graph's handle node. Tools read it; the model never sees it.
current_customer: contextvars.ContextVar[str] = contextvars.ContextVar("current_customer")

# Injected once at startup via init().
_backend = None
_kb = None


def init(backend, knowledge_base) -> None:
    """Inject live dependencies at app startup.

    Keeping tools thin makes them easy to test and easy to replace with real
    services later.
    """
    global _backend, _kb
    _backend = backend
    _kb = knowledge_base


def _require_backend():
    # Fail fast with a clear message if a student imports and calls a tool
    # without first running app bootstrap or tools.init(...).
    if _backend is None:
        raise RuntimeError("Support tools are not initialized.")
    return _backend


def _require_kb():
    if _kb is None:
        raise RuntimeError("Knowledge-base tools are not initialized.")
    return _kb


def _cid() -> str:
    # The graph sets this ContextVar once per specialist run. The model cannot
    # change it, which protects tools from acting on a hallucinated account id.
    cid = current_customer.get(None)
    if cid is None:
        raise RuntimeError("No customer is scoped for this tool call.")
    return cid


def _as_bool(value) -> bool:
    # Tool-call arguments are model-generated. Normalize common strings so
    # "false" does not become True just because non-empty strings are truthy.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if low in {"false", "no", "n", "off", "disable", "disabled"}:
            return False
    return bool(value)


# ── Billing tools ─────────────────────────────────────────────────────────────
@tool
def get_subscription() -> str:
    """Get the current plan, seat count, monthly cost and payment method for the customer."""
    s = _require_backend().get_subscription(_cid())
    return (f"Plan: {s['plan']} | Seats: {s['seats']} | MRR: ${s['mrr']} | "
            f"Payment: {s['payment_method']} | Customer since: {s['since']}")


@tool
def list_invoices() -> str:
    """List the customer's recent invoices with status (paid/failed/refunded)."""
    inv = _require_backend().list_invoices(_cid())
    if not inv:
        return "No invoices on file (likely a Free plan)."
    return "\n".join(f"{i['id']} | {i['date']} | ${i['amount']} | {i['status']}" for i in inv)


@tool
def request_refund(amount: float, reason: str) -> str:
    """Request a refund of `amount` dollars for the stated `reason`. This does not
    process the refund directly — it is checked against refund policy first."""
    # This tool returns a confirmation of intent only. The graph applies the
    # real refund policy and decides whether money is moved.
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("amount must be greater than zero")
    except (TypeError, ValueError) as exc:
        return f"Could not request refund: {exc}"
    reason = str(reason).strip() or "No reason provided"
    return (f"Refund of ${amount:.2f} recorded for policy review (reason: {reason}). "
            f"It will be auto-approved if within policy, otherwise sent for human approval.")


@tool
def change_plan(new_plan: str) -> str:
    """Change the customer's plan. Valid values: 'Free', 'Pro', 'Business'."""
    try:
        r = _require_backend().change_plan(_cid(), new_plan)
        return f"Plan changed to {r['plan']} (${r['mrr']}/mo, {r['seats']} seat(s))."
    except ValueError as e:
        return f"Could not change plan: {e}"


# ── Technical tools ───────────────────────────────────────────────────────────
@tool
def search_help_center(query: str) -> str:
    """Search the CloudNova help center for how-to and troubleshooting articles.
    Use this for any product/technical question to ground the answer in real docs."""
    return _require_kb().search_text(query, k=2)


@tool
def open_ticket(summary: str, priority: str = "normal") -> str:
    """Escalate an unresolved technical issue by opening a support ticket."""
    try:
        t = _require_backend().create_ticket(_cid(), summary, priority)
        return f"Opened ticket {t['id']} (priority: {t['priority']})."
    except ValueError as e:
        return f"Could not open ticket: {e}"


@tool
def check_service_status() -> str:
    """Check the live operational status of CloudNova's services (web, API, sync,
    billing). Use this when a customer reports something is down or not working."""
    return _require_backend().service_status()


# ── Account tools ─────────────────────────────────────────────────────────────
@tool
def send_password_reset() -> str:
    """Email the customer a one-time password reset link."""
    r = _require_backend().reset_password(_cid())
    return f"Password reset link sent (valid {r['expires_in']})."


@tool
def update_seats(seats: int) -> str:
    """Set the number of paid seats on the customer's workspace."""
    try:
        r = _require_backend().update_seats(_cid(), seats)
        return f"Seats updated to {r['seats']}."
    except ValueError as e:
        return f"Could not update seats: {e}"


@tool
def set_two_factor(enabled: bool) -> str:
    """Enable or disable two-factor authentication for the customer."""
    r = _require_backend().set_two_factor(_cid(), _as_bool(enabled))
    return f"Two-factor authentication is now {'ON' if r['two_factor'] else 'OFF'}."


# ── Registry: which tools each specialist is allowed to use ───────────────────
# This allow-list is a practical safety boundary: each specialist only sees the
# tools relevant to its job.
TOOLSETS = {
    "billing": [get_subscription, list_invoices, request_refund, change_plan],
    "technical": [search_help_center, check_service_status, open_ticket],
    "account": [get_subscription, send_password_reset, update_seats, set_two_factor],
}

# Flat lookup name -> tool, for the agent loop to execute a chosen tool.
# LangChain gives tool calls by name; the agent loop uses this map to run them.
BY_NAME = {t.name: t for ts in TOOLSETS.values() for t in ts}
