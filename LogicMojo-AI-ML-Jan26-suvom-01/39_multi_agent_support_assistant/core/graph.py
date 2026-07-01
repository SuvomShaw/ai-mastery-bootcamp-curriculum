"""
The orchestration graph (LangGraph) — this is the spine of the whole app.

Flow for every customer message:

    START
      │
   recall_memory      load durable facts about this customer (Long-term Memory)
      │
    triage            classify category / priority / sentiment (Structured Output)
      │
   supervise          pick the right specialist (Multi-Agent Supervisor)
      │
   ┌──┴── direct?  ── direct_reply ──┐      small talk needs no specialist
   │                                 │
  handle ── refund? ── approval ─────┤      ReAct specialist; refunds over the
   │  (Human-in-the-Loop interrupt)  │      limit pause for a human to approve
   └──────────────┬──────────────────┘
                review               critique & polish the reply (Reflection)
                  │
               remember              save any new facts (Long-term Memory)
                  │
                 END

The graph is compiled with a checkpointer so the Human-in-the-Loop `interrupt`
can pause mid-run and resume later with the human's decision.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from config import REFUND_AUTO_APPROVE_LIMIT
from core import agents
from core import tools as tools_mod
from core.agents import ev


# ── Shared state passed between nodes ─────────────────────────────────────────
class SupportState(TypedDict):
    # `add_messages` appends new messages instead of replacing the list. This is
    # how the graph keeps short-term chat history inside one thread.
    messages: Annotated[list, add_messages]      # full conversation
    customer_id: str
    triage: dict
    route: dict
    reply: str
    qa: dict
    memory_facts: list
    pending_action: Optional[dict]               # a refund waiting for approval
    trace: Annotated[list, operator.add]         # human-readable run log


def build_graph(backend, store):
    """Wire the nodes into a compiled graph. `backend` and `store` are injected
    so the graph has no global state and is easy to test."""

    def recall_memory(state: SupportState):
        # Long-term memory is loaded before triage so every downstream agent can
        # use saved customer preferences or context.
        facts = store.get_facts(state["customer_id"])
        detail = "; ".join(facts) if facts else "no prior facts on file"
        return {"memory_facts": facts,
                "trace": [ev("Memory", "🧠", f"Recalled {len(facts)} fact(s)", detail)]}

    def triage_node(state: SupportState):
        t = agents.triage(state["messages"])
        return {"triage": t,
                "trace": [ev("Triage", "🏷️",
                             f"{t['category']} · {t['priority']} · {t['sentiment']}",
                             t["summary"])]}

    def supervise_node(state: SupportState):
        r = agents.supervise(state["triage"], state["messages"])
        return {"route": r,
                "trace": [ev("Supervisor", "🧭", f"Routed to → {r['specialist']}", r["reason"])]}

    def route_decider(state: SupportState):
        return "direct" if state["route"]["specialist"] == "direct" else "handle"

    def handle_node(state: SupportState):
        cid = state["customer_id"]
        specialist = state["route"]["specialist"]
        profile = backend.get_account(cid)

        # Scope every tool call to THIS customer for the duration of the node.
        # The model never receives or invents a customer id for tool calls.
        token = tools_mod.current_customer.set(cid)
        try:
            res = agents.run_specialist(specialist, state["messages"], profile,
                                        state["memory_facts"])
        finally:
            tools_mod.current_customer.reset(token)

        # If a live-push callback was active, specialist events were already streamed
        # to the UI one-by-one; skip them here to avoid duplicates in the trace list.
        trace = [] if agents.live_push_active() else list(res["events"])
        pending = None
        proposal = res["proposal"]

        # Refund policy lives HERE, not in the tool — one place, can't be bypassed.
        if proposal:
            amt, reason = proposal["amount"], proposal["reason"]
            if amt <= REFUND_AUTO_APPROVE_LIMIT:
                backend.apply_refund(cid, amt, reason)
                trace.append(ev("Refund Policy", "⚖️", f"Auto-approved ${amt:.2f}",
                                f"within the ${REFUND_AUTO_APPROVE_LIMIT:.0f} auto-approve limit"))
            else:
                pending = {"amount": amt, "reason": reason, "customer": profile["name"]}
                trace.append(ev("Refund Policy", "⚖️", f"${amt:.2f} needs human approval",
                                f"exceeds the ${REFUND_AUTO_APPROVE_LIMIT:.0f} limit — pausing"))

        return {"reply": res["reply"], "pending_action": pending, "trace": trace}

    def pending_decider(state: SupportState):
        return "approval" if state.get("pending_action") else "review"

    def approval_node(state: SupportState):
        # ⏸ Human-in-the-Loop: the graph pauses here until the UI resumes it.
        action = state["pending_action"]
        # `interrupt()` returns control to Streamlit. When the user clicks
        # Approve/Deny, app.py resumes this exact node with Command(resume=...).
        decision = interrupt({
            "type": "refund_approval",
            "amount": action["amount"],
            "reason": action["reason"],
            "customer": action["customer"],
        })
        approved = str(decision).strip().lower().startswith("y")
        if approved:
            backend.apply_refund(state["customer_id"], action["amount"], action["reason"])
            note = (f"\n\n✅ Good news — a refund of ${action['amount']:.2f} has been "
                    f"approved and processed. It should appear in 3–5 business days.")
            trace = [ev("Human Approval", "✅", f"APPROVED refund ${action['amount']:.2f}",
                        "a human agent approved the action")]
        else:
            note = (f"\n\nI checked with a manager and I'm unable to approve a refund of "
                    f"${action['amount']:.2f} right now. I've noted your request and a "
                    f"specialist will follow up with options.")
            trace = [ev("Human Approval", "🚫", f"DENIED refund ${action['amount']:.2f}",
                        "a human agent declined the action")]
        return {"reply": state["reply"] + note, "pending_action": None, "trace": trace}

    def direct_node(state: SupportState):
        profile = backend.get_account(state["customer_id"])
        reply = agents.direct_reply(state["messages"], profile, state["memory_facts"])
        return {"reply": reply,
                "trace": [ev("Assistant", "💬", "Replied directly", "no specialist needed")]}

    def review_node(state: SupportState):
        triage = state["triage"]
        # Reflection is a cost — only run it when the customer is unhappy or it's
        # high priority. Routine replies are sent as-is.
        if not agents.should_review(triage):
            qa = {"approved": True, "issues": "routine reply; review skipped",
                  "final": state["reply"]}
            return {"qa": qa, "messages": [AIMessage(state["reply"])],
                    "trace": [ev("Quality Reviewer", "⏭️", "Skipped review",
                                 "routine request — no extra reflection pass needed")]}
        qa = agents.review(state["reply"], triage)
        verdict = "Approved the draft" if qa["approved"] else "Revised the draft"
        detail = "reply met the quality bar" if qa["approved"] else qa.get("issues", "")
        # Commit the final reply to the message history here.
        return {"reply": qa["final"], "qa": qa,
                "messages": [AIMessage(qa["final"])],
                "trace": [ev("Quality Reviewer", "🔎", verdict, detail)]}

    def remember_node(state: SupportState):
        cid = state["customer_id"]
        # Memory extraction runs after the reply so one turn can update durable
        # customer facts for future conversations.
        facts = agents.extract_facts(state["messages"])
        added = [f for f in facts if store.add_fact(cid, f)]
        detail = "; ".join(added) if added else "nothing new worth remembering"
        return {"memory_facts": store.get_facts(cid),
                "trace": [ev("Memory", "💾", f"Saved {len(added)} new fact(s)", detail)]}

    g = StateGraph(SupportState)
    g.add_node("recall_memory", recall_memory)
    g.add_node("triage", triage_node)
    g.add_node("supervise", supervise_node)
    g.add_node("handle", handle_node)
    g.add_node("approval", approval_node)
    g.add_node("direct", direct_node)
    g.add_node("review", review_node)
    g.add_node("remember", remember_node)

    # The section below is the graph topology. Read it like a flowchart:
    # node -> next node, or conditional node -> possible branches.
    g.add_edge(START, "recall_memory")
    g.add_edge("recall_memory", "triage")
    g.add_edge("triage", "supervise")
    g.add_conditional_edges("supervise", route_decider,
                            {"handle": "handle", "direct": "direct"})
    g.add_conditional_edges("handle", pending_decider,
                            {"approval": "approval", "review": "review"})
    g.add_edge("approval", "review")
    g.add_edge("direct", "review")
    g.add_edge("review", "remember")
    g.add_edge("remember", END)

    # Checkpointer is REQUIRED for the interrupt()/resume in approval_node.
    return g.compile(checkpointer=MemorySaver())
