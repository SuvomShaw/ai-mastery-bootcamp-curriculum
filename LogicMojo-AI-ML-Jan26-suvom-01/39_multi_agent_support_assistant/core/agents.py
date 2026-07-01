"""
The reasoning units: triage, supervisor, specialists, reflection, memory.

Each function here is small and does ONE job, so the graph in graph.py reads
like a flowchart. The patterns from the course map directly onto these:

  • triage / route / review / extract_facts → Structured Output
  • run_specialist                          → ReAct (reason → act → observe loop)
  • technical specialist's search tool      → Agentic RAG
  • review                                  → Reflection (self-critique)
"""

from __future__ import annotations

import contextvars
import math
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import MAX_TOOL_STEPS, COMPANY, PRODUCT
from core import tools as tools_mod
from core.llm import get_llm

# Why no `with_structured_output` here:
# this qwen3.5 build only follows a JSON schema when its slow "thinking" phase is
# on (~50s/call) — with reasoning off it's fast (~5s) but ignores the schema. For a
# responsive demo we instead use fast reasoning-off calls and parse a tiny, well-
# constrained text format with safe fallbacks. (core/schemas.py keeps the strict
# Pydantic schemas as the reference approach you'd use on a stronger/cloud model.)

# Set by app.py before each graph.stream() call so run_specialist can push
# individual events to the UI in real time instead of waiting for the node to finish.
# ContextVar keeps parallel Streamlit sessions from sharing a callback.
_live_push = contextvars.ContextVar("live_push", default=None)

CATEGORIES = ("billing", "technical", "account", "general")
PRIORITIES = ("low", "normal", "high", "urgent")
SENTIMENTS = ("happy", "neutral", "frustrated", "angry")
SPECIALISTS = ("billing", "technical", "account", "direct")


def _chat(prompt: str) -> str:
    """One fast, plain (reasoning-off) LLM call returning trimmed text."""
    return (get_llm(reasoning=False).invoke(prompt).content or "").strip()


def _pick(text: str, options, default: str) -> str:
    """Return the first option that appears in `text` (case-insensitive).

    Local models sometimes answer with extra words. This helper gives us a
    small, reliable parser for short classification decisions.
    """
    low = text.lower()
    for opt in options:
        if opt in low:
            return opt
    return default


# ── System prompts for each specialist (their "job description") ──────────────
SPECIALIST_PROMPTS = {
    "billing": (
        "You are the Billing specialist for {company}. The customer is {name} on the "
        "{plan} plan. Help with invoices, payments, plan changes and refunds. "
        "Always check real data with your tools before answering.\n"
        "REFUND RULE (important): when the customer asks for a refund, you MUST call the "
        "request_refund tool with the exact amount and a clear reason. Calling request_refund "
        "is REQUIRED to issue any refund — never tell the customer a refund is done unless you "
        "have actually called request_refund. You may call list_invoices first to confirm the "
        "amount, but you must then go on to call request_refund. Do NOT stop after only "
        "checking invoices. Be concise and warm."),
    "technical": (
        "You are the Technical Support specialist for {product}. The customer is {name}. "
        "For any product question you MUST call search_help_center first and base your "
        "answer on what it returns — do not guess. If the issue is not covered or needs "
        "engineering, call open_ticket. Give clear, step-by-step help."),
    "account": (
        "You are the Account specialist for {company}. The customer is {name} on the "
        "{plan} plan. Help with login, password resets, seats and security/2FA, using "
        "your tools to actually perform the change. Confirm what you did in plain words."),
}

DIRECT_PROMPT = (
    "You are {assistant} for {company}. The customer is {name}. "
    "This message is small talk or a greeting — reply briefly and warmly, and invite "
    "them to share what they need help with. Known notes about them: {memory}.")


# Display name + icon for each specialist, used in the UI trace timeline.
AGENT_META = {
    "billing": ("Billing Agent", "💳"),
    "technical": ("Technical Agent", "🛠️"),
    "account": ("Account Agent", "🔐"),
}


def ev(agent: str, icon: str, title: str, detail: str = "") -> dict:
    """Build one structured trace event (one row in the live Agent Activity panel)."""
    detail = str(detail)
    if len(detail) > 300:
        detail = detail[:300] + "…"
    return {"agent": agent, "icon": icon, "title": title, "detail": detail}


def set_live_push(callback):
    return _live_push.set(callback)


def reset_live_push(token) -> None:
    _live_push.reset(token)


def live_push_active() -> bool:
    return _live_push.get() is not None


def _ctx(profile: dict, memory: list[str]) -> dict:
    from config import ASSISTANT_NAME
    return {
        "company": COMPANY, "product": PRODUCT, "assistant": ASSISTANT_NAME,
        "name": profile["name"], "plan": profile["plan"],
        "memory": "; ".join(memory) if memory else "none",
    }


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _coerce_positive_amount(value) -> float:
    """Tool-call arguments come from a model, so normalize common money strings."""
    if isinstance(value, str):
        value = re.sub(r"[$,\s]", "", value)
    try:
        amount = float(value)
    except (TypeError, ValueError):
        raise ValueError("Refund amount must be a number.") from None
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Refund amount must be greater than zero.")
    return amount


# ── Triage: classify the message (category / priority / sentiment) ────────────
def triage(messages: list) -> dict:
    last = messages[-1].content
    # The prompt asks for a tiny pipe-separated format. This is easier for a
    # small local model than strict JSON, and the fallback parser keeps the app
    # moving even if the model includes extra prose.
    raw = _chat(
        "Classify this customer support message. Reply on ONE line, exactly in the form:\n"
        "category | priority | sentiment | short summary\n"
        f"category ∈ {{{', '.join(CATEGORIES)}}}\n"
        f"priority ∈ {{{', '.join(PRIORITIES)}}}\n"
        f"sentiment ∈ {{{', '.join(SENTIMENTS)}}}\n\n"
        f"Message: {last}")
    line = next((l for l in raw.splitlines() if "|" in l), raw)
    parts = [p.strip() for p in line.split("|")]
    summary = parts[3] if len(parts) > 3 and parts[3] else last[:80]
    return {
        "category": _pick(parts[0] if parts else "", CATEGORIES, "general"),
        "priority": _pick(parts[1] if len(parts) > 1 else "", PRIORITIES, "normal"),
        "sentiment": _pick(parts[2] if len(parts) > 2 else "", SENTIMENTS, "neutral"),
        "summary": summary[:120],
    }


# ── Supervisor: route to a specialist (one-word decision) ─────────────────────
def supervise(triage_info: dict, messages: list) -> dict:
    last = messages[-1].content
    # The supervisor is intentionally simple: it chooses exactly one route.
    # Students can extend this to multi-hop routing as an exercise.
    raw = _chat(
        "You are a support supervisor. Pick ONE specialist and reply with ONLY that "
        "one word: billing, technical, account, or direct.\n"
        "- billing   : payments, invoices, refunds, plan/pricing changes\n"
        "- technical : product not working, how-to, bugs, integrations\n"
        "- account   : login, password, seats/members, security, 2FA\n"
        "- direct    : greeting or small talk that needs no specialist\n\n"
        f"Message: {last}")
    # Fall back to the triaged category if the word is unclear.
    fallback = triage_info["category"] if triage_info["category"] in SPECIALISTS else "direct"
    specialist = _pick(raw, SPECIALISTS, fallback)
    reason = raw.replace("\n", " ").strip()[:80] or f"matches {specialist}"
    return {"specialist": specialist, "reason": reason}


# ── Specialist: a bounded ReAct loop (reason → act → observe) ─────────────────
def run_specialist(specialist: str, messages: list, profile: dict, memory: list[str]) -> dict:
    """Run one specialist as a bounded ReAct loop.

    Returns the reply text, a list of structured trace `events` (reasoning, each
    tool call, each observation — these drive the live UI), and any refund the
    agent proposed (so the graph can apply refund policy)."""
    name, icon = AGENT_META[specialist]
    toolset = tools_mod.TOOLSETS[specialist]
    system = SystemMessage(SPECIALIST_PROMPTS[specialist].format(**_ctx(profile, memory)))
    llm = get_llm(reasoning=False).bind_tools(toolset)

    convo = [system] + messages
    events: list[dict] = []
    proposal = None

    def _emit(e: dict):
        """Append event and, if a live-push callback is registered, stream it immediately."""
        events.append(e)
        callback = _live_push.get()
        if callback is not None:
            callback(e)

    _emit(ev(name, icon, "Took over the request",
             f"tools available: {', '.join(t.name for t in toolset)}"))

    for _ in range(MAX_TOOL_STEPS):
        # ReAct step 1: ask the model what to do next. Because tools are bound to
        # the LLM, the model can either answer directly or request tool calls.
        ai = llm.invoke(convo)
        convo.append(ai)

        if not getattr(ai, "tool_calls", None):
            # No tool wanted → this content is the final answer.
            return {"reply": ai.content, "events": events, "proposal": proposal}

        if (ai.content or "").strip():
            _emit(ev(name, "💭", "Reasoning", ai.content.strip()))

        for tc in ai.tool_calls:
            tname, args = tc["name"], tc.get("args", {})
            _emit(ev(name, "🔧", f"Calling tool · {tname}", _fmt_args(args) or "no arguments"))
            if tname == "request_refund":
                # Capture intent only; the graph decides whether to execute it.
                # This separation is the main safety lesson in the billing path.
                try:
                    args = dict(args)
                    args["amount"] = _coerce_positive_amount(args.get("amount"))
                    proposal = {"amount": args["amount"],
                                "reason": str(args.get("reason") or "no reason given")}
                except ValueError as exc:
                    output = f"Could not request refund: {exc}"
                    _emit(ev(name, "👁️", "Observation", output))
                    convo.append(ToolMessage(content=output, tool_call_id=tc["id"]))
                    continue
            tool = tools_mod.BY_NAME.get(tname)
            # ReAct step 2: execute the tool and feed the observation back into
            # the conversation so the model can produce a grounded final answer.
            output = tool.invoke(args) if tool else f"Unknown tool: {tname}"
            _emit(ev(name, "👁️", "Observation", output))
            convo.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

    # Hit the step cap — ask for a clean closing reply instead of looping forever.
    _emit(ev(name, "🧷", "Reached tool-step limit", "wrapping up with a summary"))
    final = get_llm(reasoning=False).invoke(
        convo + [HumanMessage("Summarise the resolution for the customer in 2-3 friendly sentences.")])
    return {"reply": final.content, "events": events, "proposal": proposal}


# ── Direct reply for greetings / small talk ───────────────────────────────────
def direct_reply(messages: list, profile: dict, memory: list[str]) -> str:
    system = SystemMessage(DIRECT_PROMPT.format(**_ctx(profile, memory)))
    return get_llm(reasoning=False).invoke([system] + messages).content


# ── Reflection: review the draft before it is sent (self-critique) ────────────
def should_review(triage_info: dict) -> bool:
    """Only spend a reflection pass when it matters — an unhappy or high-priority
    customer. Routine replies skip it (a realistic 'spend compute where it counts')."""
    return (triage_info.get("sentiment") in ("frustrated", "angry")
            or triage_info.get("priority") in ("high", "urgent"))


def review(reply: str, triage_info: dict) -> dict:
    raw = _chat(
        "You are a support quality reviewer. If the draft reply is accurate, clear and "
        f"appropriately empathetic for a {triage_info['sentiment']} customer, reply with "
        "exactly the single word APPROVED. Otherwise, reply with ONLY an improved version "
        "of the message (no preamble).\n\n"
        f"Draft reply:\n{reply}")
    if raw.upper().startswith("APPROVED") or len(raw) < 15:
        return {"approved": True, "issues": "met the quality bar", "final": reply}
    return {"approved": False, "issues": "tightened tone & clarity", "final": raw}


# ── Memory: extract durable facts to remember ─────────────────────────────────
def extract_facts(messages: list) -> list[str]:
    # Look only at the customer's latest message — that's where new facts appear.
    user_turns = [m.content for m in messages if isinstance(m, HumanMessage)]
    if not user_turns:
        return []
    raw = _chat(
        "From the customer's message, list any DURABLE facts worth remembering for "
        "future conversations (preferences, context, recurring issues) — one per line, "
        "no bullets. Ignore one-off questions and pleasantries. If there is nothing "
        "durable, reply with exactly: NONE.\n\n"
        f"Customer said: {user_turns[-1]}")
    if raw.strip().upper().startswith("NONE"):
        return []
    facts = [l.strip(" -•\t") for l in raw.splitlines() if l.strip()]
    return [f for f in facts if 3 < len(f) < 160 and f.upper() != "NONE"][:4]
