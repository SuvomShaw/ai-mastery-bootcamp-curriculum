"""
Nova Support Copilot — Streamlit front end.

Run from this folder:   streamlit run app.py

Layout:
  • Sidebar       — pick the customer, see their profile + long-term memory + status
  • Left column   — the chat
  • Right column  — "Agent Activity": the pipeline STREAMS here live, step by step
                    (triage → supervisor → specialist tool calls → review → memory)

The interesting bits:
  1. We use `graph.stream(..., stream_mode="updates")` so each node's output arrives
     as it finishes — we render the timeline as it happens, so you can watch the
     agents work instead of just seeing a final answer.
  2. Human-in-the-Loop: when a refund needs approval the graph PAUSES. Streamlit
     reruns top-to-bottom, so we stash the run in session_state, show an Approve/Deny
     card, and resume with Command(resume=...).
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config import ASSISTANT_NAME
from core import agents as agents_mod
from core import tools as tools_mod
from core.agents import ev
from core.backend import SupportBackend
from core.graph import build_graph
from core.knowledge_base import KnowledgeBase
from core.llm import health_check
from core.memory import MemoryStore
from ui import components as C
from ui.styles import CSS

st.set_page_config(page_title=ASSISTANT_NAME, page_icon="💬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ── App-wide singletons (built once, shared across reruns) ────────────────────
@st.cache_resource(show_spinner=False)
def bootstrap():
    """Create shared app objects once.

    Streamlit reruns this script after every widget interaction. Without
    `cache_resource`, the backend, memory store, knowledge base and graph would
    be rebuilt on every click.
    """
    backend = SupportBackend()
    kb = KnowledgeBase()
    store = MemoryStore()
    tools_mod.init(backend, kb)        # wire tools to the live backend + KB
    graph = build_graph(backend, store)
    return backend, kb, store, graph


backend, kb, store, graph = bootstrap()


@st.cache_data(show_spinner=False, ttl=15)
def cached_health_check():
    """Avoid probing Ollama on every Streamlit rerun/widget click."""
    return health_check()


# ── Per-customer conversation state ───────────────────────────────────────────
def reset_thread(cid: str):
    """Start a fresh LangGraph thread for one selected customer.

    A thread id is important because LangGraph's checkpointer uses it to keep
    short-term conversation state and to resume after an interrupt.
    """
    st.session_state.customer_id = cid
    st.session_state.thread_id = f"{cid}-{st.session_state.get('run', 0)}"
    st.session_state.history = []        # list of (role, text, meta)
    st.session_state.pending = None      # interrupt payload while awaiting approval
    st.session_state.run_input = None    # work to stream on the next rerun
    st.session_state.live_events = []     # events for the in-progress turn
    st.session_state.cur = {}            # triage/route/qa/reply captured while streaming
    st.session_state.active_user = ""    # the message currently being processed
    st.session_state.last_meta = None    # most recent finished turn (for the right panel)


if "customer_id" not in st.session_state:
    st.session_state.run = 0
    reset_thread(backend.list_customers()[0]["id"])


def thread_config():
    """Return the LangGraph config used by stream/resume calls."""
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Session")
    customers = backend.list_customers()
    labels = {c["id"]: f"{c['name']} · {c['plan']}" for c in customers}
    ids = list(labels)
    chosen = st.selectbox("Acting as customer", ids,
                          index=ids.index(st.session_state.customer_id),
                          format_func=lambda i: labels[i])
    if chosen != st.session_state.customer_id:
        reset_thread(chosen)
        st.rerun()

    C.profile_card(backend.get_account(st.session_state.customer_id))
    st.divider()
    C.memory_card(store.get_facts(st.session_state.customer_id))
    if st.button("🧹 Forget this customer's memory", use_container_width=True):
        store.clear(st.session_state.customer_id)
        st.rerun()

    st.divider()
    hc = cached_health_check()
    if hc["ok"]:
        st.success(f"Ollama ready · {hc['model']}")
    else:
        st.error("Ollama not reachable — run `ollama serve` and pull the model.")
        st.caption(hc.get("error", ""))

    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.run += 1
        reset_thread(st.session_state.customer_id)
        st.rerun()


# ── Two-column layout ─────────────────────────────────────────────────────────
col_chat, col_trace = st.columns([3, 2], gap="large")

with col_trace:
    st.markdown("#### 🤖 Agent Activity")
    st.caption("Watch the multi-agent pipeline run, step by step.")
    C.pipeline_graph()        # static architecture diagram (collapsible)
    trace_ph = st.empty()

with col_chat:
    C.header()
    st.caption("Ask about **billing**, a **technical** problem, or your **account**. "
               "The agents pick who handles it.")

    for role, text, meta in st.session_state.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)
            if role == "assistant" and meta:
                C.trace_expander(meta)

    # Refund approval gate (Human-in-the-Loop).
    if st.session_state.pending:
        with st.chat_message("assistant"):
            st.info("⏸️ This action needs your approval before I can continue.")
            decision = C.approval_card(st.session_state.pending)
        if decision:
            # Resume the paused graph. The next script rerun will pass
            # Command(resume=...) into graph.stream().
            st.session_state.run_input = {"mode": "resume", "decision": decision}
            st.session_state.pending = None
            st.rerun()

    # Normal input (hidden while a run is streaming or awaiting approval).
    if not st.session_state.run_input and not st.session_state.pending:
        prompt = st.chat_input("Type your message…")
        if prompt:
            # Queue work, then rerun. This keeps Streamlit rendering predictable:
            # first show the user's message, then the next pass streams the graph.
            st.session_state.history.append(("user", prompt, None))
            st.session_state.live_events = []
            st.session_state.cur = {}
            st.session_state.active_user = prompt
            st.session_state.run_input = {"mode": "start", "text": prompt}
            st.rerun()


# ── Streaming engine (runs when there is queued work) ─────────────────────────
def render_live():
    trace_ph.markdown(
        C.render_timeline(st.session_state.live_events, st.session_state.active_user),
        unsafe_allow_html=True,
    )


if st.session_state.run_input:
    work = st.session_state.run_input
    if work["mode"] == "start":
        # A normal turn starts with a HumanMessage plus the selected customer id.
        graph_input = {"messages": [HumanMessage(work["text"])],
                       "customer_id": st.session_state.customer_id}
    else:  # resume after the human approved/denied
        # A paused run resumes from the approval_node interrupt point.
        graph_input = Command(resume=work["decision"])

    events = st.session_state.live_events
    cur = st.session_state.cur
    paused = False

    # Register the live-push callback so run_specialist streams each tool-call
    # event to the UI the moment it happens, not after the whole node finishes.
    def _live_push(event: dict):
        events.append(event)
        render_live()

    live_push_token = agents_mod.set_live_push(_live_push)
    render_live()  # show whatever we have so far immediately
    try:
        for chunk in graph.stream(graph_input, thread_config(), stream_mode="updates"):
            if "__interrupt__" in chunk:
                # LangGraph returns an interrupt payload when approval_node pauses.
                # Store it so the next Streamlit render can show Approve/Deny.
                st.session_state.pending = chunk["__interrupt__"][0].value
                paused = True
                break
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                # Note: specialist events inside handle_node were already pushed
                # via _live_push above; update.get("trace") skips them (graph.py
                # returns an empty list when _live_push is set). All other nodes
                # (triage, supervise, review, remember) arrive here normally.
                events.extend(update.get("trace", []))
                for key in ("triage", "route", "qa", "reply"):
                    if key in update:
                        cur[key] = update[key]
            render_live()  # refresh after each node for non-specialist steps
    except Exception as exc:
        events.append(ev("System", "⚠️", "Something went wrong", str(exc)))
        cur["reply"] = f"⚠️ I hit an error talking to the model: {exc}"
        render_live()
    finally:
        agents_mod.reset_live_push(live_push_token)  # always clear, even on error

    st.session_state.run_input = None
    if paused:
        st.rerun()  # show the approval card; events are preserved for resume
    else:
        # Save the finished answer and its trace so students can inspect old runs
        # from the expander below each assistant message.
        meta = {"events": list(events), "triage": cur.get("triage"),
                "route": cur.get("route"), "qa": cur.get("qa")}
        st.session_state.history.append(
            ("assistant", cur.get("reply", "(no reply produced)"), meta))
        st.session_state.last_meta = meta
        st.session_state.live_events = []
        st.session_state.cur = {}
        st.rerun()
else:
    # Idle: keep a paused approval trace visible; otherwise show the last turn.
    last = st.session_state.get("last_meta")
    if st.session_state.get("pending") and st.session_state.get("live_events"):
        events = st.session_state.live_events
        title = st.session_state.get("active_user")
    else:
        events = last["events"] if last else []
        title = None
    trace_ph.markdown(
        C.render_timeline(events, title),
        unsafe_allow_html=True,
    )
