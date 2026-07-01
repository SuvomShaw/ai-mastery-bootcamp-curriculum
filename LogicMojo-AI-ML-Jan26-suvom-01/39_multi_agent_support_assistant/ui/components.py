"""
Reusable Streamlit render helpers.

Keeping the markup here keeps app.py focused on flow/state, not HTML strings.
"""

from __future__ import annotations

import html

import streamlit as st

from config import ASSISTANT_NAME, PRODUCT


def header() -> None:
    st.markdown(
        f"""<div class="nova-header">
              <div class="logo">N</div>
              <div>
                <div class="nova-title">{ASSISTANT_NAME}</div>
                <div class="nova-sub">{PRODUCT}</div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def _row(label: str, value: str) -> str:
    return f'<div class="profile-row"><span>{label}</span><span>{html.escape(str(value))}</span></div>'


def profile_card(profile: dict) -> None:
    st.markdown("**Customer**")
    st.markdown(
        '<div class="card">'
        + _row("Name", profile["name"])
        + _row("Email", profile["email"])
        + _row("Plan", f"{profile['plan']} · {profile['seats']} seat(s)")
        + _row("MRR", f"${profile['mrr']}/mo")
        + _row("Customer since", profile["since"])
        + _row("Open tickets", profile["open_tickets"])
        + "</div>",
        unsafe_allow_html=True,
    )


def memory_card(facts: list[str]) -> None:
    st.markdown("**What I remember**")
    if not facts:
        st.caption("Nothing yet — facts are saved as the conversation goes on.")
        return
    st.markdown(
        "".join(f'<div class="memory-pill">• {html.escape(f)}</div>' for f in facts),
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{html.escape(text)}</span>'


def triage_badges(triage: dict) -> str:
    if not triage:
        return ""
    return (_badge(triage.get("category", "?"), triage.get("category", "general"))
            + _badge(triage.get("priority", "?"), triage.get("priority", "normal"))
            + _badge(triage.get("sentiment", "?"), triage.get("sentiment", "neutral")))


def render_timeline(events: list[dict], title: str | None = None) -> str:
    """Build the Agent Activity timeline HTML from structured trace events."""
    if not events:
        return ('<div class="tl-empty">Send a message and the agents\' work — '
                'triage, routing, tool calls, observations, review — will stream here.</div>')
    rows = []
    if title:
        rows.append(f'<div class="tl-title-bar">Pipeline for: “{html.escape(title)}”</div>')
    for e in events:
        detail = (f'<div class="tl-detail">{html.escape(e["detail"])}</div>'
                  if e.get("detail") else "")
        rows.append(
            f'<div class="tl-item"><div class="tl-dot">{e.get("icon","•")}</div>'
            f'<div class="tl-body">'
            f'<div class="tl-agent">{html.escape(e.get("agent",""))}</div>'
            f'<div class="tl-head">{html.escape(e.get("title",""))}</div>'
            f'{detail}</div></div>'
        )
    return '<div class="tl">' + "".join(rows) + "</div>"


def trace_expander(meta: dict) -> None:
    """Collapsible per-message trace, so older turns stay inspectable in the chat."""
    n = len(meta.get("events", []))
    with st.expander(f"🔎 Agent trace for this reply · {n} steps"):
        if meta.get("triage"):
            st.markdown(triage_badges(meta["triage"]), unsafe_allow_html=True)
        st.markdown(render_timeline(meta.get("events", [])), unsafe_allow_html=True)


# Static picture of the LangGraph pipeline — the same nodes/edges wired in
# core/graph.py. Rendered with st.graphviz_chart (in-browser, no system deps).
PIPELINE_DOT = """
digraph Pipeline {
  rankdir=TB;
  bgcolor="transparent";
  pad=0.2;
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11
        color="#5b6cff" fillcolor="#eef0ff" fontcolor="#1f2330" margin="0.16,0.08"];
  edge [color="#9aa0b5" fontname="Helvetica" fontsize=9 fontcolor="#6b7280" arrowsize=0.8];

  start    [label="START" shape=circle fillcolor="#5b6cff" fontcolor="white" width=0.35 fixedsize=true];
  recall   [label="recall_memory\\n(long-term memory)"];
  triage   [label="triage\\n(classify)"];
  supervise[label="supervise\\n(route to specialist)"];
  handle   [label="handle\\n(ReAct + tools)" fillcolor="#eafaf0" color="#0a7d3a"];
  approval [label="approval  ⏸\\n(human-in-the-loop)" fillcolor="#fffaf0" color="#f0b429"];
  direct   [label="direct\\n(small talk)"];
  review   [label="review\\n(reflection)"];
  remember [label="remember\\n(save facts)"];
  stop     [label="END" shape=circle fillcolor="#1f2330" fontcolor="white" width=0.35 fixedsize=true];

  start -> recall -> triage -> supervise;
  supervise -> handle  [label="specialist"];
  supervise -> direct  [label="small talk"];
  handle    -> approval[label="refund > limit"];
  handle    -> review  [label="else"];
  approval  -> review;
  direct    -> review;
  review    -> remember -> stop;
}
"""


def pipeline_graph() -> None:
    """Show the static pipeline diagram (the architecture, before it runs live)."""
    with st.expander("🗺️ Pipeline graph (the full route every message takes)"):
        st.graphviz_chart(PIPELINE_DOT, use_container_width=True)
        st.caption("Green = ReAct specialist with tools · amber = human approval gate. "
                   "Watch it run, step by step, below.")


def approval_card(payload: dict) -> str | None:
    """Render the Human-in-the-Loop refund gate. Returns 'yes'/'no'/None."""
    st.markdown('<div class="approve-card">', unsafe_allow_html=True)
    st.markdown(f"### ⏸️ Approval needed — refund ${payload['amount']:.2f}")
    st.write(f"**Customer:** {payload['customer']}")
    st.write(f"**Reason:** {payload['reason']}")
    st.caption("This exceeds the auto-approve limit, so a human must decide.")
    c1, c2, _ = st.columns([1, 1, 3])
    decision = None
    if c1.button("✅ Approve", type="primary", use_container_width=True):
        decision = "yes"
    if c2.button("🚫 Deny", use_container_width=True):
        decision = "no"
    st.markdown("</div>", unsafe_allow_html=True)
    return decision
