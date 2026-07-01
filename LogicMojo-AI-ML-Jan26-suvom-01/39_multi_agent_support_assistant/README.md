# Nova Support Copilot

Nova Support Copilot is a reference implementation of a multi-agent customer
support assistant. It is built for students who want to understand how a real
agentic application is assembled from small, testable parts.

The app runs locally with Streamlit, LangGraph, LangChain tool calling, and
Ollama. It demonstrates routing, specialist agents, ReAct-style tool use,
retrieval augmented generation, long-term memory, reflection, and a
human-in-the-loop approval step.

For a longer lecture-style walkthrough, also see `readme_2.md`. This README is
the main setup and student reference guide.

## What Students Should Learn

After studying this project, you should be able to explain:

- How one user message moves through a LangGraph workflow.
- How a supervisor routes work to specialist agents.
- How ReAct agents decide when to call tools.
- Why tools should receive intent arguments, not hidden account IDs.
- How RAG grounds technical answers in a small help-center knowledge base.
- How long-term memory is different from chat history.
- How `interrupt()` pauses a graph for human approval.
- How to test business rules separately from the LLM.

## Application Flow

Every customer message goes through this pipeline:

```text
START
  -> recall_memory
  -> triage
  -> supervise
  -> specialist handler OR direct reply
  -> optional refund approval pause
  -> review
  -> remember
END
```

The right-side Agent Activity panel shows this pipeline live while the app runs.
Students can use it to connect the UI behavior to the code in `core/graph.py`.

## Main Concepts

| Concept | File | What to inspect |
|---|---|---|
| Streamlit app state | `app.py` | `st.session_state`, streaming, resume after approval |
| Graph orchestration | `core/graph.py` | nodes, edges, conditional routing, `interrupt()` |
| Agent logic | `core/agents.py` | triage, supervisor, specialist ReAct loop, reflection |
| Tools | `core/tools.py` | function calling, customer scoping, tool registry |
| Mock backend | `core/backend.py` | account state, refunds, tickets, plan changes |
| RAG | `core/knowledge_base.py` | markdown article loading, embeddings, keyword fallback |
| Long-term memory | `core/memory.py` | JSON-backed durable facts |
| Model access | `core/llm.py` | Ollama model resolution and health check |
| UI components | `ui/components.py` | cards, trace rendering, approval UI |
| Tests | `tests/test_core.py` | backend policy, memory, tools, RAG fallback |

## Folder Structure

```text
39_multi_agent_support_assistant/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── core/
│   ├── agents.py
│   ├── backend.py
│   ├── graph.py
│   ├── knowledge_base.py
│   ├── llm.py
│   ├── memory.py
│   ├── schemas.py
│   └── tools.py
├── data/
│   ├── customers.json
│   └── knowledge_base.md
├── tests/
│   └── test_core.py
└── ui/
    ├── components.py
    └── styles.py
```

## Setup

### 1. Install Ollama

Install Ollama from:

```text
https://ollama.com
```

Start Ollama:

```bash
ollama serve
```

In another terminal, pull the recommended models:

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

If your machine is slower, the app can still start, but answers may take time.
If the embedding model is missing, the knowledge base automatically falls back
to keyword search.

### 2. Install Python Dependencies

From this folder:

```bash
cd 39_multi_agent_support_assistant
python -m pip install -r requirements.txt
```

Using a virtual environment is recommended for students:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

Streamlit usually opens:

```text
http://localhost:8501
```

## Quick Verification

Run these before making changes:

```bash
python -m compileall app.py core ui tests
python -m unittest discover -s tests
```

Expected result:

```text
Ran 10 tests
OK
```

If optional LLM packages are missing, one agent-helper test may be skipped, but
the backend, memory, tool, and RAG tests should still run.

## Demo Prompts

Open the app, choose a customer in the sidebar, and try these prompts.

### Billing Agent

```text
What plan am I on and what am I paying?
```

Expected behavior:

- Supervisor routes to Billing Agent.
- Agent calls `get_subscription`.
- Final answer includes plan, seats, and monthly cost.

```text
I was double-charged $30, can I get a refund?
```

Expected behavior:

- Billing Agent calls `request_refund`.
- Graph auto-approves because amount is below the refund limit.

```text
I want a $400 refund for the failed charge.
```

Expected behavior:

- Billing Agent requests refund.
- Graph pauses at the human approval node.
- UI shows Approve and Deny buttons.

### Technical Agent

```text
My board is not syncing. What should I do?
```

Expected behavior:

- Supervisor routes to Technical Agent.
- Agent calls `search_help_center`.
- Answer is grounded in `data/knowledge_base.md`.

```text
Is the sync service down right now?
```

Expected behavior:

- Technical Agent may call `check_service_status`.
- Response mentions the simulated Realtime Sync incident.

### Account Agent

```text
I forgot my password. Can you reset it?
```

Expected behavior:

- Supervisor routes to Account Agent.
- Agent calls `send_password_reset`.

```text
Turn on two-factor authentication.
```

Expected behavior:

- Account Agent calls `set_two_factor`.
- Backend state changes in memory for the current app run.

### Long-Term Memory

```text
I prefer email updates instead of phone calls.
```

Expected behavior:

- Memory extraction stores a durable fact.
- Click New conversation and ask:

```text
What do you remember about me?
```

The assistant should use the saved memory fact.

## How to Study the Code

Use this order when reading the project:

1. Start with `app.py`.
   Understand how Streamlit stores chat history, pending approval state, and
   live trace events.

2. Read `core/graph.py`.
   This is the backbone. Every node returns a partial state update. Conditional
   edges decide whether to call a specialist, reply directly, or pause for
   approval.

3. Read `core/agents.py`.
   Notice that triage and supervisor are small LLM calls, while specialists use
   a loop: model decides, tool runs, observation goes back to model.

4. Read `core/tools.py`.
   Tools never accept a customer ID from the model. The graph sets the current
   customer through a `ContextVar`, which prevents the model from accidentally
   acting on the wrong account.

5. Read `core/backend.py`.
   This is a fake business system. It behaves like an API but only mutates an
   in-memory copy of `data/customers.json`.

6. Read `core/knowledge_base.py`.
   The technical agent uses this for RAG. Embeddings are preferred, keyword
   search is the fallback.

7. Read `tests/test_core.py`.
   The tests avoid calling the LLM. They verify deterministic business behavior
   such as refund validation, seat rules, memory persistence, and tool errors.

## Important Design Decisions

### 1. The graph owns policy

The refund tool only records the agent's intent. It does not move money. The
graph applies the rule:

```text
amount <= NOVA_REFUND_LIMIT -> auto approve
amount > NOVA_REFUND_LIMIT  -> pause for human approval
```

This is safer because business policy is centralized in one deterministic place.

### 2. Tools are scoped to one customer

The model never passes `customer_id` into tools. The graph sets the customer
context before running a specialist:

```python
token = tools_mod.current_customer.set(cid)
```

Each tool reads that scoped value. This prevents accidental cross-account
actions caused by a hallucinated customer ID.

### 3. Reflection is conditional

The quality reviewer does not run for every message. It runs only for angry,
frustrated, high-priority, or urgent requests. This keeps routine turns faster
while still showing the reflection pattern.

### 4. RAG has a fallback

If Ollama embeddings are unavailable, the help-center search falls back to a
simple keyword ranker. This keeps the demo usable even on machines where the
embedding model has not been pulled yet.

### 5. Tests focus on deterministic code

LLM outputs can vary. The unit tests focus on code paths that should never vary:
refund amount validation, plan and seat rules, ticket creation, memory
persistence, and tool error messages.

## Configuration

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `NOVA_MODEL` | `qwen3.5:4b` | Ollama chat model |
| `NOVA_EMBED_MODEL` | `qwen3-embedding:0.6b` | Ollama embedding model |
| `NOVA_REFUND_LIMIT` | `50` | Refunds at or below this amount auto-approve |

Examples:

```bash
NOVA_REFUND_LIMIT=100 streamlit run app.py
NOVA_MODEL=qwen3.5:9b streamlit run app.py
```

## Troubleshooting

### App opens but says Ollama is not reachable

Start Ollama:

```bash
ollama serve
```

Then confirm the model exists:

```bash
ollama list
ollama pull qwen3.5:4b
```

### Technical answers say no relevant article found

Check `data/knowledge_base.md`. The simple fallback search works best when the
customer's words overlap with article titles or body text.

### Streamlit does not refresh after code edits

This project disables Streamlit's file watcher in `.streamlit/config.toml` to
avoid noisy dependency scanning in heavy ML environments. Stop and restart:

```bash
streamlit run app.py
```

### A test fails after you change business rules

Update the tests only if the rule intentionally changed. For example, changing
the maximum number of seats should be reflected in `tests/test_core.py`.

## Suggested Student Exercises

1. Add a new tool named `export_data`.
2. Add a Security Agent for suspicious login or 2FA issues.
3. Replace the keyword fallback with FAISS or Chroma.
4. Save backend changes to a JSON file instead of memory only.
5. Add a SQLite LangGraph checkpointer so approval pauses survive app restarts.
6. Extend the supervisor so one message can call two specialists.
7. Add tests for invalid plan names and ticket creation.

## Submission Checklist For Students

Before submitting your modified version:

- App starts with `streamlit run app.py`.
- `python -m unittest discover -s tests` passes.
- README explains what you changed.
- New business rules have tests.
- No API keys or secrets are committed.
- `data/memory_store.json` is not submitted unless your instructor asks for it.

## License And Use

This project is intended as a classroom reference for the LogicMojo AI/ML course.
Students are encouraged to run it, modify it, add tests, and explain each part in
their own words.
