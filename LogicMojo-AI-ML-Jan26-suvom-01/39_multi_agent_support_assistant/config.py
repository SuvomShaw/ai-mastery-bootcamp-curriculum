"""
Central configuration for the Nova Support Copilot.

Everything tunable lives here so the rest of the code never hard-codes a model
name, file path, or business rule. In a real deployment these would come from
environment variables / a secrets manager — we read a few from os.environ and
fall back to sensible local defaults.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.json")
KB_PATH = os.path.join(DATA_DIR, "knowledge_base.md")
MEMORY_PATH = os.path.join(DATA_DIR, "memory_store.json")

# ── Models (local, via Ollama) ────────────────────────────────────────────────
# The chat model must support native tool/function calling — qwen3.5 does.
MODEL = os.environ.get("NOVA_MODEL", "qwen3.5:4b")
# Tried in order if MODEL is not installed locally.
MODEL_FALLBACKS = ["qwen3.5:9b", "qwen2.5:7b", "qwen2.5:3b", "llama3.1:8b"]
# Embedding model for knowledge-base retrieval (Agentic RAG).
EMBED_MODEL = os.environ.get("NOVA_EMBED_MODEL", "qwen3-embedding:0.6b")
NUM_CTX = 8192

# ── Business rules (the kind of policy that belongs in ONE place) ──────────────
# Refunds at or below this amount auto-approve; anything above pauses for a human.
REFUND_AUTO_APPROVE_LIMIT = float(os.environ.get("NOVA_REFUND_LIMIT", "50"))
# Safety cap so a specialist's tool loop can never run forever.
MAX_TOOL_STEPS = 4

# ── Branding / copy ───────────────────────────────────────────────────────────
COMPANY = "CloudNova"
PRODUCT = "CloudNova — a project & task management SaaS"
ASSISTANT_NAME = "Nova Support Copilot"
