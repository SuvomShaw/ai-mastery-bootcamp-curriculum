"""
Reference Pydantic schemas for the structured parts of the agent pipeline.

On a stronger/cloud model you would bind these with `llm.with_structured_output`.
This local course demo keeps them here as the clean production target, while
`core/agents.py` uses fast constrained-text parsing for better responsiveness on
small Ollama models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Triage(BaseModel):
    """Classification of the incoming customer message — shown in the UI."""
    category: Literal["billing", "technical", "account", "general"] = Field(
        description="the main topic of the message")
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        description="how time-sensitive this is for the customer")
    sentiment: Literal["happy", "neutral", "frustrated", "angry"] = Field(
        description="the customer's emotional tone")
    summary: str = Field(description="one short sentence summarising the request")


class Route(BaseModel):
    """The supervisor's decision: which specialist handles this turn."""
    specialist: Literal["billing", "technical", "account", "direct"] = Field(
        description="billing=payments/refunds/plans; technical=product issues; "
                    "account=login/seats/security; direct=greeting or small talk")
    reason: str = Field(description="one short phrase explaining the choice")


class QAReview(BaseModel):
    """Reflection step: a reviewer grades the drafted reply before it is sent."""
    approved: bool = Field(description="true if the reply is accurate, clear and on-tone")
    issues: str = Field(description="what to fix, or 'none'")
    improved_reply: str = Field(
        description="if not approved, a corrected reply; otherwise repeat the reply unchanged")


class MemoryExtraction(BaseModel):
    """Durable facts worth remembering about this customer across sessions."""
    facts: list[str] = Field(
        default_factory=list,
        description="short third-person facts, e.g. 'prefers email over phone'. "
                    "Empty if nothing durable was shared.")
