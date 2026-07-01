"""
LLM factory + health check.

One place that knows how to talk to Ollama. Every agent asks `get_llm()` for a
fresh chat model so we never scatter model names or connection settings around.
The model name is resolved ONCE against what is actually installed locally, with
graceful fallbacks, so the app still starts if the preferred model is missing.
"""

from __future__ import annotations

import ollama
from langchain_ollama import ChatOllama

from config import MODEL, MODEL_FALLBACKS, NUM_CTX

_resolved_model: str | None = None


def _installed_models() -> list[str]:
    """Return the list of model tags Ollama has pulled locally (best effort)."""
    try:
        listing = ollama.list()
        # The client returns objects with a `.model` attribute; older versions
        # return dicts. Handle both so we are version-tolerant.
        models = getattr(listing, "models", None) or listing.get("models", [])
        out = []
        for m in models:
            out.append(getattr(m, "model", None) or m.get("model"))
        return [m for m in out if m]
    except Exception:
        return []


def resolve_model() -> str:
    """Pick the first of [MODEL, *fallbacks] that is installed; cache the choice."""
    global _resolved_model
    if _resolved_model:
        return _resolved_model

    available = _installed_models()
    # This lets students change machines without editing code. If the preferred
    # model is missing but a fallback is installed, the app still runs.
    for candidate in [MODEL, *MODEL_FALLBACKS]:
        if candidate in available:
            _resolved_model = candidate
            return candidate

    # Nothing matched (or Ollama unreachable) — keep the preferred name so the
    # error surfaces clearly the first time we actually call the model.
    _resolved_model = MODEL
    return MODEL


def get_llm(temperature: float = 0.0, reasoning: bool | None = None, **kwargs) -> ChatOllama:
    """Build a ChatOllama bound to the resolved model.

    reasoning=False is useful for the execution-heavy nodes: qwen3.5 is a
    reasoning model and skipping the silent "thinking" phase makes plain
    generation faster and avoids occasional empty responses.
    """
    opts = dict(model=resolve_model(), temperature=temperature, num_ctx=NUM_CTX)
    if reasoning is not None:
        opts["reasoning"] = reasoning
    opts.update(kwargs)
    return ChatOllama(**opts)


def health_check() -> dict:
    """Cheap probe used by the UI to show a green/red status banner."""
    try:
        model = resolve_model()
        installed = _installed_models()
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "ok"}],
            options={"num_predict": 1},
        )
        return {"ok": True, "model": model, "installed": installed}
    except Exception as exc:  # Ollama not running / model missing
        return {"ok": False, "model": resolve_model(), "error": str(exc), "installed": []}
