"""
Knowledge base for Agentic RAG.

Loads the help-center markdown, splits it into articles (one per "## " heading)
and answers `search(query)` with the most relevant articles. It embeds articles
with a local Ollama embedding model and ranks by cosine similarity. If embeddings
are unavailable (model not pulled / Ollama down) it transparently falls back to
keyword overlap so the app never hard-fails on this path.
"""

from __future__ import annotations

import re

from config import EMBED_MODEL, KB_PATH

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from",
    "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "our",
    "the", "to", "what", "when", "why", "with", "your",
}


class KnowledgeBase:
    def __init__(self, path: str = KB_PATH):
        self.articles = self._load(path)
        self._vectors = None          # list[list[float]] once embedded
        self._embed_ready = None      # tri-state: None=unknown, True/False=tried
        self._embedder = None

    # ── Loading / parsing ─────────────────────────────────────────────────────
    @staticmethod
    def _load(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        articles = []
        # Each "##" heading in the markdown file becomes one retrievable article.
        # This keeps the RAG example transparent for students.
        for block in re.split(r"\n## ", text):
            if "\n" not in block:
                continue
            title, *body = block.strip().split("\n", 1)
            title = title.lstrip("# ").strip()
            body = (body[0].strip() if body else "")
            # Skip the top H1 banner and any empty sections.
            if title and body and not title.startswith("CloudNova Help Center"):
                articles.append({"title": title, "body": body})
        return articles

    # ── Embedding (lazy, computed once) ───────────────────────────────────────
    def _ensure_embeddings(self) -> bool:
        if self._embed_ready is not None:
            return self._embed_ready
        try:
            # Embeddings are lazy: the app only pays this cost when the technical
            # agent actually searches the help center.
            from langchain_ollama import OllamaEmbeddings
            self._embedder = OllamaEmbeddings(model=EMBED_MODEL)
            texts = [f"{a['title']}. {a['body']}" for a in self.articles]
            self._vectors = self._embedder.embed_documents(texts)
            self._embed_ready = True
        except Exception:
            self._embed_ready = False  # fall back to keyword search
        return self._embed_ready

    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-9)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}

    def _keyword_rank(self, query: str, k: int) -> list[int]:
        words = self._terms(query)
        if not words:
            return []
        scored = []
        for i, a in enumerate(self.articles):
            title_terms = self._terms(a["title"])
            body_terms = self._terms(a["body"])
            # Title matches are weighted higher because they usually identify
            # the main topic of the help-center article.
            score = 3 * len(words & title_terms) + len(words & body_terms)
            scored.append((score, -i, i))
        scored.sort(reverse=True)
        return [i for score, _, i in scored[:k] if score > 0]

    # ── Public API ────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 2) -> list[dict]:
        if not self.articles:
            return []
        if self._ensure_embeddings():
            try:
                qv = self._embedder.embed_query(query)
                sims = sorted(
                    ((self._cosine(qv, v), i) for i, v in enumerate(self._vectors)),
                    reverse=True,
                )
                idx = [i for _, i in sims[:k]]
            except Exception:
                # If Ollama goes away after startup, degrade gracefully instead
                # of breaking the technical support flow.
                self._embed_ready = False
                idx = self._keyword_rank(query, k)
        else:
            idx = self._keyword_rank(query, k)
        return [self.articles[i] for i in idx]

    def search_text(self, query: str, k: int = 2) -> str:
        hits = self.search(query, k)
        if not hits:
            return "No relevant help-center article found."
        return "\n\n".join(f"### {h['title']}\n{h['body']}" for h in hits)
