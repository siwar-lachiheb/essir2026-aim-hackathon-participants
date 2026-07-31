"""Find the chunks most relevant to a question.

Three retrieval strategies:

- **standard** (Level 1 & 2): single query, optional history-based rewrite.
- **multiquery** (Level 3): LLM decomposes the question into sub-queries,
  retrieves for each, deduplicates.
- **sections** (Level 3): fetches all chunks, groups by section, picks the
  most relevant per section so every section is represented.

The router ``_retrieve_level3`` picks between multiquery and sections.
Swap its logic to change how Level 3 questions are routed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..llm.base import Message
from ..llm.factory import get_client
from ..vectorstore.qdrant_store import get_store
from .embeddings import get_embedder


@dataclass
class Context:
    text: str
    page: int
    score: float
    section: str = ""


# ---------------------------------------------------------------------------
# Level 2: query rewriting
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = (
    "Given the conversation history and a follow-up question, rewrite the "
    "follow-up into a standalone search query that contains all necessary "
    "context. If the question is already standalone, return it unchanged.\n\n"
    "History:\n{history}\n\n"
    "Follow-up: {question}\n\n"
    "Standalone query:"
)


def rewrite_query(question: str, history: list[Message]) -> str:
    """Resolve a follow-up into a standalone search query using the LLM."""
    if not history:
        return question
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    try:
        client = get_client()
        rewritten = client.chat([{
            "role": "user",
            "content": _REWRITE_PROMPT.format(history=history_text, question=question),
        }])
        return rewritten.strip() or question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# Strategy: standard (Level 1 & 2)
# ---------------------------------------------------------------------------

def _retrieve_standard(question: str, top_k: int, history: list[Message]) -> list[Context]:
    """Single query retrieval with optional history-based rewrite."""
    query = rewrite_query(question, history)
    vector = get_embedder().embed([query], is_query=True)[0]
    hits = get_store().search(vector, top_k)
    return [
        Context(
            text=str(h.payload.get("text", "")),
            page=int(h.payload.get("page", 0)),
            score=float(h.score),
            section=str(h.payload.get("section", "")),
        )
        for h in hits
    ]


# ---------------------------------------------------------------------------
# Strategy: multi-query fan-out (Level 3 — multi-hop, table+text)
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = (
    "Break this question into 2-4 independent sub-queries that together "
    "cover all the evidence needed to answer it. Return a JSON array of strings.\n\n"
    "Question: {question}\n\n"
    "Sub-queries:"
)


def _retrieve_multiquery(question: str, top_k: int) -> list[Context]:
    """Decompose into sub-queries, retrieve for each, deduplicate."""
    client = get_client()
    embedder = get_embedder()
    store = get_store()

    # Ask LLM to decompose the question
    try:
        raw = client.chat([{
            "role": "user",
            "content": _DECOMPOSE_PROMPT.format(question=question),
        }])
        sub_queries = json.loads(raw.strip())
        if not isinstance(sub_queries, list):
            sub_queries = [question]
    except Exception:
        sub_queries = [question]

    # Always include the original question
    if question not in sub_queries:
        sub_queries.insert(0, question)

    # Retrieve for each sub-query
    seen_texts: set[str] = set()
    contexts: list[Context] = []
    for sq in sub_queries:
        vector = embedder.embed([sq], is_query=True)[0]
        hits = store.search(vector, top_k)
        for h in hits:
            text = str(h.payload.get("text", ""))
            if text in seen_texts:
                continue
            seen_texts.add(text)
            contexts.append(Context(
                text=text,
                page=int(h.payload.get("page", 0)),
                score=float(h.score),
                section=str(h.payload.get("section", "")),
            ))

    # Sort by score descending
    contexts.sort(key=lambda c: c.score, reverse=True)
    return contexts


# ---------------------------------------------------------------------------
# Strategy: section-aware fetch (Level 3 — synthesis)
# ---------------------------------------------------------------------------

def _retrieve_sections(question: str, top_k_per_section: int = 2) -> list[Context]:
    """Fetch all chunks, group by section, return top-N per section by relevance."""
    embedder = get_embedder()
    store = get_store()

    q_vector = np.array(embedder.embed([question], is_query=True)[0])
    records = store.scroll_all()

    # Group by section
    by_section: dict[str, list] = defaultdict(list)
    for r in records:
        section = str(r.payload.get("section", ""))
        by_section[section].append(r)

    # Score and pick top-N per section
    contexts: list[Context] = []
    for section, recs in by_section.items():
        # Embed all chunks in this section
        texts = [str(r.payload.get("text", "")) for r in recs]
        vectors = embedder.embed(texts, is_query=False)
        # Cosine similarity with the question
        scored = []
        for rec, vec in zip(recs, vectors):
            sim = float(np.dot(q_vector, np.array(vec)) / (
                np.linalg.norm(q_vector) * np.linalg.norm(vec) + 1e-10
            ))
            scored.append((rec, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        for rec, sim in scored[:top_k_per_section]:
            contexts.append(Context(
                text=str(rec.payload.get("text", "")),
                page=int(rec.payload.get("page", 0)),
                score=round(sim, 4),
                section=section,
            ))

    contexts.sort(key=lambda c: c.score, reverse=True)
    return contexts


# ---------------------------------------------------------------------------
# Level 3 router — swap this to change routing logic
# ---------------------------------------------------------------------------

_ROUTE_PROMPT = (
    "You are a query router. Given a question about a document, classify it "
    "into exactly one category.\n\n"
    "- SECTIONS: the question asks for a broad overview, full summary, or "
    "coverage of all/many parts of the document.\n"
    "- MULTIQUERY: the question requires combining evidence from multiple "
    "specific parts of the document (comparison, multi-hop reasoning, "
    "cross-referencing data).\n\n"
    "Reply with a single word: SECTIONS or MULTIQUERY.\n\n"
    "Question: {question}\n\nCategory:"
)


def _classify_level3(question: str) -> str:
    """Ask the LLM whether the question needs sections or multiquery."""
    try:
        client = get_client()
        raw = client.chat([{
            "role": "user",
            "content": _ROUTE_PROMPT.format(question=question),
        }]).strip().upper()
        if "SECTIONS" in raw:
            return "sections"
        return "multiquery"
    except Exception:
        return "multiquery"


def _retrieve_level3(question: str, top_k: int) -> list[Context]:
    """Route Level 3 questions via LLM classification."""
    strategy = _classify_level3(question)
    if strategy == "sections":
        return _retrieve_sections(question)
    return _retrieve_multiquery(question, top_k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    top_k: int,
    history: list[Message] | None = None,
    level: int = 1,
) -> list[Context]:
    if level == 3:
        return _retrieve_level3(question, top_k)
    return _retrieve_standard(question, top_k, history or [])
