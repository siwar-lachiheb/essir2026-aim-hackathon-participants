"""Answer a question end to end.

    history + retrieved context  ->  prompt  ->  LLM  ->  grounded answer

This is what `POST /query` calls. You send only a question and its level; the system
assigns the id, threads the conversation (level-2 follow-ups share memory), produces the
answer, and writes it to `data/out/` as a JSON file you can later copy into `submission/`.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..llm.base import LLMError, Message
from ..llm.factory import get_client
from ..models import Diagnostics, QueryRequest, QueryResponse, Source
from . import memory
from .retrieve import Context, retrieve

SYSTEM_PROMPT = (
    "You answer questions about a single document using only the context provided. "
    "If the context does not contain the answer, say so plainly rather than guessing. "
    "Be specific and concise."
)


def _build_messages(question: str, contexts: list[Context], history: list[Message]) -> list[Message]:
    context_block = "\n\n".join(
        f"[Section: {c.section}] [page {c.page}]\n{c.text}" for c in contexts
    ) or "(no context retrieved)"
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Prior turns give the model the conversation so far (Level 2). Retrieval still
    # needs the rewritten query — history in the prompt is necessary but not sufficient.
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context from the document:\n{context_block}\n\nQuestion: {question}",
        }
    )
    return messages


def _sources_from(contexts: list[Context]) -> list[Source]:
    # Baseline: the retrieved units become the citations, truncated to a short quote.
    # TODO(level-1): a page (or chunk) is not a precise citation. Return the specific
    #   sentence that supports the answer, with its correct page — not the whole unit.
    out: list[Source] = []
    for c in contexts:
        quote = c.text.strip().replace("\n", " ")
        if len(quote) > 300:
            quote = quote[:300].rsplit(" ", 1)[0] + "…"
        out.append(Source(page=c.page, quote=quote, score=round(c.score, 4)))
    return out


def _save(response: QueryResponse, when: datetime) -> None:
    """Write the answer to data/out/q_<id>_level_<level>_<datetime>.json."""
    out_dir = Path(get_settings().out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    name = f"q_{response.question_id}_level_{response.level}_{stamp}.json"
    (out_dir / name).write_text(response.model_dump_json(indent=2), encoding="utf-8")


def answer(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    client = get_client()
    top_k = req.top_k or settings.top_k

    # The system owns the ids. Level-N questions share one conversation, so level-2
    # follow-ups automatically see the earlier turns of the same level.
    question_id = "q" + uuid.uuid4().hex[:6]
    conversation_id = f"level-{req.level}"

    history = memory.get_history(conversation_id)
    now = datetime.now(UTC)
    started = time.perf_counter()

    contexts = retrieve(req.question, top_k, history, level=req.level)
    messages = _build_messages(req.question, contexts, history)

    try:
        answer_text = client.chat(messages)
    except LLMError as e:
        # Degrade rather than 500: return the retrieved evidence so the pipeline is
        # still usable (and debuggable) without a running LLM.
        answer_text = (
            f"[LLM unavailable: {e}] Retrieved context is attached as sources; "
            "no generated answer."
        )

    memory.append(conversation_id, req.question, answer_text)
    latency_ms = int((time.perf_counter() - started) * 1000)

    response = QueryResponse(
        question_id=question_id,
        level=req.level,
        question=req.question,
        answer=answer_text,
        conversation_id=conversation_id,
        sources=_sources_from(contexts),
        diagnostics=Diagnostics(
            provider=settings.llm_provider,
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            retrieved_chunks=len(contexts),
            tokens=None,          # TODO: report real token usage if your provider returns it
            latency_ms=latency_ms,
            timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )
    _save(response, now)
    return response
