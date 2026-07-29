"""Conversation memory with history compression.

Stores conversation turns per conversation_id.  When history exceeds
``MAX_RECENT_TURNS`` pairs, older turns are compressed into a single summary
using the LLM, keeping recent turns verbatim.
"""

from __future__ import annotations

from ..llm.base import Message

# conversation_id -> ordered list of messages
_STORE: dict[str, list[Message]] = {}

# conversation_id -> summary of older turns
_SUMMARIES: dict[str, str] = {}

# Keep the last N user+assistant pairs verbatim; summarize older ones.
MAX_RECENT_TURNS = 2  # 2 pairs = 4 messages

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely, preserving all key facts, "
    "entities, and conclusions that would be needed to understand follow-up questions. "
    "Be specific — keep names, numbers, and terminology.\n\n{history}"
)


def get_history(conversation_id: str | None) -> list[Message]:
    if not conversation_id:
        return []
    messages: list[Message] = []
    summary = _SUMMARIES.get(conversation_id)
    if summary:
        messages.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        })
    messages.extend(_STORE.get(conversation_id, []))
    return messages


def append(conversation_id: str | None, user: str, assistant: str) -> None:
    if not conversation_id:
        return
    history = _STORE.setdefault(conversation_id, [])
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": assistant})

    # Compress if history exceeds the limit
    max_messages = MAX_RECENT_TURNS * 2
    if len(history) > max_messages:
        _compress(conversation_id)


def _compress(conversation_id: str) -> None:
    """Summarize old turns, keep only the most recent ones."""
    from ..llm.factory import get_client

    history = _STORE[conversation_id]
    max_messages = MAX_RECENT_TURNS * 2
    old = history[:-max_messages]
    recent = history[-max_messages:]

    # Build text from old turns (+ any existing summary)
    parts: list[str] = []
    existing = _SUMMARIES.get(conversation_id)
    if existing:
        parts.append(f"Previous summary: {existing}")
    for msg in old:
        parts.append(f"{msg['role'].upper()}: {msg['content']}")

    try:
        client = get_client()
        summary = client.chat([{
            "role": "user",
            "content": _SUMMARIZE_PROMPT.format(history="\n".join(parts)),
        }])
        _SUMMARIES[conversation_id] = summary
    except Exception:
        # If summarization fails, just keep the raw text as summary
        _SUMMARIES[conversation_id] = "\n".join(parts)

    _STORE[conversation_id] = recent


def reset(conversation_id: str) -> None:
    _STORE.pop(conversation_id, None)
    _SUMMARIES.pop(conversation_id, None)
