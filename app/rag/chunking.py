"""Turn PDF pages into retrievable units.

Section-aware, sentence-based chunking using NLTK's Punkt tokenizer.
Pages are concatenated, section headings are detected, text is split into
sentences, and sentences are grouped into chunks of up to ``chunk_size``
characters.  Each chunk carries its page number and section title.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

import nltk
from nltk.tokenize import sent_tokenize

from ..config import get_settings

# Download Punkt data once (no-op after the first time)
nltk.download("punkt_tab", quiet=True)

_HEADING_RE = re.compile(r"\n(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)")


@dataclass
class Chunk:
    text: str
    page: int      # 1-indexed
    index: int     # position within the document
    section: str = ""


def _build_full_text(pages: list[str]) -> tuple[str, list[int]]:
    """Concatenate pages and return (full_text, page_char_offsets)."""
    offsets: list[int] = []
    pos = 0
    for text in pages:
        offsets.append(pos)
        pos += len(text) + 1
    return "\n".join(pages), offsets


def _detect_sections(full_text: str) -> list[tuple[int, str]]:
    """Return (char_offset, section_title) pairs."""
    sections: list[tuple[int, str]] = [(0, "Preamble")]
    for m in _HEADING_RE.finditer(full_text):
        sections.append((m.start(), f"{m.group(1)} {m.group(2).strip()}"))
    return sections


def _lookup(pos: int, offsets: list[int]) -> int:
    """Binary-search *pos* in sorted *offsets*, return the 1-indexed bucket."""
    return bisect.bisect_right(offsets, pos)


def chunk_pages(pages: list[str]) -> list[Chunk]:
    full_text, page_offsets = _build_full_text(pages)
    sections = _detect_sections(full_text)
    section_offsets = [o for o, _ in sections]
    section_titles = [t for _, t in sections]

    sentences = sent_tokenize(full_text)
    max_size = get_settings().chunk_size

    chunks: list[Chunk] = []
    idx = 0
    buf: list[str] = []
    buf_len = 0
    buf_start = 0
    char_pos = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Locate this sentence in the full text
        found = full_text.find(sent, char_pos)
        if found >= 0:
            char_pos = found

        if buf and buf_len + len(sent) + 1 > max_size:
            page = _lookup(buf_start, page_offsets)
            sec_idx = bisect.bisect_right(section_offsets, buf_start) - 1
            chunks.append(Chunk(
                text=" ".join(buf), page=page, index=idx,
                section=section_titles[sec_idx],
            ))
            idx += 1
            buf.clear()
            buf_len = 0
            buf_start = char_pos

        if not buf:
            buf_start = char_pos
        buf.append(sent)
        buf_len += len(sent) + (1 if buf_len else 0)
        char_pos += len(sent)

    if buf:
        page = _lookup(buf_start, page_offsets)
        sec_idx = bisect.bisect_right(section_offsets, buf_start) - 1
        chunks.append(Chunk(
            text=" ".join(buf), page=page, index=idx,
            section=section_titles[sec_idx],
        ))

    return chunks
