"""Split a paper into retrievable chunks, one per section.

This is the configuration the `rag-eda/` experiment measured as best for arXiv
papers: section-aligned chunks, no overlap, no heading prefix. Against a
fixed-size baseline it retrieved the right chunk at rank 1 more often (0.729 vs
0.593) and reached 97% of its Recall@10 using 23% of the context, because
arXiv sections are mostly small -- the resulting chunks average ~218 tokens.

Chunks are *not* rolled up through `arxiv_fulltext.subtree`. Section 3, 3.1 and
3.2 stay three separate chunks. Rolling them together would produce a different
and much larger chunking than the one that was measured.

The split threshold is in **characters, not tokens**, on purpose. bge-m3 runs
4.04 characters per token over LaTeXML text, so the measured 1024-token
threshold is ~4,100 characters. Counting exactly would mean shipping the
`tokenizers` package and downloading a tokenizer from HuggingFace at runtime,
to decide a boundary that is coarse anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..arxiv_fulltext import Document

#: ~1024 bge-m3 tokens at the measured 4.04 chars/token. Sections shorter than
#: this stay whole; longer ones split at paragraph boundaries.
SPLIT_CHARS = 4100

#: bge-m3 accepts 8192 tokens (~33,000 characters). Nothing here should come
#: close, but a single unbroken paragraph could, and silently sending an
#: over-length chunk means the provider truncates it without telling us.
MAX_CHARS = 30_000

#: Chunks shorter than this are headings with no body -- a section title
#: followed immediately by its first subsection. They match nothing useful.
MIN_CHARS = 40

_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, carrying everything a citation needs."""

    id: str
    arxiv_id: str
    title: str
    section: str
    #: "3.2" when the heading is numbered, "" otherwise.
    number: str
    text: str
    index: int

    @property
    def citation(self) -> str:
        where = f" §{self.section}" if self.section else ""
        return f"arXiv:{self.arxiv_id}{where}"

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def _split_paragraphs(text: str, limit: int = SPLIT_CHARS) -> list[str]:
    """Pack paragraphs up to `limit`, never splitting one unless it alone is over."""
    parts, current = [], ""
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > limit:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)

    # A single paragraph over the embedding limit still has to be cut, or the
    # provider truncates it silently. Sentence boundaries are the least-bad place.
    bounded = []
    for part in parts:
        while len(part) > MAX_CHARS:
            cut = part.rfind(". ", 0, MAX_CHARS)
            cut = cut + 1 if cut > MAX_CHARS // 2 else MAX_CHARS
            bounded.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            bounded.append(part)
    return bounded


def chunk_document(document: Document) -> list[Chunk]:
    """One chunk per section, splitting oversized sections at paragraph breaks."""
    title = document.title or document.arxiv_id
    chunks: list[Chunk] = []

    for section in document.sections:
        body = (section.text or "").strip()
        if len(body) < MIN_CHARS:
            continue

        pieces = [body] if len(body) <= SPLIT_CHARS else _split_paragraphs(body)
        for piece in pieces:
            if len(piece) < MIN_CHARS:
                continue
            chunks.append(
                Chunk(
                    # Deterministic, so re-ingesting a paper upserts over its
                    # own chunks instead of duplicating them.
                    id=f"{document.arxiv_id}#{len(chunks):04d}",
                    arxiv_id=document.arxiv_id,
                    title=title,
                    section=section.heading or "",
                    number=section.number or "",
                    text=piece,
                    index=len(chunks),
                )
            )
    return chunks
