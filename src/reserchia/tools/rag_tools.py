"""Searching the library of papers the agent has already read.

This is the tool the agent reaches for *before* the arXiv API when asked about
a paper's contents. It answers from stored chunks, and every result carries the
paper id, section heading and abstract URL so the answer can be cited.

Where it hands back to the API is deliberately split in two:

- **Whether the paper is in the library** is decided exactly, by an id lookup.
  There is no score threshold here, because thresholds were measured and do not
  work: correct top-1 hits scored as low as 0.385 while questions the corpus
  could not answer scored 0.492-0.498, so any cutoff rejecting the latter also
  rejects real answers.
- **Whether the retrieved chunks actually answer the question** is left to the
  agent, which is reading them anyway. The tool says plainly that escalating to
  `get_arxiv_fulltext` is available.
"""

from __future__ import annotations

from langchain_core.tools import tool

from ..arxiv_client import normalize_id
from ..rag import store
from ..rag.search import search_library

MAX_RESULTS = 10
#: Enough to answer from, short enough that five of them stay cheap.
SNIPPET = 900


def _unavailable(exc: Exception) -> str:
    """A broken library must degrade to a message, never a traceback.

    The store lives on disk and can be missing, unwritable or corrupt. When it
    is, the agent should fall through to the arXiv API and still answer the
    question, exactly as it would for a paper it has not read.
    """
    return (
        f"The paper library is unavailable ({type(exc).__name__}). "
        "Use get_arxiv_fulltext to read the paper directly this time."
    )


def _format(hit) -> str:
    text = hit.text if len(hit.text) <= SNIPPET else hit.text[:SNIPPET].rstrip() + " [...]"
    lines = [f"[{hit.citation}] {hit.title}".rstrip()]
    lines.append(f"relevance {hit.score:.3f}")
    lines.append(text)
    if hit.abs_url:
        lines.append(hit.abs_url)
    return "\n".join(lines)


@tool
def search_paper_library(
    query: str,
    paper_id: str | None = None,
    max_results: int = 5,
) -> str:
    """Search papers already read and stored, before fetching anything new.

    Try this FIRST whenever the user asks what a paper says, wants it
    summarised, or asks about its methods, results or any specific claim. It is
    far cheaper than fetching the paper again, and its results carry the section
    references needed to cite an answer.

    If the paper is not in the library, or the passages returned do not answer
    the question, call get_arxiv_fulltext next -- doing so also adds the paper
    to the library for future questions.

    Args:
        query: What to look for, in plain words. Use the user's own question;
            it is matched both semantically and on exact wording, so specific
            names like 'PP-DocLayout-V3' work well.
        paper_id: Optional arXiv identifier to restrict the search to one paper,
            e.g. '2404.16130' or 'hep-th/9603067'. Omit it to search every paper
            in the library, which is how to answer questions spanning several.
        max_results: How many passages to return, 1 to 10. Defaults to 5.
    """
    if not (query or "").strip():
        return "Error: query is empty. Pass the question to search for."

    identifier = None
    if paper_id:
        identifier = normalize_id(str(paper_id))
        if not identifier:
            return (
                f"Error: {paper_id!r} is not an arXiv identifier. They look like "
                "1706.03762 or hep-th/9603067. Omit paper_id to search the whole "
                "library."
            )

    try:
        if store.count() == 0:
            return (
                "The paper library is empty -- no papers have been read yet. "
                "Use get_arxiv_fulltext to fetch a paper; it is added to the "
                "library automatically."
            )

        if identifier and not store.has_paper(identifier):
            known = store.papers()
            listed = ", ".join(f"arXiv:{pid}" for pid, _, _ in known[:8]) or "none yet"
            return (
                f"arXiv:{store.base_id(identifier)} is not in the library.\n"
                f"Call get_arxiv_fulltext with this identifier to read it -- that "
                f"also stores it, so later questions will not need the API.\n"
                f"Papers currently in the library: {listed}."
            )

        limit = max(1, min(int(max_results), MAX_RESULTS))
        hits = search_library(query, identifier, limit)
    except Exception as exc:  # noqa: BLE001 - degrade to the API path
        return _unavailable(exc)

    if not hits:
        return (
            f"No passages in the library matched {query!r}. "
            "Use search_arxiv to find a relevant paper, then get_arxiv_fulltext "
            "to read it."
        )

    scope = f"arXiv:{store.base_id(identifier)}" if identifier else "the library"
    blocks = [f"{len(hits)} passage(s) from {scope}, best first:"]
    blocks += [_format(hit) for hit in hits]
    blocks.append(
        "Cite each claim with the bracketed reference above. If these passages "
        "do not answer the question, call get_arxiv_fulltext for the full text."
    )
    return "\n\n".join(blocks)


@tool
def list_paper_library() -> str:
    """List the papers already read and stored, with their identifiers.

    Use this when the user asks what has been read so far, or when you need an
    arXiv identifier for a paper the conversation has already covered.
    """
    try:
        known = store.papers()
    except Exception as exc:  # noqa: BLE001 - degrade to the API path
        return _unavailable(exc)
    if not known:
        return (
            "The paper library is empty. Papers are added automatically when "
            "get_arxiv_fulltext reads one."
        )
    lines = [f"{len(known)} paper(s) in the library:"]
    for arxiv_id, title, chunks in known:
        lines.append(f"  arXiv:{arxiv_id}  {title or '(untitled)'}  ({chunks} passages)")
    return "\n".join(lines)
