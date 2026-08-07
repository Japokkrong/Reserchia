"""arXiv paper retrieval: search, browse by category, fetch by identifier.

Three tools rather than one general query tool, because the arXiv query
language is easy to get subtly wrong -- a misplaced quote widens a search by
four orders of magnitude without erroring. Each tool here owns the syntax for
one access pattern so the model only has to supply plain arguments.

Docstrings are the model's specification, so they are written for it.
Bad input returns a readable error string rather than raising, which gives the
model a chance to correct itself on the next turn.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BeforeValidator

from ..arxiv_client import (
    MAX_RESULTS,
    SORT_FIELDS,
    Results,
    fetch,
    format_results,
    month_range,
    normalize_id,
)
from ..arxiv_fulltext import (
    PDF_URL,
    fetch_document,
    find_section,
    render,
    section_index,
    subtree,
)
from ..rag.ingest import ingest_later

_EXAMPLE_CATEGORIES = ("cs.HC", "cs.AI", "cs.LG", "math.CO", "physics.optics")
_EXAMPLE_IDS = ("1706.03762", "2404.16130v1", "hep-th/9603067")

#: arXiv's first submissions were in August 1991.
_FIRST_YEAR = 1991

#: Roughly 15k tokens. A typical paper is well under this; a survey can be ten
#: times over it, and because the graph carries one growing message list, an
#: oversized body would be resent on every later turn of the conversation.
FULLTEXT_LIMIT = 60_000

#: Shape check only -- the real list runs to ~150 categories and changes, so a
#: wrong-but-plausible one is caught by its empty result set instead.
_CATEGORY_RE = re.compile(r"[a-zA-Z-]+(?:\.[a-zA-Z-]{2,})?")

#: Signals that the caller already wrote arXiv query syntax, which should then
#: be passed through untouched rather than rebuilt.
_HAS_SYNTAX = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|id|all):|\b(?:AND|OR|ANDNOT)\b")


def _search_query(query: str) -> str:
    """Turn a plain request into an arXiv query, or pass syntax through.

    Bare words matter here: arXiv treats an unquoted multi-word string as a
    loose OR, so "attention is all you need" would match half the archive.
    Terms are ANDed instead, with double-quoted runs kept together as phrases.
    """
    if _HAS_SYNTAX.search(query):
        return query

    terms = [term for term in re.findall(r'"[^"]+"|\S+', query) if term.strip('"')]
    if not terms:
        return query
    return " AND ".join(f"all:{term}" for term in terms)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def _as_text(value: object) -> str:
    """Coerce whatever the model emitted for an identifier into a string.

    ``1706.03762`` looks like a number, and models do send it as one -- which a
    ``str | list[str]`` schema rejected, costing two wasted tool calls before
    the model recovered. Declaring a plain string is what fixes that in
    practice; this is the safety net for when it does not.

    Numbers are a lossy path (JSON ``2404.16130`` has already become
    ``2404.1613`` by the time it arrives, trailing zero gone), so the value of
    the string annotation is that this branch stays unused.
    """
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


#: A string in the schema, so the model quotes identifiers instead of
#: arithmetic-ing them; tolerant of lists and numbers at runtime regardless.
PaperIds = Annotated[str, BeforeValidator(_as_text)]


def _category_error(category: str) -> str:
    examples = ", ".join(_EXAMPLE_CATEGORIES)
    return (
        f"Error: {category!r} is not a valid arXiv category name. Categories look "
        f"like 'archive.SUBCATEGORY', e.g. {examples}, or a bare archive such as "
        "'hep-th'. Retry with a corrected category."
    )


@tool
def search_arxiv(
    query: str,
    category: str | None = None,
    max_results: int = 5,
    sort_by: str = "relevance",
    start: int = 0,
) -> str:
    """Search arXiv for papers by topic, title words, author, or abstract text.

    Use this whenever the user asks what research exists on a subject, or asks
    for papers by an author, and does not already have an arXiv identifier.
    Results include the abstract and a link to each paper -- cite those links
    rather than recalling papers from memory.

    Args:
        query: What to search for, in plain words, e.g. 'eye tracking in
            virtual reality'. Put a phrase in double quotes to require it
            verbatim, e.g. '"attention is all you need"'. You may also write
            arXiv field syntax directly and it is used as-is:
            'au:Hinton', 'ti:transformer AND abs:efficiency'.
        category: Optional arXiv category to restrict the search to, e.g.
            'cs.HC', 'cs.LG', 'math.CO'. Omit to search all of arXiv.
        max_results: How many papers to return, 1 to 25. Defaults to 5. Keep it
            small unless the user asked for a broad survey.
        sort_by: 'relevance' (default), 'submittedDate' for newest first, or
            'lastUpdatedDate'. Use 'submittedDate' when the user asks for
            recent or latest work.
        start: Offset for paging, defaults to 0. To show the next page after
            returning 5 results, call again with start=5.
    """
    if not (query or "").strip():
        return "Error: query is empty. Pass the words to search arXiv for."

    if sort_by not in SORT_FIELDS:
        return (
            f"Error: sort_by must be one of {', '.join(SORT_FIELDS)} "
            f"(got {sort_by!r})."
        )

    search = _search_query(query.strip())
    if category:
        if not _CATEGORY_RE.fullmatch(category.strip()):
            return _category_error(category)
        search = f"({search}) AND cat:{category.strip()}"

    max_results = _clamp(max_results, 1, MAX_RESULTS)
    start = max(0, int(start))

    # Raw strings, deliberately -- see the encoding note in arxiv_client.
    results = fetch(
        {
            "search_query": search,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
    )
    if isinstance(results, str):
        return results

    if not results.papers:
        return (
            f"No arXiv papers found for {query!r}"
            + (f" in category {category}." if category else ".")
            + " Try fewer or broader words, or drop the category filter."
        )

    header = f"matching {query!r}" + (f" in {category}" if category else "")
    return format_results(
        results,
        f"{header} (sorted by {sort_by})",
        start=start,
        abstract_chars=400,
        author_limit=5,
    )


@tool
def browse_arxiv(
    category: str,
    year: int,
    month: int | None = None,
    max_results: int = 10,
    start: int = 0,
) -> str:
    """List papers submitted to one arXiv category in a given year or month.

    Use this when the user wants to see what appeared in a field over a period
    -- 'what came out in cs.HC in January 2019' -- rather than searching for a
    topic. For a topic search use search_arxiv instead.

    Counts can differ by a few from the arXiv website's listing pages: this
    filters on submission date, while the website lists by announcement date.

    Args:
        category: The arXiv category to list, e.g. 'cs.HC', 'cs.AI', 'math.CO',
            or a bare archive such as 'hep-th'. Required.
        year: Four-digit year, e.g. 2019. arXiv starts in 1991.
        month: Optional month number, 1 to 12. Omit to cover the whole year.
        max_results: How many papers to return, 1 to 25. Defaults to 10.
        start: Offset for paging, defaults to 0. The reply reports the total, so
            call again with a larger start to continue through it.
    """
    if not (category or "").strip() or not _CATEGORY_RE.fullmatch(category.strip()):
        return _category_error(category)

    try:
        year = int(year)
    except (TypeError, ValueError):
        return f"Error: year must be a four-digit number (got {year!r})."

    latest = datetime.now().year + 1
    if not _FIRST_YEAR <= year <= latest:
        return (
            f"Error: year must be between {_FIRST_YEAR} and {latest} "
            f"(got {year}). arXiv's first submissions were in August 1991."
        )

    if month is not None:
        try:
            month = int(month)
        except (TypeError, ValueError):
            return f"Error: month must be a number from 1 to 12 (got {month!r})."
        if not 1 <= month <= 12:
            return f"Error: month must be between 1 and 12 (got {month})."

    category = category.strip()
    since, until = month_range(year, month)
    max_results = _clamp(max_results, 1, MAX_RESULTS)
    start = max(0, int(start))

    results = fetch(
        {
            "search_query": f"cat:{category} AND submittedDate:[{since} TO {until}]",
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    if isinstance(results, str):
        return results

    period = f"{year}-{month:02d}" if month else str(year)
    if not results.papers:
        return (
            f"No arXiv papers found in {category} for {period}. "
            f"Check the category name -- valid ones look like "
            f"{', '.join(_EXAMPLE_CATEGORIES)} -- or try a wider period."
        )

    return format_results(
        results,
        f"submitted to {category} in {period}",
        start=start,
        abstract_chars=400,
        author_limit=5,
    )


@tool
def get_arxiv_paper(paper_ids: PaperIds) -> str:
    """Fetch specific arXiv papers by identifier, with their full abstracts.

    Use this whenever the user names an arXiv identifier or links to a paper,
    instead of answering about it from memory. Prefer it over search_arxiv when
    an identifier is already known.

    Args:
        paper_ids: One or more arXiv identifiers, as a string -- always quoted,
            never as a bare number, since '1706.03762' is an identifier and not
            a decimal. Accepts the bare id ('1706.03762'), a version
            ('2404.16130v1'), the older style ('hep-th/9603067'), an
            'arXiv:1706.03762' prefix, or a pasted https://arxiv.org/abs/...
            URL. Separate several with commas: '1706.03762, 2404.16130'.
    """
    raw = re.split(r"[,\s]+", paper_ids)

    wanted: list[str] = []
    invalid: list[str] = []
    for item in (part.strip() for part in raw):
        if not item:
            continue
        identifier = normalize_id(item)
        if identifier:
            wanted.append(identifier)
        else:
            invalid.append(item)

    if not wanted:
        examples = ", ".join(_EXAMPLE_IDS)
        listed = ", ".join(repr(item) for item in invalid) or "nothing"
        return (
            f"Error: no valid arXiv identifier in {listed}. Identifiers look "
            f"like {examples}. If you do not have one, use search_arxiv instead."
        )

    results = fetch({"id_list": ",".join(wanted), "max_results": len(wanted)})
    if isinstance(results, str):
        return results

    # arXiv returns id_list matches in its own order, so restore the caller's,
    # and match on the versionless id since the request may omit a version the
    # response carries.
    def base(identifier: str) -> str:
        return re.sub(r"v\d+$", "", identifier)

    found = {base(paper.arxiv_id): paper for paper in results.papers}
    ordered = [found[base(w)] for w in wanted if base(w) in found]
    missing = [w for w in wanted if base(w) not in found]

    if not ordered:
        return (
            f"No arXiv papers found for {', '.join(wanted)}. "
            "Check the identifier, or use search_arxiv to find the paper by title."
        )

    blocks = [
        format_results(
            Results(total=len(ordered), papers=ordered),
            f"for {', '.join(wanted)}",
            abstract_chars=None,
            author_limit=None,
        )
    ]
    if missing:
        blocks.append(f"Not found on arXiv: {', '.join(missing)}.")
    if invalid:
        blocks.append(
            f"Skipped, not arXiv identifiers: {', '.join(repr(i) for i in invalid)}."
        )
    return "\n\n".join(blocks)


@tool
def get_arxiv_fulltext(paper_id: PaperIds, section: str | None = None) -> str:
    """Read the full body text of one arXiv paper, not just its abstract.

    Use this whenever the user wants a summary of a paper, asks what it did,
    how it works, what its method, experiments, results or conclusions were, or
    asks about any specific claim in it. An abstract is not enough to answer
    those -- fetch the paper.

    Long papers come back as a list of section headings instead of the whole
    text. When that happens, call this again with the section you need.

    Args:
        paper_id: One arXiv identifier, as a string -- always quoted, never as
            a bare number, since '1706.03762' is an identifier and not a
            decimal. A bare id, a version ('2404.16130v1'), the older style
            ('hep-th/9603067'), or a pasted https://arxiv.org/abs/... URL. If
            you only know the title, use search_arxiv first to get the id.
        section: Optional section to return on its own, named either by number
            ('3', '3.2') or by heading ('Positional Encoding'). Omit it to get
            the whole paper.
    """
    identifier = normalize_id(paper_id)
    if not identifier:
        return (
            f"Error: {paper_id!r} is not an arXiv identifier. They look like "
            f"{', '.join(_EXAMPLE_IDS)}. Use search_arxiv to find the paper first."
        )

    document = fetch_document(identifier)
    if isinstance(document, str):
        return (
            f"{document}\n\n"
            f"PDF: {PDF_URL.format(identifier)}\n"
            "Use get_arxiv_paper for the abstract, and tell the user the full "
            "text could not be retrieved rather than describing the paper from "
            "memory."
        )

    # The second of the two paths. The full text goes back to the agent now, so
    # this turn is answered immediately; meanwhile the paper is chunked,
    # embedded and stored in the background so the next question about it can be
    # served from the library without touching the API. `fetch_document` caches
    # the parsed document, so the worker re-uses it rather than downloading
    # again. A failure here is swallowed -- it costs a repeat fetch later, which
    # is not worth interrupting an answer for.
    ingest_later(identifier)

    header = [f"arXiv:{identifier}"]
    if document.title:
        header.append(f"Title: {document.title}")
    header.append(f"Source: {document.source}")
    if document.lower_fidelity:
        header.append(
            "Note: this text was extracted from the PDF, so layout, equations "
            "and some word spacing are unreliable. Treat quotations with care."
        )
    prefix = "\n".join(header)

    if section:
        found = find_section(document, section)
        if found is None:
            index = section_index(document)
            return f"{prefix}\n\nNo section matching {section!r}." + (
                f" Available sections:\n{index}"
                if index
                else " This paper has no section headings."
            )

        # Subsections included: LaTeXML stores "3" and "3.1" as flat siblings,
        # and someone asking for section 3 means the whole of it.
        parts = subtree(document, found)
        text = render(parts)
        if len(text) > FULLTEXT_LIMIT:
            listing = "\n".join(
                f"  {part.heading} ({len(part.text):,} chars)"
                for part in parts[1:]
                if part.heading
            )
            return (
                f"{prefix}\n\nSection {parts[0].heading!r} is long "
                f"({len(text):,} characters). Call this tool again naming one "
                f"of its subsections:\n{listing}"
            )
        return f"{prefix}\n\n{text}"

    body = document.text
    if len(body) <= FULLTEXT_LIMIT:
        return f"{prefix}\n\n{body}"

    # Too long to return whole: lead with the abstract so the model still has
    # something to work from, then the index so it can ask for what it needs.
    opening = next(
        (s for s in document.sections if "abstract" in s.heading.casefold() and s.text),
        next((s for s in document.sections if s.text), None),
    )
    return "\n\n".join(
        filter(
            None,
            [
                prefix,
                f"This paper is long ({len(body):,} characters), so here is its "
                "abstract and contents rather than the whole text. Call this "
                "tool again with section='...' for the part you need.",
                f"## {opening.heading}\n\n{opening.text}" if opening else "",
                f"Sections:\n{section_index(document)}",
            ],
        )
    )
