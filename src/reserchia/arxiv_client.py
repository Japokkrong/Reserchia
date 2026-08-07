"""arXiv API client -- HTTP, throttling, Atom parsing, and formatting.

The model-facing tools live in ``tools/arxiv_tools.py``; this module is the
plumbing under them, deliberately free of LangChain imports so it can be
exercised without an API key.

Everything here talks to the official API at ``export.arxiv.org/api/query``
rather than scraping the ``arxiv.org/list/...`` browse pages, because the API
is the interface arXiv publishes terms of use for.

Two things in here are load-bearing and easy to "clean up" into breakage:

**The throttle.** arXiv's terms of use say to "make no more than one request
every three seconds, and limit requests to a single connection at a time". A
ReAct loop will happily fire three searches back to back, so the gap is
enforced here, under a lock, rather than trusted to callers.

**Raw query encoding.** The API manual documents its examples pre-encoded --
``ti:%22quantum+theory%22``. That is only correct when hand-concatenating a URL
string. Handing those same characters to an HTTP client double-encodes them
(``%22`` -> ``%2522``), and arXiv does not complain -- it silently drops the
phrase grouping and returns a much larger, much worse result set::

    ti:"attention is all you need"          ->      35 results
    ti:%22attention is all you need%22      -> 459,755 results

So queries are built as raw strings in a params dict and httpx does the
encoding. Do not "fix" them to match the manual.
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

API_URL = "https://export.arxiv.org/api/query"

#: arXiv terms of use: one request per three seconds, one connection.
MIN_INTERVAL = 3.0
#: Our own ceiling on results per call, to protect the context window.
#: arXiv itself allows far more.
MAX_RESULTS = 25
TIMEOUT = 30.0

USER_AGENT = "Reserchia/0.1 (LangGraph research agent; +https://arxiv.org/help/api)"

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

SORT_FIELDS = ("relevance", "submittedDate", "lastUpdatedDate")
SORT_ORDERS = ("ascending", "descending")

# arXiv identifiers come in two generations: post-2007 "1902.00098" and the
# older "hep-th/9603067". Both may carry a version suffix.
_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?"


# -- HTTP ---------------------------------------------------------------------

_client = httpx.Client(
    headers={"User-Agent": USER_AGENT},
    timeout=TIMEOUT,
    follow_redirects=True,
)

_lock = threading.Lock()
_last_request = 0.0


def _throttle() -> None:
    """Hold the gap arXiv's terms of use require, serialising callers."""
    global _last_request
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    published: str
    updated: str
    primary_category: str
    categories: tuple[str, ...]
    summary: str
    comment: str
    journal_ref: str
    doi: str
    abs_url: str
    pdf_url: str


@dataclass(frozen=True)
class Results:
    #: Total matches arXiv holds, which is usually more than were returned.
    total: int
    papers: list[Paper]


def fetch(params: dict) -> Results | str:
    """Run one API query.

    Returns ``Results``, or a human-readable error string -- never raises, so
    the tools above can hand the message straight to the model and let it
    correct itself on the next turn.
    """
    _throttle()
    try:
        response = _client.get(API_URL, params=params)
    except httpx.RequestError as exc:
        return (
            f"Error: could not reach the arXiv API ({type(exc).__name__}). "
            "It may be down or the network may be unavailable. "
            "Tell the user rather than guessing an answer."
        )

    # Deliberately no raise_for_status(): arXiv reports bad input as HTTP 400
    # with an Atom body whose entry carries the actual reason, and that reason
    # is the most useful thing we can return.
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return (
            f"Error: arXiv returned an unreadable response "
            f"(HTTP {response.status_code}). Try again in a moment."
        )

    message = _error_message(root)
    if message:
        return f"Error from arXiv: {message}"
    if response.status_code != 200:
        return f"Error: arXiv returned HTTP {response.status_code}."

    return Results(
        total=_int(root.findtext("opensearch:totalResults", namespaces=NS)),
        papers=[_parse_entry(entry) for entry in root.findall("a:entry", NS)],
    )


def _error_message(root: ET.Element) -> str:
    """arXiv's own description of a rejected request, if this is one.

    An error feed holds a single entry whose id points at
    ``arxiv.org/api/errors#...``; that id is the reliable marker, not the
    entry's title, which a real paper could coincidentally share.
    """
    for entry in root.findall("a:entry", NS):
        if "/api/errors" in (entry.findtext("a:id", "", NS) or ""):
            return _clean(entry.findtext("a:summary", "", NS))
    return ""


# -- parsing ------------------------------------------------------------------


def _clean(text: str | None) -> str:
    """arXiv wraps titles and abstracts at source; collapse that back."""
    return " ".join((text or "").split())


def _int(text: str | None) -> int:
    try:
        return int(text or 0)
    except ValueError:
        return 0


def _parse_entry(entry: ET.Element) -> Paper:
    raw_id = entry.findtext("a:id", "", NS) or ""
    arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", raw_id)

    abs_url, pdf_url = "", ""
    for link in entry.findall("a:link", NS):
        href = (link.get("href") or "").replace("http://", "https://", 1)
        if link.get("title") == "pdf":
            pdf_url = href
        elif link.get("rel") == "alternate":
            abs_url = href

    primary = entry.find("arxiv:primary_category", NS)

    return Paper(
        arxiv_id=arxiv_id,
        title=_clean(entry.findtext("a:title", "", NS)),
        authors=tuple(
            _clean(author.findtext("a:name", "", NS))
            for author in entry.findall("a:author", NS)
        ),
        published=(entry.findtext("a:published", "", NS) or "")[:10],
        updated=(entry.findtext("a:updated", "", NS) or "")[:10],
        primary_category=primary.get("term", "") if primary is not None else "",
        categories=tuple(
            category.get("term", "") for category in entry.findall("a:category", NS)
        ),
        summary=_clean(entry.findtext("a:summary", "", NS)),
        comment=_clean(entry.findtext("arxiv:comment", "", NS)),
        journal_ref=_clean(entry.findtext("arxiv:journal_ref", "", NS)),
        doi=_clean(entry.findtext("arxiv:doi", "", NS)),
        abs_url=abs_url,
        pdf_url=pdf_url,
    )


# -- identifiers and dates ----------------------------------------------------


def normalize_id(raw: str) -> str | None:
    """Reduce anything that names a paper to a bare arXiv identifier.

    Accepts the bare id, an ``arXiv:`` prefix, and pasted abs/pdf URLs, since
    a model will produce all of those. Returns ``None`` if it is not an
    identifier at all, so the caller can say so before spending a request.
    """
    text = (raw or "").strip()
    if not text:
        return None

    url = re.search(r"/(?:abs|pdf)/(.+)$", text)
    if url:
        text = url.group(1)
    text = re.sub(r"^arxiv[:\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text.strip().rstrip("/"), flags=re.IGNORECASE)

    if re.fullmatch(_NEW_ID, text) or re.fullmatch(_OLD_ID, text):
        return text
    return None


def month_range(year: int, month: int | None = None) -> tuple[str, str]:
    """The ``submittedDate`` bounds for a whole year, or one month of it.

    arXiv wants ``YYYYMMDDHHMM`` in GMT. The end bound is the first instant of
    the following period, so December has to roll the year over.
    """
    if month is None:
        start, end = (year, 1), (year + 1, 1)
    elif month == 12:
        start, end = (year, 12), (year + 1, 1)
    else:
        start, end = (year, month), (year, month + 1)
    return (
        f"{start[0]:04d}{start[1]:02d}010000",
        f"{end[0]:04d}{end[1]:02d}010000",
    )


# -- formatting ---------------------------------------------------------------


def format_paper(
    paper: Paper,
    abstract_chars: int | None = None,
    author_limit: int | None = None,
) -> str:
    """One paper as a labelled plain-text block.

    ``abstract_chars`` truncates the abstract (search results list many papers;
    full abstracts would crowd out everything else). ``None`` keeps it whole.
    """
    authors = list(paper.authors)
    if author_limit and len(authors) > author_limit:
        shown = f"{', '.join(authors[:author_limit])} (+{len(authors) - author_limit} more)"
    else:
        shown = ", ".join(authors) or "unknown"

    categories = [c for c in paper.categories if c != paper.primary_category]
    category_line = ", ".join(
        filter(None, [f"{paper.primary_category} (primary)", *categories])
    )

    summary = paper.summary
    if abstract_chars and len(summary) > abstract_chars:
        summary = summary[:abstract_chars].rstrip() + " [...]"

    lines = [
        f"arXiv:{paper.arxiv_id}",
        f"Title: {paper.title}",
        f"Authors: {shown}",
        f"Submitted: {paper.published}",
    ]
    if paper.updated and paper.updated != paper.published:
        lines.append(f"Last updated: {paper.updated}")
    lines.append(f"Categories: {category_line}")
    if paper.journal_ref:
        lines.append(f"Journal ref: {paper.journal_ref}")
    if paper.doi:
        lines.append(f"DOI: {paper.doi}")
    if paper.comment:
        lines.append(f"Comment: {paper.comment}")
    lines.append(f"Abstract: {summary}")
    lines.append(f"Abstract page: {paper.abs_url}")
    if paper.pdf_url:
        lines.append(f"PDF: {paper.pdf_url}")
    return "\n".join(lines)


def format_results(
    results: Results,
    header: str,
    start: int = 0,
    abstract_chars: int | None = None,
    author_limit: int | None = None,
) -> str:
    """A result page, led by a count line so the model can offer to page on."""
    count = len(results.papers)
    first, last = start + 1, start + count
    noun = "paper" if results.total == 1 else "papers"
    shown = f"showing {first}" if count == 1 else f"showing {first}-{last}"
    lines = [f"Found {results.total} {noun} {header}; {shown}."]
    lines += [
        format_paper(paper, abstract_chars, author_limit) for paper in results.papers
    ]

    remaining = results.total - last
    if remaining > 0:
        lines.append(
            f"{remaining} more available -- call again with start={last} to continue."
        )
    return "\n\n".join(lines)
