"""Resolving a citation the model wrote back to the passage it came from.

The agent is told to cite as `[arXiv:2603.10910 §3.1 Public Benchmarks]`, and
it does -- approximately. In practice it abbreviates: `[arXiv:1706.03762 §3.5]`
for a section headed "3.5 Positional Encoding", `[arXiv:1706.03762 §Residual
Dropout]` for one with no number at all. Matching stored labels exactly would
miss most real citations, so `resolve` works down from exact to progressively
looser matches.

The registry is process-wide and never cleared per turn. The model happily
cites something retrieved several questions ago, and re-retrieving it just to
render a link would be absurd.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict

from .search import Hit

#: Passages worth keeping addressable. Each is a chunk of a paper, so a couple
#: of hundred is a few conversations' worth without unbounded growth.
LIMIT = 200

#: "arXiv:2603.10910 §3.1 Public Benchmarks" -> id and section reference.
#: The section part is optional; a bare paper citation is still a citation.
_LABEL = re.compile(
    r"arxiv:\s*(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-zA-Z-]+(?:\.[a-zA-Z-]+)?/\d{7}(?:v\d+)?)"
    r"(?:\s*[§#]\s*(?P<section>.+))?$",
    re.IGNORECASE,
)

_lock = threading.Lock()
_hits: OrderedDict[str, Hit] = OrderedDict()


def _base(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", (arxiv_id or "").strip())


def parse_label(label: str) -> tuple[str, str] | None:
    """Split a citation into (arxiv id, section reference).

    Returns None when the text is not a citation at all, which is how the UI
    knows to leave a bracketed phrase alone rather than linkify it.
    """
    match = _LABEL.match((label or "").strip().strip("[]").strip())
    if not match:
        return None
    return _base(match.group("id")), (match.group("section") or "").strip()


def remember(hits: list[Hit]) -> None:
    """Record retrieved passages so their citations can be resolved later."""
    with _lock:
        for hit in hits:
            _hits[hit.id] = hit
            _hits.move_to_end(hit.id)
        while len(_hits) > LIMIT:
            _hits.popitem(last=False)


def remember_sections(arxiv_id: str, title: str, sections) -> None:
    """Record sections read from full text, not just retrieved from the library.

    The first time a paper is discussed the answer comes from
    `get_arxiv_fulltext`, and every citation in it names a section that library
    search never returned. Without this, exactly the answers most likely to be
    read carefully are the ones whose citations cannot be opened.

    Score is 0.0 because nothing was ranked -- the section was read directly.
    """
    identifier = _base(arxiv_id)
    hits = [
        Hit(
            id=f"fulltext:{identifier}#{index}",
            arxiv_id=identifier,
            title=title or "",
            section=section.heading or "",
            text=section.text or "",
            score=0.0,
            abs_url=f"https://arxiv.org/abs/{identifier}",
        )
        for index, section in enumerate(sections)
        if (section.text or "").strip()
    ]
    remember(hits)


def resolve(label: str) -> Hit | None:
    """Find the passage a citation refers to, tolerating abbreviation."""
    parsed = parse_label(label)
    if not parsed:
        return None
    arxiv_id, section = parsed

    with _lock:
        candidates = [hit for hit in _hits.values() if _base(hit.arxiv_id) == arxiv_id]
    if not candidates:
        return None

    if not section:
        # A bare paper citation: the best passage retrieved from it will do,
        # and insertion order puts the most recent last.
        return candidates[-1]

    wanted = section.casefold().rstrip(".")

    for hit in candidates:  # exact heading
        if hit.section.casefold() == wanted:
            return hit

    for hit in candidates:  # "3.5" against "3.5 Positional Encoding"
        heading = hit.section.casefold()
        if heading.startswith(f"{wanted} ") or heading == wanted:
            return hit

    for hit in candidates:  # "Residual Dropout" appearing in the heading
        if wanted in hit.section.casefold():
            return hit

    return None


def known() -> int:
    with _lock:
        return len(_hits)


def clear() -> None:
    with _lock:
        _hits.clear()
