"""Chunking strategies, and the arXiv-specific question of what they break.

Every strategy returns **character spans into the source text**, not strings.
That is deliberate: the retrieval test set defines ground truth as a character
span, so one test set can score every configuration in the grid. Chunk ids
could not do that -- they differ per strategy, so a chunk-id gold set would
only ever be valid for the configuration that produced it.

The four strategies climb a ladder of how much document structure they use:

    fixed      none -- token windows, the naive baseline
    recursive  punctuation -- paragraph, then line, then sentence
    paragraph  blank lines -- never splits a paragraph
    section    the document's own section tree

For arXiv papers the last one is the interesting case, because LaTeXML gives
the real tree while a PDF only yields whatever a heading regex can recover.
The gap between those two is one of the results worth having.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common import Chunk, n_tokens, token_offsets

Span = tuple[int, int]

#: Coarse to fine, the order a recursive splitter should prefer to cut on.
SEPARATORS = ("\n\n", "\n", ". ", " ")

#: A numbered heading on its own line ("3 Model Architecture", "3.2 Attention"),
#: or an all-caps one ("ABSTRACT"). This is all a PDF gives us to rebuild the
#: section tree from -- the comparison against LaTeXML's real tree is the point.
_PDF_HEADING = re.compile(
    r"^[ \t]*(?:(\d+(?:\.\d+)*)[ \t.]+([A-Z][^\n]{2,70})|([A-Z][A-Z \t]{3,40}))[ \t]*$",
    re.M,
)
_MD_HEADING = re.compile(r"^##[ \t]+(.+)$", re.M)


@dataclass(frozen=True)
class Config:
    strategy: str
    size: int
    overlap: float
    heading_prefix: bool

    @property
    def name(self) -> str:
        overlap = f"o{int(self.overlap * 100)}"
        prefix = "hp1" if self.heading_prefix else "hp0"
        return f"{self.strategy}-{self.size}-{overlap}-{prefix}"


def grid() -> list[Config]:
    """The configurations to sweep.

    Overlap is only applied where it is meaningful: `paragraph` and `section`
    cut on structural boundaries, and sliding them by a token fraction would
    destroy the property that makes them worth testing.
    """
    configs = []
    for heading_prefix in (False, True):
        for size in (256, 512, 1024):
            for overlap in (0.0, 0.15):
                configs.append(Config("fixed", size, overlap, heading_prefix))
                configs.append(Config("recursive", size, overlap, heading_prefix))
            configs.append(Config("paragraph", size, 0.0, heading_prefix))
            configs.append(Config("section", size, 0.0, heading_prefix))
    return configs


# -- strategies ---------------------------------------------------------------


def fixed_spans(text: str, size: int, overlap: float) -> list[Span]:
    """Token windows, ignoring every boundary in the document."""
    offsets = token_offsets(text)
    if not offsets:
        return []
    step = max(1, size - int(size * overlap))
    spans = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + size]
        if not window:
            break
        spans.append((window[0][0], window[-1][1]))
        if start + size >= len(offsets):
            break
    return spans


def _recursive_split(text: str, start: int, end: int, seps, size: int) -> list[Span]:
    if end <= start:
        return []
    if n_tokens(text[start:end]) <= size:
        return [(start, end)]
    if not seps:
        # Nothing left to cut on: fall back to a hard token window so a single
        # unbroken run can never exceed the budget.
        return [
            (start + a, start + b) for a, b in fixed_spans(text[start:end], size, 0.0)
        ]

    separator, rest = seps[0], seps[1:]
    pieces, cursor = [], start
    for match in re.finditer(re.escape(separator), text[start:end]):
        cut = start + match.end()
        if cut > cursor:
            pieces.append((cursor, cut))
            cursor = cut
    if cursor < end:
        pieces.append((cursor, end))

    out: list[Span] = []
    for piece_start, piece_end in pieces:
        out.extend(_recursive_split(text, piece_start, piece_end, rest, size))
    return out


def _pack(text: str, units: list[Span], size: int, overlap: float) -> list[Span]:
    """Greedily merge adjacent units up to `size`, then apply overlap."""
    if not units:
        return []
    merged: list[list[Span]] = []
    current: list[Span] = []

    for unit in units:
        candidate = current + [unit]
        span = (candidate[0][0], candidate[-1][1])
        if current and n_tokens(text[span[0] : span[1]]) > size:
            merged.append(current)
            current = [unit]
        else:
            current = candidate
    if current:
        merged.append(current)

    spans = [(group[0][0], group[-1][1]) for group in merged]
    if overlap <= 0:
        return spans

    # Back each chunk's start up by `overlap` of its budget, in tokens.
    want = int(size * overlap)
    overlapped = [spans[0]]
    for index in range(1, len(spans)):
        start, end = spans[index]
        previous_start = spans[index - 1][0]
        offsets = token_offsets(text[previous_start:start])
        if offsets:
            back = offsets[max(0, len(offsets) - want)][0]
            start = previous_start + back
        overlapped.append((start, end))
    return overlapped


def recursive_spans(text: str, size: int, overlap: float) -> list[Span]:
    units = _recursive_split(text, 0, len(text), SEPARATORS, size)
    return _pack(text, units, size, overlap)


def paragraph_spans(text: str, size: int, _overlap: float = 0.0) -> list[Span]:
    """Pack whole paragraphs. A paragraph longer than `size` stays whole.

    That is the point of the strategy, and the resulting oversized chunks are a
    measurement, not a bug -- they show how often arXiv prose exceeds a budget.
    """
    units, cursor = [], 0
    for match in re.finditer(r"\n\s*\n", text):
        if match.end() > cursor:
            units.append((cursor, match.end()))
            cursor = match.end()
    if cursor < len(text):
        units.append((cursor, len(text)))
    return _pack(text, units, size, 0.0)


def section_spans(text: str, sections: list[dict], size: int) -> list[Span]:
    """One chunk per section, splitting oversized ones at paragraph breaks."""
    spans = []
    for start, end, _heading in locate_sections(text, sections):
        if n_tokens(text[start:end]) <= size:
            spans.append((start, end))
            continue
        for a, b in paragraph_spans(text[start:end], size):
            spans.append((start + a, start + b))
    return spans


# -- section geometry ---------------------------------------------------------


def locate_sections(text: str, sections: list[dict]) -> list[tuple[int, int, str]]:
    """Char span and heading of each section within the extracted text.

    HTML text is `render(sections)`, where every heading is prefixed with
    "## ". PDF text has no such marker -- headings are ordinary lines -- so a
    search for "## Introduction" finds nothing and every chunk comes back
    orphaned. Sections carrying their own offset (which `pdf_sections` records
    while scanning) skip the search entirely.
    """
    located, cursor = [], 0
    for section in sections:
        heading = section.get("heading") or ""

        if section.get("start") is not None:
            found = int(section["start"])
        else:
            needle = f"## {heading}" if heading else (section.get("text") or "")[:60]
            if not needle:
                continue
            found = text.find(needle, cursor)
            if found < 0 and heading:
                found = text.find(heading, cursor)
            if found < 0:
                continue
            cursor = found + len(needle)

        located.append([found, len(text), heading])
        if len(located) > 1:
            located[-2][1] = found
    return [(a, b, h) for a, b, h in located]


def pdf_sections(text: str) -> list[dict]:
    """Recover a section tree from PDF text using heading shapes alone.

    A PDF has no structure to read, only typography that survived extraction.
    What this finds versus what LaTeXML hands over is exactly the cost of
    chunking the PDF instead of the HTML.
    """
    found = []
    for match in _PDF_HEADING.finditer(text):
        number, titled, capped = match.groups()
        heading = f"{number} {titled}".strip() if titled else (capped or "").strip()
        if heading:
            found.append((match.start(), heading))
    if not found:
        return [{"number": "", "heading": "", "text": text, "start": 0}]

    sections = []
    for index, (start, heading) in enumerate(found):
        end = found[index + 1][0] if index + 1 < len(found) else len(text)
        # The offset is recorded here so `locate_sections` never has to search
        # for a heading that carries no "## " marker.
        sections.append(
            {"number": "", "heading": heading, "text": text[start:end], "start": start}
        )
    return sections


def heading_for(span: Span, located: list[tuple[int, int, str]]) -> str:
    for start, end, heading in located:
        if start <= span[0] < end:
            return heading
    return ""


# -- assembly -----------------------------------------------------------------


def build(
    text: str,
    sections: list[dict],
    paper: str,
    source: str,
    config: Config,
) -> list[Chunk]:
    if config.strategy == "fixed":
        spans = fixed_spans(text, config.size, config.overlap)
    elif config.strategy == "recursive":
        spans = recursive_spans(text, config.size, config.overlap)
    elif config.strategy == "paragraph":
        spans = paragraph_spans(text, config.size)
    elif config.strategy == "section":
        spans = section_spans(text, sections, config.size)
    else:
        raise ValueError(f"unknown strategy {config.strategy!r}")

    located = locate_sections(text, sections)
    chunks = []
    for index, (start, end) in enumerate(spans):
        body = text[start:end].strip()
        if not body:
            continue
        heading = heading_for((start, end), located)
        # The prefix goes into what gets embedded and indexed, while the span
        # still points at the untouched source -- so gold matching is unaffected
        # by whether the prefix is on.
        payload = f"{heading} -- {body}" if config.heading_prefix and heading else body
        chunks.append(
            Chunk(
                id=f"{paper}:{source}:{config.name}:{index:04d}",
                paper=paper,
                source=source,
                config=config.name,
                text=payload,
                start=start,
                end=end,
                section=heading,
                n_tokens=n_tokens(payload),
            )
        )
    return chunks
