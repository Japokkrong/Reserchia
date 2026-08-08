"""Full paper text, from arXiv's LaTeXML HTML with a PDF fallback.

The metadata API in ``arxiv_client`` gives abstracts. This gives bodies, which
is what summarising a paper actually needs.

Three sources, tried in order, because no single one covers the archive:

1. ``arxiv.org/html/<id>`` -- arXiv's own LaTeXML rendering. Covers recent
   papers, including ones ar5iv has not converted yet.
2. ``ar5iv.labs.arxiv.org/html/<id>`` -- LaTeXML over the back catalogue.
   **It reports failure by redirecting to /abs/, not with a status code**, so
   the final URL is what has to be checked.
3. ``arxiv.org/pdf/<id>`` -- text extraction, which works everywhere and reads
   badly. Two-column layouts interleave and math does not survive, so this
   tier is labelled as lower fidelity for the model's benefit.

Probed coverage: of eight papers spanning 1996 to 2026, arXiv HTML had three,
ar5iv had five, and the two together had seven. Only a 1996 hep-th paper
needed the PDF.

Parsing is stdlib ``html.parser``; no beautifulsoup or lxml. LaTeXML output is
regular enough to make that easy, and one detail makes it worthwhile: every
``<math>`` element carries an ``alttext`` attribute holding the original LaTeX,
so formulas come out as ``$h_{t}$`` rather than as flattened MathML rubble.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx

from .arxiv_client import client, throttle
from .observability import track

HTML_URL = "https://arxiv.org/html/{}"
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{}"
PDF_URL = "https://arxiv.org/pdf/{}"

SOURCE_HTML = "arXiv HTML (LaTeXML)"
SOURCE_AR5IV = "ar5iv (LaTeXML)"
SOURCE_PDF = "PDF text extraction"

#: A converted survey can exceed 3 MB of HTML; this only stops the pathological.
MAX_BYTES = 12 * 1024 * 1024

#: Parsed documents are large, so keep few. Enough that asking for several
#: sections of one paper, or comparing two, costs no re-download.
CACHE_LIMIT = 4

_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img",
    "input", "link", "meta", "source", "track", "wbr",
}
_SKIP_TAGS = {
    "script", "style", "nav", "header", "footer",
    "button", "form", "select", "option", "svg",
}
#: LaTeXML/arxiv.org page furniture that is not part of the paper.
_SKIP_CLASSES = (
    "ltx_bibliography",
    "ltx_page_navbar",
    "ltx_page_header",
    "ltx_page_footer",
    "ltx_authors",
    "package-alerts",
    "extra-services",
)
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BREAKS = {"p", "div", "section", "li", "tr", "br", "figcaption", "table", "figure"}

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", re.S)

#: PDF text keeps typographic ligatures as single codepoints, which break word
#: matching. The spurious mid-word spaces PDF extraction also produces
#: ("bla ck hole") are left alone -- there is no safe way to tell them from
#: real ones, which is part of why that tier is labelled lower fidelity.
_LIGATURES = str.maketrans(
    {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬆ": "st", "ﬅ": "ft"}
)


@dataclass(frozen=True)
class Section:
    #: "3.2" for a numbered heading, "" for the title or an unnumbered one.
    number: str
    heading: str
    text: str


@dataclass(frozen=True)
class Document:
    arxiv_id: str
    title: str
    source: str
    sections: tuple[Section, ...]

    @property
    def text(self) -> str:
        parts = []
        for section in self.sections:
            if section.heading:
                parts.append(f"## {section.heading}")
            if section.text:
                parts.append(section.text)
        return "\n\n".join(parts)

    @property
    def lower_fidelity(self) -> bool:
        return self.source == SOURCE_PDF


# -- parsing ------------------------------------------------------------------


class _LatexmlParser(HTMLParser):
    """Pull readable text and a section index out of LaTeXML HTML.

    Emits nothing until ``<article>`` opens, which is what keeps arxiv.org's
    page chrome -- cookie dialogs, the fundraising banner, the feedback form --
    out of the extracted paper.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.sections: list[Section] = []
        self._buffer: list[str] = []
        self._heading: list[str] = []
        self._open_heading = ""
        self._depth = 0
        self._skip_below: int | None = None
        self._in_article = False
        self._article_depth = 0
        self._collecting_heading = False
        self._want_title = True

    # -- state helpers

    @property
    def _skipping(self) -> bool:
        return self._skip_below is not None

    def _emit(self, text: str) -> None:
        if self._in_article and not self._skipping:
            self._buffer.append(text)

    def _close_section(self) -> None:
        text = _tidy("".join(self._buffer))
        self._buffer = []
        if not (self._open_heading or text):
            return
        match = _NUMBERED.match(self._open_heading)
        self.sections.append(
            Section(
                number=match.group(1) if match else "",
                heading=self._open_heading,
                text=text,
            )
        )

    # -- HTMLParser hooks

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _VOID:
            if tag == "br":
                self._emit("\n")
            return

        self._depth += 1
        attributes = dict(attrs)

        if tag == "article" and not self._in_article:
            self._in_article = True
            self._article_depth = self._depth
            return

        if self._skipping:
            return

        classes = attributes.get("class", "")
        if tag in _SKIP_TAGS or any(name in classes for name in _SKIP_CLASSES):
            self._skip_below = self._depth - 1
            return

        if tag == "math":
            # The LaTeX source, rather than the MathML spelling of it.
            alttext = attributes.get("alttext")
            if alttext:
                self._emit(f" ${alttext}$ ")
            self._skip_below = self._depth - 1
            return

        if tag in _HEADINGS and self._in_article:
            self._close_section()
            self._collecting_heading = True
            self._heading = []
            return

        if tag in _BREAKS:
            self._emit("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return

        if self._collecting_heading and tag in _HEADINGS:
            self._collecting_heading = False
            self._open_heading = _tidy("".join(self._heading))
            if self._want_title and self._open_heading:
                self.title = self._open_heading
                self._want_title = False

        if self._skipping and self._depth <= (self._skip_below or 0) + 1:
            self._skip_below = None

        if self._in_article and tag == "article" and self._depth == self._article_depth:
            self._in_article = False

        self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skipping or not self._in_article:
            return
        if self._collecting_heading:
            self._heading.append(data)
            return
        self._buffer.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._close_section()


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_html(html: str, arxiv_id: str, source: str) -> Document:
    parser = _LatexmlParser()
    parser.feed(html)
    parser.close()
    sections = [s for s in parser.sections if s.heading or s.text]
    return Document(
        arxiv_id=arxiv_id,
        title=parser.title,
        source=source,
        sections=tuple(sections),
    )


# -- fetching -----------------------------------------------------------------


@dataclass(frozen=True)
class Fetched:
    status: int
    #: After redirects -- ar5iv's failure signal lives here, not in `status`.
    url: str
    body: bytes
    encoding: str

    def text(self) -> str:
        return self.body.decode(self.encoding or "utf-8", errors="replace")


def _download(url: str) -> Fetched | str:
    """One throttled GET, capped in size. Returns the body or an error.

    Streamed rather than buffered so an unexpectedly huge conversion is
    abandoned partway instead of after it has all arrived.
    """
    with track("arxiv.throttle", "tool"):
        throttle()
    try:
        with track("arxiv.download", "tool", url=url.rsplit("/", 2)[-2]) as span, \
                client.stream("GET", url) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    return (
                        f"Error: {url} is larger than "
                        f"{MAX_BYTES // (1024 * 1024)} MB; refusing to load it."
                    )
            span.set(status=response.status_code, bytes=len(body))
            return Fetched(
                status=response.status_code,
                url=str(response.url),
                body=bytes(body),
                encoding=response.charset_encoding or "utf-8",
            )
    except httpx.RequestError as exc:
        return (
            f"Error: could not reach {url} ({type(exc).__name__}). "
            "Tell the user rather than guessing the paper's contents."
        )


def _from_pdf(data: bytes, arxiv_id: str) -> Document | str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf raises many shapes
        return (
            f"Error: could not read the PDF for arXiv:{arxiv_id} "
            f"({type(exc).__name__})."
        )

    text = _tidy("\n\n".join(pages)).translate(_LIGATURES)
    if not text.strip():
        return (
            f"Error: the PDF for arXiv:{arxiv_id} has no extractable text "
            "(it is probably scanned images)."
        )
    return Document(
        arxiv_id=arxiv_id,
        title="",
        source=SOURCE_PDF,
        sections=(Section(number="", heading="", text=text),),
    )


_CACHE: dict[str, Document] = {}


def fetch_document(arxiv_id: str) -> Document | str:
    """Get a paper's body, trying each source in turn.

    Returns a ``Document``, or a readable error string -- never raises, so the
    tool layer can hand the message straight to the model.
    """
    if arxiv_id in _CACHE:
        return _CACHE[arxiv_id]

    errors: list[str] = []

    for url, source in (
        (HTML_URL.format(arxiv_id), SOURCE_HTML),
        (AR5IV_URL.format(arxiv_id), SOURCE_AR5IV),
    ):
        fetched = _download(url)
        if isinstance(fetched, str):
            errors.append(fetched)
            continue
        if fetched.status != 200:
            continue
        # ar5iv bounces to the abstract page when it has no conversion; that
        # redirect is the only signal, since the status is still 200.
        if "/abs/" in fetched.url:
            continue
        document = parse_html(fetched.text(), arxiv_id, source)
        if document.sections:
            return _remember(arxiv_id, document)

    fetched = _download(PDF_URL.format(arxiv_id))
    if isinstance(fetched, str):
        errors.append(fetched)
    elif fetched.status == 200:
        document = _from_pdf(fetched.body, arxiv_id)
        if isinstance(document, Document):
            return _remember(arxiv_id, document)
        errors.append(document)

    if errors:
        return errors[0]
    return (
        f"Error: no full text is available for arXiv:{arxiv_id}. arXiv has no "
        "LaTeX-derived HTML for it and the PDF could not be read."
    )


def _remember(arxiv_id: str, document: Document) -> Document:
    _CACHE[arxiv_id] = document
    while len(_CACHE) > CACHE_LIMIT:
        del _CACHE[next(iter(_CACHE))]
    return document


# -- section lookup -----------------------------------------------------------


def find_section(document: Document, wanted: str) -> int | None:
    """Resolve a section by number ('3', '3.2') or name ('Attention').

    Returns its position, because callers need the sections that follow it too.
    """
    target = (wanted or "").strip().rstrip(".").casefold()
    if not target:
        return None

    for index, section in enumerate(document.sections):
        if section.heading.casefold() == target or section.number.casefold() == target:
            return index

    for index, section in enumerate(document.sections):
        match = _NUMBERED.match(section.heading)
        title = (
            match.group(2).strip().casefold() if match else section.heading.casefold()
        )
        if title == target:
            return index

    for index, section in enumerate(document.sections):
        if target in section.heading.casefold():
            return index
    return None


def subtree(document: Document, index: int) -> list[Section]:
    """A section together with its subsections.

    LaTeXML stores "3", "3.1" and "3.2" as three flat siblings, so returning
    only the matched one hands back the paragraph before the first subsection
    -- 556 characters of a 60,000-character chapter. Anyone asking for section
    3 means all of it.
    """
    root = document.sections[index]
    collected = [root]
    if not root.number:
        # An unnumbered heading has no subsections to gather.
        return collected

    for section in document.sections[index + 1 :]:
        if section.number and not section.number.startswith(f"{root.number}."):
            break
        collected.append(section)
    return collected


def render(sections: list[Section]) -> str:
    parts = []
    for section in sections:
        if section.heading:
            parts.append(f"## {section.heading}")
        if section.text:
            parts.append(section.text)
    return "\n\n".join(parts)


def section_index(document: Document) -> str:
    """The contents listing, for when a paper is too long to return whole.

    Sizes are cumulative over subsections, so they say what asking for that
    section would actually cost.
    """
    lines = []
    for index, section in enumerate(document.sections):
        if not section.heading:
            continue
        size = sum(len(part.text) for part in subtree(document, index))
        lines.append(f"  {section.heading} ({size:,} chars)")
    return "\n".join(lines)
