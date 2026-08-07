"""Extract both representations of each paper, and measure how they differ.

PDF text and LaTeXML HTML are two views of the same paper, and the gap between
them is the first EDA result: whichever chunking strategy wins, it is operating
on one of these, and the input quality bounds everything downstream.

Run: python rag-eda/01_extract.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA,
    EXTRACTED,
    PAPERS,
    ensure_dirs,
    lexical_tokens,
    n_tokens,
    split_references,
    write_json,
)

from reserchia.arxiv_fulltext import _from_pdf, fetch_document  # noqa: E402

#: Typographic ligatures, counted on RAW pypdf output -- `_from_pdf` normalises
#: them, so measuring post-normalisation would always report zero and say
#: nothing about what the PDF actually contained.
_LIGATURES = re.compile(r"[ﬀ-ﬆ]")
_HEADING = re.compile(r"^#{2}\s+(.+)$", re.M)
#: "[12] Author, Title, 2023." -- bibliography rather than prose.
_CITATION = re.compile(r"\[\d{1,3}\]")


def artefacts(text: str) -> dict:
    """The concrete ways extracted text can be wrong."""
    return {
        "chars": len(text),
        "tokens": n_tokens(text),
        # An odd count means at least one formula is not closed.
        "unbalanced_math": text.count("$") % 2,
        "math_spans": text.count("$") // 2,
        "headings": len(_HEADING.findall(text)),
        "citation_markers": len(_CITATION.findall(text)),
        "references_chars": len(split_references(text)[1]),
        "references_share": round(len(split_references(text)[1]) / max(1, len(text)), 3),
    }


def raw_ligatures(paper: str) -> int:
    """What the PDF held before `_from_pdf` normalised it."""
    from pypdf import PdfReader

    reader = PdfReader(DATA / f"{paper}.pdf")
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return len(_LIGATURES.findall(raw))


def corruption(text: str, reference: str) -> dict:
    """Vocabulary this source has that the other does not, bodies only.

    Detecting kerning splits ("bla ck hole") by pattern is hopeless -- any
    regex loose enough to catch them also matches ordinary English like "of
    the". But the same paper exists in two extractions, so words present in one
    and absent from the other measure what differs, with no dictionary needed.

    Both sides are stripped of their bibliography first. Without that, this
    measured nothing but the fact that PDF text keeps references and LaTeXML
    drops them -- the "fragments" it surfaced were author surnames and venue
    abbreviations (acl, al, aug, bai), not extraction damage.
    """
    mine = set(lexical_tokens(split_references(text)[0]))
    theirs = set(lexical_tokens(split_references(reference)[0]))
    only = mine - theirs
    fragments = {word for word in only if len(word) <= 3 and word.isalpha()}
    return {
        "body_vocabulary": len(mine),
        "absent_from_other_source": len(only),
        "short_fragments": len(fragments),
        "fragment_examples": sorted(fragments)[:12],
    }


def extract_pdf(paper: str) -> dict:
    pdf = DATA / f"{paper}.pdf"
    document = _from_pdf(pdf.read_bytes(), paper)
    if isinstance(document, str):
        sys.exit(f"{paper}: {document}")
    return document_payload(document, "pdf")


def extract_html(paper: str) -> dict:
    document = fetch_document(paper)
    if isinstance(document, str):
        sys.exit(f"{paper}: {document}")
    return document_payload(document, "html")


def document_payload(document, source: str) -> dict:
    text = document.text
    return {
        "paper": document.arxiv_id,
        "source": source,
        "origin": document.source,
        "title": document.title,
        "text": text,
        "sections": [
            {"number": s.number, "heading": s.heading, "text": s.text}
            for s in document.sections
        ],
        "stats": artefacts(text),
    }


def main() -> None:
    ensure_dirs()
    rows = []

    for paper, label in PAPERS.items():
        print(f"\n{paper} -- {label}")
        payloads = {"pdf": extract_pdf(paper), "html": extract_html(paper)}

        for source, payload in payloads.items():
            other = payloads["html" if source == "pdf" else "pdf"]["text"]
            payload["corruption"] = corruption(payload["text"], other)
            if source == "pdf":
                payload["stats"]["raw_ligatures"] = raw_ligatures(paper)
            write_json(EXTRACTED / f"{paper}.{source}.json", payload)

            stats = payload["stats"]
            rows.append(payload)
            print(
                f"  {source:5s} {payload['origin']:22s} "
                f"{stats['chars']:>8,} chars  {stats['tokens']:>7,} tok  "
                f"{len(payload['sections']):>3} sections"
            )

    print(f"\n{'paper':16s} {'src':5s} {'sections':>9s} {'math':>6s} {'refs%':>7s} "
          f"{'body vocab':>11s} {'only-here':>10s} {'fragments':>10s}")
    for payload in rows:
        stats, corrupt = payload["stats"], payload["corruption"]
        print(
            f"{payload['paper']:16s} {payload['source']:5s} "
            f"{len(payload['sections']):>9,} {stats['math_spans']:>6,} "
            f"{100 * stats['references_share']:>6.0f}% {corrupt['body_vocabulary']:>11,} "
            f"{corrupt['absent_from_other_source']:>10,} "
            f"{corrupt['short_fragments']:>10,}"
        )

    print("\nbody-only fragments unique to the PDF extraction (bibliography excluded):")
    for payload in rows:
        if payload["source"] == "pdf":
            examples = payload["corruption"]["fragment_examples"]
            print(f"  {payload['paper']}: {', '.join(examples) or '(none)'}")
            print(f"    raw ligatures in the PDF: {payload['stats']['raw_ligatures']:,}")

    write_json(
        EXTRACTED / "summary.json",
        [
            {
                "paper": p["paper"],
                "source": p["source"],
                "origin": p["origin"],
                "sections": len(p["sections"]),
                **p["stats"],
                **{k: v for k, v in p["corruption"].items() if k != "fragment_examples"},
            }
            for p in rows
        ],
    )
    print("\nwrote data/extracted/")


if __name__ == "__main__":
    main()
