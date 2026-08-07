"""Shared paths, tokenisation and IO for the RAG EDA.

The two corpus papers are arXiv submissions, so `reserchia.arxiv_fulltext`
already knows how to get both of their representations -- the PDF text tier and
the LaTeXML HTML tier with its section tree. Nothing here re-implements that.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXTRACTED = DATA / "extracted"
CHUNKS = DATA / "chunks"
CHROMA = DATA / "chroma"
CACHE = DATA / "embedding-cache"
FIGURES = ROOT / "figures"

#: The corpus. Both are OCR/VLM papers, and they overlap enough -- benchmarks,
#: layout analysis, compact-model tradeoffs -- that cross-paper confusion is a
#: real retrieval signal rather than a contrived one.
PAPERS = {
    "2601.14722v1": "Typhoon OCR (Thai document extraction)",
    "2603.10910v2": "GLM-OCR Technical Report",
}

SOURCES = ("pdf", "html")

EMBED_MODEL = "baai/bge-m3"
EMBED_DIM = 1024
#: bge-m3's own limit, per OpenRouter's model metadata.
EMBED_CONTEXT = 8192


def ensure_dirs() -> None:
    for path in (EXTRACTED, CHUNKS, CHROMA, CACHE, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


# -- tokenisation -------------------------------------------------------------

_tokenizer = None
_tokenizer_kind = ""


def _load_tokenizer():
    """bge-m3's real tokenizer if we can reach it, a labelled proxy if not.

    Chunk sizes are the whole point of this EDA, so counting in the model's own
    tokens matters. ``tokenizers`` pulls just the tokenizer.json -- no torch,
    no transformers.
    """
    global _tokenizer, _tokenizer_kind
    if _tokenizer is not None:
        return _tokenizer
    try:
        from tokenizers import Tokenizer

        _tokenizer = Tokenizer.from_pretrained("BAAI/bge-m3")
        _tokenizer_kind = "bge-m3 (exact)"
    except Exception:  # noqa: BLE001 - offline, HF down, any of it
        import tiktoken

        _tokenizer = tiktoken.get_encoding("cl100k_base")
        _tokenizer_kind = "cl100k_base (APPROXIMATE -- bge-m3 unavailable)"
    return _tokenizer


def tokenizer_kind() -> str:
    _load_tokenizer()
    return _tokenizer_kind


def encode(text: str) -> list[int]:
    tokenizer = _load_tokenizer()
    if hasattr(tokenizer, "encode_ordinary"):  # tiktoken
        return tokenizer.encode_ordinary(text)
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode(tokens: list[int]) -> str:
    tokenizer = _load_tokenizer()
    return tokenizer.decode(tokens)


def n_tokens(text: str) -> int:
    return len(encode(text))


def token_offsets(text: str) -> list[tuple[int, int]]:
    """Character span of every token, so token windows map back to the source.

    Ground truth is matched on character spans, so a chunker that cuts on token
    boundaries still has to say where in the document it cut.
    """
    tokenizer = _load_tokenizer()
    if hasattr(tokenizer, "encode_ordinary"):  # tiktoken has no offset map
        # Proportional estimate. Only reached when the real tokenizer is
        # unavailable, and `tokenizer_kind()` already says so loudly.
        ids = tokenizer.encode_ordinary(text)
        step = len(text) / max(1, len(ids))
        return [(int(i * step), int((i + 1) * step)) for i in range(len(ids))]
    return tokenizer.encode(text, add_special_tokens=False).offsets


# -- records ------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    id: str
    paper: str
    source: str
    config: str
    text: str
    #: Character span in the extracted text of (paper, source). Ground truth is
    #: matched against these, so a single test set works across every chunking
    #: configuration -- chunk ids could not do that, they differ per config.
    start: int
    end: int
    section: str
    n_tokens: int


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, produced_by: str) -> None:
    """Fail loudly and usefully when a pipeline stage was skipped."""
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} -- run {produced_by} first")


# -- text helpers -------------------------------------------------------------

#: Model names and identifiers must survive tokenisation intact: PP-DocLayout-V3
#: and bge-m3 are exactly the queries where lexical retrieval should win, and a
#: tokenizer that splits on '-' throws that away.
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")


def lexical_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text)]


#: A line that is only the word "References" (or "REFERENCES"/"Bibliography"),
#: which is how both corpus papers begin their bibliography in PDF text. The
#: LaTeXML tier already drops the bibliography, so this only ever fires on PDF.
_REFERENCES = re.compile(r"^[ \t]*(?:##\s*)?(?:references|bibliography)[ \t]*$", re.I | re.M)


def split_references(text: str) -> tuple[str, str]:
    """Separate body prose from the bibliography.

    Worth doing explicitly: in the corpus PDFs the reference list is 21-27% of
    the extracted text, and it is pure retrieval noise -- author names and
    venue abbreviations that match queries lexically while answering nothing.
    Treating it as body text would flatter BM25 and pollute the chunk stats.
    """
    matches = list(_REFERENCES.finditer(text))
    if not matches:
        return text, ""
    # The last one, so a "References" mention inside the body does not win.
    cut = matches[-1].start()
    return text[:cut].rstrip(), text[cut:].strip()
