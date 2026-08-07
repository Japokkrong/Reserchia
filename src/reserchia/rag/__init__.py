"""The paper library: chunk, embed, store and search papers the agent has read.

Settings here are not arbitrary -- they are the configuration measured as best
in `rag-eda/`: section-aligned chunks, bge-m3 as the encoder, ChromaDB with
cosine, and dense/lexical fusion at alpha 0.5. No code is shared with that
experiment; only its conclusions.
"""

# Modules are exported under their own names; the functions inside them are
# named so they can never shadow one -- `ingest_paper` not `ingest`,
# `search_library` not `search`. Getting that wrong turns
# `from .rag import search` into a function and every `search.foo()` into an
# AttributeError at runtime.
from . import ingest, search, store
from .chunking import Chunk, chunk_document
from .ingest import ingest_later, ingest_paper
from .search import Hit, search_library
from .store import base_id, count, has_paper, papers

__all__ = [
    "Chunk",
    "Hit",
    "base_id",
    "chunk_document",
    "count",
    "has_paper",
    "ingest",
    "ingest_later",
    "ingest_paper",
    "papers",
    "search",
    "search_library",
    "store",
]
