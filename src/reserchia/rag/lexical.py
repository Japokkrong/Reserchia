"""BM25 over the library, for the lexical half of hybrid search.

Chroma does ship a native hybrid search with `Knn` and `Rrf` ranking
expressions, but it raises `NotImplementedError` on a local `PersistentClient`
-- it is implemented only for distributed and hosted Chroma. So the lexical
side stays in Python.

Scaling note, measured rather than guessed: `rank_bm25` scores every document
in the corpus, taking 0.09 ms at 79 chunks, 4.8 ms at 7,900 and 37.8 ms at
39,500. That is fine for a personal library and still far below the ~2 s the
query-embedding round trip costs. Past roughly ten thousand chunks it becomes
the dominant local cost and this should move to `bm25s` or a real inverted
index.
"""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from . import store

#: Model names and identifiers must survive tokenisation whole. PP-DocLayout-V3
#: and bge-m3 are exactly the queries where lexical retrieval beats the encoder,
#: and splitting on '-' throws that advantage away.
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text)]


class _Index:
    __slots__ = ("bm25", "ids", "rows", "size")

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.ids = [row["id"] for row in rows]
        self.size = len(rows)
        self.bm25 = BM25Okapi([tokenize(row["text"]) for row in rows]) if rows else None


_lock = threading.Lock()
_cache: dict[str, _Index] = {}


def invalidate() -> None:
    """Called after a write, since the corpus statistics have changed."""
    with _lock:
        _cache.clear()


def _index(arxiv_id: str | None) -> _Index:
    key = arxiv_id or "*"
    with _lock:
        cached = _cache.get(key)
        # The count is a cheap staleness check: an ingest between queries
        # changes it, and BM25's idf depends on the whole corpus.
        expected = store.count() if arxiv_id is None else None
        if cached is not None and (expected is None or cached.size == expected):
            return cached

        index = _Index(store.fetch_all(arxiv_id))
        _cache[key] = index
        return index


def search(query: str, limit: int, arxiv_id: str | None = None) -> dict[str, float]:
    """Best lexical matches as {id: BM25 score}."""
    index = _index(arxiv_id)
    if index.bm25 is None:
        return {}
    scores = index.bm25.get_scores(tokenize(query))
    ranked = sorted(zip(index.ids, scores), key=lambda pair: -pair[1])[:limit]
    return {cid: float(score) for cid, score in ranked}


def rows_by_id(arxiv_id: str | None = None) -> dict[str, dict]:
    return {row["id"]: row for row in _index(arxiv_id).rows}
