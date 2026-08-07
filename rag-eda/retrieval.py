"""Dense, lexical, and the fusion of the two, plus ranking metrics.

The one thing that must not be got wrong here is score normalisation. Chroma
returns cosine **distance** (smaller is better, roughly 0..2) and BM25 returns
an unbounded relevance score (larger is better, corpus-dependent). Adding them
raw and calling the coefficient "alpha" produces a sweep that measures nothing
-- whichever score happens to have the larger magnitude dominates at every
alpha. So both are min-max normalised over the same candidate pool per query
before they are combined, and the sanity check for it is that alpha=1 must
reproduce pure dense ranking and alpha=0 pure BM25, exactly.

RRF is included alongside as a scale-free comparator: it fuses ranks rather
than scores, so it cannot suffer the problem at all.
"""

from __future__ import annotations

import math

from common import lexical_tokens
from rank_bm25 import BM25Okapi

POOL = 50
RRF_K = 60


class Index:
    """One chunking configuration, searchable both ways."""

    def __init__(self, collection, rows: list[dict]) -> None:
        self.collection = collection
        self.rows = rows
        self.by_id = {row["id"]: row for row in rows}
        self.ids = [row["id"] for row in rows]
        # Hyphenated identifiers stay whole: PP-DocLayout-V3 and bge-m3 are
        # exactly the queries lexical retrieval should win, and a tokenizer
        # that splits on '-' throws that advantage away.
        self.bm25 = BM25Okapi([lexical_tokens(row["text"]) for row in rows])

    def dense(self, query_vector: list[float], k: int = POOL) -> dict[str, float]:
        result = self.collection.query(
            query_embeddings=[query_vector], n_results=min(k, len(self.ids))
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Cosine distance -> similarity, so bigger is better for both retrievers.
        return {cid: 1.0 - dist for cid, dist in zip(ids, distances)}

    def lexical(self, query: str, k: int = POOL) -> dict[str, float]:
        scores = self.bm25.get_scores(lexical_tokens(query))
        ranked = sorted(zip(self.ids, scores), key=lambda pair: -pair[1])[:k]
        return {cid: float(score) for cid, score in ranked}


#: Floor for a retrieved document's normalised score. Plain min-max maps the
#: worst item in a pool to exactly 0.0, which makes it indistinguishable from a
#: document that never appeared in that pool at all -- and then at alpha=1.0 an
#: item BM25 found but dense did not can tie with dense's own last result and
#: win on sort order. Being in the pool must always beat being absent from it.
FLOOR = 1e-6


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return {key: 1.0 for key in scores}
    return {
        key: FLOOR + (1.0 - FLOOR) * (value - low) / (high - low)
        for key, value in scores.items()
    }


def fuse_weighted(dense, lexical, alpha: float) -> list[str]:
    """alpha=1 is pure dense, alpha=0 is pure BM25, both exactly."""
    dense_n, lexical_n = _normalise(dense), _normalise(lexical)
    combined = {}
    for key in set(dense_n) | set(lexical_n):
        # Absent from a pool scores a true 0 -- below FLOOR, so it always ranks
        # under anything that retriever actually returned.
        combined[key] = alpha * dense_n.get(key, 0.0) + (1 - alpha) * lexical_n.get(key, 0.0)
    # Ties break on id so a run is reproducible.
    return [key for key, _ in sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))]


def fuse_rrf(dense, lexical, k: int = RRF_K) -> list[str]:
    ranks: dict[str, float] = {}
    for scores in (dense, lexical):
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        for position, (key, _) in enumerate(ordered):
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (k + position + 1)
    return [key for key, _ in sorted(ranks.items(), key=lambda kv: (-kv[1], kv[0]))]


# -- relevance and metrics ----------------------------------------------------


def is_relevant(row: dict, gold: dict, coverage: float = 0.5) -> bool:
    """A chunk counts when it covers at least half the gold span.

    Span-based, not id-based: the 72 configurations produce different chunk
    ids, so only a positional definition of "correct" can score them all.
    """
    start, end = gold["start"], gold["end"]
    if end <= start:
        return False
    overlap = min(row["end"], end) - max(row["start"], start)
    return overlap > 0 and overlap >= coverage * (end - start)


def relevance_vector(ranking: list[str], by_id: dict, gold: dict) -> list[int]:
    return [1 if is_relevant(by_id[cid], gold) else 0 for cid in ranking if cid in by_id]


def recall_at(rel: list[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(1.0, sum(rel[:k]) / total_relevant)


def hit_at(rel: list[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def mrr_at(rel: list[int], k: int = 10) -> float:
    for index, value in enumerate(rel[:k], start=1):
        if value:
            return 1.0 / index
    return 0.0


def ndcg_at(rel: list[int], k: int = 10) -> float:
    gain = sum(value / math.log2(index + 1) for index, value in enumerate(rel[:k], start=1))
    ideal_hits = min(sum(rel), k)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return gain / ideal if ideal else 0.0
