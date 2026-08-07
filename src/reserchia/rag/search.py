"""Hybrid search over the library: dense and lexical, mixed at alpha.

alpha defaults to 0.5, which the `rag-eda/` sweep measured as the peak on
Recall@1, @5, @10 and MRR simultaneously -- there is no recall/precision
tradeoff in this parameter. The top is flat from 0.4 to 0.6, so the exact value
is not worth defending; both *extremes* are clearly worse, with dense-only the
worst setting tested on this kind of corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from . import embeddings, lexical, store

#: Candidates taken from each retriever before fusing.
POOL = 50

#: Floor for a normalised score. Plain min-max maps a pool's worst item to
#: exactly 0.0, which makes it indistinguishable from an item that never
#: appeared in that pool at all -- and then at alpha=1.0 something only BM25
#: found can tie with dense's own last hit and win on sort order. Being in a
#: pool must always beat being absent from it. This is a real bug that the
#: alpha=1.0-equals-pure-dense check catches; do not remove the floor.
FLOOR = 1e-6


@dataclass(frozen=True)
class Hit:
    id: str
    arxiv_id: str
    title: str
    section: str
    text: str
    score: float
    abs_url: str

    @property
    def citation(self) -> str:
        return f"arXiv:{self.arxiv_id}" + (f" §{self.section}" if self.section else "")


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


def fuse(dense: dict, lexical_scores: dict, alpha: float) -> list[tuple[str, float]]:
    """alpha=1 is pure dense, alpha=0 is pure BM25, both exactly."""
    dense_n = _normalise(dense)
    lexical_n = _normalise(lexical_scores)
    combined = {
        key: alpha * dense_n.get(key, 0.0) + (1 - alpha) * lexical_n.get(key, 0.0)
        for key in set(dense_n) | set(lexical_n)
    }
    # Ties break on id, so repeated queries rank identically.
    return sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))


def search_library(
    query: str,
    paper_id: str | None = None,
    limit: int = 5,
    settings: Settings | None = None,
) -> list[Hit]:
    """Hybrid search over stored chunks.

    Named `search_library`, not `search`, so importing it from the package
    cannot shadow this module -- the same trap `ingest_paper` avoids.
    """
    settings = settings or get_settings()
    if not store.count(settings):
        return []

    arxiv_id = store.base_id(paper_id) if paper_id else None
    vector = embeddings.embed_one(query, settings)
    dense = store.query_dense(vector, POOL, arxiv_id, settings)
    lexical_scores = lexical.search(query, POOL, arxiv_id)

    rows = lexical.rows_by_id(arxiv_id)
    hits = []
    for chunk_id, score in fuse(dense, lexical_scores, settings.rag_alpha)[:limit]:
        row = rows.get(chunk_id)
        if row is None:
            continue
        hits.append(
            Hit(
                id=chunk_id,
                arxiv_id=row.get("arxiv_id", ""),
                title=row.get("title", ""),
                section=row.get("section", ""),
                text=row.get("text", ""),
                score=score,
                abs_url=row.get("abs_url", ""),
            )
        )
    return hits
