"""The paper library: a persistent Chroma collection of chunks.

One collection, `hnsw:space=cosine`. bge-m3 hands back L2-normalised vectors,
so cosine distance and dot product agree and no renormalisation is needed.

The interesting function here is `has_paper`. It is a metadata-filtered count,
so it answers "have I read this paper?" exactly and cheaply -- which is what
the agent's fallback rule turns on. That matters because the alternative,
thresholding a similarity score, was measured and does not work: correct top-1
hits scored as low as 0.385 while out-of-corpus questions scored 0.492-0.498,
so no cutoff separates them.
"""

from __future__ import annotations

import threading

import chromadb
from chromadb.api.models.Collection import Collection

from ..config import Settings, get_settings
from .chunking import Chunk

COLLECTION = "papers"

_lock = threading.Lock()
_collection: Collection | None = None


def collection(settings: Settings | None = None) -> Collection:
    """The library, created on first use. Safe to call from any thread."""
    global _collection
    with _lock:
        if _collection is None:
            settings = settings or get_settings()
            settings.store_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(settings.store_dir / "chroma"))
            _collection = client.get_or_create_collection(
                name=COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return _collection


def has_paper(arxiv_id: str, settings: Settings | None = None) -> bool:
    """Exactly whether this paper is in the library. No scores involved."""
    if not arxiv_id:
        return False
    found = collection(settings).get(
        where={"arxiv_id": base_id(arxiv_id)}, limit=1, include=[]
    )
    return bool(found["ids"])


def base_id(arxiv_id: str) -> str:
    """Drop a version suffix. v1 and v2 of a paper are the same paper here."""
    import re

    return re.sub(r"v\d+$", "", (arxiv_id or "").strip())


def papers(settings: Settings | None = None) -> list[tuple[str, str, int]]:
    """Every paper in the library, as (arxiv_id, title, chunk count)."""
    rows = collection(settings).get(include=["metadatas"])
    counts: dict[str, list] = {}
    for metadata in rows["metadatas"] or []:
        key = metadata.get("arxiv_id", "")
        entry = counts.setdefault(key, [metadata.get("title", ""), 0])
        entry[1] += 1
    return sorted(
        ((key, title, count) for key, (title, count) in counts.items()),
        key=lambda row: row[0],
    )


def upsert(chunks: list[Chunk], vectors: list[list[float]], settings=None) -> int:
    """Write chunks. Ids are deterministic, so re-ingesting replaces in place."""
    if not chunks:
        return 0
    collection(settings).upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=vectors,
        metadatas=[
            {
                "arxiv_id": base_id(chunk.arxiv_id),
                "title": chunk.title,
                "section": chunk.section,
                "number": chunk.number,
                "index": chunk.index,
                "abs_url": chunk.abs_url,
            }
            for chunk in chunks
        ],
    )
    return len(chunks)


def count(settings: Settings | None = None) -> int:
    return collection(settings).count()


def fetch_all(arxiv_id: str | None = None, settings=None) -> list[dict]:
    """Documents and metadata, optionally for one paper. Feeds the BM25 index."""
    where = {"arxiv_id": base_id(arxiv_id)} if arxiv_id else None
    rows = collection(settings).get(where=where, include=["documents", "metadatas"])
    return [
        {"id": cid, "text": text, **(metadata or {})}
        for cid, text, metadata in zip(
            rows["ids"], rows["documents"] or [], rows["metadatas"] or []
        )
    ]


def query_dense(
    vector: list[float],
    limit: int,
    arxiv_id: str | None = None,
    settings=None,
) -> dict[str, float]:
    """Nearest chunks as {id: cosine similarity}."""
    total = count(settings)
    if not total:
        return {}
    where = {"arxiv_id": base_id(arxiv_id)} if arxiv_id else None
    result = collection(settings).query(
        query_embeddings=[vector],
        n_results=min(limit, total),
        where=where,
        include=["distances"],
    )
    ids = result["ids"][0]
    distances = result["distances"][0]
    # Cosine distance -> similarity, so both retrievers agree that bigger is better.
    return {cid: 1.0 - distance for cid, distance in zip(ids, distances)}


def reset_cache() -> None:
    """Drop the cached handle. For tests that repoint the store directory."""
    global _collection
    with _lock:
        _collection = None
