"""Embed every chunking configuration into its own Chroma collection.

One collection per (configuration, source) so the retrieval sweep can compare
them without re-indexing. Vectors come back from bge-m3 already L2-normalised,
so cosine distance and dot product agree and Chroma's cosine space is correct.

Run: python rag-eda/03_embed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb  # noqa: E402
import chunkers  # noqa: E402
from common import (  # noqa: E402
    CHROMA,
    CHUNKS,
    EMBED_CONTEXT,
    SOURCES,
    ensure_dirs,
    read_jsonl,
    require,
)
from embedding import Embedder  # noqa: E402


def collection_name(config: str, source: str) -> str:
    return f"{config}__{source}".replace(".", "_")


def main() -> None:
    ensure_dirs()
    require(CHUNKS / "stats.json", "rag-eda/02_chunk_eda.py")

    client = chromadb.PersistentClient(path=str(CHROMA))
    embedder = Embedder()

    jobs = [
        (config.name, source)
        for config in chunkers.grid()
        for source in SOURCES
        if (CHUNKS / f"{config.name}.{source}.jsonl").exists()
    ]

    total_chunks = sum(
        len(read_jsonl(CHUNKS / f"{c}.{s}.jsonl")) for c, s in jobs
    )
    print(f"{len(jobs)} collections, {total_chunks:,} chunks total\n")

    oversized = 0
    for index, (config, source) in enumerate(jobs, 1):
        rows = read_jsonl(CHUNKS / f"{config}.{source}.jsonl")
        if not rows:
            continue

        name = collection_name(config, source)
        # Rebuild from scratch so a re-run can never mix two indexings.
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - absent is the normal case
            pass
        collection = client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

        texts = [row["text"] for row in rows]
        over = [row for row in rows if row["n_tokens"] > EMBED_CONTEXT]
        oversized += len(over)

        vectors = embedder.embed(texts, label=f"[{index}/{len(jobs)}] {name}")
        collection.add(
            ids=[row["id"] for row in rows],
            documents=texts,
            embeddings=vectors,
            metadatas=[
                {
                    "paper": row["paper"],
                    "source": row["source"],
                    "start": row["start"],
                    "end": row["end"],
                    "section": row["section"] or "",
                    "n_tokens": row["n_tokens"],
                }
                for row in rows
            ],
        )
        print(f"[{index}/{len(jobs)}] {name:44s} {len(rows):>4} chunks")

    print(f"\n{embedder.report()}")
    print(f"chunks over bge-m3's {EMBED_CONTEXT}-token limit: {oversized}")
    print(f"wrote {CHROMA.relative_to(CHROMA.parent.parent)}/")


if __name__ == "__main__":
    main()
