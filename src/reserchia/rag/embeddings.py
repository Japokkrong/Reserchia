"""bge-m3 embeddings through OpenRouter.

OpenRouter's embedding models are absent from the chat catalogue at
`/api/v1/models` -- they are listed at `/api/v1/embeddings/models` -- but the
endpoint is OpenAI-shaped, so the `openai` SDK already pulled in by
`langchain-openai` talks to it with only a base-URL swap.

bge-m3 returns L2-normalised vectors, which is why the Chroma collection can
use cosine space directly with no renormalisation step.
"""

from __future__ import annotations

import time

from openai import OpenAI

from ..config import Settings, get_settings

DIMENSIONS = 1024

#: Requests are batched by a **token budget, not an item count**. bge-m3's 8192
#: context applies to the whole request, and exceeding it comes back as an
#: unhelpful HTTP 429 "engine is currently overloaded" rather than a size error.
#: Measured against the live endpoint: 6,390 tokens in one request succeeded,
#: 12,408 failed. 6,000 leaves margin for the estimate being an estimate.
BATCH_TOKENS = 6_000
#: Belt and braces for many tiny chunks, where the token budget alone would
#: allow an unreasonably long input list.
BATCH_ITEMS = 32
#: bge-m3 runs ~4.04 characters per token on LaTeXML text, so characters are a
#: good enough proxy to pack batches without shipping a tokenizer.
CHARS_PER_TOKEN = 4.04

#: Generous, because the other thing a 429 means is a shared provider being
#: briefly overloaded, which clears on its own. Backoff runs 2/4/8/16/32s.
MAX_RETRIES = 6
BACKOFF_BASE = 2.0
MAX_BACKOFF = 32.0

_client: OpenAI | None = None


def _openai(settings: Settings) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    return _client


def _batches(texts: list[str]) -> list[list[str]]:
    """Group texts so no request exceeds the model's context window."""
    budget = int(BATCH_TOKENS * CHARS_PER_TOKEN)
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0

    for text in texts:
        length = len(text)
        # A single text over budget still has to go alone; the model truncates
        # it, which is better than failing the whole ingest.
        if current and (size + length > budget or len(current) >= BATCH_ITEMS):
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += length
    if current:
        batches.append(current)
    return batches


def embed(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed texts, preserving input order."""
    if not texts:
        return []
    settings = settings or get_settings()
    client = _openai(settings)

    vectors: list[list[float]] = []
    for batch in _batches(texts):
        response = _call(client, settings.embed_model, batch)
        # `data` carries an explicit index; do not trust list order.
        for item in sorted(response.data, key=lambda item: item.index):
            if len(item.embedding) != DIMENSIONS:
                raise ValueError(
                    f"{settings.embed_model} returned {len(item.embedding)} "
                    f"dimensions, expected {DIMENSIONS}"
                )
            vectors.append(item.embedding)

    if len(vectors) != len(texts):
        raise ValueError(
            f"embedded {len(vectors)} vectors for {len(texts)} texts -- "
            "batching lost or duplicated an input"
        )
    return vectors


def embed_one(text: str, settings: Settings | None = None) -> list[float]:
    return embed([text], settings)[0]


def _retry_after(exc: Exception) -> float | None:
    """The provider's own advice on when to come back, if it gave any."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        return float(header.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _call(client: OpenAI, model: str, batch: list[str]):
    for attempt in range(MAX_RETRIES):
        try:
            return client.embeddings.create(model=model, input=batch)
        except Exception as exc:  # noqa: BLE001 - rate limits, 5xx, timeouts
            if attempt == MAX_RETRIES - 1:
                raise
            wait = _retry_after(exc) or min(BACKOFF_BASE ** (attempt + 1), MAX_BACKOFF)
            time.sleep(wait)
    raise RuntimeError("unreachable")
