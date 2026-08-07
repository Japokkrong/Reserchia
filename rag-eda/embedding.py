"""bge-m3 embeddings through OpenRouter.

OpenRouter's embedding models are not in the chat catalogue at
``/api/v1/models`` -- they live at ``/api/v1/embeddings/models`` -- but the
endpoint itself is OpenAI-shaped, so the ``openai`` SDK already in the project
talks to it with nothing more than a base URL swap.

Every vector is cached on disk under ``sha256(model + text)``. The sweep
embeds the same chunk text many times over (a chunk that survives unchanged
between two configurations is the same string), and re-running the pipeline
should cost nothing at all.
"""

from __future__ import annotations

import hashlib
import json
import time

from common import CACHE, EMBED_DIM, EMBED_MODEL
from openai import OpenAI

from reserchia.config import get_settings

BATCH = 64
MAX_RETRIES = 4


class Embedder:
    def __init__(self, model: str = EMBED_MODEL) -> None:
        settings = get_settings()
        self.model = model
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.cache_dir = CACHE / model.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.prompt_tokens = 0

    def _path(self, text: str):
        digest = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _cached(self, text: str) -> list[float] | None:
        path = self._path(text)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        return None

    def _store(self, text: str, vector: list[float]) -> None:
        path = self._path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vector))

    def _call(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:  # noqa: BLE001 - rate limits, 5xx, timeouts
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                print(f"    retry in {wait}s after {type(exc).__name__}: {exc}")
                time.sleep(wait)
                continue

            usage = getattr(response, "usage", None)
            if usage is not None:
                self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            # data is not guaranteed ordered; index says where each belongs.
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str], label: str = "") -> list[list[float]]:
        vectors: list[list[float] | None] = [self._cached(t) for t in texts]
        todo = [i for i, vector in enumerate(vectors) if vector is None]
        self.misses += len(todo)

        for start in range(0, len(todo), BATCH):
            indices = todo[start : start + BATCH]
            batch = [texts[i] for i in indices]
            fresh = self._call(batch)
            for index, vector in zip(indices, fresh):
                if len(vector) != EMBED_DIM:
                    raise ValueError(
                        f"{self.model} returned {len(vector)} dims, expected {EMBED_DIM}"
                    )
                self._store(texts[index], vector)
                vectors[index] = vector
            if label:
                done = min(start + BATCH, len(todo))
                print(f"    {label}: embedded {done}/{len(todo)}", end="\r")

        if label and todo:
            print(" " * 60, end="\r")
        return [v for v in vectors if v is not None]

    def report(self) -> str:
        total = self.hits + self.misses
        # $0.01 per 1M tokens, per OpenRouter's model metadata for bge-m3.
        cost = self.prompt_tokens / 1_000_000 * 0.01
        return (
            f"{total:,} texts | {self.hits:,} cached, {self.misses:,} embedded | "
            f"{self.prompt_tokens:,} tokens billed | ~${cost:.4f}"
        )
