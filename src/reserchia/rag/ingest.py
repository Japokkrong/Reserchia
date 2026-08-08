"""Adding a paper to the library, in the foreground or behind the answer.

This is the second of the two paths that run when the agent fetches a paper.
The first hands the full text straight back so the current question can be
answered; this one chunks, embeds and stores it so the *next* question needs no
API call at all.

It runs on a single worker thread rather than a pool: ingests are short, Chroma
writes are better serialised, and one worker makes "is this paper already being
ingested?" a simple set membership test.

The thread is deliberately **not** a daemon. A daemon would be killed mid-write
when the process exits, leaving a half-indexed paper in the library; `wait()`
gives the CLI something to call on the way out instead.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as await_futures

from ..arxiv_fulltext import Document, fetch_document
from ..config import Settings, get_settings
from ..observability import track
from . import embeddings, lexical, store
from .chunking import chunk_document

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()
_in_flight: set[str] = set()
_queued: list[Future] = []
_failures: list[str] = []


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reserchia-ingest"
        )
    return _executor


def ingest_paper(document: Document, settings: Settings | None = None) -> int:
    """Chunk, embed and store one paper. Returns the number of chunks written.

    Named `ingest_paper`, not `ingest`, so it cannot shadow this module when
    both are imported from the package.
    """
    settings = settings or get_settings()
    chunks = chunk_document(document)
    if not chunks:
        return 0
    with track("ingest", "chain", paper=document.arxiv_id, chunks=len(chunks)):
        vectors = embeddings.embed([chunk.text for chunk in chunks], settings)
        written = store.upsert(chunks, vectors, settings)
    # BM25's idf depends on the whole corpus, so any write invalidates it.
    lexical.invalidate()
    return written


def ingest_later(arxiv_id: str) -> Future | None:
    """Queue a paper for background indexing.

    Returns None when there is nothing to do -- already stored, or already
    queued. `fetch_document` caches the parsed document in process, so the
    worker re-uses it and never downloads the paper a second time.
    """
    identifier = store.base_id(arxiv_id)
    if not identifier:
        return None

    try:
        if store.has_paper(identifier):
            return None
    except Exception:  # noqa: BLE001 - an unreadable store must not block an answer
        return None

    with _lock:
        if identifier in _in_flight:
            return None
        _in_flight.add(identifier)

    future = _pool().submit(_run, identifier)
    with _lock:
        _queued.append(future)
    return future


def _run(identifier: str) -> int:
    try:
        document = fetch_document(identifier)
        if isinstance(document, str):
            _failures.append(f"{identifier}: {document}")
            return 0
        return ingest_paper(document)
    except Exception as exc:  # noqa: BLE001 - background work must never surface
        # Swallowed on purpose: the user is reading an answer that was already
        # produced from the full text. A failed index costs a repeat fetch next
        # time, which is not worth interrupting them for.
        _failures.append(f"{identifier}: {type(exc).__name__}: {exc}")
        return 0
    finally:
        with _lock:
            _in_flight.discard(identifier)


def pending() -> int:
    with _lock:
        return len(_in_flight)


def failures() -> list[str]:
    return list(_failures)


def wait(timeout: float | None = None) -> bool:
    """Block until queued ingests finish. True if the queue drained in time.

    The CLI calls this on exit so a paper is not left half-written. A timeout
    is honoured properly here -- `ThreadPoolExecutor.shutdown` takes no timeout,
    so the futures are awaited directly and only then is the pool torn down.
    """
    global _executor
    with _lock:
        futures = list(_queued)
        _queued.clear()

    done = True
    if futures:
        _, not_done = await_futures(futures, timeout=timeout)
        done = not not_done

    if done:
        with _lock:
            executor, _executor = _executor, None
        if executor is not None:
            executor.shutdown(wait=True)
    return done
