"""Timing every step, to a LangSmith trace and to a local log.

Two consumers, one instrumentation point. `track()` opens a LangSmith span when
tracing is on, and always appends the same measurement to the turn's local
record. Neither can drift from the other because there is only one place that
measures.

Why both, rather than just LangSmith:

- **LangSmith cannot see the interesting steps.** `embeddings.embed`,
  `store.query_dense`, `lexical.search`, the arXiv throttle and the Kroki render
  are plain functions, not LangChain runnables, so nothing traces them
  automatically -- and they are where the time actually goes. Measured:
  bge-m3 ~2,000 ms, arXiv throttle ~3,000 ms, Kroki ~2,000 ms against Chroma at
  1.8 ms and BM25 at 0.16 ms. Without these spans a trace just says
  `search_paper_library` took two seconds.
- **`/stats` and the offline report need local data.** LangSmith's records live
  on their servers, so a REPL command would need a network round trip and would
  return nothing at all with tracing off.

Two behaviours here were measured rather than assumed:

**A dead key floods stderr.** A single probe with an invalid key produced 21
`Failed to multipart ingest runs: ... 403 Forbidden` lines from LangSmith's
background uploader. In the REPL that reads as a broken app when the agent is
fine, so the uploader's logger is muted unless explicitly asked for.

**Spans are not free when tracing is off** -- 1000 cost 84 ms, so 0.084 ms
each. That is half the cost of a BM25 query, which is why the hot paths check
`enabled()` before building one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings

#: Loggers LangSmith's background uploader shouts through when it cannot reach
#: the API. Muted by default: an expired key must not look like a crash.
_NOISY = ("langsmith.client", "langsmith._internal", "langsmith.utils")


def _quieten() -> None:
    if os.getenv("RESERCHIA_LOG_LEVEL", "").lower() == "debug":
        return
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.CRITICAL)


_quieten()


def tracing() -> bool:
    """Whether LangSmith tracing is on. Cheap enough to call in a hot path."""
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("1", "true", "yes")


def logging_enabled() -> bool:
    return os.getenv("RESERCHIA_LOGGING", "true").strip().lower() not in ("0", "false", "no")


@dataclass
class Step:
    name: str
    kind: str
    ms: float = 0.0
    fields: dict = field(default_factory=dict)


@dataclass
class TurnRecord:
    started: float = 0.0
    question: str = ""
    steps: list[Step] = field(default_factory=list)


_lock = threading.Lock()
_turn = TurnRecord()
#: Kept in memory so `/stats` can answer for the current session without
#: re-reading the log file.
_session: list[dict] = []


def start_turn(question: str = "") -> None:
    global _turn
    with _lock:
        _turn = TurnRecord(started=time.perf_counter(), question=question)


class Handle:
    """Returned by `track`, for attaching results measured inside the block."""

    __slots__ = ("_step", "_span")

    def __init__(self, step: Step, span) -> None:
        self._step = step
        self._span = span

    def set(self, **fields) -> None:
        self._step.fields.update(fields)


@contextlib.contextmanager
def track(name: str, kind: str = "tool", **fields):
    """Time a step, into the local record and (when on) a LangSmith span.

    `kind` maps to LangSmith's run_type, so a retrieval shows up as retrieval
    rather than as an anonymous chain.
    """
    step = Step(name=name, kind=kind, fields=dict(fields))
    started = time.perf_counter()

    if not tracing():
        # Fast path. No span object is built at all -- see the module docstring
        # on why 0.084 ms matters against a 0.16 ms BM25 call.
        try:
            yield Handle(step, None)
        finally:
            step.ms = (time.perf_counter() - started) * 1000
            _record(step)
        return

    from langsmith import trace as ls_trace

    run_type = kind if kind in ("tool", "retriever", "llm", "embedding", "chain") else "tool"
    with ls_trace(name=name, run_type=run_type, inputs=dict(fields)) as span:
        handle = Handle(step, span)
        try:
            yield handle
        finally:
            step.ms = (time.perf_counter() - started) * 1000
            with contextlib.suppress(Exception):
                span.end(outputs={"ms": round(step.ms, 2), **step.fields})
            _record(step)


def _record(step: Step) -> None:
    with _lock:
        _turn.steps.append(step)


def finish_turn(usage=None) -> dict:
    """Close the turn, append it to the log, and return the record."""
    with _lock:
        record = _turn

    elapsed = (time.perf_counter() - record.started) * 1000 if record.started else 0.0
    payload = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ms": round(elapsed, 1),
        "question_chars": len(record.question),
        "steps": [
            {"name": s.name, "kind": s.kind, "ms": round(s.ms, 2), **s.fields}
            for s in record.steps
        ],
    }
    if usage is not None:
        payload.update(
            {
                "calls": usage.calls,
                "input_tokens": usage.input,
                "output_tokens": usage.output,
                "cached_tokens": usage.cached,
                "embed_tokens": usage.embedded,
                "cost_usd": round(usage.cost, 8) if usage.cost else 0.0,
            }
        )

    _session.append(payload)
    if logging_enabled():
        _append(payload)
    return payload


def log_dir() -> Path:
    configured = os.getenv("RESERCHIA_LOG_DIR", "").strip()
    path = Path(configured) if configured else get_settings().store_dir / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append(payload: dict) -> None:
    try:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        with (log_dir() / f"turns-{month}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:  # noqa: BLE001 - logging must never break a turn
        pass


# -- reporting ----------------------------------------------------------------


def read_log(limit: int = 500) -> list[dict]:
    """Recent turns from disk, newest last."""
    rows: list[dict] = []
    try:
        files = sorted(log_dir().glob("turns-*.jsonl"))
    except Exception:  # noqa: BLE001
        return rows
    for path in files[-3:]:
        with contextlib.suppress(Exception):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            rows.append(json.loads(line))
    return rows[-limit:]


def summarise(turns: list[dict]) -> dict:
    """Totals and the step breakdown that says where the time went."""
    if not turns:
        return {}
    by_step: dict[str, list[float]] = {}
    for turn in turns:
        for step in turn.get("steps", []):
            by_step.setdefault(step["name"], []).append(step.get("ms", 0.0))

    total_ms = sum(t.get("ms", 0.0) for t in turns)
    return {
        "turns": len(turns),
        "total_ms": total_ms,
        "median_ms": sorted(t.get("ms", 0.0) for t in turns)[len(turns) // 2],
        "tokens": sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in turns),
        "cost_usd": sum(t.get("cost_usd", 0.0) for t in turns),
        "steps": sorted(
            (
                {
                    "name": name,
                    "count": len(times),
                    "total_ms": sum(times),
                    "mean_ms": sum(times) / len(times),
                    "share": (sum(times) / total_ms) if total_ms else 0.0,
                }
                for name, times in by_step.items()
            ),
            key=lambda row: -row["total_ms"],
        ),
    }


def stats_text() -> str:
    """The `/stats` report: this session, then recent history from disk."""
    lines = []
    session = summarise(_session)
    if session:
        lines.append(
            f"this session   {session['turns']} turn(s) · "
            f"{session['tokens']:,} tokens · ${session['cost_usd']:.4f} · "
            f"median {session['median_ms'] / 1000:.1f}s"
        )
        for row in session["steps"][:5]:
            lines.append(
                f"    {row['name']:<22} {row['count']:>3}x  "
                f"{row['total_ms']:>8,.0f} ms  {row['share'] * 100:>4.0f}%"
            )
    else:
        lines.append("this session   no turns yet")

    history = summarise(read_log())
    if history and history["turns"] > len(_session):
        lines.append("")
        lines.append(
            f"logged history {history['turns']} turn(s) · "
            f"{history['tokens']:,} tokens · ${history['cost_usd']:.4f} · "
            f"median {history['median_ms'] / 1000:.1f}s"
        )
        for row in history["steps"][:5]:
            lines.append(
                f"    {row['name']:<22} {row['count']:>3}x  "
                f"{row['total_ms']:>8,.0f} ms  {row['share'] * 100:>4.0f}%"
            )
        lines.append(f"    -> {log_dir()}")

    if tracing():
        project = os.getenv("LANGSMITH_PROJECT", "default")
        lines.append(f"\nLangSmith tracing on, project {project!r}")
    return "\n".join(lines)
