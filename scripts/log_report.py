"""Summarise Reserchia's step logs: where the time goes, and what it cost.

    python scripts/log_report.py [path/to/turns-*.jsonl]

Reads the local JSONL that every turn appends to. Works with tracing off and
without a network connection, which is the point of keeping a local mirror
alongside LangSmith.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> int:
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        from reserchia.observability import log_dir

        paths = sorted(log_dir().glob("turns-*.jsonl"))

    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no log files found -- run a turn first")
        return 1

    turns = load(paths)
    if not turns:
        print(f"no turns recorded in {', '.join(str(p) for p in paths)}")
        return 1

    latencies = [t.get("ms", 0.0) for t in turns]
    print(f"{len(turns)} turn(s) from {len(paths)} file(s)\n")
    print("latency")
    print(f"  p50 {percentile(latencies, 0.5) / 1000:>7.1f}s")
    print(f"  p95 {percentile(latencies, 0.95) / 1000:>7.1f}s")
    print(f"  max {max(latencies) / 1000:>7.1f}s")

    tokens_in = sum(t.get("input_tokens", 0) for t in turns)
    tokens_out = sum(t.get("output_tokens", 0) for t in turns)
    cached = sum(t.get("cached_tokens", 0) for t in turns)
    cost = sum(t.get("cost_usd", 0.0) for t in turns)
    print("\nusage")
    print(f"  input   {tokens_in:>10,}" + (f"  ({cached / tokens_in * 100:.0f}% cached)" if tokens_in else ""))
    print(f"  output  {tokens_out:>10,}")
    print(f"  embed   {sum(t.get('embed_tokens', 0) for t in turns):>10,}")
    print(f"  cost    {cost:>10.5f} USD   ({cost / len(turns):.6f}/turn)")

    by_name: dict[str, list[float]] = defaultdict(list)
    for turn in turns:
        for step in turn.get("steps", []):
            by_name[step["name"]].append(step.get("ms", 0.0))

    if not by_name:
        print("\nno steps recorded")
        return 0

    total_step_ms = sum(sum(v) for v in by_name.values())
    total_turn_ms = sum(latencies)
    print(f"\nwhere the time goes  ({total_step_ms / 1000:.1f}s measured "
          f"of {total_turn_ms / 1000:.1f}s total; the rest is the model)")
    print(f"  {'step':<20} {'calls':>6} {'total':>10} {'mean':>9} {'share':>7}")
    for name, times in sorted(by_name.items(), key=lambda kv: -sum(kv[1])):
        share = sum(times) / total_turn_ms * 100 if total_turn_ms else 0
        print(
            f"  {name:<20} {len(times):>6} {sum(times) / 1000:>9.1f}s "
            f"{statistics.mean(times):>8.0f}ms {share:>6.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
