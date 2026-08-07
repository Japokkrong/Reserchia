"""Run the chunking grid and measure what each strategy does to an arXiv paper.

The question is not "which chunker is tidiest" but which one preserves the
things arXiv papers carry that generic prose does not: section structure,
inline LaTeX, and a bibliography that should never be retrieved as an answer.

Run: python rag-eda/02_chunk_eda.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunkers  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import (  # noqa: E402
    CHUNKS,
    EMBED_CONTEXT,
    EXTRACTED,
    FIGURES,
    PAPERS,
    SOURCES,
    ensure_dirs,
    read_json,
    require,
    split_references,
    tokenizer_kind,
    write_json,
    write_jsonl,
)


def load(paper: str, source: str) -> dict:
    path = EXTRACTED / f"{paper}.{source}.json"
    require(path, "rag-eda/01_extract.py")
    return read_json(path)


def quality(chunks, text: str, refs_start: int, located) -> dict:
    """The arXiv-specific ways a chunk can be wrong."""
    lengths = [c.n_tokens for c in chunks]
    broken = mid_sentence = orphan = contaminated = split_section = 0

    for chunk in chunks:
        body = text[chunk.start : chunk.end]
        # An odd number of delimiters means a formula was cut in half.
        if body.count("$") % 2:
            broken += 1
        stripped = body.lstrip()
        if stripped and stripped[0].islower():
            mid_sentence += 1
        if not chunk.section:
            orphan += 1
        # More than half of this chunk sits in the bibliography.
        overlap = max(0, min(chunk.end, len(text)) - max(chunk.start, refs_start))
        if overlap > (chunk.end - chunk.start) / 2:
            contaminated += 1
        covering = [
            1 for start, end, _ in located if start < chunk.end and end > chunk.start
        ]
        if len(covering) > 1:
            split_section += 1

    count = max(1, len(chunks))
    return {
        "chunks": len(chunks),
        "tok_mean": round(statistics.mean(lengths), 1) if lengths else 0,
        "tok_median": int(statistics.median(lengths)) if lengths else 0,
        "tok_p95": int(sorted(lengths)[int(len(lengths) * 0.95)]) if lengths else 0,
        "tok_max": max(lengths) if lengths else 0,
        "over_context": sum(1 for n in lengths if n > EMBED_CONTEXT),
        "broken_math_pct": round(100 * broken / count, 1),
        "mid_sentence_pct": round(100 * mid_sentence / count, 1),
        "orphan_pct": round(100 * orphan / count, 1),
        "references_pct": round(100 * contaminated / count, 1),
        "crosses_section_pct": round(100 * split_section / count, 1),
    }


def main() -> None:
    ensure_dirs()
    print(f"tokenizer: {tokenizer_kind()}\n")

    documents = {
        (paper, source): load(paper, source)
        for paper in PAPERS
        for source in SOURCES
    }

    # A PDF has no section tree, so one is recovered from heading shapes. The
    # difference against LaTeXML's real tree is itself a result.
    for (paper, source), payload in documents.items():
        if source == "pdf":
            payload["sections"] = chunkers.pdf_sections(payload["text"])
            print(
                f"  {paper} pdf: recovered {len(payload['sections'])} headings by regex "
                f"vs {len(documents[(paper, 'html')]['sections'])} real ones in the HTML"
            )

    #: Counts sum across the two papers; percentages and lengths average.
    SUMMED = ("chunks", "over_context")
    PEAK = ("tok_max",)

    rows = []
    for config in chunkers.grid():
        for source in SOURCES:
            chunks = []
            # Defect metrics are computed per paper and then pooled, because the
            # bibliography offset and the section map are per document.
            pooled = []
            for paper in PAPERS:
                payload = documents[(paper, source)]
                mine = chunkers.build(
                    payload["text"], payload["sections"], paper, source, config
                )
                chunks.extend(mine)

                body, refs = split_references(payload["text"])
                refs_start = len(body) if refs else len(payload["text"])
                located = chunkers.locate_sections(payload["text"], payload["sections"])
                pooled.append(quality(mine, payload["text"], refs_start, located))

            stats = {}
            for key in pooled[0]:
                values = [p[key] for p in pooled]
                if key in SUMMED:
                    stats[key] = sum(values)
                elif key in PEAK:
                    stats[key] = max(values)
                else:
                    stats[key] = round(sum(values) / len(values), 1)

            rows.append({"config": config.name, "source": source, **stats})
            write_jsonl(CHUNKS / f"{config.name}.{source}.jsonl", chunks)

    write_json(CHUNKS / "stats.json", rows)

    header = (
        f"{'config':26s} {'src':5s} {'chunks':>7s} {'mean':>7s} {'p95':>6s} "
        f"{'max':>6s} {'>ctx':>5s} {'math':>6s} {'midsent':>8s} "
        f"{'orphan':>7s} {'refs':>6s} {'xsect':>6s}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for row in rows:
        print(
            f"{row['config']:26s} {row['source']:5s} {row['chunks']:>7,} "
            f"{row['tok_mean']:>7.0f} {row['tok_p95']:>6.0f} {row['tok_max']:>6,} "
            f"{row['over_context']:>5} {row['broken_math_pct']:>5.1f}% "
            f"{row['mid_sentence_pct']:>7.1f}% {row['orphan_pct']:>6.1f}% "
            f"{row['references_pct']:>5.1f}% {row['crosses_section_pct']:>5.1f}%"
        )

    figures(rows)
    print(f"\nwrote data/chunks/ ({len(rows)} configurations) and figures/")


def figures(rows) -> None:
    strategies = ["fixed", "recursive", "paragraph", "section"]
    metrics = [
        ("broken_math_pct", "equations split"),
        ("mid_sentence_pct", "starts mid-sentence"),
        ("orphan_pct", "no section context"),
        ("references_pct", "mostly bibliography"),
        ("crosses_section_pct", "spans >1 section"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.2), sharey=False)
    for axis, (key, title) in zip(axes, metrics):
        for source, colour in (("pdf", "#c44"), ("html", "#348")):
            values = [
                statistics.mean(
                    [r[key] for r in rows if r["source"] == source and r["config"].startswith(s)]
                )
                for s in strategies
            ]
            offset = -0.2 if source == "pdf" else 0.2
            axis.bar(
                [i + offset for i in range(len(strategies))],
                values,
                width=0.4,
                label=source,
                color=colour,
            )
        axis.set_xticks(range(len(strategies)))
        axis.set_xticklabels(strategies, rotation=30, ha="right")
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("% of chunks")
        axis.legend(fontsize=8)
    fig.suptitle("Chunk defects by strategy and source", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "chunk_defects.png", dpi=130)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for strategy in strategies:
        for size in (256, 512, 1024):
            picked = [
                r
                for r in rows
                if r["source"] == "html" and r["config"].startswith(f"{strategy}-{size}-")
            ]
            if picked:
                axis.scatter(
                    [size] * len(picked),
                    [r["tok_mean"] for r in picked],
                    label=strategy if size == 256 else None,
                    alpha=0.7,
                )
    axis.plot([256, 512, 1024], [256, 512, 1024], "k--", lw=1, label="target")
    axis.set_xlabel("target chunk size (bge-m3 tokens)")
    axis.set_ylabel("actual mean chunk length")
    axis.set_title("Does each strategy hit its budget? (HTML source)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "chunk_sizes.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
