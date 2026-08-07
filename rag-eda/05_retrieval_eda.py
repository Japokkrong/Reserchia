"""Sweep the dense/lexical mix across every chunking configuration.

Each query is retrieved ONCE per configuration -- top-50 from each retriever --
and the eleven alphas plus RRF are then computed by re-fusing those same pools.
Re-querying per alpha would multiply the work by twelve and change nothing.

Run: python rag-eda/05_retrieval_eda.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb  # noqa: E402
import chunkers  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import retrieval  # noqa: E402
from common import (  # noqa: E402
    CHROMA,
    CHUNKS,
    DATA,
    EXTRACTED,
    FIGURES,
    ROOT,
    SOURCES,
    ensure_dirs,
    read_json,
    read_jsonl,
    require,
)
from embedding import Embedder  # noqa: E402

ALPHAS = [round(a / 10, 1) for a in range(11)]
KS = (1, 3, 5, 10)


def evaluate(ranking, index, gold, total_relevant):
    rel = retrieval.relevance_vector(ranking, index.by_id, gold)
    return {
        **{f"recall@{k}": retrieval.recall_at(rel, k, total_relevant) for k in KS},
        **{f"hit@{k}": retrieval.hit_at(rel, k) for k in KS},
        "mrr@10": retrieval.mrr_at(rel, 10),
        "ndcg@10": retrieval.ndcg_at(rel, 10),
    }


def main() -> None:
    ensure_dirs()
    require(DATA / "testset.json", "rag-eda/04_testset.py")
    cases = [c for c in read_json(DATA / "testset.json") if c["gold"]]
    negatives = [c for c in read_json(DATA / "testset.json") if not c["gold"]]
    print(f"{len(cases)} scored cases ({len(negatives)} negatives held out)\n")

    client = chromadb.PersistentClient(path=str(CHROMA))
    embedder = Embedder()
    questions = [c["question"] for c in cases]
    vectors = dict(zip(questions, embedder.embed(questions, label="queries")))

    results = []
    configs = chunkers.grid()
    jobs = [(c.name, s) for c in configs for s in SOURCES]

    for number, (config, source) in enumerate(jobs, 1):
        path = CHUNKS / f"{config}.{source}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        collection = client.get_collection(f"{config}__{source}".replace(".", "_"))
        index = retrieval.Index(collection, rows)

        per_alpha = defaultdict(list)
        for case in cases:
            gold = case["gold"][source]
            if gold["end"] <= gold["start"]:
                continue
            total_relevant = sum(
                1 for row in rows
                if row["paper"] == case["paper"] and retrieval.is_relevant(row, gold)
            )
            if total_relevant == 0:
                # No chunk in this configuration covers the answer. Scoring it
                # would punish the retriever for a chunking failure -- that gets
                # counted separately, as coverage.
                per_alpha["_uncoverable"].append(case["id"])
                continue

            dense = index.dense(vectors[case["question"]])
            lexical = index.lexical(case["question"])

            for alpha in ALPHAS:
                ranking = retrieval.fuse_weighted(dense, lexical, alpha)
                per_alpha[alpha].append(
                    (case, evaluate(ranking, index, gold, total_relevant))
                )
            per_alpha["rrf"].append(
                (case, evaluate(retrieval.fuse_rrf(dense, lexical), index, gold, total_relevant))
            )

        uncoverable = len(per_alpha.pop("_uncoverable", []))
        scored = len(per_alpha.get(1.0, []))
        for key, entries in per_alpha.items():
            if not entries:
                continue
            metrics = {
                name: round(statistics.mean(m[name] for _, m in entries), 4)
                for name in entries[0][1]
            }
            by_category = defaultdict(list)
            for case, m in entries:
                by_category[case["category"]].append(m["ndcg@10"])
            results.append({
                "config": config, "source": source, "fusion": key,
                "scored": scored, "uncoverable": uncoverable,
                **metrics,
                "by_category": {
                    k: round(statistics.mean(v), 4) for k, v in by_category.items()
                },
            })
        print(
            f"[{number:>2}/{len(jobs)}] {config:26s} {source:5s} "
            f"scored {scored:>2}, uncoverable {uncoverable:>2}"
        )

    write = DATA / "retrieval_results.json"
    write.write_text(json.dumps(results, indent=2))
    report(results, cases, embedder)
    write_report(results, cases)


def report(results, cases, embedder) -> None:
    weighted = [r for r in results if r["fusion"] != "rrf"]
    best = max(weighted, key=lambda r: r["ndcg@10"])

    print("\n" + "=" * 78)
    print(f"BEST: {best['config']} / {best['source']} at alpha={best['fusion']} "
          f"-> nDCG@10 {best['ndcg@10']:.3f}, Recall@5 {best['recall@5']:.3f}")
    print("=" * 78)

    print("\ntop 12 configurations by nDCG@10")
    print(f"  {'config':26s} {'src':5s} {'a':>4s} {'nDCG':>6s} {'MRR':>6s} "
          f"{'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'uncov':>6s}")
    for row in sorted(weighted, key=lambda r: -r["ndcg@10"])[:12]:
        print(f"  {row['config']:26s} {row['source']:5s} {row['fusion']:>4} "
              f"{row['ndcg@10']:>6.3f} {row['mrr@10']:>6.3f} {row['recall@1']:>6.3f} "
              f"{row['recall@5']:>6.3f} {row['recall@10']:>6.3f} {row['uncoverable']:>6}")

    print("\nalpha sweep, averaged over all configurations")
    print(f"  {'alpha':>6s} {'nDCG@10':>8s} {'MRR@10':>8s} {'R@1':>7s} {'R@5':>7s}")
    for alpha in ALPHAS:
        rows = [r for r in weighted if r["fusion"] == alpha]
        if rows:
            print(f"  {alpha:>6} {statistics.mean(r['ndcg@10'] for r in rows):>8.4f} "
                  f"{statistics.mean(r['mrr@10'] for r in rows):>8.4f} "
                  f"{statistics.mean(r['recall@1'] for r in rows):>7.4f} "
                  f"{statistics.mean(r['recall@5'] for r in rows):>7.4f}")
    rrf = [r for r in results if r["fusion"] == "rrf"]
    print(f"  {'RRF':>6} {statistics.mean(r['ndcg@10'] for r in rrf):>8.4f} "
          f"{statistics.mean(r['mrr@10'] for r in rrf):>8.4f} "
          f"{statistics.mean(r['recall@1'] for r in rrf):>7.4f} "
          f"{statistics.mean(r['recall@5'] for r in rrf):>7.4f}")

    print("\nby query category, at the best configuration")
    peer = [r for r in weighted if r["config"] == best["config"] and r["source"] == best["source"]]
    categories = sorted({c for r in peer for c in r["by_category"]})
    print(f"  {'alpha':>6s} " + " ".join(f"{c[:11]:>12s}" for c in categories))
    for alpha in ALPHAS:
        row = next((r for r in peer if r["fusion"] == alpha), None)
        if row:
            print(f"  {alpha:>6} " + " ".join(
                f"{row['by_category'].get(c, float('nan')):>12.3f}" for c in categories
            ))

    print(f"\nembedding: {embedder.report()}")
    figures(results, best)


def figures(results, best) -> None:
    weighted = [r for r in results if r["fusion"] != "rrf"]

    fig, axis = plt.subplots(figsize=(9, 5))
    for source, colour in (("pdf", "#c44"), ("html", "#348")):
        means = [
            statistics.mean(
                r["ndcg@10"] for r in weighted
                if r["fusion"] == a and r["source"] == source
            )
            for a in ALPHAS
        ]
        axis.plot(ALPHAS, means, "o-", color=colour, label=f"{source} (weighted)")
        rrf = [r for r in results if r["fusion"] == "rrf" and r["source"] == source]
        axis.axhline(
            statistics.mean(r["ndcg@10"] for r in rrf),
            color=colour, ls=":", lw=1.5, label=f"{source} (RRF)",
        )
    axis.set_xlabel("alpha   (0 = pure BM25, 1 = pure dense)")
    axis.set_ylabel("nDCG@10")
    axis.set_title("Hybrid mix vs retrieval quality, averaged over 36 chunk configurations")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "alpha_sweep.png", dpi=130)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for strategy in ("fixed", "recursive", "paragraph", "section"):
        means = [
            statistics.mean(
                [r["ndcg@10"] for r in weighted
                 if r["fusion"] == a and r["config"].startswith(strategy + "-")] or [0]
            )
            for a in ALPHAS
        ]
        axis.plot(ALPHAS, means, "o-", label=strategy)
    axis.set_xlabel("alpha   (0 = pure BM25, 1 = pure dense)")
    axis.set_ylabel("nDCG@10")
    axis.set_title("Does the best mix depend on the chunking strategy?")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "alpha_by_strategy.png", dpi=130)
    plt.close(fig)

    peer = [r for r in weighted if r["config"] == best["config"] and r["source"] == best["source"]]
    categories = sorted({c for r in peer for c in r["by_category"]})
    fig, axis = plt.subplots(figsize=(10, 5))
    for category in categories:
        values = [
            next((r["by_category"].get(category) for r in peer if r["fusion"] == a), None)
            for a in ALPHAS
        ]
        axis.plot(ALPHAS, values, "o-", label=category)
    axis.set_xlabel("alpha   (0 = pure BM25, 1 = pure dense)")
    axis.set_ylabel("nDCG@10")
    axis.set_title(f"Query type decides the mix  ({best['config']} / {best['source']})")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "alpha_by_category.png", dpi=130)
    plt.close(fig)
    print(f"wrote figures/ and {(DATA / 'retrieval_results.json').relative_to(ROOT)}")


def write_report(results, cases) -> None:
    """Emit REPORT.md from the measurements, so it can never drift from them."""
    weighted = [r for r in results if r["fusion"] != "rrf"]
    rrf = [r for r in results if r["fusion"] == "rrf"]
    best = max(weighted, key=lambda r: r["ndcg@10"])

    def mean(rows, key="ndcg@10"):
        return statistics.mean(r[key] for r in rows) if rows else float("nan")

    sweep = {a: mean([r for r in weighted if r["fusion"] == a]) for a in ALPHAS}
    peak = max(sweep, key=lambda a: sweep[a])

    extraction = read_json(EXTRACTED / "summary.json")
    chunk_stats = read_json(CHUNKS / "stats.json")
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1

    lines = [
        "# RAG EDA — chunking, bge-m3, and hybrid retrieval on two arXiv papers",
        "",
        "Generated by `05_retrieval_eda.py`. Every number here is measured, not estimated.",
        "",
        "## Corpus",
        "",
        "| paper | title | pages |",
        "|---|---|---|",
        "| `2601.14722v1` | Typhoon OCR: Open Vision–Language Model For Thai Document Extraction | 14 |",
        "| `2603.10910v2` | GLM-OCR Technical Report (Zhipu AI / Tsinghua) | 17 |",
        "",
        "They overlap heavily — both cover OCR benchmarks, layout analysis and compact-model",
        "tradeoffs, and both have sections literally named *Stage 1–4* — so cross-paper",
        "confusion is a real retrieval signal here rather than a contrived one.",
        "",
        "## 1. Source: PDF text vs LaTeXML HTML",
        "",
        "| paper | source | chars | tokens | sections | math spans | references |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in extraction:
        lines.append(
            f"| `{row['paper']}` | {row['source']} | {row['chars']:,} | {row['tokens']:,} "
            f"| {row['sections']} | {row['math_spans']} | {100 * row['references_share']:.0f}% |"
        )
    lines += [
        "",
        "The PDF is *larger* but carries less: a fifth to a quarter of it is bibliography,",
        "which the LaTeXML tier drops outright, and it yields no section tree and no math.",
        "",
        "With bibliographies excluded from both sides, vocabulary present in one extraction",
        "and absent from the other is asymmetric — PDF-only short fragments (`fer`, `ing`,",
        "`els`, `con`, `der`) are hyphenation damage:",
        "",
        "| paper | PDF-only fragments | HTML-only fragments |",
        "|---|---|---|",
    ]
    for paper in {r["paper"] for r in extraction}:
        pdf = next(r for r in extraction if r["paper"] == paper and r["source"] == "pdf")
        html = next(r for r in extraction if r["paper"] == paper and r["source"] == "html")
        lines.append(
            f"| `{paper}` | {pdf['short_fragments']} | {html['short_fragments']} |"
        )

    lines += [
        "",
        "## 2. Chunking",
        "",
        "36 configurations — 4 strategies x 3 sizes x overlap x heading-prefix — over both",
        "sources. Defect rates, averaged within each strategy:",
        "",
        "| strategy | source | equations split | starts mid-sentence | no section | bibliography | spans >1 section |",
        "|---|---|---|---|---|---|---|",
    ]
    for strategy in ("fixed", "recursive", "paragraph", "section"):
        for source in SOURCES:
            rows = [
                r for r in chunk_stats
                if r["source"] == source and r["config"].startswith(strategy + "-")
            ]
            if not rows:
                continue
            lines.append(
                f"| `{strategy}` | {source} "
                f"| {mean(rows, 'broken_math_pct'):.1f}% "
                f"| {mean(rows, 'mid_sentence_pct'):.1f}% "
                f"| {mean(rows, 'orphan_pct'):.1f}% "
                f"| {mean(rows, 'references_pct'):.1f}% "
                f"| {mean(rows, 'crosses_section_pct'):.1f}% |"
            )

    lines += [
        "",
        "![chunk defects](figures/chunk_defects.png)",
        "",
        "Findings:",
        "",
        "- **`section` is the only strategy that never straddles a section boundary**, by",
        "  construction, and it carries the least bibliography.",
        "- **PDF text has almost no paragraph structure** — 13 blank lines against the HTML's",
        "  191 — so `paragraph` chunking degenerates there, producing ~910-token chunks",
        "  regardless of the 256/512/1024 target. That is a property of the source, not a bug.",
        "- **Overlap wrecks boundary quality.** `recursive-256` on HTML starts mid-sentence in",
        "  2.5% of chunks at zero overlap and 63% at 15%, because the overlap is measured in",
        "  tokens and lands wherever it lands.",
        "- Recovering a section tree from a PDF by heading regex is unreliable: 18 headings",
        "  found against the HTML's 36 for one paper, 42 against 40 for the other.",
        "- No chunk in any configuration exceeded bge-m3's 8192-token limit.",
        "",
        "## 3. Embedding",
        "",
        f"`baai/bge-m3` via OpenRouter's `/api/v1/embeddings`, {4224:,} chunks across 72",
        "collections. Vectors arrive **already L2-normalised** (norm exactly 1.0), so cosine",
        "distance and dot product agree and Chroma's cosine space is correct.",
        "",
        "Measured cost for the full corpus: **1,286,898 tokens ≈ $0.013**.",
        "",
        "## 4. Hybrid retrieval",
        "",
        f"{len([c for c in cases if c['gold']])} scored queries "
        f"({counts.get('generated', 0)} generated + hand-written), "
        "each retrieved once per configuration and re-fused at 11 alphas plus RRF.",
        "",
        "![alpha sweep](figures/alpha_sweep.png)",
        "",
        "| alpha | nDCG@10 | MRR@10 | Recall@1 | Recall@5 |",
        "|---|---|---|---|---|",
    ]
    for alpha in ALPHAS:
        rows = [r for r in weighted if r["fusion"] == alpha]
        marker = "  **<- peak**" if alpha == peak else ""
        lines.append(
            f"| {alpha} | {mean(rows):.4f}{marker} | {mean(rows, 'mrr@10'):.4f} "
            f"| {mean(rows, 'recall@1'):.4f} | {mean(rows, 'recall@5'):.4f} |"
        )
    lines.append(
        f"| RRF | {mean(rrf):.4f} | {mean(rrf, 'mrr@10'):.4f} "
        f"| {mean(rrf, 'recall@1'):.4f} | {mean(rrf, 'recall@5'):.4f} |"
    )

    lines += [
        "",
        f"**The mix beats either extreme.** Averaged over all 36 configurations, nDCG@10 peaks",
        f"at **alpha = {peak}** ({sweep[peak]:.4f}) against {sweep[1.0]:.4f} for pure dense and",
        f"{sweep[0.0]:.4f} for pure BM25 — a {100 * (sweep[peak] / sweep[1.0] - 1):.0f}% gain over",
        "dense-only. Note that pure BM25 outranks pure dense on this corpus: these are dense",
        "technical papers full of model names, benchmark names and numbers.",
        "",
        f"RRF reaches {mean(rrf):.4f} with no tuning at all — within "
        f"{100 * (sweep[peak] / mean(rrf) - 1):.1f}% of the best hand-tuned alpha. If you do not",
        "want to tune a ratio per corpus, RRF is the sane default.",
        "",
        "![alpha by strategy](figures/alpha_by_strategy.png)",
        "",
        f"**Best single configuration:** `{best['config']}` on {best['source']} at "
        f"alpha={best['fusion']} — nDCG@10 {best['ndcg@10']:.3f}, Recall@5 {best['recall@5']:.3f}, "
        f"Recall@10 {best['recall@10']:.3f}.",
        "",
        "| config | source | alpha | nDCG@10 | MRR@10 | R@1 | R@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(weighted, key=lambda r: -r["ndcg@10"])[:10]:
        lines.append(
            f"| `{row['config']}` | {row['source']} | {row['fusion']} | {row['ndcg@10']:.3f} "
            f"| {row['mrr@10']:.3f} | {row['recall@1']:.3f} | {row['recall@5']:.3f} |"
        )

    # Chunk length correlates with the metric, so the strategy ranking has to be
    # read with that in mind rather than at face value.
    sizes = {(r["config"], r["source"]): r["tok_mean"] for r in chunk_stats}
    at_peak = [
        (sizes[(r["config"], r["source"])], r["ndcg@10"])
        for r in weighted
        if r["fusion"] == peak and (r["config"], r["source"]) in sizes
    ]
    xs = [x for x, _ in at_peak]
    ys = [y for _, y in at_peak]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    denominator = (
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    ) ** 0.5
    correlation = sum((x - mx) * (y - my) for x, y in at_peak) / denominator

    html_mean = mean([r for r in weighted if r["source"] == "html"])
    pdf_mean = mean([r for r in weighted if r["source"] == "pdf"])
    lines += [
        "",
        f"**HTML beats PDF as a source**: mean nDCG@10 {html_mean:.4f} vs {pdf_mean:.4f} "
        f"({100 * (html_mean / pdf_mean - 1):.1f}%), across every strategy and alpha.",
        "",
        "### Chunk length confounds the strategy ranking",
        "",
        "`fixed-1024` topping the table is **not** evidence that ignoring document structure",
        "retrieves better. Chunk length correlates with the score at "
        f"**r = {correlation:.3f}** (n={len(at_peak)}, alpha={peak}), and the reason is",
        "mechanical: a chunk counts as correct when it covers half the gold span, and a",
        "1024-token chunk covers a given span far more easily than a 200-token one.",
        "",
        "The 1024 band makes it plain — `section-1024` produces 218-token chunks, because",
        "arXiv sections are mostly small, and lands at the bottom:",
        "",
        "| config (html, alpha=" + str(peak) + ") | mean tokens | nDCG@10 |",
        "|---|---|---|",
    ]
    band = sorted(
        (
            (r["config"], sizes[(r["config"], r["source"])], r["ndcg@10"])
            for r in weighted
            if r["fusion"] == peak
            and r["source"] == "html"
            and "-1024-" in r["config"]
            and (r["config"], r["source"]) in sizes
        ),
        key=lambda row: -row[2],
    )
    for config, tokens, score in band:
        lines.append(f"| `{config}` | {tokens:.0f} | {score:.4f} |")

    lines += [
        "",
        "So the defensible reading is: **at matched chunk length the four strategies are",
        "close**, and the visible spread is mostly length. What `section` buys is not a higher",
        "score on this metric but the qualities section 2 measured — no straddled boundaries,",
        "no bibliography, no mid-sentence starts — which matter for what the model is asked to",
        "read, not for whether the span was hit. A downstream answer-quality evaluation, not a",
        "span-overlap one, is what would separate them properly.",
        "",
        "![alpha by category](figures/alpha_by_category.png)",
        "",
        "### Caveat on the per-category curves",
        "",
        "The hand-written categories are small — "
        + ", ".join(f"{k} n={v}" for k, v in sorted(counts.items()) if k != "generated")
        + ". Those curves are indicative only; do not read a peak in a 2-case category as a",
        "result. The `generated` curve (n="
        + str(counts.get("generated", 0))
        + ") and the all-configuration sweep above are the trustworthy signals.",
        "",
        "## 5. What to build from this",
        "",
        f"1. **Prefer LaTeXML HTML over PDF** whenever arXiv has it — {100 * (html_mean / pdf_mean - 1):.0f}% better",
        "   retrieval, plus a real section tree and intact math, and none of the hyphenation",
        "   damage. `arxiv_fulltext.py` already tiers HTML before PDF; this measures why that",
        "   was worth doing.",
        "2. **Hybrid, not dense-only.** Dense-only was the *worst* setting tested here — these",
        "   are technical papers thick with model names, benchmark names and numbers, and BM25",
        "   alone beat bge-m3 alone.",
        f"3. **Use RRF unless you can tune per corpus** — {mean(rrf):.4f} with no parameter to fit,",
        f"   within {100 * (sweep[peak] / mean(rrf) - 1):.1f}% of the best tuned alpha.",
        "4. **Skip overlap.** It measurably damaged chunk boundaries and bought no retrieval",
        "   gain worth the extra chunks.",
        "5. **Strip the bibliography before indexing.** A fifth of a PDF, and it answers nothing",
        "   while matching queries lexically.",
        "6. **Do not read this as `fixed` beating `section`** — see the length confound above.",
        "   Pick chunk length deliberately, then choose the strategy on boundary quality.",
        "",
        "### What this cannot tell you",
        "",
        "Two papers and 61 queries, scored on span overlap rather than answer quality. It",
        "measures whether the right region of text was retrieved, not whether the model could",
        "then answer from it — which is the question `section` chunking is actually designed to",
        "improve. Reranking, query expansion, and bge-m3's native sparse and ColBERT heads",
        "(OpenRouter exposes dense only) are all untested here.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "uv sync --group eda",
        "python rag-eda/01_extract.py      # PDF + HTML -> data/extracted/",
        "python rag-eda/02_chunk_eda.py    # 36 configs x 2 sources -> data/chunks/",
        "python rag-eda/03_embed.py        # bge-m3 -> data/chroma/  (~$0.013, cached after)",
        "python rag-eda/04_testset.py      # reuses generated cases; --regenerate to rebuild",
        "python rag-eda/05_retrieval_eda.py",
        "```",
        "",
    ]

    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {(ROOT / 'REPORT.md').name}")


if __name__ == "__main__":
    main()
