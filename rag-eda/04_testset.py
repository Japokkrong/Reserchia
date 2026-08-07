"""Build the retrieval test set: generated cases plus a hand-written hard set.

Ground truth is a **character span in each source's text**, never a chunk id.
Chunk ids differ between the 72 configurations, so a chunk-id gold set could
only ever score the configuration that produced it. Spans let one test set
score all of them, and let the same question be scored against PDF-derived and
HTML-derived chunks alike.

The generated half is validated twice, and both matter:

- **answerability** -- if the model answers correctly without seeing the chunk,
  the question tests world knowledge, not retrieval.
- **lexical overlap** -- if the question copies a long run of words from the
  chunk, BM25 finds it trivially and the whole alpha sweep tilts. Generators do
  this constantly.

Run: python rag-eda/04_testset.py
"""

from __future__ import annotations


import random

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA,
    EXTRACTED,
    PAPERS,
    SOURCES,
    ensure_dirs,
    lexical_tokens,
    read_json,
    read_jsonl,
    require,
    write_json,
)
from common import CHUNKS  # noqa: E402

from reserchia.config import get_settings  # noqa: E402
from reserchia.llm import build_llm  # noqa: E402

#: Questions are generated from this configuration's chunks -- real section
#: structure at a size that holds a self-contained fact.
SOURCE_CONFIG = "section-512-o0-hp0"
SAMPLE = 50
#: A question sharing this many consecutive words with its chunk is a giveaway.
MAX_SHARED_RUN = 8

random.seed(20260808)


GENERATE = """You are building a retrieval benchmark from a research paper.

Below is one passage from "{title}".

<passage>
{passage}
</passage>

Write ONE question that:
- is answered by this passage and would be hard to answer without it
- a researcher might plausibly ask
- does NOT copy long phrases from the passage; use your own wording
- does NOT mention "the passage", "the text", or "this paper"
- is a single sentence

Reply with only the question."""

ANSWERABLE = """Answer this question from your own knowledge, in one short sentence.
If you do not know, reply exactly: UNKNOWN

Question: {question}"""

JUDGE = """A retrieval benchmark question must not be answerable from general knowledge.

Question: {question}
Reference answer from the source passage: {passage}
Candidate answer given WITHOUT the passage: {answer}

Does the candidate answer convey the same specific fact as the reference?
Reply with exactly one word: YES or NO."""


def longest_shared_run(question: str, passage: str) -> int:
    q = lexical_tokens(question)
    p = lexical_tokens(passage)
    if not q or not p:
        return 0
    positions = {}
    for index, word in enumerate(p):
        positions.setdefault(word, []).append(index)

    best = 0
    for start in range(len(q)):
        for begin in positions.get(q[start], []):
            length = 0
            while (
                start + length < len(q)
                and begin + length < len(p)
                and q[start + length] == p[begin + length]
            ):
                length += 1
            best = max(best, length)
    return best


def locate(gold_text: str, target: str, window_step: int = 120) -> tuple[int, int]:
    """Find where `gold_text` lives inside `target`, tolerating extraction drift.

    An exact search fails across sources: PDF text hyphenates, reflows and
    keeps furniture the LaTeXML render drops. Matching on token containment
    instead finds the same passage in both, which is what lets one question be
    scored against PDF chunks and HTML chunks alike.
    """
    exact = target.find(gold_text)
    if exact >= 0:
        return exact, exact + len(gold_text)

    wanted = set(lexical_tokens(gold_text))
    if not wanted:
        return 0, 0
    width = max(200, len(gold_text))

    best, best_score = (0, min(width, len(target))), 0.0
    for start in range(0, max(1, len(target) - width + 1), window_step):
        window = target[start : start + width]
        found = set(lexical_tokens(window))
        score = len(wanted & found) / len(wanted)
        if score > best_score:
            best_score, best = score, (start, start + len(window))
    return best if best_score >= 0.35 else (0, 0)


def gold_spans(gold_text: str, texts: dict) -> dict:
    spans = {}
    for source in SOURCES:
        start, end = locate(gold_text, texts[source])
        spans[source] = {"start": start, "end": end}
    return spans


HAND_CASES = [
    # --- exact lexical: an identifier only one paper contains -----------------
    ("lexical", "2603.10910v2", "What does PP-DocLayout-V3 do in the pipeline?",
     "PP-DocLayout-V3 first performs layout analysis, followed by parallel region-level recognition"),
    ("lexical", "2603.10910v2", "What is CogViT?",
     "It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder"),
    # --- paraphrase: no shared vocabulary with the passage --------------------
    ("paraphrase", "2603.10910v2", "How did they make the model generate output faster?",
     "GLM-OCR introduces a Multi-Token Prediction (MTP) mechanism that predicts multiple tokens per step, significantly improving decoding throughput"),
    ("paraphrase", "2601.14722v1", "Why is this language harder for OCR than English?",
     "Thai presents additional challenges due to script complexity from non-latin letters, the absence of explicit word boundaries"),
    # --- cross-paper: both papers discuss model sizes and stages --------------
    ("cross-paper", "2603.10910v2", "Which of these models has 0.9 billion parameters?",
     "GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding"),
    ("cross-paper", "2601.14722v1", "Which model targets Thai document extraction?",
     "This paper presents Typhoon OCR, an open VLM for document extraction tailored for Thai and English"),
    ("cross-paper", "2603.10910v2", "What happens in the reinforcement learning stage of training?",
     "Stage 4: Reinforcement Learning"),
    ("cross-paper", "2601.14722v1", "What is in stage 1 of the dataset creation pipeline?",
     "Dataset Creation Pipeline"),
    # --- numeric / table lookup ----------------------------------------------
    # The quote must be unique. "OmniDocBench" alone occurs 7 times, and the
    # span then landed on a figure caption rather than the result -- scoring
    # retrieval down for finding the passage that actually answers the question.
    ("numeric", "2603.10910v2", "How does it score on OmniDocBench v1.5?",
     "GLM-OCR achieves 94.6 on OmniDocBench v1.5 [24], ranking first among all evaluated models"),
    ("numeric", "2601.14722v1", "Where can the 7B checkpoint be downloaded?",
     "Typhoon OCR 7B: https://huggingface.co/scb10x/typhoon-ocr-7b"),
    # --- negatives: neither paper answers these ------------------------------
    ("negative", None, "What learning rate schedule does BERT use for pretraining?", None),
    ("negative", None, "How much did the authors pay for their GPU cluster?", None),
]


def main() -> None:
    ensure_dirs()
    require(EXTRACTED / "summary.json", "rag-eda/01_extract.py")

    texts = {
        paper: {source: read_json(EXTRACTED / f"{paper}.{source}.json")["text"]
                for source in SOURCES}
        for paper in PAPERS
    }
    titles = {
        paper: read_json(EXTRACTED / f"{paper}.html.json")["title"] or paper
        for paper in PAPERS
    }

    cases = []

    # -- hand-written ---------------------------------------------------------
    for category, paper, question, quote in HAND_CASES:
        if paper is None:
            cases.append({
                "id": f"hand-{len(cases):03d}", "origin": "hand", "category": category,
                "question": question, "paper": None, "gold": None, "answerable": False,
            })
            continue
        cases.append({
            "id": f"hand-{len(cases):03d}", "origin": "hand", "category": category,
            "question": question, "paper": paper,
            "gold": gold_spans(quote, texts[paper]), "gold_quote": quote,
            "answerable": True,
        })
    print(f"hand-written: {len(cases)} cases")

    # -- generated ------------------------------------------------------------
    # Reused unless --regenerate. A benchmark that produces different questions
    # on every run cannot be compared against its own earlier results, and the
    # hand-written half is the part that gets edited.
    existing = DATA / "testset.json"
    if existing.exists() and "--regenerate" not in sys.argv:
        previous = [c for c in read_json(existing) if c["origin"] == "generated"]
        if previous:
            for case in previous:
                case["gold"] = gold_spans(case["gold_quote"], texts[case["paper"]])
            cases.extend(previous)
            print(f"reusing {len(previous)} generated cases (--regenerate to rebuild)")
            finish(cases)
            return

    pool = []
    for source_file in sorted(CHUNKS.glob(f"{SOURCE_CONFIG}.html.jsonl")):
        pool = [r for r in read_jsonl(source_file) if r["n_tokens"] >= 80]
    if not pool:
        sys.exit(f"no chunks for {SOURCE_CONFIG}; run rag-eda/02_chunk_eda.py")

    # Stratify across papers so one does not dominate the benchmark.
    by_paper = {p: [c for c in pool if c["paper"] == p] for p in PAPERS}
    sample = []
    for paper, rows in by_paper.items():
        random.shuffle(rows)
        sample.extend(rows[: SAMPLE // len(PAPERS)])
    print(f"\ngenerating from {len(sample)} chunks (>=80 tokens, stratified)")

    llm = build_llm(get_settings())
    kept, dropped = [], {"overlap": 0, "world_knowledge": 0, "malformed": 0}

    for index, chunk in enumerate(sample, 1):
        passage = chunk["text"]
        question = llm.invoke(
            GENERATE.format(title=titles[chunk["paper"]], passage=passage)
        ).content.strip()
        question = question.strip('"').split("\n")[0].strip()

        if len(question) < 15 or "?" not in question:
            dropped["malformed"] += 1
            continue

        run = longest_shared_run(question, passage)
        if run > MAX_SHARED_RUN:
            dropped["overlap"] += 1
            print(f"  [{index:>2}] dropped: {run} shared words -- {question[:60]}")
            continue

        blind = llm.invoke(ANSWERABLE.format(question=question)).content.strip()
        if not blind.upper().startswith("UNKNOWN"):
            verdict = llm.invoke(
                JUDGE.format(question=question, passage=passage[:900], answer=blind)
            ).content.strip().upper()
            if verdict.startswith("YES"):
                dropped["world_knowledge"] += 1
                print(f"  [{index:>2}] dropped: answerable without it -- {question[:60]}")
                continue

        kept.append({
            "id": f"gen-{len(kept):03d}", "origin": "generated", "category": "generated",
            "question": question, "paper": chunk["paper"],
            "gold": gold_spans(passage, texts[chunk["paper"]]),
            "gold_quote": passage[:300], "answerable": True,
            "shared_run": run,
        })
        print(f"  [{index:>2}] kept: {question[:72]}")

    cases.extend(kept)
    print(f"\ngenerated kept {len(kept)}/{len(sample)}; dropped {dict(dropped)}")
    finish(cases)


def finish(cases: list[dict]) -> None:
    write_json(DATA / "testset.json", cases)
    print(f"total cases: {len(cases)}")

    by_category: dict[str, int] = {}
    for case in cases:
        by_category[case["category"]] = by_category.get(case["category"], 0) + 1
    print("by category:", by_category)

    thin = [k for k, v in by_category.items() if v < 5 and k != "generated"]
    if thin:
        print(
            f"NOTE: {', '.join(thin)} have under 5 cases each -- their per-category "
            "curves are indicative only, not significant."
        )

    unlocated = [
        c["id"] for c in cases
        if c["gold"] and any(g["end"] == 0 for g in c["gold"].values())
    ]
    if unlocated:
        print(f"WARNING: gold span not found in some source for: {unlocated}")
    print("wrote data/testset.json")


if __name__ == "__main__":
    main()
