# rag-eda

An experiment, not a feature. Nothing in `src/reserchia/` depends on any of this.

Reserchia's agent currently retrieves whole papers and hands them to the model intact, which
works only because DeepSeek V4 has a 1M context. Chunked retrieval is the next step, and this
measures what it should look like for arXiv papers before any of it gets built:

- which **chunking** strategy suits arXiv document structure
- how **bge-m3** (via OpenRouter) behaves as the encoder
- what **dense/BM25 ratio** actually retrieves best, in ChromaDB

Findings are in **[REPORT.md](REPORT.md)**, regenerated from the measurements on every run.

## Running it

The EDA dependencies live in an optional group, so the agent's runtime stays lean:

```bash
uv sync --group eda
```

Then, in order — each stage reads the previous one's output:

```bash
python rag-eda/01_extract.py       # PDF + LaTeXML HTML -> data/extracted/
python rag-eda/02_chunk_eda.py     # 36 configs x 2 sources -> data/chunks/ + figures
python rag-eda/03_embed.py         # bge-m3 -> 72 Chroma collections
python rag-eda/04_testset.py       # retrieval test cases -> data/testset.json
python rag-eda/05_retrieval_eda.py # alpha sweep -> figures + REPORT.md
```

`03_embed.py` costs about **$0.013** the first time and nothing afterwards — every vector is
cached on disk under `sha256(model + text)`. `04_testset.py` reuses its generated questions
unless you pass `--regenerate`, so the benchmark stays comparable between runs.

## The pieces

| file | role |
|---|---|
| `common.py` | paths, bge-m3 tokenisation with char offsets, bibliography splitting |
| `chunkers.py` | the four strategies, all returning character spans |
| `embedding.py` | bge-m3 through OpenRouter: batching, retry, disk cache |
| `retrieval.py` | Chroma + BM25, weighted and RRF fusion, ranking metrics |

Extraction is not reimplemented here — `reserchia.arxiv_fulltext` already fetches both the PDF
text and the LaTeXML HTML with its section tree, so `01_extract.py` just calls it.

## Three things that are easy to get wrong

**Ground truth is a character span, not a chunk id.** The 72 configurations produce different
chunk ids, so a chunk-id gold set could only ever score the configuration that produced it. A
chunk counts as correct when it covers at least half the gold span, which lets one test set
score every configuration — and score PDF-derived and HTML-derived chunks alike.

**Scores must be normalised before they are mixed.** Chroma returns cosine *distance* (0..2,
lower is better); BM25 returns an unbounded score an order of magnitude larger. Combine them raw
and whichever has the bigger magnitude wins at every alpha — the sweep measures nothing. The
test for it is that `alpha=1.0` reproduces pure dense ranking exactly and `alpha=0.0` reproduces
pure BM25. That check caught a real bug here: plain min-max maps a pool's *worst* item to 0.0,
making it indistinguishable from an item that never appeared in that pool, so BM25-only results
could tie with dense's last hit and win on sort order. Hence `FLOOR` in `retrieval.py`.

**Generated questions flatter BM25.** An LLM asked to write a question about a passage tends to
reuse its wording, and BM25 then finds it trivially. `04_testset.py` drops any question sharing
more than 8 consecutive words with its source chunk, and drops any the model can answer without
the chunk at all. On this run the overlap guard never fired — measured max run was 6 words — but
that is a result, not a reason to remove the guard.

## Caveat worth keeping in view

Two papers, 61 queries. The all-configuration alpha sweep and the 49 generated cases carry
enough weight to act on; the hand-written categories have 2–4 cases each and their per-category
curves are indicative only. Do not read a peak in a 2-case category as a finding.
