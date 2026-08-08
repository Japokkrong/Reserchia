# Reserchia

A LangGraph agent running DeepSeek V4 Flash through OpenRouter, with one tool: the current
date and time.

## Setup

```bash
uv sync
cp .env.example .env      # then paste your key into OPENROUTER_API_KEY
```

Get a key at <https://openrouter.ai/keys>.

## Run

```bash
uv run reserchia
```

```
Reserchia — deepseek/deepseek-v4-flash-0731 (reasoning off)
Commands: /reset to clear memory, /exit to quit.

> what time is it in Tokyo?

  [tool] get_current_datetime(timezone='Asia/Tokyo')
    ISO 8601: 2026-08-04T02:14:52+09:00
    Date: 2026-08-04
    Day of week: Tuesday
    ... (+3 more lines)

It's just past 2:14 AM on Tuesday, August 4th in Tokyo.
```

With `OPENROUTER_REASONING=enabled` the model's thinking streams first, under `[thinking]`, and
reasoning tokens are broken out in the count.

Conversation memory lasts for the life of the process (`InMemorySaver`); `/reset` starts a
fresh thread.

### Token usage

Every turn ends with a usage line:

```
  [tokens] 2 calls · in 9,703 (8,192 cached) · out 166 · embed 6 · turn 9,875 · session 22,726
```

- **calls** — a ReAct turn is several model calls, not one; the model is re-invoked after each
  tool result.
- **in** grows every turn, because `MessagesState` resends the whole conversation each call.
  Watching it climb (5,573 → 7,070 → 9,703 over three turns above) is the clearest picture of
  why retrieval beats pasting whole papers into the history.
- **cached** is prompt-cache reuse, billed at a discount — large because the system prompt and
  history repeat.
- **embed** is bge-m3 spend for library search, and includes any background indexing that
  finished during the turn.

Counts come from the provider's own `usage` field, not an estimate. Cost is not shown: OpenRouter
reports it, but `langchain_openai` discards the field before it reaches us — the same class of
gap that `llm.py` works around for reasoning.

## Where the time goes

Every step is timed. `/stats` in the REPL, or `python scripts/log_report.py` over the log:

```
where the time goes  (0.9s measured of 10.0s total; the rest is the model)
  step                  calls      total      mean   share
  embed                     1       0.8s      840ms    8.4%
  bm25.build                1       0.0s       22ms    0.2%
  chroma.query              1       0.0s        8ms    0.1%
  bm25.score                1       0.0s        1ms    0.0%
```

Timings always go to a local JSONL under `<store_dir>/logs/` — no account, no network. Setting
`LANGSMITH_TRACING=true` additionally sends the whole trace tree to LangSmith; `langsmith` is
already installed, being a dependency of `langchain-core`.

**Both, rather than just LangSmith, for two reasons.** LangSmith traces LangChain runnables, so
the steps that actually cost time — the bge-m3 round trip, Chroma, BM25, arXiv's rate limiter,
Kroki — are invisible to it, being ordinary functions. And `/stats` cannot query a remote service
with tracing off. So `observability.track()` feeds both from one measurement, and the
sub-steps are added as explicit `langsmith.trace()` spans. Verified against the API: all eight
appear in the trace tree with latencies matching the local log.

Three details worth keeping:

- **The throttle is timed separately from the request.** A three-second wait for arXiv's rate
  limit is not a slow API, and one combined number would say the opposite.
- **A dead key must not look like a crash.** An expired key makes LangSmith's uploader emit a
  `Failed to multipart ingest runs ... 403` line per batch — 21 from a single probe. Those
  loggers are muted, so the agent answers normally and stderr stays clean.
- **Spans are not free.** LangSmith's own cost 84 µs each, which is half a BM25 query, so
  `track()` checks whether tracing is on before building one — measured at 2 µs when off.

**Cost is the real billed figure**, not an estimate. OpenRouter reports it in `usage.cost` when
the request asks; `langchain_openai` drops the field, so `ChatOpenRouter` rescues it the same way
it already rescues reasoning. Hosted tools can only estimate from a price table that may not list
this model.

## Diagrams and equations

Papers are full of things prose is bad at — pipelines, architectures, training stages, the
formula a method turns on. Two tools let the agent show them instead of describing them:

| Tool | For |
|---|---|
| `render_diagram(mermaid, caption=)` | structure worth seeing: a pipeline, an architecture, a multi-stage process |
| `render_equation(latex, caption=)` | a display equation, captioned and numbered |

![Diagram and equation rendering](docs/chainlit-visuals.png)

Ordinary inline maths needs no tool — `$d_k$` is typeset automatically, in answers *and* in
citation side panels, which matters because retrieved arXiv passages keep their original LaTeX.
That is the `latex = true` flag in `.chainlit/config.toml`; Chainlit ships KaTeX but leaves it
off by default.

The prompt is written as *when*, not *that*, because the failure mode is an agent that draws two
boxes for every answer. A plain factual question still comes back as one sentence.

**Rendering is the validation.** Mermaid's grammar is large and its parser is JavaScript, so a
Python check would be a heuristic that still waves through diagrams mermaid rejects. Diagrams go
to [Kroki](https://kroki.io), which runs the real parser and hands back its error verbatim —

```
Diagram not rendered, mermaid rejected it: Error 400: SyntaxError:
Parse error on line 3: ...lowchart TD  A --> ---------------------^
```

— so the model can fix it and retry *before* the answer is written. That is also why the render
is eager rather than deferred to the UI: a lazy render would surface the error too late to act
on. PNGs are cached under `<store_dir>/diagrams/<sha256>.png`, so a repeat costs nothing.

**This sends the diagram source to kroki.io.** It is model-authored text about public papers,
not your data, but it is an outbound call and you should know it happens. The local alternative,
`@mermaid-js/mermaid-cli`, needs a headless Chromium (~200 MB) — rejected as too much weight for
one feature.

In the terminal, which cannot draw, the CLI prints the mermaid source and the path of the PNG
that was rendered anyway.

## Web UI

A second front end over the same agent and the same library:

```bash
uv sync --group ui
uv run reserchia-ui          # or: chainlit run ui/app.py
```

![Chainlit UI](docs/chainlit-ui.png)

- **Reasoning is a one-click toggle** — the brain button beside the paperclip. It is a Chainlit
  *command* with `button` and `persistent` set, so it latches as a mode and its state arrives as
  `message.command` on every turn. Reading it there, rather than in a settings callback, means
  the mode can never disagree with the button the user is looking at.

  Flipping it rebuilds the agent with `reasoning` swapped and hands it the *same* checkpointer,
  so the conversation survives — verified by toggling mid-chat and asking a question that needs
  the earlier turn. That works only because `llm.py` strips stale `reasoning_details` when
  reasoning goes off; without it, replaying a history recorded with reasoning on would 400.
- **Citations are clickable** — `arXiv:2404.16130 §3.1` opens the passage the claim came from,
  with its section heading and relevance score.
- **Tool calls are collapsible steps** showing the JSON arguments and the result.
- **Token usage** sits under each answer, the same fields as the CLI.

Both UIs share `turn.py`, so neither has its own opinion about what the graph's event stream
means — the rules for it (tool results also arrive on the `messages` channel, tool calls repeat
across chunks, usage arrives once per model call) live in one place.

### The one fragile thing: how citations become clickable

Chainlit's `prepareContent` scans a finished message for **element names as plain text** and
wraps each occurrence in a link itself. So `ui/app.py` names each element after *the model's own
citation wording* and then emits that wording **bare** — no markdown link. Writing your own link
makes Chainlit rewrite the text inside it and the page shows a literal `](`.

Unresolved citations are the opposite case: no element exists, so they get a real markdown link
to the abstract page and Chainlit leaves them alone. Same code path, two behaviours, and a
citation is never a dead end.

Passages come from `rag/citations.py`, fed by both `search_paper_library` **and**
`get_arxiv_fulltext`. The second matters more than it looks: the first answer about any paper is
written from full text, so without registering those sections the answers most worth checking
would be exactly the ones whose citations could not be opened.

## The graph

A ReAct loop wired explicitly in `agent.py` — `agent` calls the model, `tools_condition`
routes on whether the reply carried tool calls, and `ToolNode` runs them and hands control
back. The cycle repeats until the model answers without calling a tool.

```mermaid
graph TD
    start(["__start__"])
    agent("agent")
    tools("tools")
    finish(["__end__"])

    start --> agent
    agent -. "tool calls present" .-> tools
    agent -. "no tool calls" .-> finish
    tools -- "ToolMessage appended" --> agent
```

State is `MessagesState`, so every node appends to one growing message list — which is why
the reasoning round trip in `llm.py` matters: each trip around the loop resends the whole
history, assistant messages included.

## Layout

| File | Role |
|---|---|
| `config.py` | Environment → `Settings` |
| `llm.py` | `ChatOpenRouter` — the model, plus the reasoning round trip below |
| `arxiv_client.py` | arXiv API HTTP, throttling, Atom parsing, formatting |
| `arxiv_fulltext.py` | Full paper text — LaTeXML HTML parsing, PDF fallback, cache |
| `rag/` | The paper library — chunking, bge-m3, ChromaDB, hybrid search |
| `tools/datetime_tools.py` | `get_current_datetime` |
| `tools/arxiv_tools.py` | The four arXiv tools |
| `tools/rag_tools.py` | `search_paper_library`, `list_paper_library` |
| `visuals.py` | Mermaid rendering, LaTeX checking, per-turn visual registry |
| `tools/visual_tools.py` | `render_diagram`, `render_equation` |
| `tools/__init__.py` | `TOOLS` registry |
| `agent.py` | The `StateGraph` |
| `cli.py` | The REPL |

## arXiv tools

| Tool | For |
|---|---|
| `search_arxiv(query, category=, sort_by=, …)` | "what research exists on X" |
| `browse_arxiv(category, year, month=, …)` | the `arxiv.org/list/cs.HC/2019-01` pattern |
| `get_arxiv_paper(paper_ids)` | metadata + abstract, by identifier |
| `get_arxiv_fulltext(paper_id, section=)` | the paper's actual body text |

The first three go through the official API at `export.arxiv.org/api/query`, not the browse
pages. Results carry real `arxiv.org/abs/` links, and the system prompt tells the model to cite
those rather than recall papers from memory — and to reach for `get_arxiv_fulltext`, not the
abstract, whenever it is asked to summarize a paper or discuss its details.

Two things in `arxiv_client.py` are load-bearing:

**The throttle.** arXiv's [terms of use](https://info.arxiv.org/help/api/tou.html) require no
more than one request every three seconds on a single connection. A ReAct loop will fire
several searches back to back, so the gap is enforced in `_throttle()` under a lock rather than
trusted to callers. Expect a visible pause when the model chains two lookups.

**Raw query encoding.** The API manual documents its examples pre-encoded — `ti:%22quantum
theory%22`. That is only correct when hand-building a URL string. Handing those characters to
an HTTP client double-encodes them to `%2522`, and arXiv does not error — it silently drops
the phrase grouping:

```
ti:"attention is all you need"      ->      35 results
ti:%22attention is all you need%22  -> 459,755 results
```

So queries are built as raw strings in a params dict and `httpx` does the encoding. "Fixing"
them to match the manual re-breaks it invisibly.

One honest caveat: `browse_arxiv` filters on *submission* date while the website's listing
pages order by *announcement* date, so counts differ slightly at period boundaries — cs.HC for
2019-01 is 62 here against the site's 64.

### Full text

`get_arxiv_fulltext` tries three sources in order, because no single one covers the archive:

1. **`arxiv.org/html/<id>`** — arXiv's own LaTeXML rendering. Recent papers, *including ones
   ar5iv has not converted yet*.
2. **`ar5iv.labs.arxiv.org/html/<id>`** — LaTeXML over the back catalogue. It reports failure by
   **redirecting to `/abs/`**, not with a status code, so `arxiv_fulltext` checks the final URL.
   Miss this and you parse the abstract page as if it were the paper.
3. **`arxiv.org/pdf/<id>`** → `pypdf`. Works everywhere, reads worst.

Measured over eight papers spanning 1996–2026: arXiv HTML had three, ar5iv five, the two
together seven. Only a 1996 `hep-th` paper needed the PDF.

Parsing is stdlib `html.parser` — no beautifulsoup, no lxml. LaTeXML output is regular enough,
and one detail earns it: **every `<math>` carries an `alttext` holding the original LaTeX**, so
formulas come out as `$h_{t}$` instead of flattened MathML. Extraction starts at
`<article class="ltx_document">`, which is what keeps arxiv.org's cookie dialog and fundraising
banner out of the paper.

PDF-derived text is labelled lower-fidelity in the tool's own output, so the model hedges on it.
That is not defensive boilerplate — the 1996 paper extracts as `bla ck hole`, `vi olation`,
`eﬀect`. Ligatures are normalized; the kerning-induced word splits can't be, since there is no
safe way to tell them from real spaces.

**Size.** A typical paper is ~10k tokens, but *A Survey of Large Language Models* is ~153k — and
`MessagesState` resends whatever lands in history on every later turn. So past
`FULLTEXT_LIMIT` (60k chars) the tool returns the abstract plus a section index instead, and the
model asks for the section it needs. Parsed papers are cached (4 deep), so those follow-ups cost
no download and no throttle wait.

## The paper library

Papers the agent has read are kept in a local ChromaDB collection, so asking about one twice
does not fetch it twice.

```
"summarize arXiv 1706.03762"
  |
  +-> search_paper_library    empty / paper absent -> escalate
  +-> get_arxiv_fulltext      answer this turn from the full text
        |
        +-> background: chunk -> bge-m3 -> ChromaDB

next session, same paper
  +-> search_paper_library    5 passages, no API call at all
```

`get_arxiv_fulltext` returns immediately and indexes on a single background worker, so the two
paths really do run at once. The worker is deliberately **not** a daemon — the CLI waits for it
on exit (up to 30s), because a daemon killed mid-write leaves a half-indexed paper.

**Where the fallback decision is made matters.** Whether a paper is *in* the library is decided
exactly, by an id lookup. Whether the retrieved passages *answer the question* is left to the
model, which is reading them anyway. It is tempting to use a similarity threshold for the
second part — that was measured and it does not work: correct top-1 hits scored as low as
0.385 while questions the corpus could not answer scored 0.492–0.498. Any cutoff rejecting the
latter also rejects real answers.

Settings come from `rag-eda/` (see [REPORT.md](rag-eda/REPORT.md)): section-aligned chunks, no
overlap, `baai/bge-m3`, and dense+BM25 fused at **α = 0.5** — the peak on Recall@1/@5/@10 and
MRR simultaneously, with both extremes clearly worse. `RESERCHIA_RAG_ALPHA` exposes it.

Two things in `rag/` are load-bearing:

- **Embedding batches are sized by token budget, not item count.** bge-m3's 8192-token context
  applies to the whole request, and exceeding it returns HTTP 429 *"engine is currently
  overloaded"* rather than a size error — badly misleading. Measured: 6,390 tokens per request
  succeeded, 12,408 failed.
- **Score normalisation floors at epsilon, not zero.** Plain min-max maps a pool's worst item to
  exactly 0.0, making it indistinguishable from an item that never appeared in that pool; a
  BM25-only result could then tie with dense's last hit and win on sort order. The test is that
  α=1.0 reproduces pure dense ranking exactly and α=0.0 pure BM25.

`/library` lists what has been read. Chroma's own hybrid search (`Knn`, `Rrf`) is hosted-only
and raises `NotImplementedError` on a local `PersistentClient`, which is why BM25 is in Python.

## Adding a tool

1. Write a `@tool`-decorated function in `src/reserchia/tools/`. Its docstring is what the
   model sees — write it for the model, and describe every argument.
2. Add it to `TOOLS` in `tools/__init__.py`.

That's it — `agent.py` binds whatever is in `TOOLS`, and `ToolNode` executes it.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `OPENROUTER_MODEL` | `deepseek/deepseek-v4-flash-0731` | Any OpenRouter model id |
| `OPENROUTER_REASONING` | `disabled` | `enabled` \| `disabled` |
| `OPENROUTER_REASONING_EFFORT` | `low` | `minimal`…`max`; only sent when reasoning is on |
| `OPENROUTER_TEMPERATURE` | `0` | |
| `OPENROUTER_SITE_URL` | — | Optional `HTTP-Referer`, for OpenRouter's leaderboards |
| `OPENROUTER_APP_NAME` | — | Optional `X-Title` |
| `OPENROUTER_EMBED_MODEL` | `baai/bge-m3` | Library encoder; changing it invalidates stored vectors |
| `RESERCHIA_STORE_DIR` | `~/.local/share/reserchia` | Where the paper library lives |
| `RESERCHIA_RAG_ALPHA` | `0.5` | Library search mix: 0 = BM25, 1 = embeddings |

## Why `llm.py` looks the way it does

**Do not delete the overrides in `llm.py` as dead code — they are what makes reasoning mode
usable at all.**

OpenRouter returns reasoning in two non-standard fields, `reasoning` (plaintext) and
`reasoning_details` (structured), and
[its docs](https://openrouter.ai/docs/use-cases/reasoning-tokens) require the structured one
to come back when the conversation continues — which, in an agent, it always does:

> When passing back `reasoning_details`, preserve the exact sequence returned by the model —
> no rearranging or modification permitted.

`langchain_openai` drops both. From its own `_convert_message_to_dict` docstring:

> Non-standard response fields added by third-party providers (e.g. `reasoning_content`,
> `reasoning_details`) are **not** extracted or preserved.

So a stock `ChatOpenAI` reasons on the first call and loses it the moment a tool result comes
back. `ChatOpenRouter` closes the round trip in three overrides:

- `_create_chat_result` — capture reasoning from non-streaming responses
- `_convert_chunk_to_generation_chunk` — capture it from streaming deltas. Not optional: the
  REPL streams, and LangGraph's `stream_mode="messages"` attaches a callback handler that
  forces the streaming path even though the graph node calls `llm.invoke()`.
- `_get_request_payload` — re-inject it into outbound assistant messages

### The streaming trap

`reasoning_details` arrive as deltas, and LangChain merges chunk `additional_kwargs` with
`merge_lists`, which field-merges any two list items sharing an integer `index` — concatenating
*every* string field, not just the payload. Two chunks of one detail merge to
`format: "deepseek-v4deepseek-v4"`, corrupting precisely the structure OpenRouter says to send
back unmodified.

So deltas are stored as fragments with `index` renamed to `__or_index`, which makes
`merge_lists` append them verbatim, and `_coalesce()` stitches them back at request time —
concatenating only `text`/`summary`/`data` and passing every other field through untouched.

With `OPENROUTER_REASONING=disabled` there is nothing to carry and all three overrides degrade
to no-ops, so one class covers both modes. `_get_request_payload` also *strips* stale reasoning
when reasoning is off, so flipping the toggle mid-conversation stays valid too.

### Note on `langchain-openrouter`

There is a separate `langchain-openrouter` package with its own `ChatOpenRouter`. This project
does not use it: it pulls in a second SDK, and the goal here was the OpenAI-compatible path.
The local class is unrelated despite the shared name.
