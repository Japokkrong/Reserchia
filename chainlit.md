# Reserchia

A research assistant over arXiv, running DeepSeek V4 Flash through OpenRouter.

**Ask it about a paper** — by identifier (`summarize arXiv 1706.03762`) or by name
(`what does the GraphRAG paper say about community summaries?`).

It searches papers it has already read before fetching anything, so the second question about a
paper costs no API call. Papers get read once and remembered.

### Reading an answer

- **Citations are clickable.** `[arXiv:1706.03762 §3.5]` opens the passage the claim came from,
  in a side panel, with its relevance score.
- **Tool calls are collapsible steps.** Expand one to see the arguments and what came back.
- **Token usage** sits under each answer: model calls, input (with the cached share), output,
  and embedding tokens for library search.

Input tokens climb every turn — the whole conversation is resent on each model call. That is
what the library is for.
