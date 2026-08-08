"""Chainlit front end for Reserchia.

    chainlit run ui/app.py

Same agent, same paper library, same token accounting as the terminal REPL --
`turn.interpret` is shared, so neither UI has its own opinion about what the
graph's event stream means. What this adds is what a terminal cannot do:
citations you can click to read the passage the answer was drawn from.

How the citations work, because it is not obvious and breaks silently:
Chainlit's markdown renderer intercepts **anchors**. A link `[Label](url)`
becomes a clickable side-panel reference when `Label` exactly matches an
attached element's name, and stays an ordinary external link otherwise. So the
elements here are named after whatever the model wrote -- `arXiv:1706.03762
§3.5` -- rather than after the passage's real heading. Element names are
arbitrary; the model's wording is not, and rewriting it would be worse than
leaving a citation unresolved.
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import chainlit as cl
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reserchia.agent import build_app  # noqa: E402
from reserchia.config import ConfigError, get_settings  # noqa: E402
from reserchia.rag import citations, ingest, store  # noqa: E402
from reserchia.turn import (  # noqa: E402
    Reasoning,
    StreamState,
    Token,
    ToolCall,
    ToolResult,
    Usage,
    interpret,
)

#: A citation as the model writes it: `[arXiv:2603.10910 §3.1 Title]`, sometimes
#: already carrying a markdown link. The trailing `(...)` is captured so it can
#: be consumed -- otherwise rewriting the bracket part leaves a stray URL behind.
CITATION = re.compile(
    r"\[(?P<label>arXiv:[^\]\[]+)\](?:\((?P<url>[^)\s]+)\))?", re.IGNORECASE
)

#: Lucide icon names, so a glance distinguishes library hits from API calls.
ICONS = {
    "search_paper_library": "library",
    "list_paper_library": "library-big",
    "get_arxiv_fulltext": "book-open",
    "get_arxiv_paper": "file-text",
    "search_arxiv": "search",
    "browse_arxiv": "list",
    "get_current_datetime": "clock",
}


def _greeting() -> str:
    settings = get_settings()
    mode = "reasoning on" if settings.reasoning_enabled else "reasoning off"
    lines = [f"**Reserchia** — `{settings.model}` ({mode})"]
    try:
        papers = store.papers()
    except Exception:  # noqa: BLE001 - a broken library must not block the UI
        papers = []
    if papers:
        lines.append(f"\n{len(papers)} paper(s) already in the library:")
        lines += [
            f"- `arXiv:{pid}` {title or '(untitled)'} — {n} passages"
            for pid, title, n in papers
        ]
        lines.append("\nAsk about one of these and it answers without touching the API.")
    else:
        lines.append(
            "\nThe library is empty. Ask about any arXiv paper — it will be read "
            "once and remembered."
        )
    return "\n".join(lines)


@cl.on_chat_start
async def start() -> None:
    try:
        get_settings()
    except ConfigError as exc:
        await cl.Message(content=f"**Configuration error**\n\n```\n{exc}\n```").send()
        return

    cl.user_session.set("app", build_app(checkpointer=InMemorySaver()))
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    cl.user_session.set("usage", Usage())
    await cl.Message(content=_greeting()).send()


def link_citations(text: str) -> tuple[str, dict[str, str]]:
    """Resolve `[arXiv:...]` citations, returning the answer and their passages.

    Pure on purpose -- it returns `(rewritten answer, {label: passage markdown})`
    rather than Chainlit elements, because constructing a `cl.Text` needs a live
    session and that would make this logic testable only through a browser.

    The two cases are rewritten differently, and the reason is not obvious:

    - **Resolved** citations are reduced to the *bare label*, with no markdown
      link at all. Chainlit's `prepareContent` scans the finished content for
      element names and wraps every occurrence in a link itself. Writing our own
      link means it rewrites the text *inside* ours, and the page ends up
      showing a literal `](`.
    - **Unresolved** citations get a real markdown link to the abstract page.
      No element exists for them, so `prepareContent` leaves them alone, and the
      citation still goes somewhere useful instead of being a dead end.
    """
    passages: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        label = match.group("label").strip()
        parsed = citations.parse_label(label)
        if not parsed:
            # Not a citation at all -- some other bracketed phrase. Leave it.
            return match.group(0)
        arxiv_id, _ = parsed
        url = match.group("url") or f"https://arxiv.org/abs/{arxiv_id}"

        hit = citations.resolve(label)
        if hit is None:
            return f"[{label}]({url})"

        if label not in passages:
            body = [f"### {hit.section or 'Passage'}", ""]
            if hit.title:
                body += [f"*{hit.title}*", ""]
            body += [
                hit.text,
                "",
                "---",
                f"`arXiv:{hit.arxiv_id}` · relevance {hit.score:.3f} · [{url}]({url})",
            ]
            passages[label] = "\n".join(body)
        # Bare: Chainlit turns the element name into the reference link.
        return label

    return CITATION.sub(replace, text), passages


def _elements(passages: dict[str, str]) -> list[cl.Text]:
    """Wrap resolved passages as side-panel elements.

    The element name is the model's own citation wording, because Chainlit's
    anchor renderer matches link text against element names exactly.
    """
    return [
        cl.Text(name=label, content=content, display="side")
        for label, content in passages.items()
    ]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    app = cl.user_session.get("app")
    if app is None:
        await cl.Message(content="Not configured — check OPENROUTER_API_KEY.").send()
        return

    usage: Usage = cl.user_session.get("usage")
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    state = StreamState()
    usage.start_turn()

    answer = cl.Message(content="")
    # Keyed by call id, not name: one turn can call the same tool twice -- a
    # library miss followed by a full-text fetch -- and keying on name would
    # close the first step with the second call's output.
    steps: dict[str, cl.Step] = {}
    thinking: cl.Step | None = None
    body: list[str] = []

    async def close(step: cl.Step, output: str | None = None) -> None:
        if output is not None:
            step.output = output
        await step.update()

    try:
        async for mode, payload in app.astream(
            {"messages": [{"role": "user", "content": message.content}]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            for event in interpret(mode, payload, state, usage):
                if isinstance(event, Reasoning):
                    if thinking is None:
                        thinking = cl.Step(name="thinking", type="llm")
                        await thinking.send()
                    await thinking.stream_token(event.text)

                elif isinstance(event, Token):
                    if thinking is not None:
                        await close(thinking)
                        thinking = None
                    body.append(event.text)
                    await answer.stream_token(event.text)

                elif isinstance(event, ToolCall):
                    step = cl.Step(
                        name=event.name,
                        type="tool",
                        show_input="json",
                        icon=ICONS.get(event.name),
                    )
                    step.input = event.args
                    await step.send()
                    steps[event.id] = step

                elif isinstance(event, ToolResult):
                    step = steps.pop(event.id, None)
                    if step is not None:
                        await close(step, event.content)
    finally:
        # However this ended, leave no step spinning forever.
        if thinking is not None:
            await close(thinking)
        for step in steps.values():
            await close(step, "(no result)")

    linked, passages = link_citations("".join(body))
    answer.content = f"{linked}\n\n---\n*{usage.finish_turn()}*"
    answer.elements = _elements(passages)
    await answer.update()


@cl.on_chat_end
async def stop() -> None:
    # A paper may still be indexing in the background; dropping it costs a
    # repeat fetch next time. Same reasoning as the CLI's exit handler.
    if ingest.pending():
        ingest.wait(timeout=30.0)
