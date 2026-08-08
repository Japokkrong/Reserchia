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
import dataclasses
from pathlib import Path

import chainlit as cl
from chainlit.types import ThreadDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reserchia.agent import build_app  # noqa: E402
from reserchia.config import ConfigError, get_settings  # noqa: E402
from reserchia import observability, persistence, visuals  # noqa: E402
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


#: A one-click toggle in the composer, beside the upload button. `button` puts
#: it there rather than in the slash-command list, and `persistent` makes it
#: latch instead of applying to a single message -- so it reads as a mode, which
#: is what it is. Its state arrives as `message.command` on every turn.
REASONING_COMMAND = "Reasoning"


def _commands(on: bool) -> list[dict]:
    return [
        {
            "id": REASONING_COMMAND,
            "icon": "brain",
            "description": (
                "Think before answering — slower, more tokens, "
                "and the thinking is shown as its own step"
            ),
            "button": True,
            "persistent": True,
            "selected": on,
        }
    ]


def _agent_for(on: bool):
    """The agent with reasoning on or off, rebuilt only when the mode changes.

    Reasoning is fixed at construction -- it decides the `reasoning` block sent
    to OpenRouter and whether `ChatOpenRouter` replays `reasoning_details` -- so
    flipping it means a new graph. The conversation survives because the history
    lives in the checkpointer, and the same saver and thread id are reused.
    """
    if cl.user_session.get("reasoning_on") == on:
        return cl.user_session.get("app")

    settings = dataclasses.replace(
        get_settings(), reasoning="enabled" if on else "disabled"
    )
    app = build_app(checkpointer=cl.user_session.get("saver"), settings=settings)
    cl.user_session.set("app", app)
    cl.user_session.set("reasoning_on", on)
    return app


if persistence.enabled():

    @cl.data_layer
    def _data_layer():
        """Threads to Postgres, files to RustFS. Only when DATABASE_URL is set."""
        return persistence.data_layer()

    @cl.header_auth_callback
    def _auth(headers) -> cl.User | None:
        """Identify, rather than authenticate.

        Chainlit will not list threads without a user, and this returns the same
        one unconditionally -- no login page. It is only reasonable because
        compose publishes the port on 127.0.0.1; on any other interface this is
        an open door.
        """
        return cl.User(identifier=persistence.LOCAL_USER)


async def _begin(thread_id: str | None = None) -> None:
    """Session state shared by a new chat and a resumed one."""
    settings = get_settings()
    cl.user_session.set("saver", InMemorySaver())
    # Reuse the persisted thread's id so the checkpointer and the stored
    # transcript agree on which conversation this is.
    cl.user_session.set("thread_id", thread_id or str(uuid.uuid4()))
    cl.user_session.set("usage", Usage())
    cl.user_session.set("reasoning_on", None)
    _agent_for(settings.reasoning_enabled)
    await cl.context.emitter.set_commands(_commands(settings.reasoning_enabled))


@cl.on_chat_start
async def start() -> None:
    try:
        settings = get_settings()
    except ConfigError as exc:
        await cl.Message(content=f"**Configuration error**\n\n```\n{exc}\n```").send()
        return

    await _begin()
    await cl.Message(content=_greeting()).send()


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    """Reopen a stored conversation, and give the agent its memory back.

    Chainlit redraws the transcript from Postgres on its own. That is only half
    of it: the graph's checkpointer starts empty, so without this the user would
    be looking at a full conversation while the agent saw an empty one, and the
    first follow-up would answer as if nothing had been said.

    Replaying the stored user and assistant turns into the checkpointer under
    the thread's own id closes that gap. Tool calls and reasoning are not
    replayed -- only the conversation as the user sees it -- which is enough for
    context and avoids resurrecting `reasoning_details` that may no longer match
    the current mode.
    """
    try:
        get_settings()
    except ConfigError as exc:
        await cl.Message(content=f"**Configuration error**\n\n```\n{exc}\n```").send()
        return

    await _begin(thread_id=thread.get("id"))

    history = []
    for step in thread.get("steps") or []:
        kind, text = step.get("type"), (step.get("output") or "").strip()
        if not text:
            continue
        if kind == "user_message":
            history.append(HumanMessage(content=text))
        elif kind == "assistant_message":
            # Strip the token footer; it is display furniture, and feeding it
            # back would have the model treat its own accounting as content.
            history.append(AIMessage(content=_without_footer(text)))

    if history:
        app = cl.user_session.get("app")
        config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
        await app.aupdate_state(config, {"messages": history})


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


#: The usage footer this UI appends after each answer, e.g.
#: "2 calls · in 9,703 (8,192 cached) · out 166 · turn 9,875 · session 22,726".
_FOOTER = re.compile(r"\n*-{3,}\n*\*?\d+ calls? · .*$", re.S)


def _without_footer(text: str) -> str:
    return _FOOTER.sub("", text).strip()


def _diagrams(drawn) -> list[cl.Image]:
    """Rendered diagrams, shown under the answer.

    `display="inline"` rather than "side": a diagram is the point of the answer,
    not a reference to check, so it should be visible without a click.
    """
    return [
        cl.Image(
            name=diagram.caption or f"diagram-{index + 1}",
            content=diagram.png,
            display="inline",
            size="large",
        )
        for index, diagram in enumerate(drawn)
    ]


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
    if cl.user_session.get("saver") is None:
        await cl.Message(content="Not configured — check OPENROUTER_API_KEY.").send()
        return

    # The toggle's state rides in on every message: the command id while it is
    # latched on, absent while off. Reading it here rather than in a callback
    # means the mode can never disagree with the button the user is looking at.
    app = _agent_for(message.command == REASONING_COMMAND)

    usage: Usage = cl.user_session.get("usage")
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    state = StreamState()
    usage.start_turn()
    visuals.start_turn()
    observability.start_turn(message.content)

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
    shown = visuals.take()

    # Equations go into the content as $$...$$ so KaTeX typesets them -- better
    # than an image in every way that matters here: selectable, theme-aware and
    # sharp at any zoom. Diagrams have no such option and become images.
    blocks = [linked]
    for equation in shown.equations:
        label = f" — {equation.caption}" if equation.caption else ""
        blocks.append(f"$$\n{equation.latex}\n$$\n\n*({equation.number}){label}*")

    answer.content = "\n\n".join(blocks)
    answer.elements = _elements(passages) + _diagrams(shown.diagrams)
    await answer.update()

    # The footer goes in its own message rather than at the end of this one.
    # Chainlit always renders elements after a message's content, so a diagram
    # would otherwise appear *below* the token line -- which reads as though the
    # answer had ended before the picture arrived.
    line = usage.finish_turn()
    observability.finish_turn(usage)
    await cl.Message(content=f"*{line}*").send()


@cl.on_chat_end
async def stop() -> None:
    # A paper may still be indexing in the background; dropping it costs a
    # repeat fetch next time. Same reasoning as the CLI's exit handler.
    if ingest.pending():
        ingest.wait(timeout=30.0)
