"""Interactive REPL for the Reserchia agent."""

from __future__ import annotations

import os
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from .agent import build_app
from .config import ConfigError, get_settings
from . import visuals
from .rag import ingest
from .rag.store import count as rag_count
from .tools.rag_tools import list_paper_library
from .turn import (
    Reasoning,
    StreamState,
    Token,
    ToolCall,
    ToolResult,
    Usage,
    interpret,
)

#: How long to let a background paper indexing finish on the way out. Ingests
#: are a couple of API calls; killing one mid-write leaves a half-indexed paper.
INGEST_GRACE = 30.0

_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(text: str) -> str:
    return _paint(text, "2")


def cyan(text: str) -> str:
    return _paint(text, "36")


def red(text: str) -> str:
    return _paint(text, "31")


def _format_result(text: str, limit: int = 6) -> str:
    lines = text.splitlines() or [text]
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"... (+{len(lines) - limit} more lines)")
    return "\n".join(f"    {line}" for line in shown)


class Printer:
    """Renders the graph's event stream, tracking which section is open."""

    #: Sections written token-by-token, so their last line needs terminating.
    _STREAMED = ("answer", "thinking")

    def __init__(self) -> None:
        self.section: str | None = None

    def _switch(self, section: str) -> bool:
        """Open `section`, closing the previous one. True if newly opened."""
        if self.section == section:
            return False
        if self.section in self._STREAMED:
            print()  # terminate the half-written streamed line
        if self.section is not None:
            print()  # blank line between sections
        self.section = section
        return True

    def token(self, text: str) -> None:
        self._switch("answer")
        print(text, end="", flush=True)

    def thought(self, text: str) -> None:
        if self._switch("thinking"):
            print(dim("  [thinking]"))
        print(dim(text), end="", flush=True)

    def tool_call(self, call: ToolCall) -> None:
        self._switch("tool")
        print(cyan(f"  [tool] {call.signature()}"))

    def tool_result(self, text: str) -> None:
        self._switch("tool")
        print(dim(_format_result(text)))

    def finish(self) -> None:
        if self.section in self._STREAMED:
            print()
        print()
        self.section = None


def _run_turn(app, question: str, config: dict, usage: Usage) -> None:
    print()
    printer = Printer()
    state = StreamState()
    usage.start_turn()
    visuals.start_turn()

    for mode, payload in app.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        for event in interpret(mode, payload, state, usage):
            if isinstance(event, Reasoning):
                printer.thought(event.text)
            elif isinstance(event, Token):
                printer.token(event.text)
            elif isinstance(event, ToolCall):
                printer.tool_call(event)
            elif isinstance(event, ToolResult):
                printer.tool_result(event.content)

    printer.finish()
    _show_visuals()
    print(dim("  [tokens] " + usage.finish_turn()) + "\n")


def _show_visuals() -> None:
    """A terminal cannot draw, so show the source and where the render landed.

    The PNG was produced anyway -- rendering is how the diagram gets validated --
    so printing its path costs nothing and makes the work usable.
    """
    shown = visuals.take()
    for diagram in shown.diagrams:
        print(cyan(f"  [diagram] {diagram.caption or 'untitled'}"))
        for line in diagram.source.splitlines():
            print(dim(f"    {line}"))
        if diagram.path:
            print(dim(f"    -> {diagram.path}"))
        print()
    for equation in shown.equations:
        label = f" — {equation.caption}" if equation.caption else ""
        print(cyan(f"  [equation ({equation.number})]{label}"))
        print(dim(f"    {equation.latex}"))
        print()


def _shutdown() -> None:
    """Let a background paper indexing finish before the process goes away."""
    if not ingest.pending():
        return
    print(dim("indexing the last paper into the library..."), file=sys.stderr)
    if not ingest.wait(timeout=INGEST_GRACE):
        print(
            dim(f"still indexing after {INGEST_GRACE:.0f}s; it will be re-read later."),
            file=sys.stderr,
        )


def main() -> int:
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(red(f"error: {exc}"), file=sys.stderr)
        return 1

    app = build_app(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    usage = Usage()

    mode = "reasoning on" if settings.reasoning_enabled else "reasoning off"
    try:
        library = rag_count()
    except Exception:  # noqa: BLE001 - a broken library must not stop the REPL
        library = 0
    shelf = f", {library} passages in library" if library else ""
    print(f"Reserchia — {settings.model} ({mode}{shelf})")
    print(dim("Commands: /library to list papers, /reset to clear memory, /exit.\n"))

    try:
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not question:
                continue
            if question in ("/exit", "/quit"):
                return 0
            if question == "/reset":
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                print(dim("Conversation memory cleared.\n"))
                continue
            if question == "/library":
                print(list_paper_library.invoke({}) + "\n")
                continue

            try:
                _run_turn(app, question, config, usage)
            except KeyboardInterrupt:
                print(dim("\n(interrupted)\n"))
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                print(red(f"\nerror: {type(exc).__name__}: {exc}\n"), file=sys.stderr)
    finally:
        _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
