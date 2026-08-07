"""Interactive REPL for the Reserchia agent."""

from __future__ import annotations

import os
import sys
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .agent import build_app
from .config import ConfigError, get_settings
from .llm import reasoning_text
from .rag import ingest
from .rag.store import count as rag_count
from .tools.rag_tools import list_paper_library

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


def _format_call(call: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
    return f"{call.get('name')}({args})"


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

    def tool_call(self, call: dict) -> None:
        self._switch("tool")
        print(cyan(f"  [tool] {_format_call(call)}"))

    def tool_result(self, text: str) -> None:
        self._switch("tool")
        print(dim(_format_result(text)))

    def finish(self) -> None:
        if self.section in self._STREAMED:
            print()
        print()
        self.section = None


def _run_turn(app, question: str, config: dict) -> None:
    print()
    printer = Printer()
    seen_calls: set[str] = set()

    for mode, payload in app.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            chunk, meta = payload
            # Tool results stream through here too; they are rendered from the
            # "updates" branch below, so only the model's own output is taken.
            if (meta or {}).get("langgraph_node") != "agent":
                continue
            reasoning = reasoning_text(getattr(chunk, "additional_kwargs", {}))
            if reasoning:
                printer.thought(reasoning)
            if isinstance(chunk.content, str) and chunk.content:
                printer.token(chunk.content)

        elif mode == "updates":
            for node, update in (payload or {}).items():
                for message in (update or {}).get("messages", []):
                    if node == "agent" and isinstance(message, AIMessage):
                        for call in message.tool_calls or []:
                            key = call.get("id") or _format_call(call)
                            if key not in seen_calls:
                                seen_calls.add(key)
                                printer.tool_call(call)
                    elif isinstance(message, ToolMessage):
                        printer.tool_result(str(message.content))

    printer.finish()


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
                _run_turn(app, question, config)
            except KeyboardInterrupt:
                print(dim("\n(interrupted)\n"))
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                print(red(f"\nerror: {type(exc).__name__}: {exc}\n"), file=sys.stderr)
    finally:
        _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
