"""Reading the agent graph's event stream, once, for every front end.

LangGraph emits two interleaved channels and neither is quite what a UI wants.
The rules for turning them into displayable events are small but easy to get
wrong, and every one of them was learned the hard way:

- **Tool results arrive on the `messages` channel too.** Rendering everything
  from there prints tool output twice, once raw and once formatted. Only the
  `agent` node's own output belongs to the answer.
- **Tool calls repeat across chunks**, so they need de-duplicating by call id or
  they appear several times per invocation.
- **Usage arrives once per model call**, and a ReAct turn makes several -- the
  model is re-invoked after every tool result.

`interpret` is a pure function over one `(mode, payload)` pair, so the CLI can
drive it from a `for` loop and Chainlit from an `async for` without either
restating any of the above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, ToolMessage

from .llm import reasoning_text


@dataclass(frozen=True)
class Token:
    """A piece of the answer the user sees."""

    text: str


@dataclass(frozen=True)
class Reasoning:
    """A piece of the model's thinking, when reasoning is enabled."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    id: str

    def signature(self) -> str:
        rendered = ", ".join(f"{k}={v!r}" for k, v in (self.args or {}).items())
        return f"{self.name}({rendered})"


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str
    #: The call this answers. A turn can invoke the same tool twice -- a library
    #: miss then a full-text fetch, say -- so a UI keyed on name alone would
    #: attach the second result to the first call's display.
    id: str = ""


Event = Token | Reasoning | ToolCall | ToolResult


@dataclass
class StreamState:
    """Carried across one turn so repeated tool calls are only emitted once."""

    seen_calls: set[str] = field(default_factory=set)
    #: Tool call ids to names, so a result can be attributed to its call.
    call_names: dict[str, str] = field(default_factory=dict)
    #: (id, name) in call order, for results that arrive without a call id.
    pending: list[tuple[str, str]] = field(default_factory=list)


def interpret(mode: str, payload, state: StreamState, usage: "Usage") -> list[Event]:
    """Turn one streamed `(mode, payload)` pair into displayable events."""
    events: list[Event] = []

    if mode == "messages":
        chunk, meta = payload
        # Tool results stream through here as well; they are emitted from the
        # "updates" branch instead, so only the model's own output is taken.
        if (meta or {}).get("langgraph_node") != "agent":
            return events
        thinking = reasoning_text(getattr(chunk, "additional_kwargs", {}) or {})
        if thinking:
            events.append(Reasoning(thinking))
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            events.append(Token(content))
        return events

    if mode != "updates":
        return events

    for node, update in (payload or {}).items():
        for message in (update or {}).get("messages", []):
            if node == "agent" and isinstance(message, AIMessage):
                # One usage record per model call, and a turn makes several.
                usage.record(message.usage_metadata)
                for call in message.tool_calls or []:
                    name = call.get("name", "tool")
                    call_id = call.get("id") or f"{name}:{call.get('args')}"
                    if call_id in state.seen_calls:
                        continue
                    state.seen_calls.add(call_id)
                    state.call_names[call_id] = name
                    state.pending.append((call_id, name))
                    events.append(
                        ToolCall(name=name, args=call.get("args") or {}, id=call_id)
                    )
            elif isinstance(message, ToolMessage):
                call_id = getattr(message, "tool_call_id", "") or ""
                if call_id in state.call_names:
                    name = state.call_names[call_id]
                    state.pending = [p for p in state.pending if p[0] != call_id]
                elif state.pending:
                    # No id to match on: results arrive in call order.
                    call_id, name = state.pending.pop(0)
                else:
                    name = "tool"
                events.append(
                    ToolResult(
                        name=name, content=str(message.content), id=call_id
                    )
                )
    return events


class Usage:
    """Token accounting for one turn, and for the session.

    A ReAct turn is several model calls, not one, and `MessagesState` resends
    the whole conversation on each. So per-turn input tokens grow with the
    conversation, which is exactly the thing worth being able to see.
    """

    def __init__(self) -> None:
        self.turns = 0
        self.session_total = 0
        self._reset()

    def _reset(self) -> None:
        self.calls = 0
        self.input = 0
        self.output = 0
        self.cached = 0
        self.reasoning = 0
        self.embed_start = embedding_tokens()

    def start_turn(self) -> None:
        self._reset()

    def record(self, metadata: dict | None) -> None:
        if not metadata:
            return
        self.calls += 1
        self.input += metadata.get("input_tokens", 0) or 0
        self.output += metadata.get("output_tokens", 0) or 0
        details_in = metadata.get("input_token_details") or {}
        details_out = metadata.get("output_token_details") or {}
        self.cached += details_in.get("cache_read", 0) or 0
        self.reasoning += details_out.get("reasoning", 0) or 0

    @property
    def embedded(self) -> int:
        # Includes background indexing that finished during this turn.
        return max(0, embedding_tokens() - self.embed_start)

    @property
    def turn_total(self) -> int:
        return self.input + self.output + self.embedded

    def parts(self) -> list[str]:
        """The fields both front ends display, in one order."""
        return [
            f"{self.calls} call{'s' if self.calls != 1 else ''}",
            f"in {self.input:,}" + (f" ({self.cached:,} cached)" if self.cached else ""),
            f"out {self.output:,}"
            + (f" ({self.reasoning:,} thinking)" if self.reasoning else ""),
            *([f"embed {self.embedded:,}"] if self.embedded else []),
            f"turn {self.turn_total:,}",
            f"session {self.session_total:,}",
        ]

    def finish_turn(self) -> str:
        """Close the turn and return its one-line summary."""
        self.turns += 1
        self.session_total += self.turn_total
        return " · ".join(self.parts())


def embedding_tokens() -> int:
    try:
        from .rag.embeddings import tokens_used

        return tokens_used()
    except Exception:  # noqa: BLE001 - accounting must never break a turn
        return 0
