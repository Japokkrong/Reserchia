"""The agent graph: a ReAct loop wired explicitly with LangGraph."""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import get_settings
from .llm import build_llm
from .tools import TOOLS

SYSTEM_PROMPT = """You are Reserchia, a helpful assistant with access to tools.

You have no reliable internal clock and your training data is stale, so you \
cannot know the current date or time on your own. Whenever the user asks about \
the present moment -- the date, day of week, month, year, or time -- call \
get_current_datetime instead of answering from memory. If they name a place, \
pass the matching IANA timezone.

Answer conversationally and concisely. Do not read tool output back verbatim; \
state the answer the user actually asked for."""


def build_app(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the agent graph.

    Built lazily so that importing this module does not require an API key --
    only calling this does.
    """
    llm = build_llm(get_settings()).bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
