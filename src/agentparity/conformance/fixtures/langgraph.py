"""Deterministic (model-free) LangGraph agents for every behaviour.

The TOOL agent goes through LangGraph's real ``ToolNode``: a scripted node
emits an AIMessage with a tool call exactly as a model would, so the
governance path under test is the framework's own.
"""

from __future__ import annotations

import asyncio

from ...adapters.langgraph import LangGraphAdapter, governed
from ...core.types import AgentInfo
from ..spec import Behaviour, Probe


class LangGraphFixture:
    framework = "langgraph"

    def build(self, behaviour: Behaviour, probe: Probe):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.config import get_stream_writer
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode
        from langgraph.types import interrupt

        def last_human(state) -> str:
            return [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content

        g = StateGraph(MessagesState)
        # Every agent this fixture builds claims governance: the claim belongs to
        # the integration (tools wrapped with @governed), not to one behaviour.
        governs = True

        if behaviour is Behaviour.ECHO:
            g.add_node("n", lambda s: {"messages": [AIMessage("echo: " + last_human(s))]})

        elif behaviour is Behaviour.STREAM:

            async def n(s):
                w = get_stream_writer()
                for piece in ("one", " two", " three"):
                    w(piece)
                    await asyncio.sleep(0)
                return {}

            g.add_node("n", n)

        elif behaviour is Behaviour.REMEMBER:

            def n(s):
                humans = [m.content for m in s["messages"] if isinstance(m, HumanMessage)]
                return {"messages": [AIMessage(humans[-2] if len(humans) > 1 else "<none>")]}

            g.add_node("n", n)

        elif behaviour is Behaviour.ASK:

            def n(s):
                answer = interrupt("Which colour?")
                return {"messages": [AIMessage("colour: " + answer)]}

            g.add_node("n", n)

        elif behaviour is Behaviour.SLOW:

            async def n(s):
                get_stream_writer()("started")
                for _ in range(600):
                    probe.tick()
                    await asyncio.sleep(0.05)
                return {}

            g.add_node("n", n)

        elif behaviour is Behaviour.TOOL:

            @tool
            @governed
            def lookup(key: str) -> str:
                """Look up a value by key."""
                return probe.lookup(key)

            def call(s):
                return {
                    "messages": [
                        AIMessage("", tool_calls=[{"name": "lookup", "args": {"key": last_human(s)}, "id": "c1"}])
                    ]
                }

            def answer(s):
                result = [m for m in s["messages"] if isinstance(m, ToolMessage)][-1]
                return {"messages": [AIMessage(str(result.content))]}

            g.add_node("call", call)
            g.add_node("tools", ToolNode([lookup]))
            g.add_node("answer", answer)
            g.add_edge(START, "call")
            g.add_edge("call", "tools")
            g.add_edge("tools", "answer")
            g.add_edge("answer", END)
            return LangGraphAdapter(
                g.compile(checkpointer=InMemorySaver()), AgentInfo("lg-tool"), governs_tools=True
            )

        elif behaviour is Behaviour.FAIL:

            def n(s):
                raise RuntimeError("deliberate failure")

            g.add_node("n", n)

        g.add_edge(START, "n")
        g.add_edge("n", END)
        return LangGraphAdapter(
            g.compile(checkpointer=InMemorySaver()), AgentInfo(f"lg-{behaviour.value}"), governs_tools=governs
        )
