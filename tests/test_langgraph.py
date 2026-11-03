import pytest

pytest.importorskip("langgraph")

from agentparity import Message, RunRequest, Runtime, TaskState  # noqa: E402
from agentparity.conformance import Behaviour, Probe, Status, run_suite  # noqa: E402
from agentparity.conformance.fixtures.langgraph import LangGraphFixture  # noqa: E402
from agentparity.conformance.suite import text_of  # noqa: E402


async def test_langgraph_is_conformant():
    report = await run_suite(LangGraphFixture())
    print(report.render())
    assert report.ok, report.render()
    assert all(r.status is Status.PASS for r in report.results)


async def test_no_checkpointer_means_no_multi_turn_claim():
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    from agentparity import AgentInfo
    from agentparity.adapters.langgraph import LangGraphAdapter

    g = StateGraph(MessagesState)
    g.add_node("n", lambda s: {"messages": [AIMessage("hi")]})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    a = LangGraphAdapter(g.compile(), AgentInfo("x"))
    assert not a.capabilities.multi_turn and not a.capabilities.input_required
    events, _ = await Runtime(a).collect(RunRequest(Message.user("q")))
    assert events[-1].state is TaskState.COMPLETED and text_of(events) == "hi"


async def test_governed_tool_is_noop_outside_a_run():
    a = LangGraphFixture().build(Behaviour.TOOL, probe := Probe())
    out = await a.graph.ainvoke({"messages": [("user", "k")]}, {"configurable": {"thread_id": "z"}})
    assert out["messages"][-1].content == "value-of-k" and probe.lookups
