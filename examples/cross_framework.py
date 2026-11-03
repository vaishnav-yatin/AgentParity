"""A LangGraph agent delegating to a Google ADK agent over A2A.

Run:  pip install "agentparity[a2a,langgraph,adk,dev]"
      python examples/cross_framework.py

No API key needed: both agents are deterministic so the example runs
anywhere. Swap in real models and nothing about the wiring changes.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agentparity import AgentInfo, DenyList, Message, RunRequest, Runtime
from agentparity.a2a import RemoteA2AAdapter
from agentparity.a2a.testing import running_server
from agentparity.adapters.langgraph import LangGraphAdapter
from agentparity.conformance import Behaviour, Probe
from agentparity.conformance.fixtures.adk import ADKFixture
from agentparity.conformance.suite import text_of


async def main(deny: bool = False) -> None:
    # 1. An ADK agent with a tool, served over A2A. Governance runs where the
    #    tools run, so the policy is given to the *server*.
    adk_agent = ADKFixture().build(Behaviour.TOOL, Probe())
    policy = lambda: DenyList(frozenset({"lookup"} if deny else set()))  # noqa: E731

    async with running_server(adk_agent, policy_factory=policy) as url:
        specialist = await RemoteA2AAdapter.connect(url)

        # 2. A LangGraph agent whose node delegates to the ADK agent. It knows
        #    nothing about ADK - only that a remote A2A agent exists.
        async def delegate(state):
            question = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
            events, _ = await Runtime(specialist).collect(RunRequest(Message.user(question)))
            return {"messages": [AIMessage(f"the specialist says: {text_of(events)}")]}

        g = StateGraph(MessagesState)
        g.add_node("delegate", delegate)
        g.add_edge(START, "delegate")
        g.add_edge("delegate", END)
        coordinator = LangGraphAdapter(g.compile(), AgentInfo("coordinator"))

        # 3. Run the LangGraph agent through the same runtime.
        events, _ = await Runtime(coordinator).collect(RunRequest(Message.user("alpha")))
        print(f"deny={deny}: {text_of(events)!r} -> {events[-1].state.value}")
        await specialist.aclose()


if __name__ == "__main__":
    asyncio.run(main(deny=False))
    asyncio.run(main(deny=True))
