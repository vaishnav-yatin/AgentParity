import pytest

pytest.importorskip("agent_framework")

from agentparity.conformance import Status, run_suite  # noqa: E402
from agentparity.conformance.fixtures.agent_framework import AgentFrameworkFixture  # noqa: E402


async def test_agent_framework_is_conformant():
    report = await run_suite(AgentFrameworkFixture())
    print(report.render())
    assert report.ok, report.render()
    assert all(r.status is Status.PASS for r in report.results)


@pytest.mark.parametrize(
    "reply,deny_by_policy,should_run",
    [("yes", False, True), ("no", False, False), ("yes", True, False)],
)
async def test_function_approval_maps_to_input_required(reply, deny_by_policy, should_run):
    """Not a suite behaviour: approval_mode="always_require" -> INPUT_REQUIRED,
    the resumed yes/no becomes a function_approval_response, and the policy
    still gates the approved call when the framework replays it."""
    from agent_framework import Agent, tool

    from agentparity.adapters.agent_framework import AgentFrameworkAdapter
    from agentparity.conformance.fixtures.agent_framework import _scripted_client_class, _tool_script
    from agentparity.conformance.spec import Probe
    from agentparity.core.governance import AllowAll, DenyList
    from agentparity.core.runtime import Runtime
    from agentparity.core.types import Message, RunRequest, TaskState, new_id

    probe = Probe()

    @tool(name="lookup", approval_mode="always_require")
    def lookup(key: str) -> str:
        return probe.lookup(key)

    client = _scripted_client_class()(_tool_script("lookup", lambda u: {"key": u}, lambda r: "got " + r))
    adapter = AgentFrameworkAdapter(Agent(client, name="approver", tools=[lookup]))
    policy = (lambda: DenyList(frozenset({"lookup"}))) if deny_by_policy else AllowAll
    rt = Runtime(adapter, policy_factory=policy)
    ctx, task = new_id(), new_id()

    e1, h1 = await rt.collect(RunRequest(Message.user("gamma"), context_id=ctx, task_id=task))
    assert h1.final_state is TaskState.INPUT_REQUIRED
    assert not probe.lookups

    e2, h2 = await rt.collect(RunRequest(Message.user(reply), context_id=ctx, task_id=task, resuming=True))
    assert h2.final_state is TaskState.COMPLETED
    assert not h1.violations and not h2.violations
    assert bool(probe.lookups) is should_run
    if reply == "yes":
        assert [r.verdict.allowed for r in h2.ctx.authorizations] == [not deny_by_policy]
