import pytest

pytest.importorskip("claude_agent_sdk")

from agentparity.conformance import Status, run_suite  # noqa: E402
from agentparity.conformance.fixtures.claude_agent_sdk import ClaudeAgentSDKFixture  # noqa: E402


async def test_claude_agent_sdk_is_conformant():
    report = await run_suite(ClaudeAgentSDKFixture())
    print(report.render())
    assert report.ok, report.render()
    claimed = set(report.claimed)
    for r in report.results:
        if r.capability is None or r.capability in claimed:
            assert r.status is Status.PASS, report.render()
