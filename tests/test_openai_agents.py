import pytest

pytest.importorskip("agents")

from agentparity.conformance import Status, run_suite  # noqa: E402
from agentparity.conformance.fixtures.openai_agents import OpenAIAgentsFixture  # noqa: E402


async def test_openai_agents_is_conformant():
    report = await run_suite(OpenAIAgentsFixture())
    print(report.render())
    assert report.ok, report.render()
    assert all(r.status is Status.PASS for r in report.results)
