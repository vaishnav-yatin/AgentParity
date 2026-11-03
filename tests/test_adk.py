import pytest

pytest.importorskip("google.adk")

from agentparity.conformance import Status, run_suite  # noqa: E402
from agentparity.conformance.fixtures.adk import ADKFixture  # noqa: E402


async def test_adk_is_conformant():
    report = await run_suite(ADKFixture())
    print(report.render())
    assert report.ok, report.render()
    assert all(r.status is Status.PASS for r in report.results)
