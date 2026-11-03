import pytest

pytest.importorskip("crewai")

from agentparity.conformance import Status, run_suite  # noqa: E402
from agentparity.conformance.fixtures.crewai import CrewAIFixture  # noqa: E402


async def test_crewai_is_conformant():
    report = await run_suite(CrewAIFixture())
    print(report.render())
    assert report.ok, report.render()
    claimed = set(report.claimed)
    for r in report.results:
        if r.capability is None or r.capability in claimed:
            assert r.status is Status.PASS, report.render()
