"""The suite is itself tested: the reference passes, each broken variant fails
the check aimed at it. A check that cannot fail proves nothing."""

import asyncio
import dataclasses

import pytest

from agentparity import ArtifactChunk, Capabilities, TextPart
from agentparity.adapters.function import FunctionAdapter
from agentparity.conformance import Behaviour, Status, run_suite
from agentparity.conformance.reference import FULL, ReferenceFixture, say
from agentparity.core.types import AgentInfo


def by_check(report):
    return {r.check: r for r in report.results}


async def test_reference_is_conformant():
    report = await run_suite(ReferenceFixture())
    assert report.ok, report.render()
    assert all(r.status is Status.PASS for r in report.results)


async def test_unclaimed_capabilities_are_skipped():
    report = await run_suite(ReferenceFixture(Capabilities()))
    r = by_check(report)
    assert r["lifecycle.echo"].status is Status.PASS
    assert r["streaming"].status is Status.SKIP
    assert report.ok


class Broken(ReferenceFixture):
    """Reference with one behaviour replaced."""

    def __init__(self, behaviour, fn, caps=FULL):
        super().__init__(caps)
        self._b, self._fn = behaviour, fn

    def build(self, behaviour, probe):
        if behaviour is self._b:
            if self._fn is None:
                return None
            return FunctionAdapter(lambda ctx: self._fn(ctx, probe), AgentInfo("broken"), self.capabilities)
        return super().build(behaviour, probe)


async def _batched(ctx, probe):
    yield say("one two three")


async def _forgetful(ctx, probe):
    yield say("<none>")


async def _ignores_cancel(ctx, probe):
    yield say("started")

    async def work():
        for _ in range(200):
            probe.tick()
            await asyncio.sleep(0.02)

    asyncio.get_running_loop().create_task(work())  # work escapes the run
    await asyncio.sleep(10)


async def _ungoverned_tool(ctx, probe):
    yield say(probe.lookup(ctx.request.message.text))


async def _authorizes_late(ctx, probe):
    result = probe.lookup(ctx.request.message.text)
    ctx.authorize_tool("lookup", {"key": ctx.request.message.text})
    yield say(result)


async def _ignores_deny(ctx, probe):
    ctx.authorize_tool("lookup", {"key": "alpha"})
    yield say(probe.lookup("alpha"))


async def _never_asks(ctx, probe):
    yield say("colour: blue")


async def _leaky_memory(ctx, probe):
    yield say("apple")


@pytest.mark.parametrize(
    "behaviour, fn, check",
    [
        (Behaviour.STREAM, _batched, "streaming"),
        (Behaviour.STREAM, None, "streaming"),
        (Behaviour.REMEMBER, _forgetful, "multi_turn"),
        (Behaviour.REMEMBER, _leaky_memory, "multi_turn"),
        (Behaviour.SLOW, _ignores_cancel, "cancellation"),
        (Behaviour.TOOL, _ungoverned_tool, "tool_governance.allow"),
        (Behaviour.TOOL, _authorizes_late, "tool_governance.allow"),
        (Behaviour.TOOL, _ignores_deny, "tool_governance.deny"),
        (Behaviour.ASK, _never_asks, "input_required"),
    ],
)
async def test_broken_variant_fails_its_check(behaviour, fn, check):
    report = await run_suite(Broken(behaviour, fn), timeout=5)
    r = by_check(report)[check]
    assert r.status in (Status.FAIL, Status.UNVERIFIED), report.render()
    assert not report.ok
    others = [x for x in report.results if x.check != check and x.check.split(".")[0] != check.split(".")[0]]
    assert all(x.status is Status.PASS for x in others), report.render()


async def test_hang_is_reported_not_waited_forever():
    async def hang(ctx, probe):
        await asyncio.sleep(60)
        yield

    report = await run_suite(Broken(Behaviour.FAIL, hang), timeout=0.5)
    assert by_check(report)["lifecycle.failure"].status is Status.FAIL


async def test_claim_without_a_check_is_unverified():
    report = await run_suite(ReferenceFixture(dataclasses.replace(FULL, usage=True)))
    assert by_check(report)["usage"].status is Status.UNVERIFIED
    assert not report.ok
