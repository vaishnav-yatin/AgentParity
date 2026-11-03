"""Over the real A2A JSON-RPC routes: in-process ASGI for simple round trips,
a real localhost socket wherever timing matters (streaming, cancellation).

The strongest test of the bridge: serve the reference over A2A, consume it
back through RemoteA2AAdapter, and run the conformance suite against the
remote. Anything the wire drops shows up as a failed check.
"""

import contextlib

import httpx
import pytest

pytest.importorskip("a2a")

from agentparity import Capabilities, Message, RunRequest, Runtime, TaskState  # noqa: E402
from agentparity.a2a import RemoteA2AAdapter, build_app  # noqa: E402
from agentparity.a2a.testing import running_server  # noqa: E402
from agentparity.conformance import Behaviour, Status, run_suite  # noqa: E402
from agentparity.conformance.reference import ReferenceFixture  # noqa: E402
from agentparity.conformance.suite import text_of  # noqa: E402

BASE = "http://agent.test"


async def remote_for(adapter):
    app = build_app(adapter, url=BASE + "/")
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)
    return await RemoteA2AAdapter.connect(BASE, httpx_client=http)


async def test_card_advertises_only_claims():
    from agentparity.conformance import Probe

    fx = ReferenceFixture(Capabilities(streaming=False))
    remote = await remote_for(fx.build(Behaviour.ECHO, Probe()))
    assert remote.card.capabilities.streaming is False
    fx = ReferenceFixture(Capabilities(streaming=True))
    remote = await remote_for(fx.build(Behaviour.ECHO, Probe()))
    assert remote.card.capabilities.streaming is True


async def test_echo_round_trip_over_wire():
    from agentparity.conformance import Probe

    remote = await remote_for(ReferenceFixture().build(Behaviour.ECHO, Probe()))
    events, h = await Runtime(remote).collect(RunRequest(Message.user("hello wire")))
    assert events[-1].state is TaskState.COMPLETED, events
    assert text_of(events) == "echo: hello wire"
    assert h.violations == []


class RemoteFixture:
    """Every behaviour of the reference, but reached over A2A on a real socket."""

    framework = "reference via A2A"

    def __init__(self, stack):
        self.inner = ReferenceFixture()
        self.stack = stack

    def build(self, behaviour, probe):
        return _LazyRemote(self.inner.build(behaviour, probe), self.stack)


class _LazyRemote:
    """Starts the server on first use so build() can stay synchronous."""

    framework = "a2a-remote"

    def __init__(self, local, stack):
        self.local = local
        self.stack = stack
        self.remote = None

    @property
    def info(self):
        return self.local.info

    @property
    def capabilities(self):
        return self.local.capabilities

    async def _ensure(self):
        if self.remote is None:
            url = await self.stack.enter_async_context(running_server(self.local))
            self.remote = await RemoteA2AAdapter.connect(url)
        return self.remote

    async def stream(self, ctx):
        r = await self._ensure()
        async for e in r.stream(ctx):
            yield e

    async def cancel(self, ctx):
        r = await self._ensure()
        await r.cancel(ctx)


async def test_reference_conformant_through_a2a():
    async with contextlib.AsyncExitStack() as stack:
        report = await run_suite(RemoteFixture(stack), timeout=10)
    results = {r.check: r for r in report.results}
    # Governance is enforced where the agent runs. A client-side policy cannot
    # reach a remote agent's tools, so the suite must find no authorization
    # record on this side - which is the honest answer.
    for name, r in results.items():
        if name.startswith("tool_governance"):
            assert r.status is Status.FAIL, report.render()
        else:
            assert r.status is Status.PASS, report.render()
