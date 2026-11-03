"""Reference fixture: plain-Python agents that satisfy every behaviour.

It exists so the suite itself is tested: the reference must pass every
check, and the deliberately broken variants in ``tests/`` must fail the
check aimed at them.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..adapters.function import FunctionAdapter
from ..core.adapter import Adapter, RunContext
from ..core.governance import ToolDenied
from ..core.types import (
    AgentInfo,
    ArtifactChunk,
    Capabilities,
    Event,
    Message,
    StatusUpdate,
    TaskState,
    TextPart,
    ToolCall,
    ToolResult,
    new_id,
)
from .spec import Behaviour, Probe

FULL = Capabilities(
    streaming=True, multi_turn=True, input_required=True, cancellation=True, tool_governance=True
)


def say(text: str) -> ArtifactChunk:
    return ArtifactChunk(new_id(), (TextPart(text),), last_chunk=True)


class ReferenceFixture:
    framework = "python (reference)"

    def __init__(self, capabilities: Capabilities = FULL) -> None:
        self.capabilities = capabilities
        self._history: dict[str, list[str]] = {}

    def build(self, behaviour: Behaviour, probe: Probe) -> Adapter | None:
        fn = getattr(self, f"_{behaviour.value}")
        return FunctionAdapter(lambda ctx: fn(ctx, probe), AgentInfo("reference"), self.capabilities)

    async def _echo(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        yield say("echo: " + ctx.request.message.text)

    async def _stream(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        aid = new_id()
        words = ["one", "two", "three"]
        for i, w in enumerate(words):
            yield ArtifactChunk(aid, (TextPart(w if i == 0 else " " + w),), append=i > 0, last_chunk=i == 2)
            await asyncio.sleep(0)

    async def _remember(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        hist = self._history.setdefault(ctx.request.context_id, [])
        yield say(hist[-1] if hist else "<none>")
        hist.append(ctx.request.message.text)

    async def _ask(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        if not ctx.request.resuming:
            yield StatusUpdate(TaskState.INPUT_REQUIRED, Message.agent("Which colour?"))
            return
        yield say("colour: " + ctx.request.message.text)

    async def _slow(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        yield say("started")
        for _ in range(600):
            probe.tick()
            await asyncio.sleep(0.05)

    async def _tool(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        args = {"key": ctx.request.message.text}
        verdict = ctx.authorize_tool("lookup", args)
        call_id = new_id()
        if not verdict.allowed:
            yield ToolResult(call_id, "lookup", ok=False, summary=verdict.reason)
            raise ToolDenied("lookup", verdict.reason)
        yield ToolCall(call_id, "lookup", args)
        result = probe.lookup(**args)
        yield ToolResult(call_id, "lookup", ok=True)
        yield say(result)

    async def _fail(self, ctx: RunContext, probe: Probe) -> AsyncIterator[Event]:
        raise RuntimeError("deliberate failure")
        yield  # pragma: no cover
