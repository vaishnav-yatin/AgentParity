"""The runtime guarantees a well-formed lifecycle whatever the adapter does."""

import asyncio

from agentparity import (
    AgentInfo,
    ArtifactChunk,
    Message,
    RunRequest,
    Runtime,
    StatusUpdate,
    TaskState,
    TextPart,
)
from agentparity.adapters.function import FunctionAdapter


def adapter(fn):
    return FunctionAdapter(fn, AgentInfo("t"))


def req(text="hi"):
    return RunRequest(Message.user(text))


async def test_implicit_completion_and_working_first():
    async def fn(ctx):
        yield ArtifactChunk("a", (TextPart("x"),), last_chunk=True)

    events, h = await Runtime(adapter(fn)).collect(req())
    assert events[0] == StatusUpdate(TaskState.WORKING)
    assert events[-1].state is TaskState.COMPLETED
    assert h.violations == [] and h.final_state is TaskState.COMPLETED


async def test_exception_becomes_failed():
    async def fn(ctx):
        raise ValueError("boom")
        yield

    events, h = await Runtime(adapter(fn)).collect(req())
    assert events[-1].state is TaskState.FAILED
    assert "boom" in events[-1].message.text


async def test_events_after_final_are_dropped_and_recorded():
    async def fn(ctx):
        yield StatusUpdate(TaskState.COMPLETED)
        yield ArtifactChunk("a", (TextPart("late"),))

    events, h = await Runtime(adapter(fn)).collect(req())
    assert [type(e) for e in events] == [StatusUpdate, StatusUpdate]
    assert [v.rule for v in h.violations] == ["no-events-after-final"]


async def test_append_to_unknown_artifact_is_repaired_and_recorded():
    async def fn(ctx):
        yield ArtifactChunk("a", (TextPart("x"),), append=True)

    events, h = await Runtime(adapter(fn)).collect(req())
    chunk = events[1]
    assert chunk.append is False
    assert [v.rule for v in h.violations] == ["append-to-unknown-artifact"]


async def test_chunk_after_last_chunk_dropped():
    async def fn(ctx):
        yield ArtifactChunk("a", (TextPart("x"),), last_chunk=True)
        yield ArtifactChunk("a", (TextPart("y"),), append=True)

    events, h = await Runtime(adapter(fn)).collect(req())
    assert sum(isinstance(e, ArtifactChunk) for e in events) == 1
    assert [v.rule for v in h.violations] == ["artifact-after-last-chunk"]


async def test_exactly_one_final_status():
    async def fn(ctx):
        yield StatusUpdate(TaskState.INPUT_REQUIRED)
        yield StatusUpdate(TaskState.COMPLETED)

    events, h = await Runtime(adapter(fn)).collect(req())
    finals = [e for e in events if isinstance(e, StatusUpdate) and e.state.final_for_run]
    assert len(finals) == 1 and finals[0].state is TaskState.INPUT_REQUIRED


async def test_cancel_stops_producer():
    ran_after = []

    async def fn(ctx):
        yield ArtifactChunk("a", (TextPart("go"),), last_chunk=True)
        for _ in range(100):
            await asyncio.sleep(0.02)
            ran_after.append(1)

    h = Runtime(adapter(fn)).start(req())
    events = []
    async for e in h.events():
        events.append(e)
        if isinstance(e, ArtifactChunk):
            h.cancel()
    n = len(ran_after)
    await asyncio.sleep(0.1)
    assert events[-1].state is TaskState.CANCELED
    assert len(ran_after) == n


async def test_uncooperative_adapter_recorded_as_cancel_not_honoured():
    async def fn(ctx):
        yield ArtifactChunk("a", (TextPart("go"),), last_chunk=True)
        for _ in range(3):  # bounded so the test loop can still shut down
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                continue  # swallows cancellation - a broken framework

    h = Runtime(adapter(fn), cancel_grace=0.2).start(req())
    async for e in h.events():
        if isinstance(e, ArtifactChunk):
            h.cancel()
    assert any(v.rule == "cancel-not-honoured" for v in h.violations)
