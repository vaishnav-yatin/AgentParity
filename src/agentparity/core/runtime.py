"""Runtime: turns any adapter's output into a well-formed task lifecycle.

Whatever an adapter does, a consumer of :meth:`RunHandle.events` sees:

1. ``StatusUpdate(WORKING)`` first;
2. content events (artifact chunks, tool calls, usage);
3. exactly one final status (terminal or interrupted), then nothing.

Adapter mistakes are *repaired* so the stream stays valid, and *recorded* in
``RunHandle.violations`` so the conformance suite can report them. A repair
never hides a violation.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from .adapter import Adapter, RunContext
from .governance import AllowAll, ToolPolicy
from .types import (
    ArtifactChunk,
    Event,
    Message,
    RunRequest,
    StatusUpdate,
    TaskState,
)

_DONE = object()


@dataclass
class Violation:
    rule: str
    detail: str


@dataclass
class RunHandle:
    request: RunRequest
    ctx: RunContext
    violations: list[Violation] = field(default_factory=list)
    final_state: TaskState | None = None
    _events: AsyncIterator[Event] | None = None
    _cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)

    def events(self) -> AsyncIterator[Event]:
        assert self._events is not None
        return self._events

    def cancel(self) -> None:
        self._cancel_requested.set()
        self.ctx.cancelled.set()


class Runtime:
    def __init__(
        self,
        adapter: Adapter,
        *,
        policy_factory: Callable[[], ToolPolicy] = AllowAll,
        cancel_grace: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.policy_factory = policy_factory
        self.cancel_grace = cancel_grace

    def start(self, request: RunRequest) -> RunHandle:
        ctx = RunContext(request=request, policy=self.policy_factory())
        handle = RunHandle(request=request, ctx=ctx)
        handle._events = self._run(handle)
        return handle

    async def collect(self, request: RunRequest) -> tuple[list[Event], RunHandle]:
        handle = self.start(request)
        events = [e async for e in handle.events()]
        return events, handle

    # ------------------------------------------------------------------ internals

    async def _run(self, h: RunHandle) -> AsyncIterator[Event]:
        queue: asyncio.Queue = asyncio.Queue()

        async def produce() -> None:
            try:
                async for ev in self.adapter.stream(h.ctx):
                    await queue.put(ev)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - adapter errors become FAILED
                await queue.put(exc)
            finally:
                await queue.put(_DONE)

        yield StatusUpdate(TaskState.WORKING)
        producer = asyncio.create_task(produce())
        open_artifacts: set[str] = set()
        closed_artifacts: set[str] = set()
        final: StatusUpdate | None = None

        try:
            while True:
                get = asyncio.create_task(queue.get())
                cancel_wait = asyncio.create_task(h._cancel_requested.wait())
                done, _ = await asyncio.wait(
                    {get, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if get not in done:
                    get.cancel()
                cancel_wait.cancel()

                if cancel_wait in done and get not in done:
                    final = StatusUpdate(TaskState.CANCELED)
                    await self._stop(producer, h)
                    break

                item = get.result()
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    final = StatusUpdate(
                        TaskState.FAILED,
                        Message.agent(f"{type(item).__name__}: {item}"),
                    )
                    break

                ev: Event = item
                if final is not None:
                    h.violations.append(
                        Violation("no-events-after-final", f"{type(ev).__name__} after {final.state.value}")
                    )
                    continue

                if isinstance(ev, StatusUpdate):
                    if ev.state is TaskState.SUBMITTED:
                        h.violations.append(Violation("no-submitted", "adapter emitted SUBMITTED"))
                        continue
                    if ev.state is TaskState.CANCELED and not h._cancel_requested.is_set():
                        h.violations.append(
                            Violation("canceled-without-request", "adapter reported CANCELED unprompted")
                        )
                    if ev.state.final_for_run:
                        final = ev
                        continue  # hold; emitted once the producer finishes
                    yield ev
                    continue

                if isinstance(ev, ArtifactChunk):
                    aid = ev.artifact_id
                    if aid in closed_artifacts:
                        h.violations.append(
                            Violation("artifact-after-last-chunk", f"artifact {aid} was already closed")
                        )
                        continue
                    if ev.append and aid not in open_artifacts:
                        h.violations.append(
                            Violation("append-to-unknown-artifact", f"artifact {aid} was never started")
                        )
                        ev = ArtifactChunk(aid, ev.parts, ev.name, False, ev.last_chunk)
                    open_artifacts.add(aid)
                    if ev.last_chunk:
                        closed_artifacts.add(aid)
                yield ev
        finally:
            if not producer.done():
                await self._stop(producer, h)

        if h._cancel_requested.is_set() and (final is None or not final.state.final_for_run):
            final = StatusUpdate(TaskState.CANCELED)
        if final is None:
            final = StatusUpdate(TaskState.COMPLETED)
        h.final_state = final.state
        yield final

    async def _stop(self, producer: asyncio.Task, h: RunHandle) -> None:
        producer.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(producer), timeout=self.cancel_grace)
        except asyncio.TimeoutError:
            h.violations.append(
                Violation("cancel-not-honoured", f"adapter still running {self.cancel_grace}s after cancel")
            )
        except BaseException:  # noqa: BLE001
            pass
        with contextlib.suppress(BaseException):
            await self.adapter_cancel_hook(h)

    async def adapter_cancel_hook(self, h: RunHandle) -> None:
        cancel = getattr(self.adapter, "cancel", None)
        if cancel is not None:
            await cancel(h.ctx)
