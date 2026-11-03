"""Plain-Python adapter: an async generator function is the agent.

This is the reference implementation of the adapter contract and the
baseline every framework adapter is measured against.
"""

from __future__ import annotations

from typing import AsyncIterator, Awaitable, Callable

from ..core.adapter import Adapter, RunContext
from ..core.types import AgentInfo, Capabilities, Event

AgentFn = Callable[[RunContext], AsyncIterator[Event]]


class FunctionAdapter(Adapter):
    framework = "python"

    def __init__(
        self,
        fn: AgentFn,
        info: AgentInfo,
        capabilities: Capabilities = Capabilities(),
        *,
        on_cancel: Callable[[RunContext], Awaitable[None]] | None = None,
    ) -> None:
        self._fn = fn
        self._info = info
        self._caps = capabilities
        self._on_cancel = on_cancel

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        return self._fn(ctx)

    async def cancel(self, ctx: RunContext) -> None:
        if self._on_cancel is not None:
            await self._on_cancel(ctx)
