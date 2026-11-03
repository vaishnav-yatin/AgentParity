"""The adapter contract every framework integration implements."""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .governance import AllowAll, ToolPolicy, Verdict
from .types import AgentInfo, Capabilities, Event, RunRequest


@dataclass
class AuthorizationRecord:
    name: str
    arguments: dict[str, Any]
    verdict: Verdict
    at: float = field(default_factory=time.monotonic)


@dataclass
class RunContext:
    """Per-run services handed to the adapter.

    ``authorize_tool`` must be called from the framework's own pre-tool hook,
    *before* the tool body runs. ``cancelled`` is set when the caller cancels;
    adapters whose framework cannot be interrupted mid-step should poll it.
    """

    request: RunRequest
    policy: ToolPolicy = field(default_factory=AllowAll)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    authorizations: list[AuthorizationRecord] = field(default_factory=list)

    def authorize_tool(self, name: str, arguments: dict[str, Any]) -> Verdict:
        verdict = self.policy.check(name, dict(arguments))
        self.authorizations.append(AuthorizationRecord(name, dict(arguments), verdict))
        return verdict


class Adapter(abc.ABC):
    """Wraps one agent built with one framework.

    Implementations translate the framework's run/stream API into canonical
    :class:`Event` objects. They do not emit the initial WORKING status or
    handle cancellation/exception bookkeeping; :class:`~agentparity.core.runtime.Runtime`
    does that uniformly, so every adapter gets the same lifecycle for free.

    Contract for :meth:`stream`:

    * yield content as :class:`ArtifactChunk` (and optionally ToolCall/Usage);
    * finish by yielding exactly one :class:`StatusUpdate` whose state is
      COMPLETED, INPUT_REQUIRED, AUTH_REQUIRED, REJECTED or FAILED, *or* by
      returning without one (the runtime then marks the task COMPLETED);
    * raise on error (the runtime marks the task FAILED);
    * yield nothing after a final status.
    """

    framework: str = "unknown"

    @property
    @abc.abstractmethod
    def info(self) -> AgentInfo: ...

    @property
    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def stream(self, ctx: RunContext) -> AsyncIterator[Event]: ...

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None
