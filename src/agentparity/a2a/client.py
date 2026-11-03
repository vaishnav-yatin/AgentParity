"""Consume any remote A2A agent as an :class:`Adapter`.

Because a remote agent *is* an adapter, everything that works on local
adapters works on remote ones too: the runtime's lifecycle guarantees,
governance-free delegation from any framework, and the conformance suite
(point it at a URL to check someone else's server).
"""

from __future__ import annotations

import uuid
from typing import AsyncIterator

import httpx
from a2a.client import ClientConfig, create_client
from a2a.types import a2a_pb2 as pb

from ..core.adapter import Adapter, RunContext
from ..core.types import (
    AgentInfo,
    ArtifactChunk,
    Capabilities,
    Event,
    Skill,
    StatusUpdate,
    TaskState,
)
from .convert import message_from_pb, part_from_pb, part_to_pb, state_from_pb


class RemoteA2AAdapter(Adapter):
    framework = "a2a-remote"

    def __init__(self, card: pb.AgentCard, *, httpx_client: httpx.AsyncClient | None = None) -> None:
        self.card = card
        self._httpx = httpx_client
        self._client = None
        self._remote_task: dict[str, str] = {}  # local task_id -> remote task id

    @classmethod
    async def connect(cls, url: str, *, httpx_client: httpx.AsyncClient | None = None) -> "RemoteA2AAdapter":
        """Resolve the agent card at ``url`` (base URL or card URL)."""
        from a2a.client import A2ACardResolver

        owned = httpx_client is None
        http = httpx_client or httpx.AsyncClient()
        try:
            card = await A2ACardResolver(http, url).get_agent_card()
        finally:
            if owned:
                await http.aclose()
        return cls(card, httpx_client=httpx_client)

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.card.name,
            description=self.card.description,
            version=self.card.version,
            skills=tuple(Skill(s.id, s.name, s.description, tuple(s.tags)) for s in self.card.skills),
        )

    @property
    def capabilities(self) -> Capabilities:
        # Only what the card can tell us. A2A cards do not advertise
        # multi-turn or input-required; the conformance suite can establish them.
        return Capabilities(streaming=self.card.capabilities.streaming, cancellation=True)

    async def _get_client(self):
        if self._client is None:
            config = ClientConfig(streaming=self.card.capabilities.streaming, httpx_client=self._httpx)
            self._client = await create_client(self.card, client_config=config)
        return self._client

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        client = await self._get_client()
        req = ctx.request
        msg = pb.Message(
            message_id=uuid.uuid4().hex,
            role=pb.ROLE_USER,
            context_id=req.context_id,
            parts=[part_to_pb(p) for p in req.message.parts],
        )
        if req.resuming and req.task_id in self._remote_task:
            msg.task_id = self._remote_task[req.task_id]

        async for resp in client.send_message(pb.SendMessageRequest(message=msg)):
            kind = resp.WhichOneof("payload")
            if kind == "task":
                self._remote_task[req.task_id] = resp.task.id
                for art in resp.task.artifacts:
                    yield ArtifactChunk(art.artifact_id, tuple(part_from_pb(p) for p in art.parts), art.name or None)
                st = resp.task.status
                if st.state not in (pb.TASK_STATE_UNSPECIFIED, pb.TASK_STATE_SUBMITTED, pb.TASK_STATE_WORKING):
                    yield _status(st)
            elif kind == "artifact_update":
                u = resp.artifact_update
                self._remote_task.setdefault(req.task_id, u.task_id)
                yield ArtifactChunk(
                    u.artifact.artifact_id,
                    tuple(part_from_pb(p) for p in u.artifact.parts),
                    u.artifact.name or None,
                    append=u.append,
                    last_chunk=u.last_chunk,
                )
            elif kind == "status_update":
                u = resp.status_update
                self._remote_task.setdefault(req.task_id, u.task_id)
                if u.status.state in (pb.TASK_STATE_SUBMITTED, pb.TASK_STATE_WORKING):
                    continue
                yield _status(u.status)
            elif kind == "message":
                m = message_from_pb(resp.message)
                yield ArtifactChunk(uuid.uuid4().hex, m.parts, last_chunk=True)
                yield StatusUpdate(TaskState.COMPLETED)

    async def cancel(self, ctx: RunContext) -> None:
        remote = self._remote_task.get(ctx.request.task_id)
        if remote is None:
            return
        client = await self._get_client()
        await client.cancel_task(pb.CancelTaskRequest(id=remote))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def _status(st: pb.TaskStatus) -> StatusUpdate:
    message = message_from_pb(st.message) if st.HasField("message") else None
    return StatusUpdate(state_from_pb(st.state), message)
