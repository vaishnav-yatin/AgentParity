"""Serve any adapter as an A2A v1.0 agent (JSON-RPC binding)."""

from __future__ import annotations

from typing import Callable

from a2a.helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.types import a2a_pb2 as pb

from ..core.adapter import Adapter
from ..core.governance import AllowAll, ToolPolicy
from ..core.runtime import RunHandle, Runtime
from ..core.types import ArtifactChunk, RunRequest, StatusUpdate, TaskState
from .convert import agent_card, message_from_pb, part_to_pb


class AdapterExecutor(AgentExecutor):
    """Bridges :class:`Runtime` events onto an A2A event queue."""

    def __init__(self, adapter: Adapter, *, policy_factory: Callable[[], ToolPolicy] = AllowAll) -> None:
        self.runtime = Runtime(adapter, policy_factory=policy_factory)
        self._live: dict[str, RunHandle] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise ValueError("A2A request carried no message")
        task_id, context_id = context.task_id, context.context_id
        current = context.current_task
        resuming = current is not None and current.status.state in (
            pb.TASK_STATE_INPUT_REQUIRED,
            pb.TASK_STATE_AUTH_REQUIRED,
        )
        if current is None:
            task = new_task_from_user_message(context.message)
            task.id, task.context_id = task_id, context_id
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task_id, context_id)
        request = RunRequest(
            message=message_from_pb(context.message),
            context_id=context_id,
            task_id=task_id,
            resuming=resuming,
        )
        handle = self.runtime.start(request)
        self._live[task_id] = handle
        try:
            async for ev in handle.events():
                if isinstance(ev, ArtifactChunk):
                    await updater.add_artifact(
                        [part_to_pb(p) for p in ev.parts],
                        artifact_id=ev.artifact_id,
                        name=ev.name,
                        append=ev.append,
                        last_chunk=ev.last_chunk,
                    )
                elif isinstance(ev, StatusUpdate):
                    msg = (
                        updater.new_agent_message([part_to_pb(p) for p in ev.message.parts])
                        if ev.message is not None
                        else None
                    )
                    await _apply_status(updater, ev.state, msg)
                # ToolCall / ToolResult / Usage are observability events: they go to
                # tracing hooks, not onto the wire as content.
        finally:
            self._live.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        handle = self._live.get(context.task_id)
        if handle is not None:
            handle.cancel()
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


async def _apply_status(updater: TaskUpdater, state: TaskState, msg) -> None:
    if state is TaskState.WORKING:
        await updater.start_work(msg)
    elif state is TaskState.COMPLETED:
        await updater.complete(msg)
    elif state is TaskState.FAILED:
        await updater.failed(msg)
    elif state is TaskState.INPUT_REQUIRED:
        await updater.requires_input(msg)
    elif state is TaskState.AUTH_REQUIRED:
        await updater.requires_auth(msg)
    elif state is TaskState.REJECTED:
        await updater.reject(msg)
    elif state is TaskState.CANCELED:
        await updater.cancel(msg)


def build_app(
    adapter: Adapter,
    *,
    url: str = "http://localhost:9000/",
    rpc_path: str = "/",
    policy_factory: Callable[[], ToolPolicy] = AllowAll,
    task_store: TaskStore | None = None,
):
    """Return a Starlette ASGI app serving ``adapter`` over A2A."""
    from starlette.applications import Starlette

    card = agent_card(adapter.info, adapter.capabilities, url, framework=adapter.framework)
    handler = DefaultRequestHandler(
        agent_executor=AdapterExecutor(adapter, policy_factory=policy_factory),
        task_store=task_store or InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url=rpc_path)
    return Starlette(routes=routes)


def serve(adapter: Adapter, *, host: str = "127.0.0.1", port: int = 9000, **kw) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(build_app(adapter, url=f"http://{host}:{port}/", **kw), host=host, port=port)
