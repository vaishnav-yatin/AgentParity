"""Google ADK adapter.

Mapping:

* ``context_id`` -> ADK session id (one session per A2A context)
* partial (SSE) text events -> appended artifact chunks; the aggregated
  final event that ADK sends after the partials is not re-emitted
* a long-running tool call (``event.long_running_tool_ids``) -> INPUT_REQUIRED.
  On resume the caller's plain reply is turned into the FunctionResponse ADK
  expects, so A2A callers never have to know ADK's function-response format
* governance via an ADK *plugin* (``before_tool_callback``), so it applies
  to every agent in the tree without modifying them. A denied call returns
  an error result to the model instead of running the tool - ADK's own
  short-circuit semantics
"""

from __future__ import annotations

import contextvars
from typing import Any, AsyncIterator

from ..core.adapter import Adapter, RunContext
from ..core.types import (
    AgentInfo,
    ArtifactChunk,
    Capabilities,
    DataPart,
    Event,
    Message,
    Role,
    StatusUpdate,
    TaskState,
    TextPart,
    ToolCall,
    ToolResult,
    Usage,
    new_id,
)

_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("agentparity_adk_ctx", default=None)


def _governance_plugin():
    from google.adk.plugins.base_plugin import BasePlugin

    class GovernancePlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="agentparity_governance")

        async def before_tool_callback(self, *, tool, tool_args, tool_context):
            ctx = _current.get()
            if ctx is None:
                return None
            verdict = ctx.authorize_tool(tool.name, dict(tool_args))
            if verdict.allowed:
                return None
            return {"error": f"denied by policy: {verdict.reason}"}

    return GovernancePlugin()


def _has_llm_agent(agent: Any) -> bool:
    from google.adk.agents import LlmAgent

    if isinstance(agent, LlmAgent):
        return True
    return any(_has_llm_agent(s) for s in getattr(agent, "sub_agents", []) or [])


class ADKAdapter(Adapter):
    framework = "google-adk"

    def __init__(
        self,
        agent: Any,
        info: AgentInfo | None = None,
        *,
        app_name: str = "agentparity",
        user_id: str = "a2a",
        session_service: Any = None,
        streaming: bool = True,
        plugins: list[Any] | None = None,
    ) -> None:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        self.agent = agent
        self._info = info or AgentInfo(agent.name, getattr(agent, "description", "") or "")
        self._app = app_name
        self._user = user_id
        self._sessions = session_service or InMemorySessionService()
        self._streaming = streaming
        self._pending: dict[str, tuple[str, str]] = {}  # context_id -> (call_id, tool name)
        self.runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=self._sessions,
            plugins=[_governance_plugin(), *(plugins or [])],
        )

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        has_llm = _has_llm_agent(self.agent)
        return Capabilities(
            streaming=self._streaming,
            multi_turn=True,
            input_required=True,
            cancellation=True,
            tool_governance=has_llm,
            # usage events are emitted when ADK reports them, but not claimed:
            # the suite has no usage check yet.
        )

    async def _session(self, session_id: str) -> None:
        s = await self._sessions.get_session(app_name=self._app, user_id=self._user, session_id=session_id)
        if s is None:
            await self._sessions.create_session(app_name=self._app, user_id=self._user, session_id=session_id)

    def _new_message(self, ctx: RunContext):
        from google.genai import types

        req = ctx.request
        pending = self._pending.pop(req.context_id, None) if req.resuming else None
        if pending is not None:
            call_id, name = pending
            return types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id, name=name, response={"result": req.message.text}
                        )
                    )
                ],
            )
        return types.Content(role="user", parts=[types.Part(text=req.message.text)])

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        from google.adk.agents.run_config import RunConfig, StreamingMode

        req = ctx.request
        await self._session(req.context_id)
        config = RunConfig(streaming_mode=StreamingMode.SSE if self._streaming else StreamingMode.NONE)
        token = _current.set(ctx)
        open_aid: str | None = None
        try:
            async for ev in self.runner.run_async(
                user_id=self._user,
                session_id=req.context_id,
                new_message=self._new_message(ctx),
                run_config=config,
            ):
                if ev.error_code or ev.error_message:
                    raise RuntimeError(f"ADK error {ev.error_code}: {ev.error_message}")
                if ev.usage_metadata is not None and not ev.partial:
                    u = ev.usage_metadata
                    yield Usage(u.prompt_token_count or 0, u.candidates_token_count or 0)
                parts = (ev.content.parts if ev.content and ev.content.parts else []) or []
                text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))

                if ev.partial:
                    if text:
                        first = open_aid is None
                        if first:
                            open_aid = new_id()
                        yield ArtifactChunk(open_aid, (TextPart(text),), append=not first)
                    continue

                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc is not None:
                        args = dict(fc.args or {})
                        if ev.long_running_tool_ids and fc.id in ev.long_running_tool_ids:
                            self._pending[req.context_id] = (fc.id, fc.name)
                            question = args.get("question") or args.get("prompt")
                            msg = (
                                Message(Role.AGENT, (TextPart(str(question)),))
                                if question
                                else Message(Role.AGENT, (DataPart({"name": fc.name, "args": args}),))
                            )
                            yield StatusUpdate(TaskState.INPUT_REQUIRED, msg)
                            return
                        yield ToolCall(fc.id or new_id(), fc.name, args)
                    fr = getattr(p, "function_response", None)
                    if fr is not None:
                        resp = fr.response or {}
                        yield ToolResult(fr.id or "", fr.name, ok="error" not in resp)

                if text:
                    if open_aid is not None:
                        # aggregate of the partials already streamed
                        open_aid = None
                        continue
                    yield ArtifactChunk(new_id(), (TextPart(text),), last_chunk=True)
        finally:
            _current.reset(token)

