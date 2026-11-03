"""Microsoft Agent Framework adapter (``agent-framework-core``).

Mapping:

* ``context_id`` -> one ``AgentSession`` per context, created with
  ``agent.create_session(session_id=context_id)`` and kept for the adapter's
  lifetime. History lives in the framework's own history provider on that
  session, so separate contexts never share history.
* ``agent.run(..., stream=True)`` -> each text-bearing ``AgentResponseUpdate``
  becomes an appended :class:`ArtifactChunk`; the stream is then finalized
  with ``get_final_response()`` so the framework persists the turn into the
  session. Function calls/results seen in updates are reported as
  ToolCall/ToolResult; ``usage`` content as Usage (not claimed: the suite has
  no usage check).
* User-input requests (``AgentResponseUpdate.user_input_requests``) ->
  INPUT_REQUIRED. Two framework mechanisms produce them:

  - a call to a *declaration-only* ``FunctionTool`` (``func=None``): the
    framework pauses instead of running it. On the ``resuming=True`` turn the
    caller's plain-text reply becomes the ``function_result`` for that call,
    so A2A callers never need the framework's tool-result format. This is the
    path the conformance suite proves.
  - a ``function_approval_request`` (tool with
    ``approval_mode="always_require"``): the resumed reply is read as a yes/no
    and sent back as ``function_approval_response``. Replies other than a
    clear yes are treated as *deny* (fail closed). The approved call is
    replayed through the same function middleware, so governance still runs.

* Governance: a per-run ``FunctionMiddleware`` passed via
  ``agent.run(middleware=[...])``. The framework's function-invocation loop
  calls it with the tool and its (validated) arguments *before* the tool body;
  when the policy denies, it sets ``context.result`` to an error string and
  returns **without** calling ``call_next()``, so the body never runs and the
  model receives the denial as the tool's result (the framework's documented
  short-circuit). Because it is passed per run, it is installed innermost,
  after any agent/client-level middleware (e.g. argument repair), and sees the
  effective arguments.

  ``tool_governance`` is claimed only when the wrapped agent is an
  ``agent_framework`` ``RawAgent``/``Agent`` whose chat client includes
  ``FunctionInvocationLayer``; a custom ``BaseAgent`` has no function loop
  for the middleware to hook, so the claim would be unprovable. It is also
  withheld when the client's MRO puts ``ChatMiddlewareLayer`` outside
  ``FunctionInvocationLayer``: that order routes run-level function
  middleware into the chat pipeline and the function loop never sees it.

Cancellation: the runtime cancels the asyncio task consuming the response
stream; the framework's loop and the chat client's stream are plain
coroutines, so cancellation propagates into the model call / tool body.
"""

from __future__ import annotations

import inspect
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

_YES = frozenset({"y", "yes", "approve", "approved", "allow", "ok", "okay", "true", "confirm", "go ahead"})


def _governance_middleware(ctx: RunContext, allowed: list[str]):
    from agent_framework import FunctionMiddleware

    class GovernanceMiddleware(FunctionMiddleware):
        """Consults the agentparity policy before the tool body runs."""

        async def process(self, context, call_next) -> None:
            args = context.arguments
            if hasattr(args, "model_dump"):
                args = args.model_dump()
            name = context.function.name
            verdict = ctx.authorize_tool(name, dict(args or {}))
            if not verdict.allowed:
                # Short-circuit: no call_next() -> the function never executes.
                context.result = f"Error: tool {name!r} denied by policy: {verdict.reason}"
                return
            allowed.append(name)
            await call_next()

    return GovernanceMiddleware()


def _governable(agent: Any) -> bool:
    try:
        from agent_framework import FunctionInvocationLayer, RawAgent
    except ImportError:  # pragma: no cover
        return False
    if not isinstance(agent, RawAgent):
        return False
    client = getattr(agent, "client", None)
    if not isinstance(client, FunctionInvocationLayer):
        return False
    # Per-run function middleware reaches the function loop only when
    # FunctionInvocationLayer sits *outside* ChatMiddlewareLayer in the MRO;
    # the other order hands every run-level middleware to the chat pipeline
    # and the function loop never sees it (governance would silently be off).
    from agent_framework import ChatMiddlewareLayer

    mro = type(client).__mro__
    if ChatMiddlewareLayer in mro and mro.index(ChatMiddlewareLayer) < mro.index(FunctionInvocationLayer):
        return False
    return "middleware" in inspect.signature(agent.run).parameters


def _args(call: Any) -> dict[str, Any]:
    if call is None:
        return {}
    try:
        return dict(call.parse_arguments() or {})
    except Exception:  # noqa: BLE001
        return {}


def _function_result_text(content: Any) -> str:
    value = getattr(content, "result", None)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(getattr(c, "text", None) or "" for c in value)
    return "" if value is None else str(value)


class AgentFrameworkAdapter(Adapter):
    framework = "microsoft-agent-framework"

    def __init__(self, agent: Any, info: AgentInfo | None = None, *, streaming: bool = True) -> None:
        """
        :param agent: an ``agent_framework`` agent (``Agent``/``RawAgent``, or
            any ``BaseAgent``-style object with ``run(..., stream=..., session=...)``
            and ``create_session(session_id=...)``).
        :param streaming: use ``run(stream=True)``; when False the whole
            response is emitted as one artifact.
        """
        self.agent = agent
        self._info = info or AgentInfo(
            getattr(agent, "name", None) or type(agent).__name__,
            getattr(agent, "description", "") or "",
        )
        self._streaming = streaming
        self._sessions: dict[str, Any] = {}
        self._pending: dict[str, list[Any]] = {}  # context_id -> user_input_request contents
        self._governable = _governable(agent)

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=self._streaming,
            multi_turn=hasattr(self.agent, "create_session"),
            input_required=hasattr(self.agent, "create_session"),
            cancellation=True,
            tool_governance=self._governable,
        )

    # ------------------------------------------------------------------ helpers

    def _session(self, context_id: str) -> Any:
        s = self._sessions.get(context_id)
        if s is None:
            s = self.agent.create_session(session_id=context_id)
            self._sessions[context_id] = s
        return s

    def _input(self, ctx: RunContext) -> tuple[Any, list[Any]]:
        from agent_framework import Content
        from agent_framework import Message as AFMessage

        req = ctx.request
        text = req.message.text
        # A non-resuming turn abandons whatever was pending in this context.
        pending = self._pending.pop(req.context_id, None)
        if not req.resuming or not pending:
            return text, []
        replies: list[Any] = []
        for p in pending:
            if p.type == "function_approval_request":
                approved = text.strip().lower().rstrip(".!") in _YES
                replies.append(p.to_function_approval_response(approved))
            elif p.type == "function_call":
                replies.append(Content.from_function_result(call_id=p.call_id, result=text))
        if not replies:
            return text, []
        role = "tool" if all(r.type == "function_result" for r in replies) else "user"
        return [AFMessage(role=role, contents=replies)], pending

    @staticmethod
    def _question(requests: list[Any], seen: dict[str, Any]) -> Message:
        r = requests[0]
        call = r.function_call if r.type == "function_approval_request" else r
        args = _args(call)
        if not args and call is not None:
            # When streaming, the pause content for a declaration-only call
            # arrives without its arguments; they were on the original call.
            args = _args(seen.get(call.call_id or ""))
        question = args.get("question") or args.get("prompt")
        if r.type == "function_approval_request":
            return Message(
                Role.AGENT,
                (
                    TextPart(question or f"Approve call to {call.name}? (yes/no)"),
                    DataPart({"type": "approval", "name": call.name, "arguments": args}),
                ),
            )
        if question:
            return Message(Role.AGENT, (TextPart(str(question)),))
        return Message(Role.AGENT, (DataPart({"name": getattr(call, "name", None), "arguments": args}),))

    # ------------------------------------------------------------------ run

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        req = ctx.request
        kwargs: dict[str, Any] = {}
        if hasattr(self.agent, "create_session"):
            kwargs["session"] = self._session(req.context_id)
        allowed: list[str] = []  # tool names the policy let through, in order
        if self._governable:
            kwargs["middleware"] = [_governance_middleware(ctx, allowed)]
        calls: dict[str, Any] = {}  # call_id -> function_call content awaiting its result
        seen: dict[str, Any] = {}  # call_id -> every function_call streamed this run
        messages, resumed = self._input(ctx)
        for p in resumed:  # an approved call's result arrives in this run
            if p.type == "function_approval_request" and p.function_call is not None:
                calls[p.function_call.call_id or ""] = p.function_call

        requests: list[Any] = []
        open_aid: str | None = None

        if self._streaming:
            response_stream = self.agent.run(messages, stream=True, **kwargs)
            async for update in response_stream:
                for c in update.contents:
                    if c.type == "function_call" and c.call_id and c.arguments:
                        seen[c.call_id] = c
                for ev in self._content_events(update.contents, calls, allowed):
                    yield ev
                requests.extend(update.user_input_requests)
                text = "".join(c.text for c in update.contents if c.type == "text" and c.text)
                if text:
                    first = open_aid is None
                    if first:
                        open_aid = new_id()
                    yield ArtifactChunk(open_aid, (TextPart(text),), append=not first)
            # Finalize: runs the framework's after-run providers (history persistence).
            await response_stream.get_final_response()
        else:
            response = await self.agent.run(messages, **kwargs)
            for m in response.messages:
                for ev in self._content_events(m.contents, calls, allowed):
                    yield ev
            requests.extend(response.user_input_requests)
            if response.text:
                open_aid = new_id()
                yield ArtifactChunk(open_aid, (TextPart(response.text),))

        if requests:
            self._pending[req.context_id] = requests
            yield StatusUpdate(TaskState.INPUT_REQUIRED, self._question(requests, seen))
            return
        if open_aid is not None:
            yield ArtifactChunk(open_aid, (), append=True, last_chunk=True)
        yield StatusUpdate(TaskState.COMPLETED)

    def _content_events(self, contents: list[Any], calls: dict[str, Any], allowed: list[str]) -> list[Event]:
        """Tool and usage observability events.

        A ToolCall is emitted only once the call's result arrives *and* the
        governance middleware let it through, so a trace never shows a denied
        call as invoked (the model's function_call is buffered until then).
        """
        out: list[Event] = []
        for c in contents:
            if c.type == "function_call" and not c.user_input_request and c.call_id:
                calls[c.call_id] = c
            elif c.type == "function_result":
                call = calls.pop(c.call_id or "", None)
                name = getattr(call, "name", "") or ""
                ran = not self._governable or name in allowed
                if ran and call is not None:
                    if name in allowed:
                        allowed.remove(name)
                    out.append(ToolCall(c.call_id, name, _args(call)))
                ok = ran and c.exception is None
                out.append(ToolResult(c.call_id or "", name, ok=ok, summary=_function_result_text(c)[:200]))
            elif c.type == "usage" and c.usage_details:
                u = c.usage_details
                out.append(Usage(u.get("input_token_count") or 0, u.get("output_token_count") or 0))
        return out
