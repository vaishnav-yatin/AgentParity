"""OpenAI Agents SDK adapter (``pip install openai-agents``, import ``agents``).

Mapping:

* ``context_id`` -> an SDK :class:`~agents.memory.Session` (by default an
  in-memory ``SQLiteSession`` per context). The SDK prepends the session's
  history to each run and appends the new items, so multi-turn is native.
* ``Runner.run_streamed`` text deltas (``response.output_text.delta`` raw
  events) -> appended artifact chunks, one artifact per output message, as
  they arrive. The completed ``message_output_created`` item is *not*
  re-emitted for a message that already streamed; it is used only as a
  fallback for models that return text without deltas.
* governance -> a **tool input guardrail** (``FunctionTool.tool_input_guardrails``)
  installed on every function tool reachable from the agent (its tools,
  handoff targets, and agents wrapped with ``Agent.as_tool``). The SDK runs
  input guardrails inside the tool-execution step immediately before the
  body is invoked (``_execute_single_tool_body``), so a denial genuinely
  prevents execution: the guardrail returns ``reject_content`` and the
  model receives the rejection message instead of a tool result, which is
  the SDK's own short-circuit. ``RunHooks.on_tool_start`` was not used
  because it cannot veto a call (only raise and abort the whole run); a
  wrapper around ``on_invoke_tool`` was not used because guardrails are the
  SDK's supported, documented interception point and survive tool cloning.
  Installation is idempotent and mutates the tools in place; outside an
  agentparity run the guardrail allows everything, so the same agent still
  works with a plain ``Runner.run``.
* tool approval (``needs_approval=True`` on a tool; the run ends with
  ``result.interruptions``) -> INPUT_REQUIRED. The ``RunState`` is held per
  task; the resuming turn approves (or, for a reply like "no"/"deny",
  rejects with the reply as the rejection message) every pending item and
  continues the same run. The SDK's approval is a yes/no decision with no
  data channel, so the caller's reply text is exposed to the approved tool
  body through :func:`human_reply` - that is how a tool such as
  ``ask_user(question)`` returns the human's answer to the model.

Limitations (reflected in :attr:`OpenAIAgentsAdapter.capabilities`):

* Only ``FunctionTool`` s can be governed. Hosted tools (web/file search,
  code interpreter, hosted MCP) run on OpenAI's side and local non-function
  tools (computer, shell, apply_patch) have no input-guardrail hook, and a
  handoff whose target agent is not reachable statically cannot be walked;
  an agent containing any of these does not claim ``tool_governance``.
* A pending approval is held in process memory. A new (non-resuming) task in
  the same context drops it.
* One reply resolves *all* pending approvals of the interrupted run.
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

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

_REJECT_WORDS = frozenset({"no", "n", "deny", "denied", "reject", "rejected", "decline", "cancel", "stop"})


@dataclass
class _Turn:
    ctx: RunContext
    reply: str | None = None
    allowed: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    denied: dict[str, tuple[str, str]] = field(default_factory=dict)


_current: contextvars.ContextVar[_Turn | None] = contextvars.ContextVar("agentparity_oai_turn", default=None)


def human_reply() -> str | None:
    """The caller's reply on a resumed INPUT_REQUIRED turn, else ``None``.

    Call it from the body of a ``needs_approval`` tool to receive what the
    human answered (the SDK's approval itself carries no data).
    """
    turn = _current.get()
    return turn.reply if turn is not None else None


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return {"input": raw}
    return val if isinstance(val, dict) else {"input": val}


def _governance_guardrail():
    from agents import ToolGuardrailFunctionOutput, tool_input_guardrail

    @tool_input_guardrail(name="agentparity_governance")
    def agentparity_governance(data):
        turn = _current.get()
        if turn is None:
            return ToolGuardrailFunctionOutput.allow()
        tc = data.context
        args = _parse_args(tc.tool_arguments)
        verdict = turn.ctx.authorize_tool(tc.tool_name, args)
        if verdict.allowed:
            turn.allowed[tc.tool_call_id] = (tc.tool_name, args)
            return ToolGuardrailFunctionOutput.allow()
        turn.denied[tc.tool_call_id] = (tc.tool_name, verdict.reason)
        return ToolGuardrailFunctionOutput.reject_content(f"denied by policy: {verdict.reason}")

    return agentparity_governance


_GUARDRAIL = None


def _install_governance(agent: Any) -> bool:
    """Attach the guardrail to every reachable function tool.

    Returns True when every tool reachable from ``agent`` is governed.
    """
    global _GUARDRAIL
    from agents import Agent, FunctionTool

    if _GUARDRAIL is None:
        _GUARDRAIL = _governance_guardrail()

    seen: set[int] = set()
    governed = True

    def walk(a: Any) -> None:
        nonlocal governed
        if id(a) in seen:
            return
        seen.add(id(a))
        for tool in getattr(a, "tools", None) or []:
            if not isinstance(tool, FunctionTool):
                governed = False
                continue
            guards = list(tool.tool_input_guardrails or [])
            if not any(g is _GUARDRAIL for g in guards):
                tool.tool_input_guardrails = [_GUARDRAIL, *guards]
            inner = getattr(tool, "_agent_instance", None)
            if inner is not None:
                walk(inner)
        for h in getattr(a, "handoffs", None) or []:
            if isinstance(h, Agent):
                walk(h)
                continue
            ref = getattr(h, "_agent_ref", None)
            target = ref() if callable(ref) else None
            if target is None:
                governed = False
            else:
                walk(target)

    walk(agent)
    return governed


def _default_approval(reply: str, item: Any) -> bool:
    return reply.strip().lower().rstrip(".!") not in _REJECT_WORDS


class OpenAIAgentsAdapter(Adapter):
    framework = "openai-agents"

    def __init__(
        self,
        agent: Any,
        info: AgentInfo | None = None,
        *,
        session_factory: Callable[[str], Any] | None = None,
        run_config: Any = None,
        context: Any = None,
        max_turns: int | None = None,
        approval_decider: Callable[[str, Any], bool] = _default_approval,
    ) -> None:
        """
        :param agent: the starting ``agents.Agent``.
        :param session_factory: ``context_id -> Session``; defaults to an
            in-memory ``SQLiteSession``. Called once per context.
        :param context: the SDK run context object handed to tools/hooks.
        :param approval_decider: ``(reply, approval_item) -> approve?`` for
            resumed approvals; the default rejects on "no"/"deny"/... and
            approves anything else.
        """
        self.agent = agent
        self._info = info or AgentInfo(agent.name, getattr(agent, "handoff_description", None) or "")
        self._session_factory = session_factory or self._sqlite_session
        self._sessions: dict[str, Any] = {}
        self._run_config = run_config
        self._context = context
        self._max_turns = max_turns
        self._decide = approval_decider
        self._pending: dict[str, tuple[str, Any]] = {}  # task_id -> (context_id, RunState)
        self._active: dict[str, Any] = {}  # task_id -> RunResultStreaming
        self._governed = _install_governance(agent)

    @staticmethod
    def _sqlite_session(context_id: str) -> Any:
        from agents import SQLiteSession

        return SQLiteSession(context_id)

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            multi_turn=True,
            input_required=True,
            cancellation=True,
            tool_governance=self._governed,
            # Usage is emitted when the SDK reports it, but not claimed: the
            # suite has no usage check yet.
        )

    def _session(self, context_id: str) -> Any:
        s = self._sessions.get(context_id)
        if s is None:
            s = self._sessions[context_id] = self._session_factory(context_id)
        return s

    def _run_input(self, ctx: RunContext, turn: _Turn) -> Any:
        req = ctx.request
        pending = self._pending.pop(req.task_id, None)
        if pending is not None and req.resuming and pending[0] == req.context_id:
            _, state = pending
            reply = req.message.text
            turn.reply = reply
            for item in state.get_interruptions():
                if self._decide(reply, item):
                    state.approve(item)
                else:
                    state.reject(item, rejection_message=reply or None)
            return state
        # A non-resuming turn (or an unknown task) starts fresh; any approval
        # left pending in this context is abandoned.
        for tid, (cid, _) in list(self._pending.items()):
            if cid == req.context_id:
                del self._pending[tid]
        return req.message.text

    async def cancel(self, ctx: RunContext) -> None:
        result = self._active.pop(ctx.request.task_id, None)
        if result is not None and not result.is_complete:
            result.cancel()

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        from agents import Runner
        from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

        req = ctx.request
        turn = _Turn(ctx)
        run_input = self._run_input(ctx, turn)
        kwargs: dict[str, Any] = {"session": self._session(req.context_id)}
        if self._run_config is not None:
            kwargs["run_config"] = self._run_config
        if self._context is not None:
            kwargs["context"] = self._context
        if self._max_turns is not None:
            kwargs["max_turns"] = self._max_turns

        token = _current.set(turn)  # copied into the SDK's run task at creation
        try:
            result = Runner.run_streamed(self.agent, run_input, **kwargs)
        finally:
            _current.reset(token)
        self._active[req.task_id] = result

        streamed: dict[str, str] = {}  # output message item id -> artifact id
        try:
            async for ev in result.stream_events():
                if ctx.cancelled.is_set():
                    break
                if isinstance(ev, RawResponsesStreamEvent):
                    data = ev.data
                    if getattr(data, "type", None) == "response.output_text.delta" and data.delta:
                        key = data.item_id or "message"
                        first = key not in streamed
                        aid = streamed.setdefault(key, new_id())
                        yield ArtifactChunk(aid, (TextPart(data.delta),), append=not first)
                    continue
                if not isinstance(ev, RunItemStreamEvent):
                    continue
                item = ev.item
                if ev.name == "message_output_created":
                    mid = getattr(item.raw_item, "id", None)
                    if mid in streamed:
                        continue  # already delivered delta by delta
                    from agents import ItemHelpers

                    text = ItemHelpers.text_message_output(item)
                    if text:
                        yield ArtifactChunk(new_id(), (TextPart(text),), last_chunk=True)
                elif ev.name == "tool_output":
                    raw = item.raw_item
                    call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
                    if call_id in turn.allowed:
                        name, args = turn.allowed.pop(call_id)
                        yield ToolCall(call_id, name, args)
                        yield ToolResult(call_id, name, ok=True, summary=str(item.output)[:200])
                    elif call_id in turn.denied:
                        name, reason = turn.denied.pop(call_id)
                        yield ToolResult(call_id, name, ok=False, summary=f"denied by policy: {reason}")

            if ctx.cancelled.is_set():
                return

            u = result.context_wrapper.usage
            if u.input_tokens or u.output_tokens:
                yield Usage(u.input_tokens, u.output_tokens)

            interruptions = list(result.interruptions or [])
            if interruptions:
                self._pending[req.task_id] = (req.context_id, result.to_state())
                yield StatusUpdate(TaskState.INPUT_REQUIRED, self._ask_message(interruptions))
                return
        finally:
            self._active.pop(req.task_id, None)
            if not result.is_complete:
                result.cancel()

    @staticmethod
    def _ask_message(items: list[Any]) -> Message:
        first = items[0]
        args = _parse_args(getattr(first, "arguments", None))
        question = args.get("question") or args.get("prompt")
        if question and len(items) == 1:
            return Message(Role.AGENT, (TextPart(str(question)),))
        return Message(
            Role.AGENT,
            tuple(
                DataPart(
                    {
                        "approval_required": getattr(i, "name", None) or getattr(i, "tool_name", None),
                        "call_id": getattr(i, "call_id", None),
                        "arguments": _parse_args(getattr(i, "arguments", None)),
                    }
                )
                for i in items
            ),
        )
