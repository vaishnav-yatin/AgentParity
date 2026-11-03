"""Claude Agent SDK (Python) adapter.

The SDK drives the Claude Code CLI as a subprocess and talks to it over the
CLI's stream-json protocol (``claude_agent_sdk._internal.transport``).

Mapping:

* ``context_id`` -> one live :class:`ClaudeSDKClient` (one CLI process) per
  A2A context. Turns of the same context are sent to the same client with
  ``client.query()``, so the CLI keeps the conversation. If that client is
  lost (error, failed cancel, :meth:`close_context`), the next turn reconnects
  with ``ClaudeAgentOptions(resume=<the CLI session id>)``.

  Why a live client rather than "resume by session id every turn": the
  input-required flow below parks the CLI *inside* a permission prompt and
  answers it on the next A2A turn, which only a live process can do. The SDK
  (0.2.x) spawns its reader and control-request tasks detached
  (``_internal/_task_compat.spawn_detached``), so a client connected in one
  run's task can be driven from the next run's task. Resume-per-turn would
  also spawn a CLI process per turn.

* ``include_partial_messages=True`` -> ``StreamEvent`` ``text_delta``\\ s become
  appended artifact chunks as they arrive. The ``AssistantMessage`` that the
  CLI sends after the deltas carries the same text again; it is not
  re-emitted (matched by API message id).

* governance -> a ``PreToolUse`` hook, prepended to the caller's hooks, that
  calls :meth:`RunContext.authorize_tool` and answers
  ``permissionDecision: "deny"`` when the policy denies. The hook, not
  ``can_use_tool``, is the enforcement point because the CLI only consults
  ``can_use_tool`` when its permission rules evaluate to "ask": tools in
  ``allowed_tools``, ``permission_mode="bypassPermissions"`` /
  ``"acceptEdits"`` or settings ``permissions.allow`` never reach it. A
  PreToolUse hook runs for every tool call before permission evaluation, and
  across hooks ``deny`` wins over ``allow``. On allow the hook returns no
  decision, so the caller's own permission flow still applies. Policies see
  in-process SDK MCP tools by their bare name (``lookup``, not
  ``mcp__server__lookup``); every other tool by the CLI's name.

* input-required -> ``can_use_tool`` for the *ask tools* (default
  ``AskUserQuestion``). When the CLI asks permission to run one, the adapter
  ends the A2A run INPUT_REQUIRED with the question and leaves the callback
  pending; the CLI waits at its permission prompt. The resuming turn answers
  it with ``PermissionResultAllow(updated_input={..., "answers": {...}})`` and
  the same CLI turn continues. This needs the ask tool to *reach* the
  permission prompt, so the capability is not claimed when an ask tool is in
  ``allowed_tools`` or ``permission_mode`` is ``bypassPermissions``. Other
  tools that reach the prompt go to the caller's ``can_use_tool`` if one was
  given, else are denied (the CLI's own non-interactive behaviour).

* cancellation -> ``client.interrupt()`` (control request), then the
  interrupted turn's result frame is drained so the next turn starts clean.
  If the interrupt or drain does not finish within ``cancel_timeout`` the
  client is disconnected, which terminates the CLI process.

* errors -> a ``ResultMessage`` with ``is_error`` raises
  :class:`ClaudeAgentRunError`; transport failures (``CLINotFoundError``,
  ``ProcessError``, ``CLIConnectionError``) propagate as raised. A failed
  context's client is discarded.

Usage events are emitted from ``ResultMessage.usage`` but not claimed (the
suite has no usage check yet).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterable

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

_EOF = object()


class ClaudeAgentRunError(RuntimeError):
    """The CLI ended a turn with ``is_error`` set."""

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class _Ask:
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class _Session:
    context_id: str
    client: Any = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pump: asyncio.Task | None = None
    ctx: RunContext | None = None
    pending_answer: asyncio.Future | None = None
    settling: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _default_answer(tool_name: str, tool_input: dict[str, Any], reply: str) -> dict[str, Any]:
    """Turn the A2A caller's reply into the ask tool's ``updated_input``.

    ``AskUserQuestion`` takes ``answers`` keyed by question text; the single
    reply answers every question asked. Other ask tools get ``answer``.
    """
    questions = tool_input.get("questions")
    if isinstance(questions, list):
        answers = {q.get("question", ""): reply for q in questions if isinstance(q, dict)}
        return {**tool_input, "answers": answers}
    return {**tool_input, "answer": reply}


def _question_message(ask: _Ask) -> Message:
    qs = ask.tool_input.get("questions")
    if isinstance(qs, list):
        texts = [q.get("question") for q in qs if isinstance(q, dict) and q.get("question")]
        if texts:
            return Message(Role.AGENT, (TextPart("\n".join(texts)),))
    for key in ("question", "prompt"):
        if isinstance(ask.tool_input.get(key), str):
            return Message(Role.AGENT, (TextPart(ask.tool_input[key]),))
    return Message(Role.AGENT, (DataPart({"name": ask.tool_name, "args": dict(ask.tool_input)}),))


def _result_error_text(msg: Any) -> str:
    if msg.errors:
        return "; ".join(str(e) for e in msg.errors)
    if isinstance(msg.result, str) and msg.result.strip():
        return msg.result.strip()
    if msg.api_error_status is not None:
        return f"API error (HTTP {msg.api_error_status})"
    return msg.subtype or "unknown error"


class ClaudeAgentSDKAdapter(Adapter):
    framework = "claude-agent-sdk"

    def __init__(
        self,
        options: Any = None,
        info: AgentInfo | None = None,
        *,
        streaming: bool = True,
        ask_tools: Iterable[str] = ("AskUserQuestion",),
        answer_builder: Callable[[str, dict[str, Any], str], dict[str, Any]] = _default_answer,
        transport_factory: Callable[[Any], Any] | None = None,
        cancel_timeout: float = 1.5,
    ) -> None:
        """
        :param options: a ``ClaudeAgentOptions``. In-process tools go in
            ``mcp_servers`` (``create_sdk_mcp_server``) as usual.
        :param ask_tools: tool names whose permission prompt is turned into
            INPUT_REQUIRED. Pass ``()`` to disable.
        :param transport_factory: builds a custom SDK ``Transport`` from the
            final options for each client (remote CLI, tests). ``None`` uses
            the SDK's subprocess transport.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        self.options = options or ClaudeAgentOptions()
        self._info = info or AgentInfo("claude-agent", "Claude Agent SDK agent")
        self._streaming = streaming
        self._ask_tools = frozenset(ask_tools)
        self._answer = answer_builder
        self._transport_factory = transport_factory
        self._cancel_timeout = cancel_timeout
        self._sessions: dict[str, _Session] = {}
        self._session_ids: dict[str, str] = {}  # context_id -> CLI session id
        servers = self.options.mcp_servers if isinstance(self.options.mcp_servers, dict) else {}
        self._sdk_servers = frozenset(
            n for n, c in servers.items() if isinstance(c, dict) and c.get("type") == "sdk"
        )

    # ------------------------------------------------------------------ contract

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        allowed = set(self.options.allowed_tools or ())
        ask_reaches_prompt = bool(self._ask_tools - allowed) and self.options.permission_mode != "bypassPermissions"
        return Capabilities(
            streaming=self._streaming,
            multi_turn=True,
            input_required=ask_reaches_prompt,
            cancellation=True,
            tool_governance=True,
        )

    def policy_name(self, cli_name: str) -> str:
        if cli_name.startswith("mcp__"):
            parts = cli_name.split("__", 2)
            if len(parts) == 3 and parts[1] in self._sdk_servers:
                return parts[2]
        return cli_name

    # ------------------------------------------------------------------ sessions

    def _options_for(self, s: _Session) -> Any:
        from claude_agent_sdk import HookMatcher, PermissionResultAllow, PermissionResultDeny

        async def governance_hook(input_data: dict[str, Any], tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
            name = self.policy_name(str(input_data.get("tool_name", "")))
            args = dict(input_data.get("tool_input") or {})
            ctx = s.ctx
            if ctx is None:  # no A2A run is driving this session: fail closed
                reason = "no active agentparity run"
            else:
                verdict = ctx.authorize_tool(name, args)
                if verdict.allowed:
                    s.queue.put_nowait(ToolCall(tool_use_id or new_id(), name, args))
                    return {}
                reason = verdict.reason
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"denied by policy: {reason}",
                }
            }

        user_can_use_tool = self.options.can_use_tool

        async def can_use_tool(tool_name: str, tool_input: dict[str, Any], context: Any):
            if tool_name in self._ask_tools:
                fut = asyncio.get_running_loop().create_future()
                s.pending_answer = fut
                s.queue.put_nowait(_Ask(tool_name, dict(tool_input)))
                try:
                    reply = await fut
                finally:
                    if s.pending_answer is fut:
                        s.pending_answer = None
                if isinstance(reply, PermissionResultDeny):
                    return reply
                return PermissionResultAllow(updated_input=self._answer(tool_name, dict(tool_input), reply))
            if user_can_use_tool is not None:
                return await user_can_use_tool(tool_name, tool_input, context)
            return PermissionResultDeny(message=f"{tool_name} requires approval and no approver is configured")

        hooks = {k: list(v) for k, v in (self.options.hooks or {}).items()}
        hooks["PreToolUse"] = [HookMatcher(matcher=None, hooks=[governance_hook]), *hooks.get("PreToolUse", [])]
        changes: dict[str, Any] = {
            "hooks": hooks,
            "can_use_tool": can_use_tool,
            "include_partial_messages": self._streaming,
        }
        prior = self._session_ids.get(s.context_id)
        if prior and not self.options.resume:
            changes["resume"] = prior
        return dataclasses.replace(self.options, **changes)

    async def _session(self, context_id: str) -> _Session:
        s = self._sessions.get(context_id)
        if s is not None:
            return s
        from claude_agent_sdk import ClaudeSDKClient

        s = _Session(context_id)
        opts = self._options_for(s)
        transport = self._transport_factory(opts) if self._transport_factory else None
        client = ClaudeSDKClient(options=opts, transport=transport)
        await client.connect()  # CLINotFoundError etc. propagate -> FAILED
        s.client = client

        async def pump() -> None:
            try:
                async for m in client.receive_messages():
                    await s.queue.put(m)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - handed to the run
                await s.queue.put(exc)
            await s.queue.put(_EOF)

        s.pump = asyncio.create_task(pump())
        self._sessions[context_id] = s
        return s

    async def close_context(self, context_id: str) -> None:
        """Disconnect a context's client (terminates its CLI process)."""
        s = self._sessions.pop(context_id, None)
        if s is None:
            return
        if s.pending_answer is not None and not s.pending_answer.done():
            s.pending_answer.cancel()
        if s.pump is not None:
            s.pump.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(s.client.disconnect(), timeout=5)

    async def aclose(self) -> None:
        for cid in list(self._sessions):
            await self.close_context(cid)

    async def _drain_turn(self, s: _Session, timeout: float) -> bool:
        """Read to the end of the current CLI turn (its result frame)."""
        from claude_agent_sdk import ResultMessage

        async def until_result() -> bool:
            while True:
                item = await s.queue.get()
                if isinstance(item, ResultMessage):
                    if item.session_id:
                        self._session_ids[s.context_id] = item.session_id
                    return True
                if item is _EOF or isinstance(item, BaseException):
                    return False

        try:
            return await asyncio.wait_for(until_result(), timeout)
        except asyncio.TimeoutError:
            return False

    async def _interrupt(self, s: _Session) -> None:
        ok = False
        with contextlib.suppress(Exception):
            await asyncio.wait_for(s.client.interrupt(), timeout=self._cancel_timeout)
            ok = await self._drain_turn(s, self._cancel_timeout)
        if not ok:
            await self.close_context(s.context_id)

    async def _abandon_question(self, s: _Session) -> None:
        """A new task arrived while the CLI waits on an unanswered question."""
        from claude_agent_sdk import PermissionResultDeny

        fut = s.pending_answer
        if fut is not None and not fut.done():
            fut.set_result(PermissionResultDeny(message="the user started a new task instead of answering", interrupt=True))
        if not await self._drain_turn(s, self._cancel_timeout * 2):
            await self.close_context(s.context_id)

    async def cancel(self, ctx: RunContext) -> None:
        """Called by the runtime after a cancel: wait until the CLI really stopped."""
        s = self._sessions.get(ctx.request.context_id)
        if s is None or s.settling is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(s.settling), timeout=self._cancel_timeout * 2 + 0.5)
        except (asyncio.TimeoutError, Exception):
            await self.close_context(ctx.request.context_id)

    # ------------------------------------------------------------------ run

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        req = ctx.request
        s = self._sessions.get(req.context_id)
        if s is not None and s.settling is not None:
            with contextlib.suppress(BaseException):
                await s.settling
        s = await self._session(req.context_id)
        parked = False
        async with s.lock:
            s.ctx = ctx
            try:
                if req.resuming and s.pending_answer is not None and not s.pending_answer.done():
                    s.pending_answer.set_result(req.message.text)
                else:
                    if s.pending_answer is not None:
                        await self._abandon_question(s)
                        s = await self._session(req.context_id)
                        s.ctx = ctx
                    await s.client.query(req.message.text)

                streamed: set[str] = set()
                artifacts: dict[str, str] = {}  # API message id -> artifact id
                current_mid: str | None = None
                tool_names: dict[str, str] = {}

                while True:
                    item = await s.queue.get()
                    if item is _EOF:
                        raise ConnectionError("Claude Code CLI stream ended before the turn finished")
                    if isinstance(item, BaseException):
                        raise item
                    if isinstance(item, ToolCall):
                        yield item
                        continue
                    if isinstance(item, _Ask):
                        parked = True
                        yield StatusUpdate(TaskState.INPUT_REQUIRED, _question_message(item))
                        return

                    if isinstance(item, StreamEvent):
                        if item.parent_tool_use_id is not None:
                            continue  # sub-agent traffic is not this agent's reply
                        ev = item.event or {}
                        etype = ev.get("type")
                        if etype == "message_start":
                            current_mid = (ev.get("message") or {}).get("id") or new_id()
                        elif etype == "content_block_delta":
                            delta = ev.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                mid = current_mid or "partial"
                                first = mid not in artifacts
                                aid = artifacts.setdefault(mid, new_id())
                                streamed.add(mid)
                                yield ArtifactChunk(aid, (TextPart(delta["text"]),), append=not first)
                        continue

                    if isinstance(item, AssistantMessage):
                        if item.parent_tool_use_id is not None:
                            continue
                        for b in item.content:
                            if isinstance(b, ToolUseBlock):
                                tool_names[b.id] = self.policy_name(b.name)
                        if item.message_id is not None and item.message_id in streamed:
                            continue  # already delivered delta by delta
                        text = "".join(b.text for b in item.content if isinstance(b, TextBlock))
                        if text:
                            yield ArtifactChunk(new_id(), (TextPart(text),), last_chunk=True)
                        continue

                    if isinstance(item, UserMessage):
                        if isinstance(item.content, list):
                            for b in item.content:
                                if isinstance(b, ToolResultBlock):
                                    yield ToolResult(b.tool_use_id, tool_names.get(b.tool_use_id, ""), ok=not b.is_error)
                        continue

                    if isinstance(item, SystemMessage):
                        sid = item.data.get("session_id") if item.subtype == "init" else None
                        if sid:
                            self._session_ids[req.context_id] = sid
                        continue

                    if isinstance(item, ResultMessage):
                        if item.session_id:
                            self._session_ids[req.context_id] = item.session_id
                        if item.usage:
                            yield Usage(int(item.usage.get("input_tokens") or 0), int(item.usage.get("output_tokens") or 0))
                        if item.is_error:
                            raise ClaudeAgentRunError(
                                f"Claude Code returned an error result: {_result_error_text(item)}", item
                            )
                        return
            except asyncio.CancelledError:
                # Runtime cancelled this run: stop the CLI turn, then let the
                # runtime's cancel hook wait for it.
                s.settling = asyncio.create_task(self._interrupt(s))
                raise
            except BaseException:
                await self.close_context(req.context_id)
                raise
            finally:
                if not parked:
                    s.ctx = None
