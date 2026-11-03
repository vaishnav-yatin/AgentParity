"""CrewAI adapter.

Unit: a :class:`crewai.Crew`. A crew is what CrewAI runs, deploys and
streams: an ``Agent`` on its own has no task to perform, and ``Crew`` is the
object that owns ``kickoff``/``akickoff``, ``stream=True``, ``usage_metrics``
and the conversation input (``crew_chat_messages``) that ``crewai chat`` uses.
A single agent is served by wrapping it in a one-task crew.

Mapping:

* the inbound message -> kickoff input ``{message}`` (key configurable); at
  least one task must interpolate it.
* ``context_id`` -> CrewAI's native conversation input ``crew_chat_messages``
  (the JSON list of ``{"role", "content"}`` turns that ``crewai chat`` passes;
  CrewAI appends it to every task description). The adapter keeps that list
  per context. CrewAI's ``memory=True`` is semantic long-term memory backed by
  embeddings, not a per-conversation transcript, so it is not used for this.
* ``Crew(stream=True)`` -> ``akickoff`` returns a ``CrewStreamingOutput``; its
  TEXT chunks *from the crew's final task* become appended artifact chunks
  (earlier tasks are intermediate work, not the answer). If what streamed
  differs from the final ``CrewOutput.raw`` (guardrails, structured output,
  ReAct scaffolding from a model without native function calling), the
  artifact is replaced with the final text so the answer is always exact.
* ``Task(human_input=True)`` -> INPUT_REQUIRED, without touching stdin. The
  adapter installs a CrewAI ``HumanInputProvider`` (``set_provider``, a
  contextvar, so it is per run) whose ``handle_feedback`` stops the run with
  the agent's draft answer as the question. On the resuming turn the crew is
  kicked off again with the original inputs; the provider's
  ``setup_messages`` restores the paused task's conversation and appends the
  reply as CrewAI feedback, so the paused task continues where it stopped.
  One reply per pause: the revised answer is accepted (CrewAI's terminal loop
  keeps asking until an empty reply; over A2A an empty reply also accepts the
  draft unchanged). Limitation: tasks *before* the paused one are executed
  again on resume, because CrewAI exposes no resume-at-task short of disk
  checkpoints.
* governance -> CrewAI's execution-scoped ``PRE_TOOL_CALL`` interception hook
  (``crewai.hooks.dispatch.scoped_hooks`` + ``register_scoped``). CrewAI runs
  it on both tool paths (native function calling and ReAct text) before the
  tool body; raising ``HookAborted`` makes CrewAI skip the body and hand the
  model "Tool execution blocked by hook" instead. The dispatcher swallows any
  *other* hook exception (fail-open), so a policy error is converted to
  ``HookAborted`` here: governance fails closed. Scoped (contextvar) hooks
  cover only this run and never leak into other crews in the process.
* ``crew.usage_metrics`` -> one :class:`Usage` event when CrewAI reports tokens
  (emitted, not claimed: the suite has no usage check).

Cancellation is **not claimed**. Even ``Crew.akickoff`` runs each agent step
(LLM call, tool call) synchronously in a worker thread (``asyncio.to_thread``)
in crewai 1.15; cancelling the coroutine abandons that thread, it does not
stop it. The adapter does stop the crew cooperatively at the next model or
tool boundary (a scoped ``PRE_MODEL_CALL``/``PRE_TOOL_CALL`` hook raises
``HookAborted`` once the run is cancelled), but a step already running -- a
long generation or a long tool -- runs to completion, so the claim the suite
checks ("work actually stops") would be false.

Each run uses ``crew.copy()``: CrewAI interpolates inputs into its tasks in
place and an agent executor refuses concurrent invocation, so concurrent A2A
tasks need separate crews.
"""

from __future__ import annotations

import asyncio
import functools
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from ..core.adapter import Adapter, RunContext
from ..core.types import (
    AgentInfo,
    ArtifactChunk,
    Capabilities,
    Event,
    Message,
    StatusUpdate,
    TaskState,
    TextPart,
    ToolCall,
    ToolResult,
    Usage,
    new_id,
)


@dataclass
class _Paused:
    """A task stopped for human input, kept until the resuming turn."""

    inputs: dict[str, Any]
    task_index: int
    messages: list[dict[str, Any]]
    draft: Any  # crewai AgentFinish


@dataclass
class _Turn:
    """Per-run state shared with the CrewAI hooks and the input provider."""

    ctx: RunContext
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    resume: _Paused | None = None
    reply: str = ""
    paused: _Paused | None = None
    calls: dict[str, list[str]] = field(default_factory=dict)  # tool name -> open call ids

    def emit(self, ev: Any) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ev)


def _aborted_type():
    from crewai.hooks.dispatch import HookAborted

    return HookAborted


@functools.cache
def _awaiting_input_type() -> type:
    HookAborted = _aborted_type()

    class AwaitingInput(HookAborted):
        """Stops the crew for human input. A HookAborted subclass so CrewAI
        treats it as a deliberate stop (no agent retry), not as an error."""

    return AwaitingInput


def _task_index(context: Any) -> int:
    crew = getattr(context, "crew", None)
    task = getattr(context, "task", None)
    for i, t in enumerate(getattr(crew, "tasks", None) or []):
        if t is task:
            return i
    return -1


def _provider(turn: _Turn, awaiting: type):
    from crewai.core.providers.human_input import SyncHumanInputProvider

    class A2AHumanInputProvider(SyncHumanInputProvider):
        """Turns CrewAI's human feedback into INPUT_REQUIRED instead of stdin."""

        def setup_messages(self, context: Any) -> bool:
            r = turn.resume
            if r is None or _task_index(context) != r.task_index:
                return False
            context.messages.extend(dict(m) for m in r.messages)
            if turn.reply.strip():
                context.messages.append(context._format_feedback_message(turn.reply))
            return True

        def _decide(self, answer: Any, context: Any) -> Any:
            r = turn.resume
            if r is not None and _task_index(context) == r.task_index:
                turn.resume = None  # one reply per pause; this answer stands
                return answer if turn.reply.strip() else r.draft
            turn.paused = _Paused(
                inputs={},
                task_index=_task_index(context),
                messages=[dict(m) for m in context.messages],
                draft=answer,
            )
            raise awaiting(reason="human input required")

        def handle_feedback(self, formatted_answer: Any, context: Any) -> Any:
            return self._decide(formatted_answer, context)

        async def handle_feedback_async(self, formatted_answer: Any, context: Any) -> Any:
            return self._decide(formatted_answer, context)

    return A2AHumanInputProvider()


class CrewAIAdapter(Adapter):
    framework = "crewai"

    def __init__(
        self,
        crew: Any,
        info: AgentInfo | None = None,
        *,
        message_key: str = "message",
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """
        :param crew: a ``crewai.Crew``. Set ``stream=True`` on it to stream.
        :param message_key: kickoff input that receives the user's text; at
            least one task description or expected output must use it.
        :param inputs: extra constant kickoff inputs.
        """
        placeholder = "{" + message_key + "}"
        if not any(placeholder in (t.description + t.expected_output) for t in crew.tasks):
            raise ValueError(f"no task interpolates {placeholder}; the user's message would be dropped")
        self.crew = crew
        self._info = info or AgentInfo(getattr(crew, "name", None) or "crew")
        self._key = message_key
        self._inputs = dict(inputs or {})
        self._history: dict[str, list[dict[str, str]]] = {}
        self._paused: dict[str, _Paused] = {}  # task_id -> paused state

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=bool(self.crew.stream),
            multi_turn=True,
            input_required=True,
            cancellation=False,  # see module docstring
            tool_governance=True,
        )

    # ------------------------------------------------------------------ hooks

    def _install(self, turn: _Turn) -> None:
        """Register per-run CrewAI hooks and the input provider in the current
        context. Must run inside the task that drives the crew so the
        contextvars propagate into CrewAI's worker threads."""
        from crewai.core.providers.human_input import set_provider
        from crewai.hooks.dispatch import InterceptionPoint, register_scoped

        HookAborted = _aborted_type()
        ctx = turn.ctx

        def pre_tool(hc: Any) -> None:
            if ctx.cancelled.is_set():
                raise HookAborted(reason="run cancelled")
            args = dict(hc.tool_input or {})
            try:
                verdict = ctx.authorize_tool(hc.tool_name, args)
            except Exception as exc:  # fail closed: CrewAI would swallow it and run the tool
                raise HookAborted(reason=f"policy error: {type(exc).__name__}: {exc}") from exc
            if not verdict.allowed:
                raise HookAborted(reason=verdict.reason or "denied by policy")
            call_id = new_id()
            turn.calls.setdefault(hc.tool_name, []).append(call_id)
            turn.emit(ToolCall(call_id, hc.tool_name, args))

        def post_tool(hc: Any) -> None:
            open_calls = turn.calls.get(hc.tool_name)
            if open_calls:
                call_id = open_calls.pop(0)
                result = hc.tool_result if isinstance(hc.tool_result, str) else str(hc.tool_result)
                turn.emit(ToolResult(call_id, hc.tool_name, ok=True, summary=result[:200]))

        def pre_model(_hc: Any) -> None:
            if ctx.cancelled.is_set():
                raise HookAborted(reason="run cancelled")

        register_scoped(InterceptionPoint.PRE_TOOL_CALL, pre_tool)
        register_scoped(InterceptionPoint.POST_TOOL_CALL, post_tool)
        register_scoped(InterceptionPoint.PRE_MODEL_CALL, pre_model)
        set_provider(_provider(turn, _awaiting_input_type()))

    # ------------------------------------------------------------------ run

    async def _drive(self, turn: _Turn, crew: Any, inputs: dict[str, Any]) -> Any:
        from crewai.hooks.dispatch import scoped_hooks
        from crewai.types.streaming import CrewStreamingOutput, StreamChunkType

        with scoped_hooks():
            self._install(turn)
            result = await crew.akickoff(inputs=inputs)
            if isinstance(result, CrewStreamingOutput):
                last = len(crew.tasks) - 1
                async for chunk in result:
                    if chunk.chunk_type is StreamChunkType.TEXT and chunk.content and chunk.task_index == last:
                        turn.emit(chunk.content)
                result = result.result
            return result

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        req = ctx.request
        text = req.message.text
        resume = self._paused.pop(req.task_id, None) if req.resuming else None
        history = self._history.setdefault(req.context_id, [])

        if resume is not None:
            inputs = resume.inputs
        else:
            chat = [*history, {"role": "user", "content": text}]
            inputs = {**self._inputs, self._key: text, "crew_chat_messages": json.dumps(chat)}

        turn = _Turn(ctx, asyncio.get_running_loop(), asyncio.Queue(), resume=resume, reply=text if resume else "")
        crew = self.crew.copy()
        runner = asyncio.create_task(self._drive(turn, crew, inputs))

        aid: str | None = None
        streamed: list[str] = []
        awaiting = None
        try:
            while True:
                get = asyncio.ensure_future(turn.queue.get())
                done, _ = await asyncio.wait({get, runner}, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    break
                item = get.result()
                if isinstance(item, str):
                    yield ArtifactChunk(aid or (aid := new_id()), (TextPart(item),), append=bool(streamed))
                    streamed.append(item)
                else:
                    yield item
            while not turn.queue.empty():  # events the worker posted before it finished
                item = turn.queue.get_nowait()
                if isinstance(item, str):
                    yield ArtifactChunk(aid or (aid := new_id()), (TextPart(item),), append=bool(streamed))
                    streamed.append(item)
                else:
                    yield item

            try:
                result = runner.result()
            except _awaiting_input_type():
                if turn.paused is None:
                    raise
                awaiting = turn.paused
        finally:
            if not runner.done():
                runner.cancel()  # abandons the worker thread; hooks stop it at the next step

        if awaiting is not None:
            awaiting.inputs = inputs
            self._paused[req.task_id] = awaiting
            question = str(getattr(awaiting.draft, "output", "") or "")
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": question})
            yield StatusUpdate(TaskState.INPUT_REQUIRED, Message.agent(question))
            return

        raw = result.raw or ""
        if streamed and "".join(streamed) != raw:
            yield ArtifactChunk(aid, (TextPart(raw),), append=False, last_chunk=True)
        elif not streamed and raw:
            yield ArtifactChunk(new_id(), (TextPart(raw),), last_chunk=True)

        usage = getattr(result, "token_usage", None)
        if usage is not None and (usage.prompt_tokens or usage.completion_tokens):
            yield Usage(usage.prompt_tokens, usage.completion_tokens)

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": raw})
