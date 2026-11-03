"""LangGraph adapter.

Mapping:

* ``context_id`` -> ``configurable.thread_id`` (needs a checkpointer for
  multi-turn and input-required; capabilities say so honestly)
* LLM token chunks (``stream_mode="messages"``) and ``get_stream_writer()``
  output (``stream_mode="custom"``) -> artifact chunks, as they happen
* ``interrupt(value)`` -> INPUT_REQUIRED; the resuming turn sends
  ``Command(resume=<reply>)``
* tools wrapped with :func:`governed` -> policy check before the body runs
"""

from __future__ import annotations

import contextvars
import functools
import inspect
from typing import Any, AsyncIterator, Callable, Iterable

from ..core.adapter import Adapter, RunContext
from ..core.governance import ToolDenied
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
    new_id,
)

_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("agentparity_lg_ctx", default=None)


def governed(fn: Callable | None = None, *, name: str | None = None):
    """Wrap a tool function so the run's policy is consulted before it executes.

    Use under LangChain's ``@tool``::

        @tool
        @governed
        def lookup(key: str) -> str: ...

    Outside an agentparity run the wrapper is a no-op, so the same tool works
    in plain LangGraph tests.
    """

    def wrap(f: Callable) -> Callable:
        tool_name = name or f.__name__

        def check(args, kwargs):
            ctx = _current.get()
            if ctx is None:
                return
            bound = inspect.signature(f).bind_partial(*args, **kwargs)
            verdict = ctx.authorize_tool(tool_name, dict(bound.arguments))
            if not verdict.allowed:
                raise ToolDenied(tool_name, verdict.reason)

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def awrapper(*args, **kwargs):
                check(args, kwargs)
                return await f(*args, **kwargs)

            return awrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            check(args, kwargs)
            return f(*args, **kwargs)

        return wrapper

    return wrap(fn) if fn is not None else wrap


def _default_input(message: Message) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=message.text)]}


def _to_parts(value: Any) -> tuple:
    if isinstance(value, str):
        return (TextPart(value),)
    if isinstance(value, dict):
        if set(value) == {"text"} and isinstance(value["text"], str):
            return (TextPart(value["text"]),)
        return (DataPart(value),)
    return (TextPart(str(value)),)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return ""


class LangGraphAdapter(Adapter):
    framework = "langgraph"

    def __init__(
        self,
        graph: Any,
        info: AgentInfo,
        *,
        input_builder: Callable[[Message], Any] = _default_input,
        nodes: Iterable[str] | None = None,
        governs_tools: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        :param graph: a compiled LangGraph graph.
        :param nodes: if given, only messages produced by these nodes count as
            output (e.g. hide a router's or a critic's messages).
        :param governs_tools: set True when the graph's tools are wrapped with
            :func:`governed`; the conformance suite will hold you to it.
        """
        self.graph = graph
        self._info = info
        self._input = input_builder
        self._nodes = frozenset(nodes) if nodes is not None else None
        self._governs = governs_tools
        self._config = config or {}

    @property
    def info(self) -> AgentInfo:
        return self._info

    @property
    def capabilities(self) -> Capabilities:
        persistent = getattr(self.graph, "checkpointer", None) is not None
        return Capabilities(
            streaming=True,
            multi_turn=persistent,
            input_required=persistent,
            cancellation=True,
            tool_governance=self._governs,
        )

    async def stream(self, ctx: RunContext) -> AsyncIterator[Event]:
        from langchain_core.messages import AIMessage, AIMessageChunk
        from langgraph.types import Command

        req = ctx.request
        config = {**self._config}
        config["configurable"] = {**config.get("configurable", {}), "thread_id": req.context_id}
        graph_input = Command(resume=req.message.text) if req.resuming else self._input(req.message)

        token = _current.set(ctx)
        streamed: dict[str, str] = {}  # message id -> artifact id (token-streamed)
        try:
            async for mode, payload in self.graph.astream(
                graph_input, config, stream_mode=["custom", "messages", "updates"]
            ):
                if mode == "custom":
                    yield ArtifactChunk(new_id(), _to_parts(payload), last_chunk=True)

                elif mode == "messages":
                    msg, meta = payload
                    if self._nodes is not None and meta.get("langgraph_node") not in self._nodes:
                        continue
                    if isinstance(msg, AIMessageChunk):
                        text = _content_text(msg.content)
                        if not text:
                            continue
                        mid = msg.id or "chunk"
                        first = mid not in streamed
                        aid = streamed.setdefault(mid, new_id())
                        yield ArtifactChunk(aid, (TextPart(text),), append=not first)
                    elif isinstance(msg, AIMessage):
                        if msg.id in streamed:  # already delivered token by token
                            continue
                        for tc in msg.tool_calls or ():
                            yield ToolCall(tc.get("id") or new_id(), tc["name"], dict(tc.get("args") or {}))
                        text = _content_text(msg.content)
                        if text:
                            yield ArtifactChunk(new_id(), (TextPart(text),), last_chunk=True)

                elif mode == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                    intr = payload["__interrupt__"]
                    value = intr[0].value if intr else None
                    yield StatusUpdate(TaskState.INPUT_REQUIRED, Message(Role.AGENT, _to_parts(value)))
                    return
        finally:
            _current.reset(token)
