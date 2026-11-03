"""Deterministic Microsoft Agent Framework agents for every behaviour.

Every behaviour is a real ``agent_framework.Agent`` (so every adapter claims
the same capabilities). The model is replaced by :class:`ScriptedChatClient`,
a ``BaseChatClient`` composed with the framework's own
``FunctionInvocationLayer`` (and ``ChatMiddlewareLayer``) exactly as a
provider client is, so function calls the script emits are executed - or
paused - by the framework's real invocation loop, history comes from the
framework's own session/history provider, and streaming goes through its
``ResponseStream``. No network, no API key.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence

from ...adapters.agent_framework import AgentFrameworkAdapter
from ..spec import Behaviour, Probe

# A script receives the messages the framework sends to the "model" and
# yields lists of Content items; each yielded list becomes one streamed
# ChatResponseUpdate (or is merged into one ChatResponse when not streaming).
Script = Callable[[Sequence[Any]], AsyncIterator[list[Any]]]


def _scripted_client_class():
    from agent_framework import (
        BaseChatClient,
        ChatMiddlewareLayer,
        ChatResponse,
        ChatResponseUpdate,
        FunctionInvocationLayer,
    )

    class ScriptedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
        """A chat client whose 'model' is a Python script."""

        OTEL_PROVIDER_NAME = "agentparity-scripted"

        def __init__(self, script: Script, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._script = script

        def _inner_get_response(self, *, messages, stream, options, **kwargs):
            def updates() -> AsyncIterator[Any]:
                async def gen():
                    async for contents in self._script(list(messages)):
                        yield ChatResponseUpdate(role="assistant", contents=contents)

                return gen()

            if stream:
                return self._build_response_stream(updates())

            async def collect() -> Any:
                collected = [u async for u in updates()]
                return ChatResponse.from_updates(collected)

            return collect()

    return ScriptedChatClient


def _texts(messages: Sequence[Any], role: str) -> list[str]:
    out = []
    for m in messages:
        if str(getattr(m, "role", "")) == role and m.text:
            out.append(m.text)
    return out


def _last_function_result(messages: Sequence[Any]) -> Any | None:
    if not messages:
        return None
    last = messages[-1]
    for c in last.contents:
        if c.type == "function_result":
            return c
    return None


def _result_text(result: Any) -> str:
    """Flatten a function_result's payload to text."""
    value = result.result
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(getattr(c, "text", None) or str(c) for c in value)
    return str(value)


def _tool_script(call_name: str, args_from_user: Callable[[str], dict], reply: Callable[[str], str]) -> Script:
    """Emit one function call, then answer from its result (what a model does)."""

    async def script(messages):
        from agent_framework import Content

        fr = _last_function_result(messages)
        if fr is not None:
            yield [Content.from_text(reply(_result_text(fr)))]
            return
        user = _texts(messages, "user")[-1]
        yield [Content.from_function_call(call_id="call-1", name=call_name, arguments=args_from_user(user))]

    return script


class AgentFrameworkFixture:
    framework = "microsoft-agent-framework"

    def build(self, behaviour: Behaviour, probe: Probe):
        from agent_framework import Agent, Content, FunctionTool, tool

        Client = _scripted_client_class()
        tools: list[Any] = []

        if behaviour is Behaviour.TOOL:

            @tool(name="lookup", description="Look up a value by key.")
            def lookup(key: str) -> str:
                return probe.lookup(key)

            tools = [lookup]
            script = _tool_script("lookup", lambda user: {"key": user}, lambda result: result)

        elif behaviour is Behaviour.ASK:
            # Declaration-only tool: the framework never runs it; the loop
            # pauses with a user-input request the caller must answer.
            ask_user = FunctionTool(
                name="ask_user",
                description="Ask the user a question; the answer arrives later.",
                func=None,
                input_model={
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            )
            tools = [ask_user]
            script = _tool_script("ask_user", lambda user: {"question": "Which colour?"}, lambda r: "colour: " + r)

        else:

            async def script(messages):
                b = behaviour
                users = _texts(messages, "user")
                if b is Behaviour.ECHO:
                    yield [Content.from_text("echo: " + users[-1])]
                elif b is Behaviour.STREAM:
                    for piece in ("one", " two", " three"):
                        yield [Content.from_text(piece)]
                        await asyncio.sleep(0)
                elif b is Behaviour.REMEMBER:
                    yield [Content.from_text(users[-2] if len(users) > 1 else "<none>")]
                elif b is Behaviour.SLOW:
                    yield [Content.from_text("started")]
                    for _ in range(600):
                        probe.tick()
                        await asyncio.sleep(0.05)
                elif b is Behaviour.FAIL:
                    raise RuntimeError("deliberate failure")
                    yield []  # pragma: no cover - makes this an async generator

        agent = Agent(
            Client(script),
            name=f"maf_{behaviour.value}",
            description=f"agentparity {behaviour.value} fixture",
            tools=tools or None,
        )
        return AgentFrameworkAdapter(agent)
