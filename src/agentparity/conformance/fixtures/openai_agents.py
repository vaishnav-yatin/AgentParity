"""Deterministic OpenAI Agents SDK agents for every behaviour.

Every behaviour is a real ``agents.Agent`` run by ``Runner``; only the model
is replaced. :class:`ScriptedModel` implements the SDK's ``Model`` interface
(``get_response`` / ``stream_response``) and answers by script instead of by
inference, returning Responses-API output items and stream events exactly
as a real model would - text deltas, ``function_call`` items, a terminal
``response.completed``. TOOL and ASK therefore go through the SDK's own
function-calling, tool-guardrail and approval machinery. No network, no API
key, tracing disabled.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Union

from ...adapters.openai_agents import OpenAIAgentsAdapter, human_reply
from ...core.types import new_id
from ..spec import Behaviour, Probe

# What a script decides for one model call: a list of text pieces to stream,
# or a ``(tool_name, arguments)`` tuple for a function call.
Decision = Union[list, tuple]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (c.get("text") or "") if isinstance(c, dict) else (getattr(c, "text", "") or "") for c in content
        )
    return ""


def _items(input: Any) -> list[dict[str, Any]]:
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    out = []
    for it in input:
        out.append(it if isinstance(it, dict) else it.model_dump(exclude_unset=True))
    return out


def user_texts(input: Any) -> list[str]:
    return [
        _content_text(i.get("content"))
        for i in _items(input)
        if i.get("role") == "user" and i.get("type", "message") == "message"
    ]


def last_tool_output(input: Any) -> str | None:
    items = _items(input)
    if items and items[-1].get("type") == "function_call_output":
        out = items[-1].get("output")
        return out if isinstance(out, str) else json.dumps(out)
    return None


def _model_base():
    from agents.models.interface import Model

    return Model


def make_model(script: Callable[[Any], Decision], *, before_complete: Callable[[], Any] | None = None,
               fail: bool = False):
    """Build a scripted ``agents.models.interface.Model``."""
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseCreatedEvent,
        ResponseFunctionToolCall,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseUsage,
    )
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    from agents.items import ModelResponse
    from agents.usage import Usage

    def response(output: list, status: str = "completed") -> Response:
        return Response(
            id="resp-" + new_id(),
            created_at=0,
            model="scripted",
            object="response",
            output=output,
            tool_choice="auto",
            tools=[],
            top_p=None,
            parallel_tool_calls=False,
            status=status,
            usage=ResponseUsage(
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
                input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            ),
        )

    def output_for(decision: Decision) -> list:
        if isinstance(decision, tuple):
            name, args = decision
            cid = "call_" + new_id()
            return [ResponseFunctionToolCall(id="fc_" + cid, call_id=cid, type="function_call",
                                             name=name, arguments=json.dumps(args))]
        return [ResponseOutputMessage(
            id="msg_" + new_id(), type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="".join(decision), annotations=[])],
        )]

    class ScriptedModel(_model_base()):
        async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                               handoffs, tracing, *, previous_response_id=None, conversation_id=None,
                               prompt=None):
            if fail:
                raise RuntimeError("deliberate failure")
            if before_complete is not None:
                await before_complete()
            r = response(output_for(script(input)))
            return ModelResponse(output=r.output, usage=Usage(requests=1, input_tokens=3, output_tokens=2,
                                                              total_tokens=5), response_id=r.id)

        async def stream_response(self, system_instructions, input, model_settings, tools, output_schema,
                                  handoffs, tracing, *, previous_response_id=None, conversation_id=None,
                                  prompt=None) -> AsyncIterator[Any]:
            if fail:
                raise RuntimeError("deliberate failure")
            decision = script(input)
            output = output_for(decision)
            seq = 0
            yield ResponseCreatedEvent(type="response.created", sequence_number=seq,
                                       response=response([], "in_progress"))
            item = output[0]
            if isinstance(item, ResponseOutputMessage):
                seq += 1
                pending = item.model_copy(update={"status": "in_progress", "content": []})
                yield ResponseOutputItemAddedEvent(type="response.output_item.added", item=pending,
                                                   output_index=0, sequence_number=seq)
                for piece in decision:
                    seq += 1
                    yield ResponseTextDeltaEvent(type="response.output_text.delta", item_id=item.id,
                                                 output_index=0, content_index=0, delta=piece,
                                                 logprobs=[], sequence_number=seq)
                    await asyncio.sleep(0)
            if before_complete is not None:
                await before_complete()
            seq += 1
            yield ResponseOutputItemDoneEvent(type="response.output_item.done", item=item,
                                              output_index=0, sequence_number=seq)
            seq += 1
            yield ResponseCompletedEvent(type="response.completed", sequence_number=seq,
                                         response=response(output))

    return ScriptedModel()


class OpenAIAgentsFixture:
    framework = "openai-agents"

    def build(self, behaviour: Behaviour, probe: Probe):
        from agents import Agent, RunConfig, function_tool

        tools: list = []
        before_complete = None
        fail = False
        b = behaviour

        if b is Behaviour.ECHO:
            script = lambda inp: ["echo: " + user_texts(inp)[-1]]  # noqa: E731
        elif b is Behaviour.STREAM:
            script = lambda inp: ["one", " two", " three"]  # noqa: E731
        elif b is Behaviour.REMEMBER:

            def script(inp):
                texts = user_texts(inp)
                return [texts[-2] if len(texts) > 1 else "<none>"]

        elif b is Behaviour.SLOW:
            script = lambda inp: ["started"]  # noqa: E731

            async def before_complete():  # the model keeps "generating" after the first delta
                for _ in range(600):
                    probe.tick()
                    await asyncio.sleep(0.05)

        elif b is Behaviour.FAIL:
            script = lambda inp: ["unreachable"]  # noqa: E731
            fail = True
        elif b is Behaviour.TOOL:

            @function_tool
            def lookup(key: str) -> str:
                """Look up a value by key."""
                return probe.lookup(key)

            tools = [lookup]

            def script(inp):
                out = last_tool_output(inp)
                if out is not None:
                    return [out]
                return ("lookup", {"key": user_texts(inp)[-1]})

        elif b is Behaviour.ASK:

            @function_tool(needs_approval=True)
            def ask_user(question: str) -> str:
                """Ask the user a question; runs once the user has answered."""
                return human_reply() or ""

            tools = [ask_user]

            def script(inp):
                out = last_tool_output(inp)
                if out is not None:
                    return ["colour: " + out]
                return ("ask_user", {"question": "Which colour?"})

        else:  # pragma: no cover
            return None

        model = make_model(script, before_complete=before_complete, fail=fail)
        agent = Agent(name=f"oai_{b.value}", instructions="scripted", model=model, tools=tools)
        return OpenAIAgentsAdapter(agent, run_config=RunConfig(tracing_disabled=True))
