"""Deterministic ADK agents for every behaviour.

ECHO/STREAM/REMEMBER/SLOW/FAIL are custom ``BaseAgent`` subclasses. ASK and
TOOL need ADK's real tool machinery (``LlmAgent`` + function calling), so a
scripted ``BaseLlm`` stands in for the model and emits the function calls a
real model would.
"""

from __future__ import annotations

import asyncio

from ...adapters.adk import ADKAdapter
from ..spec import Behaviour, Probe


def _text_event(agent, ctx, text: str, *, partial: bool = False):
    from google.adk.events import Event
    from google.genai import types

    return Event(
        author=agent.name,
        invocation_id=ctx.invocation_id,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        partial=partial,
    )


def _user_texts(session) -> list[str]:
    out = []
    for e in session.events:
        if e.author == "user" and e.content and e.content.parts:
            t = "".join(p.text or "" for p in e.content.parts)
            if t:
                out.append(t)
    return out


class _ScriptedLlmFactory:
    """Builds a BaseLlm that answers by script instead of by inference."""

    @staticmethod
    def make(call_name: str, args_from_user, reply_from_result):
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        class ScriptedLlm(BaseLlm):
            model: str = "scripted"

            async def generate_content_async(self, llm_request, stream: bool = False):
                contents = llm_request.contents or []
                last = contents[-1] if contents else None
                fr = next((p.function_response for p in (last.parts or []) if p.function_response), None) if last else None
                if fr is not None:
                    text = reply_from_result(fr.response or {})
                    yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
                    return
                user = ""
                for c in reversed(contents):
                    if c.role == "user" and c.parts and c.parts[0].text:
                        user = c.parts[0].text
                        break
                call = types.FunctionCall(id="call-1", name=call_name, args=args_from_user(user))
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(function_call=call)]))

        return ScriptedLlm()


class ADKFixture:
    framework = "google-adk"

    def build(self, behaviour: Behaviour, probe: Probe):
        from google.adk.agents import BaseAgent, LlmAgent
        from google.adk.tools import FunctionTool, LongRunningFunctionTool

        if behaviour is Behaviour.TOOL:

            def lookup(key: str) -> str:
                """Look up a value by key."""
                return probe.lookup(key)

            llm = _ScriptedLlmFactory.make(
                "lookup",
                lambda user: {"key": user},
                lambda resp: str(resp.get("result", resp.get("error", resp))),
            )
            return ADKAdapter(LlmAgent(name="tool_agent", model=llm, tools=[FunctionTool(lookup)]))

        if behaviour is Behaviour.ASK:

            def ask_user(question: str) -> None:
                """Ask the user a question; the answer arrives later."""
                return None

            llm = _ScriptedLlmFactory.make(
                "ask_user",
                lambda user: {"question": "Which colour?"},
                lambda resp: "colour: " + str(resp.get("result", "")),
            )
            return ADKAdapter(LlmAgent(name="ask_agent", model=llm, tools=[LongRunningFunctionTool(ask_user)]))

        class Scripted(BaseAgent):
            async def _run_async_impl(self, ctx):
                b = behaviour
                if b is Behaviour.ECHO:
                    yield _text_event(self, ctx, "echo: " + _user_texts(ctx.session)[-1])
                elif b is Behaviour.STREAM:
                    for piece in ("one", " two", " three"):
                        yield _text_event(self, ctx, piece, partial=True)
                        await asyncio.sleep(0)
                    yield _text_event(self, ctx, "one two three")
                elif b is Behaviour.REMEMBER:
                    texts = _user_texts(ctx.session)
                    yield _text_event(self, ctx, texts[-2] if len(texts) > 1 else "<none>")
                elif b is Behaviour.SLOW:
                    yield _text_event(self, ctx, "started")
                    for _ in range(600):
                        probe.tick()
                        await asyncio.sleep(0.05)
                elif b is Behaviour.FAIL:
                    raise RuntimeError("deliberate failure")

        # Capabilities are uniform across the fixture: every adapter must claim
        # the same thing, so the custom agents carry an LlmAgent sub-agent that
        # never runs. That way tool_governance is claimed (and then proven by
        # the TOOL agent) rather than silently skipped.
        idle = LlmAgent(name="idle", model=_ScriptedLlmFactory.make("noop", lambda u: {}, lambda r: ""))
        return ADKAdapter(Scripted(name=f"adk_{behaviour.value}", sub_agents=[idle]))
