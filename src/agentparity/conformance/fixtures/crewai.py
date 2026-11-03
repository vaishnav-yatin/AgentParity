"""Deterministic (model-free) CrewAI crews for every behaviour it can build.

Every crew is a real ``Crew`` -> ``Task`` -> ``Agent`` run. The model is a
scripted ``BaseLLM`` subclass (CrewAI's custom-LLM interface) that answers the
way a function-calling model would: it reads the prompt CrewAI builds, emits
OpenAI-style tool calls that CrewAI's own executor dispatches (so governance
is exercised on the framework's real tool path), and, when CrewAI turns
streaming on, emits ``LLMStreamChunkEvent``s exactly as provider clients do.

SLOW is not built: the adapter does not claim cancellation (CrewAI runs each
step in a worker thread it cannot stop), so there is nothing to prove.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ...adapters.crewai import CrewAIAdapter
from ...core.types import AgentInfo
from ..spec import Behaviour, Probe

_MSG = re.compile(r"USER MESSAGE: <<<(.*?)>>>", re.S)
_FEEDBACK = re.compile(r"^User feedback: (.*?)\nInstructions:", re.S | re.M)
_HISTORY_LINE = re.compile(r"^(User|Assistant): (.*)$", re.M)

DESCRIPTION = "Answer the user.\nUSER MESSAGE: <<<{message}>>>"


def _text(m: dict[str, Any]) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else ""


def _user_message(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and (hit := _MSG.search(_text(m))):
            return hit.group(1)
    return ""


def _history(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Turns CrewAI appended to the task from ``crew_chat_messages``."""
    for m in messages:
        if m.get("role") == "user" and _MSG.search(_text(m)):
            return _HISTORY_LINE.findall(_text(m))
    return []


Script = Callable[[list[dict[str, Any]], bool], "str | list[dict[str, Any]]"]


def _scripted_llm(script: Script):
    from crewai.events.types.llm_events import LLMCallType
    from crewai.llms.base_llm import BaseLLM, llm_call_context

    class ScriptedLLM(BaseLLM):
        """Answers by script instead of by inference."""

        model: str = "scripted"

        def supports_function_calling(self) -> bool:
            return True

        def get_context_window_size(self) -> int:
            return 128_000

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            from_task=None,
            from_agent=None,
            response_model=None,
        ):
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            with llm_call_context():
                self._emit_call_started_event(
                    messages=messages, tools=tools, from_task=from_task, from_agent=from_agent
                )
                out = script(list(messages), bool(tools))
                if isinstance(out, str) and self.stream:
                    words = out.split(" ")
                    for i, w in enumerate(words):
                        self._emit_stream_chunk_event(
                            w if i == 0 else " " + w, from_task=from_task, from_agent=from_agent
                        )
                self._emit_call_completed_event(
                    response=out,
                    call_type=LLMCallType.TOOL_CALL if isinstance(out, list) else LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=messages,
                )
                return out

    return ScriptedLLM(model="scripted")


def _tool_call(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


class CrewAIFixture:
    framework = "crewai"

    def build(self, behaviour: Behaviour, probe: Probe):
        from crewai import Agent, Crew, Task
        from crewai.tools import BaseTool

        tools: list[Any] = []
        human_input = False

        if behaviour is Behaviour.ECHO:

            def script(msgs, has_tools):
                return "echo: " + _user_message(msgs)

        elif behaviour is Behaviour.STREAM:

            def script(msgs, has_tools):
                return "one two three"  # streamed word by word by the LLM

        elif behaviour is Behaviour.REMEMBER:

            def script(msgs, has_tools):
                # crew_chat_messages ends with the current turn; the one before
                # it (if any) is what the user said previously.
                users = [t for role, t in _history(msgs) if role == "User"]
                return users[-2] if len(users) > 1 else "<none>"

        elif behaviour is Behaviour.ASK:
            human_input = True

            def script(msgs, has_tools):
                for m in reversed(msgs):
                    if m.get("role") == "user" and (fb := _FEEDBACK.search(_text(m))):
                        return "colour: " + fb.group(1)
                return "Which colour?"

        elif behaviour is Behaviour.TOOL:

            class Lookup(BaseTool):
                name: str = "lookup"
                description: str = "Look up a value by key."

                def _run(self, key: str) -> str:
                    return probe.lookup(key)

            tools = [Lookup()]

            def script(msgs, has_tools):
                results = [m for m in msgs if m.get("role") == "tool"]
                if results:
                    return _text(results[-1])
                return _tool_call("lookup", {"key": _user_message(msgs)})

        elif behaviour is Behaviour.FAIL:

            def script(msgs, has_tools):
                raise RuntimeError("deliberate failure")

        else:  # SLOW: cancellation is not claimed
            return None

        agent = Agent(
            role="assistant",
            goal="Answer the user",
            backstory="A deterministic test agent.",
            llm=_scripted_llm(script),
            tools=tools,
            verbose=False,
        )
        task = Task(
            description=DESCRIPTION,
            expected_output="The answer.",
            agent=agent,
            human_input=human_input,
        )
        crew = Crew(agents=[agent], tasks=[task], stream=True, verbose=False)
        return CrewAIAdapter(crew, AgentInfo(f"crewai-{behaviour.value}"))
