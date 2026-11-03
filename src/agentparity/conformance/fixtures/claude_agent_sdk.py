"""Offline Claude Agent SDK agents for every behaviour, via a scripted CLI.

The SDK accepts a custom ``Transport`` (``ClaudeSDKClient(transport=...)``,
``claude_agent_sdk._internal.transport.Transport``). :class:`ScriptedCLI` is
such a transport: it stands where the Claude Code CLI process would and
speaks the CLI side of the stream-json protocol, with a script in place of
the model. Everything above the transport is the real SDK and the real
adapter: ``ClaudeSDKClient``, ``Query`` (initialize handshake, control
request routing, hook and ``can_use_tool`` dispatch, the in-process MCP
bridge), ``parse_message``, and every adapter code path.

What the scripted CLI does, per turn (frame shapes taken from
``claude_agent_sdk/_internal/query.py`` and ``message_parser.py``, SDK
0.2.177):

* answers ``initialize`` and records the hook callback ids it registers;
  answers ``interrupt``;
* emits ``system``/``init``, then, when the options the adapter built ask for
  ``include_partial_messages``, ``stream_event`` frames (``message_start``,
  ``content_block_delta``/``text_delta``, ...) followed by the complete
  ``assistant`` frame with the same message id, then ``result``;
* for a tool call: emits an ``assistant`` frame with a ``tool_use`` block,
  sends ``hook_callback`` control requests for every PreToolUse matcher that
  matches, honours a ``deny`` (the tool is not run; an error ``tool_result``
  goes back), then applies permission rules (``allowed_tools``, else a
  ``can_use_tool`` control request), and runs SDK MCP tools by sending
  ``mcp_message`` JSON-RPC (``initialize``, ``tools/call``) through the SDK's
  MCP bridge, so the tool body is the real in-process tool;
* SLOW: ticks the probe as "work" until an ``interrupt`` control request
  arrives or the transport is closed;
* FAIL: an ``is_error`` result, then a non-zero-exit ``ProcessError``, as the
  CLI exits after an error result.

What this does NOT prove: that the real CLI behaves as scripted here. It
proves the adapter against the protocol as read from the SDK source, not
against a live CLI and model. In particular the exact shape of the result
frame after an interrupt, the precise order in which the CLI runs PreToolUse
hooks vs. permission rules, and whether a live model would call
``AskUserQuestion`` are assumptions. A live run (real ``claude`` binary and an
API key) is the remaining check.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncIterator

from ...adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from ...core.types import AgentInfo
from ..spec import Behaviour, Probe

MODEL = "scripted-claude"
SERVER = "fx"


def _transport_base():
    from claude_agent_sdk._internal.transport import Transport

    return Transport


class ScriptedCLI(_transport_base()):  # type: ignore[misc]
    """The CLI end of the stream-json protocol, driven by a script."""

    def __init__(self, options: Any, behaviour: Behaviour, probe: Probe) -> None:
        self.options = options
        self.behaviour = behaviour
        self.probe = probe
        self.session_id = options.resume or str(uuid.uuid4())
        self.history: list[str] = []
        self._out: asyncio.Queue = asyncio.Queue()
        self._ready = False
        self._hooks: dict[str, list[dict[str, Any]]] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._n = 0
        self._interrupted = asyncio.Event()
        self._turns: set[asyncio.Task] = set()
        self._mcp_ready = False

    # ----------------------------------------------------------------- Transport

    async def connect(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        return None

    async def close(self) -> None:
        self._ready = False
        for t in list(self._turns):
            t.cancel()
        for f in self._pending.values():
            if not f.done():
                f.cancel()
        self._out.put_nowait(None)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._out.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if line.strip():
                self._on_frame(json.loads(line))

    # ----------------------------------------------------------------- protocol

    def _emit(self, frame: dict[str, Any]) -> None:
        if self._ready:
            self._out.put_nowait(frame)

    def _on_frame(self, f: dict[str, Any]) -> None:
        t = f.get("type")
        if t == "control_request":
            req = f["request"]
            sub = req.get("subtype")
            if sub == "initialize":
                for event, matchers in (req.get("hooks") or {}).items():
                    self._hooks[event] = matchers
                self._reply(f["request_id"], {"commands": [], "output_style": "default"})
            elif sub == "interrupt":
                self._interrupted.set()
                self._reply(f["request_id"], {})
            else:
                self._reply(f["request_id"], {})
        elif t == "control_response":
            resp = f["response"]
            fut = self._pending.pop(resp.get("request_id"), None)
            if fut is not None and not fut.done():
                if resp.get("subtype") == "error":
                    fut.set_exception(RuntimeError(resp.get("error")))
                else:
                    fut.set_result(resp.get("response") or {})
        elif t == "user":
            content = f["message"]["content"]
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            self._interrupted.clear()
            task = asyncio.create_task(self._turn(text))
            self._turns.add(task)
            task.add_done_callback(self._turns.discard)

    def _reply(self, request_id: str, response: dict[str, Any]) -> None:
        self._emit(
            {"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": response}}
        )

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._n += 1
        rid = f"cli_req_{self._n}"
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        self._emit({"type": "control_request", "request_id": rid, "request": request})
        return await fut

    # ----------------------------------------------------------------- frames

    def _init(self) -> None:
        self._emit(
            {
                "type": "system",
                "subtype": "init",
                "session_id": self.session_id,
                "model": MODEL,
                "tools": ["AskUserQuestion", f"mcp__{SERVER}__lookup"],
                "permissionMode": self.options.permission_mode or "default",
                "uuid": str(uuid.uuid4()),
            }
        )

    def _stream(self, mid: str, event: dict[str, Any]) -> None:
        if self.options.include_partial_messages:
            self._emit(
                {
                    "type": "stream_event",
                    "uuid": str(uuid.uuid4()),
                    "session_id": self.session_id,
                    "event": event,
                    "parent_tool_use_id": None,
                }
            )

    async def _say(self, pieces: list[str]) -> None:
        mid = "msg_" + uuid.uuid4().hex[:12]
        self._stream(mid, {"type": "message_start", "message": {"id": mid, "role": "assistant", "model": MODEL, "content": []}})
        self._stream(mid, {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        for p in pieces:
            self._stream(mid, {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": p}})
            await asyncio.sleep(0)
        self._stream(mid, {"type": "content_block_stop", "index": 0})
        self._stream(mid, {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}})
        self._stream(mid, {"type": "message_stop"})
        self._assistant(mid, [{"type": "text", "text": "".join(pieces)}], "end_turn")

    def _assistant(self, mid: str, content: list[dict[str, Any]], stop: str) -> None:
        self._emit(
            {
                "type": "assistant",
                "message": {
                    "id": mid,
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": content,
                    "stop_reason": stop,
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
                "parent_tool_use_id": None,
                "session_id": self.session_id,
                "uuid": str(uuid.uuid4()),
            }
        )

    def _tool_result(self, tool_use_id: str, text: str, is_error: bool) -> None:
        self._emit(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text, "is_error": is_error}],
                },
                "parent_tool_use_id": None,
                "session_id": self.session_id,
                "uuid": str(uuid.uuid4()),
            }
        )

    def _result(self, text: str, *, is_error: bool = False, subtype: str = "success", **extra: Any) -> None:
        self._emit(
            {
                "type": "result",
                "subtype": subtype,
                "duration_ms": 1,
                "duration_api_ms": 1,
                "is_error": is_error,
                "num_turns": 1,
                "session_id": self.session_id,
                "result": text,
                "total_cost_usd": 0.0,
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "uuid": str(uuid.uuid4()),
                **extra,
            }
        )

    # ----------------------------------------------------------------- tool pipeline

    def _matching_hooks(self, event: str, tool_name: str) -> list[str]:
        ids: list[str] = []
        for m in self._hooks.get(event, []):
            pat = m.get("matcher")
            if pat in (None, "", "*") or re.fullmatch(pat, tool_name):
                ids.extend(m.get("hookCallbackIds", []))
        return ids

    async def _use_tool(self, name: str, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Assistant calls a tool; hooks, permissions, execution. Returns (ok, result text)."""
        tid = "toolu_" + uuid.uuid4().hex[:12]
        mid = "msg_" + uuid.uuid4().hex[:12]
        self._stream(mid, {"type": "message_start", "message": {"id": mid, "role": "assistant", "model": MODEL, "content": []}})
        self._stream(mid, {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}}})
        self._stream(mid, {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)}})
        self._stream(mid, {"type": "content_block_stop", "index": 0})
        self._stream(mid, {"type": "message_stop"})
        self._assistant(mid, [{"type": "tool_use", "id": tid, "name": name, "input": tool_input}], "tool_use")

        # 1. PreToolUse hooks; deny beats everything.
        for cb in self._matching_hooks("PreToolUse", name):
            out = await self._request(
                {
                    "subtype": "hook_callback",
                    "callback_id": cb,
                    "input": {
                        "session_id": self.session_id,
                        "transcript_path": "/dev/null",
                        "cwd": "/",
                        "hook_event_name": "PreToolUse",
                        "tool_name": name,
                        "tool_input": tool_input,
                        "tool_use_id": tid,
                    },
                    "tool_use_id": tid,
                }
            )
            hso = out.get("hookSpecificOutput") or {}
            if hso.get("permissionDecision") == "deny":
                reason = hso.get("permissionDecisionReason", "denied by hook")
                self._tool_result(tid, reason, True)
                return False, reason

        # 2. Permission rules, then the permission prompt (can_use_tool).
        effective_input = tool_input
        if name not in (self.options.allowed_tools or []) and self.options.permission_mode != "bypassPermissions":
            if self.options.can_use_tool is None:
                self._tool_result(tid, "permission denied", True)
                return False, "permission denied"
            perm = await self._request(
                {"subtype": "can_use_tool", "tool_name": name, "input": tool_input, "permission_suggestions": [], "tool_use_id": tid}
            )
            if perm.get("behavior") != "allow":
                msg = perm.get("message", "denied")
                self._tool_result(tid, msg, True)
                return False, msg
            effective_input = perm.get("updatedInput", tool_input)

        # 3. Execute.
        if name == "AskUserQuestion":
            answers = effective_input.get("answers") or {}
            text = "; ".join(f"{q}={a}" for q, a in answers.items())
            self._tool_result(tid, text, False)
            return True, json.dumps(answers)
        if name.startswith(f"mcp__{SERVER}__"):
            if not self._mcp_ready:
                await self._mcp({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
                    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "claude-code", "version": "scripted"}}})
                await self._mcp({"jsonrpc": "2.0", "method": "notifications/initialized"})
                self._mcp_ready = True
            resp = await self._mcp({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": name.split("__", 2)[2], "arguments": effective_input}})
            result = resp.get("result") or {}
            text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
            ok = not result.get("isError") and "error" not in resp
            self._tool_result(tid, text, not ok)
            return ok, text
        self._tool_result(tid, f"unknown tool {name}", True)
        return False, "unknown tool"

    async def _mcp(self, message: dict[str, Any]) -> dict[str, Any]:
        out = await self._request({"subtype": "mcp_message", "server_name": SERVER, "message": message})
        return out.get("mcp_response") or {}

    # ----------------------------------------------------------------- the "model"

    async def _turn(self, user: str) -> None:
        b = self.behaviour
        previous = self.history[-1] if self.history else None
        self.history.append(user)
        self._init()
        if b is Behaviour.ECHO:
            await self._say(["echo: ", user])
            self._result("echo: " + user)
        elif b is Behaviour.STREAM:
            await self._say(["one", " two", " three"])
            self._result("one two three")
        elif b is Behaviour.REMEMBER:
            reply = previous if previous is not None else "<none>"
            await self._say([reply])
            self._result(reply)
        elif b is Behaviour.SLOW:
            await self._say(["started"])
            deadline = time.monotonic() + 30
            while not self._interrupted.is_set() and time.monotonic() < deadline:
                self.probe.tick()
                await asyncio.sleep(0.02)
            self._result("", is_error=True, subtype="error_during_execution", terminal_reason="aborted_streaming")
        elif b is Behaviour.TOOL:
            ok, text = await self._use_tool(f"mcp__{SERVER}__lookup", {"key": user})
            reply = text if ok else f"I could not look that up: {text}"
            await self._say([reply])
            self._result(reply)
        elif b is Behaviour.ASK:
            q = {"question": "Which colour?", "header": "Colour", "multiSelect": False,
                 "options": [{"label": "red", "description": ""}, {"label": "blue", "description": ""}]}
            ok, text = await self._use_tool("AskUserQuestion", {"questions": [q]})
            reply = "colour: " + json.loads(text).get("Which colour?", "") if ok else f"no answer: {text}"
            await self._say([reply])
            self._result(reply)
        elif b is Behaviour.FAIL:
            from claude_agent_sdk import ProcessError

            self._result("", is_error=True, subtype="error_during_execution", errors=["deliberate failure"])
            await asyncio.sleep(0)
            self._out.put_nowait(ProcessError("Command failed", exit_code=1))


class ClaudeAgentSDKFixture:
    framework = "claude-agent-sdk"

    def build(self, behaviour: Behaviour, probe: Probe):
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

        @tool("lookup", "Look up a value by key.", {"key": str})
        async def lookup(args: dict[str, Any]) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": probe.lookup(args["key"])}]}

        # Same options for every behaviour so every adapter claims the same
        # capabilities. The lookup tool is pre-approved by permission rules,
        # which is exactly the case can_use_tool would miss: governance has to
        # come from the PreToolUse hook.
        options = ClaudeAgentOptions(
            mcp_servers={SERVER: create_sdk_mcp_server(SERVER, tools=[lookup])},
            allowed_tools=[f"mcp__{SERVER}__lookup"],
        )
        return ClaudeAgentSDKAdapter(
            options,
            AgentInfo(f"claude-{behaviour.value}"),
            transport_factory=lambda opts: ScriptedCLI(opts, behaviour, probe),
        )
