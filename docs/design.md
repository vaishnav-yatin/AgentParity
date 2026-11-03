# agentparity — design

Status: v0.1.0, alpha. Date: 3 November 2026.

## 1. Problem

A2A v1.0 (Linux Foundation) has become the standard for agent-to-agent calls, and MCP for agent-to-tool calls. Every major framework now has a way to expose or consume an A2A agent, natively or through community wrappers, and several projects already wrap many frameworks at once.

What nobody verifies is **fidelity**. Adapters drop semantics without anyone noticing:

- a server advertises streaming but sends only the first and last events (a reviewer found exactly this in a proposed OpenAI Agents SDK A2A integration);
- cancel returns `CANCELED` while the framework's worker thread runs to completion;
- `contextId` is not mapped to the framework's session, so multi-turn silently breaks, or worse, sessions leak across contexts;
- each framework pauses for human input differently (LangGraph `interrupt`, ADK long-running tools, OpenAI `needs_approval`, Claude `can_use_tool`, CrewAI `human_input`, Agent Framework approval requests), and none of them maps that pause onto `input-required` the same way;
- tool policies are written once per framework, if at all.

The A2A interop test kit checks SDK-to-SDK combinations. Nothing checks *framework adapters*.

## 2. Positioning

agentparity does not define a protocol. It sits between frameworks and A2A:

```
 framework agent ──► Adapter ──► Runtime ──► A2A server  ◄── any A2A client
                        ▲            ▲
                 governance hook   lifecycle guarantees
                        │
                 Conformance suite (holds the adapter to its claims)
```

The differentiator is the suite. Adapters are many and cheap; a check that an adapter cannot fake is not.

## 3. Canonical model

`core/types.py`. No framework or network imports, enforced by keeping the core standard-library only.

| agentparity | A2A v1.0 |
|---|---|
| `TextPart` / `DataPart` / `FilePart` | `Part` text / data / url\|raw |
| `ArtifactChunk(artifact_id, parts, append, last_chunk)` | `TaskArtifactUpdateEvent` |
| `StatusUpdate(state, message)` | `TaskStatusUpdateEvent` |
| `TaskState` | `TaskState` (all eight) |
| `RunRequest.context_id` / `task_id` | `contextId` / `taskId` |
| `ToolCall`, `ToolResult`, `Usage` | not sent as content; observability only |

## 4. Adapter contract

An adapter implements `info`, `capabilities` and `stream(ctx)`, plus an optional `cancel(ctx)`. It yields content and at most one final status. It does **not** handle `WORKING`, exceptions or cancellation bookkeeping: the runtime does that the same way for everyone, so those behaviours can't drift between adapters.

`RunContext.authorize_tool(name, args)` is the governance seam. Each adapter calls it from its framework's native pre-tool hook, and the suite checks that the call is made *before* the tool body runs.

## 5. Runtime guarantees

For any adapter, a consumer sees: `WORKING`, then content, then exactly one final status, then nothing. When an adapter breaks this, the runtime repairs the stream (drops late events, turns an append to an unknown artifact into a start, records a cancel that wasn't honoured) and records a `Violation`. The conformance suite fails on any violation, so a repair never hides one.

## 6. Conformance suite

Fixtures build seven deterministic behaviours. Each check rests on an observation the adapter cannot fake:

| Check | Observation |
|---|---|
| lifecycle.echo | exact round-trip text, no violations |
| lifecycle.failure | a framework exception ends in FAILED, not a hang |
| streaming | at least 3 content events for a 3-word reply; batched output fails |
| multi_turn | turn 2 recalls turn 1 in the same context; a different context does **not** |
| input_required | turn 1 ends INPUT_REQUIRED; resuming the same task completes using the reply |
| cancellation | final CANCELED **and** the agent's probe stops ticking |
| tool_governance.allow | the authorization timestamp precedes the moment the tool body ran; the policy saw the real arguments |
| tool_governance.deny | a denied tool's body never runs |

Rules: a claimed capability whose behaviour the fixture can't build is `UNVERIFIED`, and a claim with no check behind it is `UNVERIFIED`. Both count as failures. All adapters a fixture builds must claim the same capabilities.

**The suite is itself tested.** The reference fixture passes every check, and each of nine deliberately broken variants fails the check aimed at it (`tests/test_conformance_suite.py`). Each adapter's governance check was also mutation-tested: with `authorize_tool` unwired, all six adapters fail both governance checks.

## 7. A2A bridge

- `build_app(adapter)` gives a Starlette app with the agent-card route and the JSON-RPC binding, built on the official `a2a-sdk` 1.x. The card advertises streaming only if it is claimed.
- `RemoteA2AAdapter.connect(url)` turns any A2A agent into an adapter.
- The reference fixture passes the full suite *through the wire* on a real socket, including cancellation with the probe check across the network. Governance shows correctly as not enforced client-side, because a policy has to run where the tools run.
- Test note: httpx's in-process ASGI transport buffers responses, so streaming and cancellation tests use `a2a.testing.running_server` on a localhost socket.

## 8. Decisions

1. **Build on A2A and MCP, invent no wire format.**
2. **Clean-room.** The project shares no code with any proprietary Leap codebase. A product that wants it depends on it.
3. **Governance at the server.** A client can't enforce a policy on someone else's tools, and the suite says so rather than pretending otherwise.
4. **Honest claims over full matrices.** CrewAI does not claim cancellation. The suite proved it can't be honoured mid-step, so the matrix shows `—` instead of a PASS that isn't true.
5. **Python first.** Five of the six target frameworks are Python-first. A TypeScript port is roadmap.

## 9. Known limitations (v0.1.0)

- Claude Agent SDK is verified against a scripted CLI transport, not a live CLI.
- CrewAI: cancellation is not claimed. On resume, tasks before the paused one run again.
- OpenAI Agents SDK: tool approvals carry no data, so the caller's reply reaches the tool through `human_reply()`. Hosted tools (web search, code interpreter, hosted MCP) can't be governed, and an agent that uses them must not claim `tool_governance`.
- Microsoft Agent Framework: a custom chat client must list `FunctionInvocationLayer` before `ChatMiddlewareLayer`. The adapter checks the order and withholds the governance claim if it's wrong.
- Usage events are emitted but not claimed: the suite has no usage check yet.
- `ToolCall`, `ToolResult` and `Usage` are not yet exported to OpenTelemetry.

## Roadmap

- **0.2:** live mode (`AGENTPARITY_LIVE=1`) running the same behaviours against real models; a usage check; OpenTelemetry export.
- **0.3:** a URL mode for the suite (check someone else's A2A server through a behaviour-negotiation extension); MCP tool governance; Strands, Pydantic AI and LlamaIndex adapters.
- **Later:** a public compatibility dashboard per framework version, and a TypeScript port.
