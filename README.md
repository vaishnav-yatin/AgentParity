# agentparity

**Connect agents across frameworks over A2A, and prove that the connection is faithful.**

Most frameworks can now expose an agent over [A2A](https://a2a-protocol.org) or call one. But "speaks A2A" doesn't tell you whether it streams for real or buffers the whole answer, whether cancel actually stops the work, whether a resumed task picks up where it paused, or whether a tool policy runs before the tool does. agentparity covers three things:

1. **One adapter contract** for LangGraph, Google ADK, OpenAI Agents SDK, Claude Agent SDK, CrewAI and Microsoft Agent Framework, with an A2A v1.0 server and client on top.
2. **A conformance suite** that holds each adapter to every capability it claims. Each check rests on something the adapter can't fake. For example, the cancellation check watches a probe the agent has to keep ticking, so an adapter that reports `CANCELED` while its framework keeps running fails the check.
3. **Uniform governance.** You write one tool policy, and each adapter wires it into its framework's own pre-tool hook.

```python
from agentparity.a2a import serve
from agentparity.adapters.langgraph import LangGraphAdapter

serve(LangGraphAdapter(graph, AgentInfo("research")), port=9000)   # now any A2A client can call it
```

```bash
$ agentparity conform langgraph
agentparity conformance - langgraph
claims: streaming, multi_turn, input_required, cancellation, tool_governance

  PASS       lifecycle.echo         round-trip text exact
  PASS       lifecycle.failure      framework exception -> FAILED
  PASS       streaming              3 incremental content events
  PASS       multi_turn             context_id carries history; separate contexts isolated
  PASS       input_required         interrupt and resume on the same task
  PASS       cancellation           canceled and the agent actually stopped
  PASS       tool_governance.allow  policy saw name+arguments before the tool body ran
  PASS       tool_governance.deny   denied tool did not run (run ended failed)

RESULT: CONFORMANT
```

## Support matrix (v0.1.0)

Every PASS below comes from the suite running against the framework's real code, with a scripted model in place of inference. No API key or network is needed. `—` means the adapter doesn't claim that capability, for the reason given.

| Framework | Version tested | Streaming | Multi-turn | Input required | Cancellation | Tool governance | Governance hook |
|---|---|---|---|---|---|---|---|
| LangGraph | 1.2.19 | PASS | PASS¹ | PASS¹ | PASS | PASS | `@governed` tool wrapper |
| Google ADK | 2.13.0 | PASS | PASS | PASS | PASS | PASS | Runner plugin `before_tool_callback` |
| OpenAI Agents SDK | 0.24.0 | PASS | PASS | PASS | PASS | PASS | tool input guardrail |
| Claude Agent SDK | 0.2.177 | PASS² | PASS² | PASS² | PASS² | PASS² | `PreToolUse` hook |
| CrewAI | 1.15.34 | PASS | PASS | PASS | —³ | PASS | scoped `PRE_TOOL_CALL` hook |
| Microsoft Agent Framework | 1.24.0 | PASS | PASS | PASS | PASS | PASS | function middleware |

1. Requires a checkpointer. Without one the adapter doesn't claim these capabilities.
2. Verified against a scripted stand-in for the Claude Code CLI that speaks its stream-json protocol, as read from the SDK source. That proves the adapter against the protocol, not against a live CLI.
3. CrewAI runs each model or tool step in a worker thread that can't be stopped. The suite caught this: when cancellation was claimed, the run reported CANCELED while the probe kept ticking. The adapter still stops at the next step boundary, but it doesn't claim cancellation.

## Install

```bash
pip install "agentparity[a2a]"                 # core + A2A server/client
pip install "agentparity[a2a,langgraph]"       # plus one framework
# extras: langgraph, adk, openai-agents, claude, crewai, agent-framework
```

The core imports nothing outside the standard library. Each adapter imports its framework only when you use it.

## Concepts

**Adapter**: wraps one agent built with one framework and translates its run or stream API into canonical events (`ArtifactChunk`, `StatusUpdate`, `ToolCall`, `Usage`). The canonical types map one-to-one onto A2A's.

**Runtime**: turns any adapter's output into a well-formed lifecycle: `WORKING` first, then content, then exactly one final status. If an adapter breaks the contract (events after the final status, an append to an artifact that was never started, a cancel it doesn't honour), the runtime repairs the stream *and* records a violation.

**Capabilities**: what an adapter claims. The A2A agent card advertises only claimed capabilities, and the suite checks each one. A claim the suite has no check for is reported as `UNVERIFIED`, which counts as a failure.

**Governance**: `ToolPolicy.check(name, args) -> Verdict`. Built-in policies are `AllowAll`, `AllowList` (fails closed), `DenyList` and `ToolBudget`. Governance runs where the tools run, so for a remote agent you give the policy to the server.

```python
from agentparity import AllowList
from agentparity.a2a import serve

serve(adapter, port=9000, policy_factory=lambda: AllowList(frozenset({"search", "lookup"})))
```

**Remote agents are adapters too.** `await RemoteA2AAdapter.connect(url)` gives you an adapter for any A2A agent, in any language and from any vendor, so the runtime and the suite work on remote agents as well.

## Cross-framework example

[`examples/cross_framework.py`](examples/cross_framework.py) has a LangGraph agent delegate to an ADK agent over A2A, with a server-side policy:

```
deny=False: 'the specialist says: value-of-alpha' -> completed
deny=True: "the specialist says: denied by policy: tool 'lookup' is denied" -> completed
```

## Adding a framework

Write an `Adapter` and a conformance *fixture*: a class that builds agents with seven trivial behaviours (echo, stream, remember, ask, slow, tool, fail). Then run `agentparity conform your.module:YourFixture`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/design.md](docs/design.md).

## Status

Alpha. The 0.x API may change between minor versions, so pin exactly. The suite runs offline. A live mode against real models is on the [roadmap](docs/design.md#roadmap).

## License

Apache-2.0. Copyright 2026 Leap AI Technologies, Inc.
