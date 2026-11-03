# Changelog

Versioning is `<major>.<minor>.<patch>`. Every code change bumps patch, and a feature or major fix bumps minor. Existing tags are never moved.

## 0.1.0 — 2026-11-03

First release.

- Core: canonical types mapped one-to-one onto A2A v1.0, the adapter contract, `Runtime` lifecycle guarantees with violation recording, and tool governance policies (`AllowAll`, `AllowList`, `DenyList`, `ToolBudget`).
- A2A bridge on `a2a-sdk` 1.x: `build_app`/`serve` (agent card + JSON-RPC) and `RemoteA2AAdapter`.
- Conformance suite: 8 checks across lifecycle, streaming, multi-turn, input-required, cancellation and tool governance. Unverifiable and unchecked claims count as failures. The suite is tested against 9 broken variants.
- Adapters and offline fixtures: LangGraph, Google ADK, OpenAI Agents SDK, Claude Agent SDK (scripted CLI transport), CrewAI (cancellation not claimed), Microsoft Agent Framework.
- CLI: `agentparity conform <fixture>` and `agentparity serve <module:adapter>`.
