"""Command line: ``agentparity conform`` and ``agentparity serve``."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys

BUILTIN_FIXTURES = {
    "reference": "agentparity.conformance.reference:ReferenceFixture",
    "langgraph": "agentparity.conformance.fixtures.langgraph:LangGraphFixture",
    "adk": "agentparity.conformance.fixtures.adk:ADKFixture",
    "openai-agents": "agentparity.conformance.fixtures.openai_agents:OpenAIAgentsFixture",
    "claude-agent-sdk": "agentparity.conformance.fixtures.claude_agent_sdk:ClaudeAgentSDKFixture",
    "crewai": "agentparity.conformance.fixtures.crewai:CrewAIFixture",
    "agent-framework": "agentparity.conformance.fixtures.agent_framework:AgentFrameworkFixture",
}


def _load(spec: str):
    spec = BUILTIN_FIXTURES.get(spec, spec)
    if ":" not in spec:
        raise SystemExit(f"expected module:attribute or one of {', '.join(BUILTIN_FIXTURES)}; got {spec!r}")
    mod, attr = spec.split(":", 1)
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _conform(args) -> int:
    from .conformance import run_suite

    fixture = _load(args.fixture)()
    report = asyncio.run(run_suite(fixture, timeout=args.timeout))
    if args.json:
        print(
            json.dumps(
                {
                    "framework": report.framework,
                    "claimed": report.claimed,
                    "ok": report.ok,
                    "results": [
                        {"check": r.check, "capability": r.capability, "status": r.status.value, "detail": r.detail}
                        for r in report.results
                    ],
                },
                indent=2,
            )
        )
    else:
        print(report.render())
    return 0 if report.ok else 1


def _serve(args) -> int:
    from .a2a import serve

    factory = _load(args.adapter)
    adapter = factory() if callable(factory) and not hasattr(factory, "stream") else factory
    serve(adapter, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentparity")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("conform", help="run the conformance suite against a fixture")
    c.add_argument("fixture", help=f"module:Class, or one of: {', '.join(BUILTIN_FIXTURES)}")
    c.add_argument("--timeout", type=float, default=10.0, help="seconds per run (default 10)")
    c.add_argument("--json", action="store_true", help="machine-readable report")
    c.set_defaults(fn=_conform)

    s = sub.add_parser("serve", help="serve an adapter over A2A")
    s.add_argument("adapter", help="module:attribute - an Adapter, or a zero-arg factory returning one")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9000)
    s.set_defaults(fn=_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
