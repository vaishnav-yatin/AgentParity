"""Governance hooks: the same policy applied whatever framework runs the agent.

Each adapter wires :meth:`RunContext.authorize_tool` into its framework's
native pre-tool hook (LangGraph tool wrapper, OpenAI Agents tool guardrail,
Claude Agent SDK ``can_use_tool``, ADK ``before_tool_callback``, CrewAI
tool hooks, Agent Framework function middleware). The policy itself is
written once, here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class Decision(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


ALLOW = Verdict(Decision.ALLOW)


@runtime_checkable
class ToolPolicy(Protocol):
    def check(self, name: str, arguments: dict[str, Any]) -> Verdict: ...


class AllowAll:
    def check(self, name: str, arguments: dict[str, Any]) -> Verdict:
        return ALLOW


@dataclass
class AllowList:
    """Allow only named tools. Unknown tools are denied (fail closed)."""

    names: frozenset[str]

    def check(self, name: str, arguments: dict[str, Any]) -> Verdict:
        if name in self.names:
            return ALLOW
        return Verdict(Decision.DENY, f"tool {name!r} is not on the allow list")


@dataclass
class DenyList:
    names: frozenset[str]

    def check(self, name: str, arguments: dict[str, Any]) -> Verdict:
        if name in self.names:
            return Verdict(Decision.DENY, f"tool {name!r} is denied")
        return ALLOW


@dataclass
class ToolBudget:
    """Deny once a run has made ``max_calls`` tool calls. Wraps another policy."""

    max_calls: int
    inner: ToolPolicy = field(default_factory=AllowAll)
    _used: int = 0

    def check(self, name: str, arguments: dict[str, Any]) -> Verdict:
        if self._used >= self.max_calls:
            return Verdict(Decision.DENY, f"tool budget of {self.max_calls} exhausted")
        v = self.inner.check(name, arguments)
        if v.allowed:
            self._used += 1
        return v


class ToolDenied(Exception):
    """Raised by adapters inside the framework's tool hook when policy denies."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")
        self.name = name
        self.reason = reason
