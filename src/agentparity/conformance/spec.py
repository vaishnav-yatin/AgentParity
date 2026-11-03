"""What a conformance fixture must build.

The suite cannot know how to make an agent in your framework do a specific
thing, so each adapter ships a *fixture* that builds agents with the
behaviours below. Behaviours are deliberately trivial so a deterministic
(model-free) agent can exhibit them; a live fixture may use a real model.

A fixture returns ``None`` for a behaviour it cannot build. If the adapter
*claims* the capability that behaviour proves, ``None`` is a failure: an
unverifiable claim is treated as a false one.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..core.adapter import Adapter


class Behaviour(str, enum.Enum):
    ECHO = "echo"
    """Reply with exactly ``"echo: " + <input text>`` and complete."""

    STREAM = "stream"
    """Reply ``"one two three"`` as at least three separate content events."""

    REMEMBER = "remember"
    """Reply with the previous user message in this context, or ``"<none>"``."""

    ASK = "ask"
    """Turn 1: end INPUT_REQUIRED asking for a colour.
    Turn 2 (resuming the same task): complete with ``"colour: " + <reply>``."""

    SLOW = "slow"
    """Emit a content event, then keep working: call ``probe.tick()`` at least
    every 50 ms for 30 s, emitting nothing further, until cancelled."""

    TOOL = "tool"
    """Call tool ``lookup`` with ``{"key": <input text>}`` through the
    framework's own tool mechanism, and reply with its result. The tool body
    must be ``probe.lookup``. If the tool is denied, do not call it."""

    FAIL = "fail"
    """Raise an exception from inside the framework's run."""


@dataclass
class Probe:
    """Shared with fixtures so the suite can observe what really executed."""

    ticks: int = 0
    lookups: list[tuple[float, str]] = field(default_factory=list)

    def tick(self) -> None:
        self.ticks += 1

    def lookup(self, key: str) -> str:
        self.lookups.append((time.monotonic(), key))
        return f"value-of-{key}"


class Fixture(Protocol):
    """Builds agents for one framework. Return a *fresh* adapter per call."""

    framework: str

    def build(self, behaviour: Behaviour, probe: Probe) -> Adapter | None: ...
