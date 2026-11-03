"""The conformance suite.

Each check proves one claim with an observation the adapter cannot fake:
the cancel check watches a probe the agent must keep ticking, so an adapter
that reports CANCELED while its framework keeps running fails; the
governance check compares the authorization timestamp with the moment the
tool body actually ran.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from typing import Callable

from ..core.adapter import Adapter
from ..core.governance import AllowAll, DenyList, ToolPolicy
from ..core.runtime import RunHandle, Runtime
from ..core.types import (
    ArtifactChunk,
    Event,
    Message,
    RunRequest,
    StatusUpdate,
    TaskState,
    TextPart,
    new_id,
)
from .spec import Behaviour, Fixture, Probe


class Status(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # capability not claimed, nothing to prove
    UNVERIFIED = "unverified"  # claimed, but the fixture cannot build the agent


@dataclass
class CheckResult:
    check: str
    capability: str | None
    status: Status
    detail: str = ""


@dataclass
class Report:
    framework: str
    claimed: list[str]
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status in (Status.PASS, Status.SKIP) for r in self.results)

    def render(self) -> str:
        lines = [f"agentparity conformance - {self.framework}", f"claims: {', '.join(self.claimed) or '(none)'}", ""]
        w = max(len(r.check) for r in self.results) if self.results else 10
        for r in self.results:
            lines.append(f"  {r.status.value.upper():<10} {r.check:<{w}}  {r.detail}")
        lines.append("")
        lines.append("RESULT: " + ("CONFORMANT" if self.ok else "NOT CONFORMANT"))
        return "\n".join(lines)


def text_of(events: list[Event]) -> str:
    out = []
    for e in events:
        if isinstance(e, ArtifactChunk):
            out.extend(p.text for p in e.parts if isinstance(p, TextPart))
    return "".join(out)


def final_of(events: list[Event]) -> StatusUpdate | None:
    for e in reversed(events):
        if isinstance(e, StatusUpdate):
            return e
    return None


class Suite:
    def __init__(self, fixture: Fixture, *, timeout: float = 10.0, cancel_grace: float = 2.0) -> None:
        self.fixture = fixture
        self.timeout = timeout
        self.cancel_grace = cancel_grace

    # ------------------------------------------------------------------ helpers

    async def _collect(
        self, adapter: Adapter, request: RunRequest, policy: Callable[[], ToolPolicy] = AllowAll
    ) -> tuple[list[Event], RunHandle]:
        rt = Runtime(adapter, policy_factory=policy, cancel_grace=self.cancel_grace)
        handle = rt.start(request)
        events: list[Event] = []

        async def drain() -> None:
            async for e in handle.events():
                events.append(e)

        try:
            await asyncio.wait_for(drain(), timeout=self.timeout)
        except asyncio.TimeoutError:
            handle.cancel()
            raise
        return events, handle

    def _build(self, b: Behaviour, probe: Probe) -> Adapter | None:
        return self.fixture.build(b, probe)

    # ------------------------------------------------------------------ run

    async def run(self) -> Report:
        probe = Probe()
        sample = self._build(Behaviour.ECHO, probe)
        if sample is None:
            raise ValueError("fixture must build the ECHO behaviour; it is the baseline")
        caps = sample.capabilities
        report = Report(self.fixture.framework, caps.claimed())

        checks: list[tuple[str, str | None, bool, Callable]] = [
            ("lifecycle.echo", None, True, self.check_echo),
            ("lifecycle.failure", None, True, self.check_failure),
            ("streaming", "streaming", caps.streaming, self.check_streaming),
            ("multi_turn", "multi_turn", caps.multi_turn, self.check_multi_turn),
            ("input_required", "input_required", caps.input_required, self.check_input_required),
            ("cancellation", "cancellation", caps.cancellation, self.check_cancellation),
            ("tool_governance.allow", "tool_governance", caps.tool_governance, self.check_tool_allow),
            ("tool_governance.deny", "tool_governance", caps.tool_governance, self.check_tool_deny),
        ]
        # A claim with no check behind it cannot be trusted: report it.
        covered = {cap for _, cap, _, _ in checks if cap}
        for cap in caps.claimed():
            if cap not in covered:
                report.results.append(
                    CheckResult(cap, cap, Status.UNVERIFIED, "claimed, but this suite version has no check for it")
                )

        for name, cap, claimed, fn in checks:
            if cap is not None and not claimed:
                report.results.append(CheckResult(name, cap, Status.SKIP, "not claimed"))
                continue
            try:
                result = await fn()
            except asyncio.TimeoutError:
                result = (Status.FAIL, f"no final status within {self.timeout}s")
            except Exception as exc:  # noqa: BLE001
                result = (Status.FAIL, f"suite error: {type(exc).__name__}: {exc}")
            status, detail = result
            report.results.append(CheckResult(name, cap, status, detail))
        return report

    # ------------------------------------------------------------------ checks

    def _violations(self, h: RunHandle) -> str | None:
        if h.violations:
            return "; ".join(f"{v.rule}: {v.detail}" for v in h.violations)
        return None

    async def check_echo(self):
        probe = Probe()
        a = self._build(Behaviour.ECHO, probe)
        events, h = await self._collect(a, RunRequest(Message.user("ping 42")))
        if bad := self._violations(h):
            return Status.FAIL, bad
        if not isinstance(events[0], StatusUpdate) or events[0].state is not TaskState.WORKING:
            return Status.FAIL, "first event was not WORKING"
        f = final_of(events)
        if f is None or f.state is not TaskState.COMPLETED:
            return Status.FAIL, f"final state {f.state.value if f else None}, expected completed"
        got = text_of(events)
        if got.strip() != "echo: ping 42":
            return Status.FAIL, f"reply {got!r}, expected 'echo: ping 42'"
        return Status.PASS, "round-trip text exact"

    async def check_failure(self):
        a = self._build(Behaviour.FAIL, Probe())
        if a is None:
            return Status.UNVERIFIED, "fixture cannot build FAIL"
        events, h = await self._collect(a, RunRequest(Message.user("x")))
        f = final_of(events)
        if f is None or f.state is not TaskState.FAILED:
            return Status.FAIL, f"framework error surfaced as {f.state.value if f else None}, expected failed"
        return Status.PASS, "framework exception -> FAILED"

    async def check_streaming(self):
        a = self._build(Behaviour.STREAM, Probe())
        if a is None:
            return Status.UNVERIFIED, "claimed but fixture cannot build STREAM"
        events, h = await self._collect(a, RunRequest(Message.user("go")))
        if bad := self._violations(h):
            return Status.FAIL, bad
        chunks = [e for e in events if isinstance(e, ArtifactChunk) and text_of([e]).strip()]
        joined = " ".join(text_of(events).split())
        if joined != "one two three":
            return Status.FAIL, f"streamed text {joined!r}, expected 'one two three'"
        if len(chunks) < 3:
            return Status.FAIL, f"{len(chunks)} content event(s); streaming claimed but output was batched"
        return Status.PASS, f"{len(chunks)} incremental content events"

    async def check_multi_turn(self):
        a = self._build(Behaviour.REMEMBER, Probe())
        if a is None:
            return Status.UNVERIFIED, "claimed but fixture cannot build REMEMBER"
        ctx = new_id()
        e1, h1 = await self._collect(a, RunRequest(Message.user("apple"), context_id=ctx))
        e2, h2 = await self._collect(a, RunRequest(Message.user("what did I say?"), context_id=ctx))
        other, _ = await self._collect(a, RunRequest(Message.user("and here?"), context_id=new_id()))
        for h in (h1, h2):
            if bad := self._violations(h):
                return Status.FAIL, bad
        if "apple" not in text_of(e2):
            return Status.FAIL, f"turn 2 reply {text_of(e2)!r} does not recall turn 1 in the same context"
        if "apple" in text_of(other):
            return Status.FAIL, "a different context_id saw this context's history (sessions leak)"
        return Status.PASS, "context_id carries history; separate contexts isolated"

    async def check_input_required(self):
        a = self._build(Behaviour.ASK, Probe())
        if a is None:
            return Status.UNVERIFIED, "claimed but fixture cannot build ASK"
        ctx, task = new_id(), new_id()
        e1, h1 = await self._collect(a, RunRequest(Message.user("start"), context_id=ctx, task_id=task))
        f1 = final_of(e1)
        if f1 is None or f1.state is not TaskState.INPUT_REQUIRED:
            return Status.FAIL, f"turn 1 ended {f1.state.value if f1 else None}, expected input_required"
        e2, h2 = await self._collect(
            a, RunRequest(Message.user("blue"), context_id=ctx, task_id=task, resuming=True)
        )
        f2 = final_of(e2)
        if f2 is None or f2.state is not TaskState.COMPLETED:
            return Status.FAIL, f"resumed turn ended {f2.state.value if f2 else None}, expected completed"
        if "colour: blue" not in text_of(e2) and "color: blue" not in text_of(e2):
            return Status.FAIL, f"resumed reply {text_of(e2)!r} did not use the answer"
        for h in (h1, h2):
            if bad := self._violations(h):
                return Status.FAIL, bad
        return Status.PASS, "interrupt and resume on the same task"

    async def check_cancellation(self):
        probe = Probe()
        a = self._build(Behaviour.SLOW, probe)
        if a is None:
            return Status.UNVERIFIED, "claimed but fixture cannot build SLOW"
        rt = Runtime(a, cancel_grace=self.cancel_grace)
        h = rt.start(RunRequest(Message.user("work")))
        events: list[Event] = []

        async def drain() -> None:
            async for e in h.events():
                events.append(e)
                if isinstance(e, ArtifactChunk):
                    await asyncio.sleep(0.2)  # let it run, then cancel
                    h.cancel()

        await asyncio.wait_for(drain(), timeout=self.timeout)
        f = final_of(events)
        if f is None or f.state is not TaskState.CANCELED:
            return Status.FAIL, f"final {f.state.value if f else None}, expected canceled"
        if bad := self._violations(h):
            return Status.FAIL, bad
        if probe.ticks == 0:
            return Status.FAIL, "SLOW agent never ticked; cannot tell whether work stopped"
        before = probe.ticks
        await asyncio.sleep(0.3)
        if probe.ticks != before:
            return Status.FAIL, f"reported CANCELED but the agent kept working ({probe.ticks - before} ticks after)"
        return Status.PASS, "canceled and the agent actually stopped"

    async def check_tool_allow(self):
        probe = Probe()
        a = self._build(Behaviour.TOOL, probe)
        if a is None:
            return Status.UNVERIFIED, "claimed but fixture cannot build TOOL"
        events, h = await self._collect(a, RunRequest(Message.user("alpha")))
        if bad := self._violations(h):
            return Status.FAIL, bad
        auths = [r for r in h.ctx.authorizations if r.name == "lookup"]
        if not probe.lookups:
            return Status.FAIL, "tool body never ran"
        if not auths:
            return Status.FAIL, "tool ran without passing through authorize_tool"
        if auths[0].at > probe.lookups[0][0]:
            return Status.FAIL, "authorization happened after the tool ran"
        if auths[0].arguments.get("key") != "alpha":
            return Status.FAIL, f"policy saw arguments {auths[0].arguments!r}, expected key='alpha'"
        if "value-of-alpha" not in text_of(events):
            return Status.FAIL, f"reply {text_of(events)!r} does not carry the tool result"
        return Status.PASS, "policy saw name+arguments before the tool body ran"

    async def check_tool_deny(self):
        probe = Probe()
        a = self._build(Behaviour.TOOL, probe)
        if a is None:
            return Status.UNVERIFIED, "claimed but fixture cannot build TOOL"
        events, h = await self._collect(
            a, RunRequest(Message.user("alpha")), policy=lambda: DenyList(frozenset({"lookup"}))
        )
        if probe.lookups:
            return Status.FAIL, "policy denied the tool but its body ran anyway"
        if not any(r.name == "lookup" and not r.verdict.allowed for r in h.ctx.authorizations):
            return Status.FAIL, "no denied authorization recorded"
        f = final_of(events)
        return Status.PASS, f"denied tool did not run (run ended {f.state.value if f else None})"


async def run_suite(fixture: Fixture, **kw) -> Report:
    return await Suite(fixture, **kw).run()
