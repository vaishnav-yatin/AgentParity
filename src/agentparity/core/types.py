"""Canonical types.

These map one-to-one onto A2A v1.0 concepts so nothing is lost when an
adapter's output crosses the wire. Framework adapters translate *into*
these types; the A2A bridge translates *out of* them. Nothing in this
module imports a framework or a networking library.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Union


def new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- parts


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class DataPart:
    data: dict[str, Any]


@dataclass(frozen=True)
class FilePart:
    """A file by URL or by inline bytes (exactly one of the two)."""

    url: str | None = None
    raw: bytes | None = None
    filename: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if (self.url is None) == (self.raw is None):
            raise ValueError("FilePart needs exactly one of url or raw")


Part = Union[TextPart, DataPart, FilePart]


class Role(str, enum.Enum):
    USER = "user"
    AGENT = "agent"


@dataclass(frozen=True)
class Message:
    role: Role
    parts: tuple[Part, ...]

    @staticmethod
    def user(text: str) -> "Message":
        return Message(Role.USER, (TextPart(text),))

    @staticmethod
    def agent(text: str) -> "Message":
        return Message(Role.AGENT, (TextPart(text),))

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))


# --------------------------------------------------------------------------- state


class TaskState(str, enum.Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def interrupted(self) -> bool:
        """The run yields control but the task is not over."""
        return self in _INTERRUPTED

    @property
    def final_for_run(self) -> bool:
        """No further events may follow in this run."""
        return self.terminal or self.interrupted


_TERMINAL = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED, TaskState.REJECTED}
)
_INTERRUPTED = frozenset({TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED})


# --------------------------------------------------------------------------- events


@dataclass(frozen=True)
class StatusUpdate:
    state: TaskState
    message: Message | None = None


@dataclass(frozen=True)
class ArtifactChunk:
    """A piece of an artifact.

    ``append=False`` starts (or replaces) the artifact; ``append=True`` adds
    parts to one already started in this task. ``last_chunk=True`` closes it.
    """

    artifact_id: str
    parts: tuple[Part, ...]
    name: str | None = None
    append: bool = False
    last_chunk: bool = False


@dataclass(frozen=True)
class ToolCall:
    """Observability event: a tool was invoked. Never sent as content.

    Adapters emit this *after* the governance hook has allowed the call, so
    a trace always pairs an authorization with an invocation.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    ok: bool
    summary: str = ""


@dataclass(frozen=True)
class Usage:
    """Token accounting reported by the framework, when it reports any."""

    input_tokens: int = 0
    output_tokens: int = 0


Event = Union[StatusUpdate, ArtifactChunk, ToolCall, ToolResult, Usage]


# --------------------------------------------------------------------------- request


@dataclass(frozen=True)
class RunRequest:
    """One inbound turn.

    ``context_id`` groups turns of one conversation (A2A contextId); adapters
    map it to the framework's own session/thread id. ``task_id`` identifies
    the task; a resumed input-required task keeps its task_id.
    """

    message: Message
    context_id: str = field(default_factory=new_id)
    task_id: str = field(default_factory=new_id)
    resuming: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    """What an adapter claims. The conformance suite holds it to every claim."""

    streaming: bool = False
    multi_turn: bool = False
    input_required: bool = False
    cancellation: bool = False
    tool_governance: bool = False
    usage: bool = False

    def claimed(self) -> list[str]:
        return [k for k, v in self.__dict__.items() if v]


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentInfo:
    name: str
    description: str = ""
    version: str = "0.0.0"
    skills: tuple[Skill, ...] = ()
