"""Canonical types <-> a2a-sdk v1 protobuf types."""

from __future__ import annotations

from google.protobuf import json_format, struct_pb2

from a2a.types import a2a_pb2 as pb

from ..core.types import (
    AgentInfo,
    Capabilities,
    DataPart,
    FilePart,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
)

_STATE_TO_PB = {
    TaskState.SUBMITTED: pb.TASK_STATE_SUBMITTED,
    TaskState.WORKING: pb.TASK_STATE_WORKING,
    TaskState.INPUT_REQUIRED: pb.TASK_STATE_INPUT_REQUIRED,
    TaskState.AUTH_REQUIRED: pb.TASK_STATE_AUTH_REQUIRED,
    TaskState.COMPLETED: pb.TASK_STATE_COMPLETED,
    TaskState.FAILED: pb.TASK_STATE_FAILED,
    TaskState.CANCELED: pb.TASK_STATE_CANCELED,
    TaskState.REJECTED: pb.TASK_STATE_REJECTED,
}
_PB_TO_STATE = {v: k for k, v in _STATE_TO_PB.items()}


def state_to_pb(state: TaskState) -> int:
    return _STATE_TO_PB[state]


def state_from_pb(value: int) -> TaskState:
    try:
        return _PB_TO_STATE[value]
    except KeyError:
        raise ValueError(f"unknown or unspecified A2A task state {value}") from None


def part_to_pb(part: Part) -> pb.Part:
    if isinstance(part, TextPart):
        return pb.Part(text=part.text)
    if isinstance(part, DataPart):
        value = struct_pb2.Value()
        json_format.ParseDict(part.data, value)
        return pb.Part(data=value)
    if isinstance(part, FilePart):
        kw = {}
        if part.filename:
            kw["filename"] = part.filename
        if part.media_type:
            kw["media_type"] = part.media_type
        if part.url is not None:
            return pb.Part(url=part.url, **kw)
        return pb.Part(raw=part.raw, **kw)
    raise TypeError(f"unsupported part {type(part).__name__}")


def part_from_pb(p: pb.Part) -> Part:
    kind = p.WhichOneof("content")
    if kind == "text":
        return TextPart(p.text)
    if kind == "data":
        data = json_format.MessageToDict(p.data)
        return DataPart(data if isinstance(data, dict) else {"value": data})
    if kind == "url":
        return FilePart(url=p.url, filename=p.filename or None, media_type=p.media_type or None)
    if kind == "raw":
        return FilePart(raw=p.raw, filename=p.filename or None, media_type=p.media_type or None)
    raise ValueError(f"empty or unknown A2A part ({kind})")


def message_from_pb(m: pb.Message) -> Message:
    role = Role.AGENT if m.role == pb.ROLE_AGENT else Role.USER
    return Message(role, tuple(part_from_pb(p) for p in m.parts))


def agent_card(
    info: AgentInfo, caps: Capabilities, url: str, *, framework: str = ""
) -> pb.AgentCard:
    """Build an Agent Card that advertises only what the adapter claims."""
    skills = [
        pb.AgentSkill(id=s.id, name=s.name, description=s.description or s.name, tags=list(s.tags) or ["general"])
        for s in info.skills
    ] or [pb.AgentSkill(id="default", name=info.name, description=info.description or info.name, tags=["general"])]
    description = info.description or info.name
    return pb.AgentCard(
        name=info.name,
        description=description,
        version=info.version,
        supported_interfaces=[pb.AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=pb.AgentCapabilities(streaming=caps.streaming, push_notifications=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=skills,
    )
