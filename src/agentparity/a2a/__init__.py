"""A2A v1.0 bridge (requires the ``a2a`` extra)."""

from .client import RemoteA2AAdapter
from .server import AdapterExecutor, build_app, serve

__all__ = ["AdapterExecutor", "RemoteA2AAdapter", "build_app", "serve"]
