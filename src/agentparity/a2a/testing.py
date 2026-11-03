"""Run an adapter on a real localhost socket for tests.

httpx's in-process ASGI transport buffers the whole response, so a
streaming or cancellable run never behaves as it would on the wire. Tests
that care about timing (streaming, cancellation) need a real server.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from typing import AsyncIterator

from ..core.adapter import Adapter
from .server import build_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.asynccontextmanager
async def running_server(adapter: Adapter, **build_kw) -> AsyncIterator[str]:
    """Serve ``adapter`` on 127.0.0.1:<free port>; yields the base URL."""
    import uvicorn

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    app = build_app(adapter, url=url + "/", **build_kw)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("test server did not start")
        yield url
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, timeout=5)
