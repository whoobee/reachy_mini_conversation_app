import socket
import asyncio
from typing import Any

import pytest
import uvicorn
from mcp import types

from reachy_mini_conversation_app.robot_mcp import build_mcp_app
from reachy_mini_conversation_app.persistent_mcp import PersistentMcpClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    """A real MCP server on localhost that can be stopped and started again."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.calls: list[str] = []
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        async def list_tools() -> list[types.Tool]:
            return [types.Tool(name="ping", inputSchema={"type": "object"})]

        async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            self.calls.append(name)
            return types.CallToolResult(content=[types.TextContent(type="text", text="pong")], isError=False)

        app = build_mcp_app("test", "tok", list_tools, call_tool)
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.02)

    async def stop(self) -> None:
        assert self._server is not None and self._task is not None
        self._server.should_exit = True
        await self._task


@pytest.mark.asyncio
async def test_survives_the_server_going_away_and_coming_back() -> None:
    """Calls fail cleanly while the server is down and work again after it restarts, once each."""
    server = _Server(_free_port())
    await server.start()
    client = PersistentMcpClient(f"http://127.0.0.1:{server.port}/mcp", headers={"Authorization": "Bearer tok"})
    try:
        assert [t.name for t in (await client.request("list_tools", idempotent=True)).tools] == ["ping"]
        await server.stop()

        with pytest.raises(Exception):
            await asyncio.wait_for(client.request("call_tool", "ping", {}), timeout=10)
        with pytest.raises(Exception):
            await asyncio.wait_for(client.request("list_tools", idempotent=True), timeout=10)

        await server.start()
        result = await asyncio.wait_for(client.request("call_tool", "ping", {}), timeout=10)
        assert result.content[0].text == "pong"
        assert server.calls == ["ping"]
    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_parallel_calls_share_one_connection() -> None:
    """Concurrent tool calls are not serialized behind each other or duplicated."""
    server = _Server(_free_port())
    await server.start()
    client = PersistentMcpClient(f"http://127.0.0.1:{server.port}/mcp", headers={"Authorization": "Bearer tok"})
    try:
        results = await asyncio.gather(*(client.request("call_tool", "ping", {}) for _ in range(5)))
        assert all(r.content[0].text == "pong" for r in results)
        assert len(server.calls) == 5
    finally:
        await client.close()
        await server.stop()
