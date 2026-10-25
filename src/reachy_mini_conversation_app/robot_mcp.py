"""Expose selected robot tools to a remote agent (e.g. Hermes) as an MCP server over HTTP.

The server runs inside the conversation app, so tool calls use the app's live robot
connection and movement manager. Requests must carry ``Authorization: Bearer <token>``.
"""

from __future__ import annotations
import hmac
import json
import asyncio
import logging
from typing import Any, Final
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import uvicorn
from mcp import types
from starlette.types import Send, Scope, Receive
from starlette.routing import Mount
from mcp.server.lowlevel import Server
from starlette.responses import JSONResponse
from starlette.applications import Starlette
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call


logger = logging.getLogger(__name__)

# Safe for a remote agent: no shutdown (go_to_sleep), no camera, no app-local memory or task plumbing.
DEFAULT_ROBOT_MCP_TOOLS: Final[tuple[str, ...]] = (
    "volume_control",
    "play_emotion",
    "stop_emotion",
    "dance",
    "stop_dance",
    "move_head",
    "sweep_look",
    "head_tracking",
    "robot_status",
)


def parse_tool_names(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated tool list, falling back to the defaults."""
    names = tuple(name.strip() for name in (value or "").split(",") if name.strip())
    return names or DEFAULT_ROBOT_MCP_TOOLS


class RobotMcpServer:
    """Streamable-HTTP MCP server offering a fixed set of robot tools."""

    def __init__(self, deps: ToolDependencies, tool_names: tuple[str, ...], host: str, port: int, token: str) -> None:
        """Prepare the server; call ``serve`` to run it."""
        if not token:
            raise ValueError("RobotMcpServer needs a bearer token")
        self._deps = deps
        self._allowed = frozenset(tool_names)
        self._host = host
        self._port = port
        self._token = token
        self._uvicorn: uvicorn.Server | None = None

    def tool_definitions(self) -> list[types.Tool]:
        """Return the MCP tool definitions for the allowed app tools."""
        return [
            types.Tool(name=spec["name"], description=spec["description"], inputSchema=spec["parameters"])
            for spec in get_tool_specs()
            if spec["name"] in self._allowed
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one allowed robot tool and return its result."""
        if name not in self._allowed:
            return {"error": f"tool not available: {name}"}
        logger.info("Robot MCP tool call: %s %s", name, arguments)
        result = await dispatch_tool_call(name, json.dumps(arguments), self._deps)
        return result if isinstance(result, dict) else {"result": result}

    def _authorized(self, scope: Scope) -> bool:
        headers = dict(scope.get("headers") or [])
        supplied = headers.get(b"authorization", b"").decode()
        return hmac.compare_digest(supplied, f"Bearer {self._token}")

    def build_app(self) -> Starlette:
        """Build the ASGI app serving MCP at any path (normally ``/mcp``)."""
        server: Server[Any, Any] = Server("reachy-robot")

        # Register request handlers directly: the SDK's decorator helpers are untyped.
        async def list_tools(_request: types.ListToolsRequest) -> types.ServerResult:
            return types.ServerResult(types.ListToolsResult(tools=self.tool_definitions()))

        async def call_tool(request: types.CallToolRequest) -> types.ServerResult:
            result = await self.call(request.params.name, dict(request.params.arguments or {}))
            content: list[types.ContentBlock] = [types.TextContent(type="text", text=json.dumps(result))]
            return types.ServerResult(types.CallToolResult(content=content, isError="error" in result))

        server.request_handlers[types.ListToolsRequest] = list_tools
        server.request_handlers[types.CallToolRequest] = call_tool

        session_manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

        async def handle(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http" and not self._authorized(scope):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
            await session_manager.handle_request(scope, receive, send)

        @asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncIterator[None]:
            async with session_manager.run():
                yield

        return Starlette(routes=[Mount("/", app=handle)], lifespan=lifespan)

    async def serve(self) -> None:
        """Serve until ``stop`` is called."""
        config = uvicorn.Config(self.build_app(), host=self._host, port=self._port, log_level="warning", lifespan="on")
        self._uvicorn = uvicorn.Server(config)
        logger.info(
            "Robot MCP server on http://%s:%d/mcp with tools %s", self._host, self._port, sorted(self._allowed)
        )
        await self._uvicorn.serve()

    async def stop(self) -> None:
        """Ask the server to exit."""
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
            await asyncio.sleep(0)
