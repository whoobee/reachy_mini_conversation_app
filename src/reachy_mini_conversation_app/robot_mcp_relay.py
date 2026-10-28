"""Always-on relay in front of the app's robot MCP server.

The conversation app's MCP server stops whenever Reachy goes to sleep. MCP clients such as
Hermes then give up and only retry every few minutes, so after waking up the robot tools
stay missing for a while. This relay runs as its own service, keeps the public port open,
and forwards every request to the app when it is running. While the app is down it keeps
offering the last known tool list and answers calls with a clear "Reachy is asleep" error,
so clients never disconnect.

Run: ``python -m reachy_mini_conversation_app.robot_mcp_relay --env /path/to/app/.env``
(``ROBOT_MCP_TOKEN`` authenticates both sides; listens on ``ROBOT_MCP_RELAY_PORT``, default
8766, and forwards to the app on ``ROBOT_MCP_PORT``, default 8767).
"""

from __future__ import annotations
import json
import asyncio
import logging
import argparse
from typing import Any, Final
from pathlib import Path
from contextlib import AsyncExitStack

import uvicorn
from mcp import types
from dotenv import dotenv_values

from reachy_mini_conversation_app.robot_mcp import build_mcp_app


logger = logging.getLogger(__name__)

ASLEEP_MESSAGE: Final[str] = (
    "Reachy is asleep or its conversation app is not running, so robot tools are unavailable. "
    "Wake Reachy (move an antenna) and try again in a moment."
)


class RobotMcpRelay:
    """Forward MCP requests to the app; keep a cached tool list while the app is down."""

    def __init__(self, upstream_url: str, token: str, cache_path: Path) -> None:
        """Relay to ``upstream_url`` (the app's MCP endpoint), authenticating with ``token``."""
        self._upstream_url = upstream_url
        self._token = token
        self._cache_path = cache_path
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._lock = asyncio.Lock()
        self._cached_tools = self._load_cache()

    def _load_cache(self) -> list[types.Tool]:
        try:
            return [types.Tool(**item) for item in json.loads(self._cache_path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            return []

    def _save_cache(self, tools: list[types.Tool]) -> None:
        data = [tool.model_dump(exclude_none=True) for tool in tools]
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.warning("Could not save the tool list cache: %s", e)

    async def _upstream(self) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        if self._session is None:
            stack = AsyncExitStack()
            try:
                headers = {"Authorization": f"Bearer {self._token}"}
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(self._upstream_url, headers=headers, timeout=5)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
            except BaseException:
                await stack.aclose()
                raise
            self._stack, self._session = stack, session
        return self._session

    async def _drop(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException as e:  # a dead transport may not close cleanly
                logger.debug("Closing upstream session failed: %s", e)

    async def _forward(self, method: str, *args: Any) -> Any:
        """Call the app, reconnecting once; raises when the app is not reachable."""
        async with self._lock:
            for attempt in (1, 2):
                try:
                    return await getattr(await self._upstream(), method)(*args)
                except Exception:
                    await self._drop()
                    if attempt == 2:
                        raise
        raise RuntimeError("unreachable")

    async def list_tools(self) -> list[types.Tool]:
        """Return the app's tools, or the last known list while the app is down."""
        try:
            tools = list((await self._forward("list_tools")).tools)
        except Exception as e:
            logger.info("App not reachable (%s); serving %d cached tools", type(e).__name__, len(self._cached_tools))
            return self._cached_tools
        if tools != self._cached_tools:
            self._cached_tools = tools
            self._save_cache(tools)
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """Forward a tool call, or explain that Reachy is asleep."""
        try:
            result: types.CallToolResult = await self._forward("call_tool", name, arguments)
            return result
        except Exception as e:
            logger.info("Tool %s while the app is down: %s", name, type(e).__name__)
            content: list[types.ContentBlock] = [
                types.TextContent(type="text", text=json.dumps({"error": ASLEEP_MESSAGE}))
            ]
            return types.CallToolResult(content=content, isError=True)


def main() -> None:
    """Run the relay as a standalone service."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", type=Path, required=True, help="The conversation app's .env file")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    env = dotenv_values(args.env)
    token = (env.get("ROBOT_MCP_TOKEN") or "").strip()
    if not token:
        raise SystemExit("ROBOT_MCP_TOKEN is not set in " + str(args.env))
    listen_port = int(env.get("ROBOT_MCP_RELAY_PORT") or 8766)
    app_port = int(env.get("ROBOT_MCP_PORT") or 8767)
    if listen_port == app_port:
        raise SystemExit("ROBOT_MCP_RELAY_PORT and ROBOT_MCP_PORT must differ")
    relay = RobotMcpRelay(
        f"http://127.0.0.1:{app_port}/mcp",
        token,
        Path.home() / ".cache" / "reachy_mini_conversation_app" / "robot_tools.json",
    )
    app = build_mcp_app("reachy-robot", token, relay.list_tools, relay.call_tool)
    logger.info("Robot MCP relay on %s:%d -> app on 127.0.0.1:%d", args.host, listen_port, app_port)
    uvicorn.run(app, host=args.host, port=listen_port, log_level="warning")


if __name__ == "__main__":
    main()
