import json
from typing import Any
from pathlib import Path

import pytest
from mcp import types

from reachy_mini_conversation_app.robot_mcp_relay import ASLEEP_MESSAGE, RobotMcpRelay


TOOLS = [types.Tool(name="get_weather", description="Weather", inputSchema={"type": "object"})]


def _relay(tmp_path: Path, monkeypatch: Any, app_up: dict[str, bool]) -> RobotMcpRelay:
    relay = RobotMcpRelay("http://127.0.0.1:8767/mcp", "secret", tmp_path / "tools.json")

    async def forward(method: str, *args: Any) -> Any:
        if not app_up["value"]:
            raise ConnectionError("app down")
        if method == "list_tools":
            return types.ListToolsResult(tools=TOOLS)
        content: list[types.ContentBlock] = [types.TextContent(type="text", text='{"ok": true}')]
        return types.CallToolResult(content=content, isError=False)

    monkeypatch.setattr(relay, "_forward", forward)
    return relay


@pytest.mark.asyncio
async def test_forwards_while_the_app_runs_and_caches_the_tool_list(tmp_path: Path, monkeypatch: Any) -> None:
    """With the app up, tools and calls pass through and the tool list is saved."""
    relay = _relay(tmp_path, monkeypatch, {"value": True})

    assert [t.name for t in await relay.list_tools()] == ["get_weather"]
    assert (await relay.call_tool("get_weather", {})).content[0].text == '{"ok": true}'  # type: ignore[union-attr]
    assert json.loads((tmp_path / "tools.json").read_text())[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_keeps_offering_tools_and_explains_while_reachy_sleeps(tmp_path: Path, monkeypatch: Any) -> None:
    """With the app down, the cached tool list stays and calls return a clear asleep error."""
    app_up = {"value": True}
    await _relay(tmp_path, monkeypatch, app_up).list_tools()  # cache filled by an earlier run
    app_up["value"] = False
    relay = _relay(tmp_path, monkeypatch, app_up)  # e.g. relay restarted while Reachy sleeps

    assert [t.name for t in await relay.list_tools()] == ["get_weather"]
    result = await relay.call_tool("get_weather", {})
    assert result.isError is True
    assert json.loads(result.content[0].text) == {"error": ASLEEP_MESSAGE}  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_no_cache_and_app_down_gives_an_empty_list(tmp_path: Path, monkeypatch: Any) -> None:
    """A first start while Reachy sleeps simply offers no tools until the app is up."""
    relay = _relay(tmp_path, monkeypatch, {"value": False})
    assert await relay.list_tools() == []
