import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

import reachy_mini_conversation_app.robot_mcp as rm_mod
from reachy_mini_conversation_app.robot_mcp import RobotMcpServer, parse_tool_names
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _spec(name: str) -> dict[str, Any]:
    return {"type": "function", "name": name, "description": f"{name} tool", "parameters": {"type": "object"}}


def _server(monkeypatch: Any, calls: list[tuple[str, str]]) -> RobotMcpServer:
    monkeypatch.setattr(rm_mod, "get_tool_specs", lambda: [_spec("volume_control"), _spec("go_to_sleep")])

    async def fake_dispatch(name: str, args_json: str, deps: Any) -> dict[str, Any]:
        calls.append((name, args_json))
        return {"status": "ok"}

    monkeypatch.setattr(rm_mod, "dispatch_tool_call", fake_dispatch)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return RobotMcpServer(deps, ("volume_control",), "127.0.0.1", 0, "secret")


def test_parse_tool_names_defaults_and_custom() -> None:
    """An empty setting means the safe default set; otherwise the listed names."""
    assert "go_to_sleep" not in parse_tool_names(None)
    assert "volume_control" in parse_tool_names("")
    assert parse_tool_names(" dance , play_emotion ") == ("dance", "play_emotion")


@pytest.mark.asyncio
async def test_only_allowed_tools_are_listed_and_callable(monkeypatch: Any) -> None:
    """Tools outside the allowlist are neither offered nor executed."""
    calls: list[tuple[str, str]] = []
    server = _server(monkeypatch, calls)

    assert [tool.name for tool in server.tool_definitions()] == ["volume_control"]
    assert await server.call("go_to_sleep", {}) == {"error": "tool not available: go_to_sleep"}
    assert await server.call("volume_control", {"level": 80}) == {"status": "ok"}
    assert calls == [("volume_control", '{"level": 80}')]


def test_mcp_endpoint_requires_the_bearer_token_and_runs_tools(monkeypatch: Any) -> None:
    """Without the token requests are rejected; with it an MCP tools/call reaches the robot tool."""
    calls: list[tuple[str, str]] = []
    server = _server(monkeypatch, calls)
    headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "volume_control", "arguments": {"level": 50}},
    }

    with TestClient(server.build_app()) as client:
        assert client.post("/mcp", json=request, headers=headers).status_code == 401
        response = client.post("/mcp", json=request, headers={**headers, "authorization": "Bearer secret"})

    assert response.status_code == 200
    result = response.json()["result"]
    assert json.loads(result["content"][0]["text"]) == {"status": "ok"}
    assert calls == [("volume_control", '{"level": 50}')]
