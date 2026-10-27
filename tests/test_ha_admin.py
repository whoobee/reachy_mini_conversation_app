from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini_conversation_app.ha_admin import HaAdmin, EngineeringMode


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _tool(name: str, description: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, inputSchema={"type": "object"})


FULL = [
    _tool("ha_get_logs", "Read Home Assistant logs and errors."),
    _tool("ha_get_automation_traces", "Get automation run traces."),
    _tool("ha_config_set_automation", "Create or update an automation."),
    _tool("ha_restart", "Restart Home Assistant."),
]
READ_ONLY = FULL[:2]


def _admin(monkeypatch: Any, calls: list[tuple[str, str, tuple[Any, ...]]]) -> tuple[HaAdmin, _Clock]:
    clock = _Clock()
    admin = HaAdmin("http://ha-mcp:8086/private_x/", EngineeringMode(15, clock=clock))

    async def fake_call(url: str, method: str, *args: Any) -> Any:
        calls.append((url, method, args))
        if method == "list_tools":
            return SimpleNamespace(tools=READ_ONLY if url.endswith("/readonly") else FULL)
        return SimpleNamespace(content=[SimpleNamespace(text="done")], isError=False)

    monkeypatch.setattr(admin, "_session_call", fake_call)
    return admin, clock


@pytest.mark.asyncio
async def test_find_tools_ranks_by_keywords_and_marks_read_or_write(monkeypatch: Any) -> None:
    """Search returns matching tools with their read/write kind."""
    admin, _ = _admin(monkeypatch, [])

    found = (await admin.find_tools({"query": "automation trace"}))["tools"]

    assert found[0]["name"] == "ha_get_automation_traces"
    assert {t["name"]: t["kind"] for t in found}["ha_config_set_automation"] == "write"


@pytest.mark.asyncio
async def test_reads_always_run_through_the_readonly_endpoint(monkeypatch: Any) -> None:
    """Read tools run without engineering mode, and only via ha-mcp's /readonly endpoint."""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    admin, _ = _admin(monkeypatch, calls)

    result = await admin.run_tool({"name": "ha_get_logs", "arguments": {"limit": 5}})

    assert result == {"tool": "ha_get_logs", "kind": "read", "is_error": False, "result": "done"}
    assert calls[-1] == ("http://ha-mcp:8086/private_x/readonly", "call_tool", ("ha_get_logs", {"limit": 5}))


@pytest.mark.asyncio
async def test_writes_need_engineering_mode_which_times_out(monkeypatch: Any) -> None:
    """Write tools are refused when off, allowed (full endpoint) while on, and refused again after the timeout."""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    admin, clock = _admin(monkeypatch, calls)
    change = {"name": "ha_config_set_automation", "arguments": {"config": {"alias": "x"}}}

    assert "engineering mode is off" in (await admin.run_tool(change))["error"]
    assert (await admin.engineering_mode({"enabled": True, "minutes": 10}))["seconds_left"] == 600
    assert (await admin.run_tool(change))["kind"] == "write"
    assert calls[-1][0] == "http://ha-mcp:8086/private_x"

    clock.now += 601
    assert admin.mode.expired_since_check() is True
    assert admin.mode.expired_since_check() is False  # announced only once
    assert "engineering mode is off" in (await admin.run_tool(change))["error"]


@pytest.mark.asyncio
async def test_engineering_mode_default_timeout_status_and_manual_off(monkeypatch: Any) -> None:
    """Without minutes the configured default applies; it can be read and turned off early."""
    admin, _ = _admin(monkeypatch, [])

    assert await admin.engineering_mode({}) == {"engineering_mode": False, "seconds_left": 0}
    assert (await admin.engineering_mode({"enabled": True}))["seconds_left"] == 15 * 60
    assert (await admin.engineering_mode({"enabled": False}))["engineering_mode"] is False
    assert admin.mode.expired_since_check() is False  # turned off by hand: nothing to announce


@pytest.mark.asyncio
async def test_unknown_tools_and_bad_arguments_are_rejected(monkeypatch: Any) -> None:
    """Unknown names and non-JSON arguments give clear errors without calling ha-mcp."""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    admin, _ = _admin(monkeypatch, calls)

    assert "unknown tool" in (await admin.run_tool({"name": "ha_nope"}))["error"]
    assert "JSON" in (await admin.run_tool({"name": "ha_get_logs", "arguments": "{bad"}))["error"]
    assert "unknown tool" in (await admin.describe_tool({"name": "ha_nope"}))["error"]
    assert all(method == "list_tools" for _, method, _ in calls)


@pytest.mark.asyncio
async def test_device_control_is_allowed_but_automations_and_protected_devices_are_gated(monkeypatch: Any) -> None:
    """Lights and blinds run always; automations, restarts and protected devices need engineering mode."""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    admin, _ = _admin(monkeypatch, calls)

    light = await admin.call_service(
        {"domain": "light", "service": "turn_on", "entity_id": "light.office", "data": {"brightness_pct": 40}}
    )
    assert light["is_error"] is False
    assert calls[-1] == (
        "http://ha-mcp:8086/private_x",
        "call_tool",
        (
            "ha_call_service",
            {"domain": "light", "service": "turn_on", "data": {"brightness_pct": 40}, "entity_id": "light.office"},
        ),
    )

    blocked = [
        {"domain": "automation", "service": "turn_off", "entity_id": "automation.tec_mainlighton"},  # the slip we saw
        {"domain": "homeassistant", "service": "restart", "entity_id": "homeassistant"},
        {"domain": "switch", "service": "turn_off", "entity_id": "switch.valve_mainwatervalve_switch_1"},
        {"domain": "light", "service": "turn_off", "entity_id": ""},  # no target: could hit every light
    ]
    count = len(calls)
    for request in blocked:
        assert "engineering mode is off" in (await admin.call_service(request))["error"]
    assert len(calls) == count  # nothing reached Home Assistant

    await admin.engineering_mode({"enabled": True})
    assert (await admin.call_service(blocked[0]))["is_error"] is False


@pytest.mark.asyncio
async def test_safe_call_service_never_forwards_raw_websocket_commands(monkeypatch: Any) -> None:
    """Only domain/service/entity/data are forwarded; ha-mcp's ws_command escape hatch is never passed through."""
    calls: list[tuple[str, str, tuple[Any, ...]]] = []
    admin, _ = _admin(monkeypatch, calls)

    await admin.call_service(
        {"domain": "light", "service": "turn_on", "entity_id": "light.a", "ws_command": "config/auth/delete"}
    )

    assert "ws_command" not in calls[-1][2][1]


def test_protected_pattern_is_configurable(monkeypatch: Any) -> None:
    """A custom regex replaces the default list of protected devices."""
    admin = HaAdmin("http://x", EngineeringMode(15), protected_pattern="aquarium")
    assert admin.service_blocked_reason("switch", "turn_off", ["switch.aquarium_pump"]) is not None
    assert admin.service_blocked_reason("switch", "turn_off", ["switch.valve_main"]) is None


@pytest.mark.asyncio
async def test_session_is_reused_and_reopened_after_a_dropped_connection(monkeypatch: Any) -> None:
    """One session per endpoint is kept open; a failed call reconnects once and retries."""
    admin = HaAdmin("http://ha-mcp/x", EngineeringMode(15))
    opened: list[str] = []
    attempts = {"n": 0}

    class _Session:
        async def list_tools(self) -> Any:
            attempts["n"] += 1
            if attempts["n"] == 2:
                raise ConnectionError("dropped")
            return SimpleNamespace(tools=[])

    async def fake_session(url: str) -> Any:
        if url not in admin._sessions:
            opened.append(url)
            admin._sessions[url] = (SimpleNamespace(aclose=_noop), _Session())
        return admin._sessions[url][1]

    monkeypatch.setattr(admin, "_session", fake_session)
    await admin._session_call("http://ha-mcp/x", "list_tools")
    await admin._session_call("http://ha-mcp/x", "list_tools")  # first try fails, reconnects, retries

    assert opened == ["http://ha-mcp/x", "http://ha-mcp/x"] and attempts["n"] == 3


async def _noop() -> None:
    return None
