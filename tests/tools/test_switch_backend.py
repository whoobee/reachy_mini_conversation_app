from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.config import persist_env_value
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.switch_backend import SwitchBackend


@pytest.mark.asyncio
async def test_switch_backend_validates_and_delegates() -> None:
    """Unknown backends are rejected; valid ones are passed to the runtime's switch callback."""
    calls: list[str] = []

    async def switch(backend: str) -> dict[str, Any]:
        calls.append(backend)
        return {"status": "switching"}

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), switch_backend=switch)
    tool = SwitchBackend()

    assert "error" in await tool(deps, backend="openai")
    assert await tool(deps, backend="Hermes") == {"status": "switching"}
    assert calls == ["hermes"]
    no_runtime = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    assert "unavailable" in (await tool(no_runtime, backend="huggingface"))["error"]


def test_persist_env_value_replaces_or_appends(tmp_path: Path, monkeypatch: Any) -> None:
    """The backend choice is written back to .env without touching other lines."""
    monkeypatch.delenv("REALTIME_BACKEND", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nREALTIME_BACKEND=hermes\nOTHER=1\n", encoding="utf-8")

    persist_env_value(env, "REALTIME_BACKEND", "huggingface")
    persist_env_value(env, "NEW_KEY", "x")

    assert env.read_text(encoding="utf-8") == "# comment\nREALTIME_BACKEND=huggingface\nOTHER=1\nNEW_KEY=x\n"
