import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

BACKENDS = {
    "huggingface": "the original Pollen Robotics conversation backend (hosted Hugging Face speech-to-speech)",
    "hermes": "the home assistant backend (wake word, Hermes agent with Home Assistant, alarms and forecasts)",
}


class SwitchBackend(Tool):
    """Switch the conversation backend; the app rebuilds it a few seconds later."""

    name = "switch_backend"
    description = (
        "Switch Reachy's conversation brain. 'huggingface' = the original Pollen Robotics conversation app "
        "backend; 'hermes' = the home assistant backend with wake word, Home Assistant control, alarms and "
        "weather forecasts. Use only when the user explicitly asks to switch modes or brains. The switch "
        "happens a few seconds after you answer, so tell the user it is switching."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "backend": {"type": "string", "enum": list(BACKENDS), "description": "Backend to switch to."},
        },
        "required": ["backend"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Ask the app to switch backends."""
        backend = str(kwargs.get("backend") or "").strip().lower()
        if backend not in BACKENDS:
            return {"error": f"backend must be one of {list(BACKENDS)}"}
        if deps.switch_backend is None:
            return {"error": "switching backends is unavailable in this runtime"}
        logger.info("Tool call: switch_backend backend=%s", backend)
        return await deps.switch_backend(backend)
