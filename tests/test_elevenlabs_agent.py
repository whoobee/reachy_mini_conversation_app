import json
import base64
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.elevenlabs_agent as el_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.elevenlabs_agent import ElevenLabsAgentHandler, build_agent_body
from reachy_mini_conversation_app.tools.core_tools import ToolSpec, ToolDependencies
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


class _FakeWebSocket:
    """Record messages sent by the handler."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))


def _handler() -> tuple[ElevenLabsAgentHandler, _FakeWebSocket]:
    handler = ElevenLabsAgentHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    ws = _FakeWebSocket()
    handler.connection = ws
    return handler, ws


def _audio_event(event_id: int, samples: list[int]) -> dict[str, Any]:
    pcm = np.array(samples, dtype=np.int16).tobytes()
    return {"type": "audio", "audio_event": {"event_id": event_id, "audio_base_64": base64.b64encode(pcm).decode()}}


def _drain(handler: ElevenLabsAgentHandler) -> list[Any]:
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def _spec(name: str) -> ToolSpec:
    return {"type": "function", "name": name, "description": f"{name} tool", "parameters": {"type": "object"}}


def test_agent_body_uses_profile_language_and_client_tools(monkeypatch: Any) -> None:
    """The synced agent gets the profile prompt, configured language, PCM audio and client tools."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "ro")
    monkeypatch.setattr(el_mod.core_tools, "get_tools", lambda: {})

    body = build_agent_body("Be Reachy.", [_spec("play_emotion"), _spec("camera")], voice_id="voice-1")

    conversation = body["conversation_config"]
    assert conversation["agent"]["language"] == "ro"
    assert conversation["agent"]["prompt"]["prompt"] == "Be Reachy."
    assert [tool["name"] for tool in conversation["agent"]["prompt"]["tools"]] == ["play_emotion"]
    assert conversation["agent"]["prompt"]["tools"][0]["type"] == "client"
    assert conversation["tts"]["voice_id"] == "voice-1"
    assert conversation["tts"]["agent_output_audio_format"] == "pcm_16000"
    assert conversation["asr"]["user_input_audio_format"] == "pcm_16000"


@pytest.mark.asyncio
async def test_interruption_clears_playback_and_drops_stale_audio() -> None:
    """Barge-in clears queued audio and ignores audio from the interrupted response."""
    handler, _ = _handler()
    clear_queue = MagicMock()
    handler._clear_queue = clear_queue

    await handler._handle_event(_audio_event(1, [1, 2, 3]))
    await handler._handle_event({"type": "interruption", "interruption_event": {"event_id": 2}})
    await handler._handle_event(_audio_event(2, [4, 5]))
    await handler._handle_event(_audio_event(3, [6]))

    clear_queue.assert_called_once()
    frames = [item for item in _drain(handler) if isinstance(item, tuple)]
    assert [frame[1].tolist() for frame in frames] == [[[1, 2, 3]], [[6]]]
    assert all(frame[0] == ElevenLabsAgentHandler.SAMPLE_RATE for frame in frames)


@pytest.mark.asyncio
async def test_ping_is_answered_with_pong() -> None:
    """Server pings get a pong with the same event ID."""
    handler, ws = _handler()

    await handler._handle_event({"type": "ping", "ping_event": {"event_id": 7}})

    assert ws.sent == [{"type": "pong", "event_id": 7}]


@pytest.mark.asyncio
async def test_tool_result_is_sent_only_when_agent_expects_it() -> None:
    """Results go back to the agent for expects_response tools and are skipped otherwise."""
    handler, ws = _handler()
    handler._expects_response = {"wait": True, "fire": False}

    def _completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="robot_status",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    await handler._handle_tool_result(_completed("fire"))
    await handler._handle_tool_result(_completed("wait"))

    assert ws.sent == [
        {"type": "client_tool_result", "tool_call_id": "wait", "result": '{"ok": true}', "is_error": False}
    ]


@pytest.mark.asyncio
async def test_user_transcript_is_forwarded() -> None:
    """Final user transcripts reach the UI queue."""
    handler, _ = _handler()

    await handler._handle_event({"type": "user_transcript", "user_transcription_event": {"user_transcript": "Salut"}})

    messages = [item.args[0] for item in _drain(handler) if isinstance(item, AdditionalOutputs)]
    assert messages == [{"role": "user", "content": "Salut"}]


@pytest.mark.asyncio
async def test_receive_sends_16khz_mono_chunks() -> None:
    """Microphone audio is downmixed and resampled to 16 kHz before streaming."""
    handler, ws = _handler()
    stereo_48k = np.zeros((480, 2), dtype=np.int16)

    await handler.receive((48000, stereo_48k))

    chunk = base64.b64decode(ws.sent[0]["user_audio_chunk"])
    assert len(chunk) == 160 * 2


def test_client_tool_parameters_all_have_descriptions(monkeypatch: Any) -> None:
    """ElevenLabs rejects client tool properties without a description, so one is filled in."""
    monkeypatch.setattr(el_mod.core_tools, "get_tools", lambda: {})
    spec = _spec("move_head")
    spec["parameters"] = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["left", "right"]},
            "speed": {"type": "number", "description": "How fast"},
        },
    }

    body = build_agent_body("Be Reachy.", [spec], voice_id=None)

    properties = body["conversation_config"]["agent"]["prompt"]["tools"][0]["parameters"]["properties"]
    assert properties["direction"]["description"] == "direction"
    assert properties["speed"]["description"] == "How fast"
