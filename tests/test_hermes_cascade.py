import sys
import json
import time
import types
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest

import reachy_mini_conversation_app.hermes_cascade as hc_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.hermes_cascade import HermesCascadeHandler, pop_sentences, looks_like_echo
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _delta(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def _handler() -> HermesCascadeHandler:
    handler = HermesCascadeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._running = True
    handler._active = True
    handler._voice_id = "voice-1"
    handler._clear_queue = MagicMock()
    return handler


def _wire_clients(handler: HermesCascadeHandler, hermes_body: bytes, tts_requests: list[dict[str, Any]]) -> None:
    def hermes(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=hermes_body, headers={"content-type": "text/event-stream"})

    def tts(request: httpx.Request) -> httpx.Response:
        tts_requests.append(json.loads(request.content))
        return httpx.Response(200, content=np.ones(160, dtype=np.int16).tobytes())

    handler._hermes_http = httpx.AsyncClient(base_url="http://hermes/v1", transport=httpx.MockTransport(hermes))
    handler._tts_http = httpx.AsyncClient(base_url="https://tts", transport=httpx.MockTransport(tts))


def _drain(handler: HermesCascadeHandler) -> list[Any]:
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def test_pop_sentences_merges_short_fragments_and_keeps_the_rest() -> None:
    """Sentences are released at punctuation; tiny fragments wait for the next one."""
    sentences, rest = pop_sentences("Da. Am aprins lumina din sufragerie. Acum mai")

    assert sentences == ["Da. Am aprins lumina din sufragerie."]
    assert rest == "Acum mai"
    assert pop_sentences(rest, final=True) == (["Acum mai"], "")


def test_looks_like_echo_detects_the_robots_own_words() -> None:
    """Transcripts made of what the robot is saying are treated as echo."""
    assert looks_like_echo("lumina din sufragerie", "Am aprins lumina din sufragerie.")
    assert not looks_like_echo("stinge tot", "Am aprins lumina din sufragerie.")


@pytest.mark.asyncio
async def test_committed_utterance_is_answered_by_hermes_and_spoken(monkeypatch: Any) -> None:
    """A committed transcript goes to Hermes; streamed sentences are synthesized and queued for playback."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "ro")
    handler = _handler()
    tts_requests: list[dict[str, Any]] = []
    progress = {"tool": "ha_call_service", "toolCallId": "c1", "status": "running"}
    _wire_clients(
        handler,
        _sse(progress, _delta("\n\nAm aprins lumina "), _delta("din sufragerie. Altceva"), _delta("?")),
        tts_requests,
    )

    await handler._handle_stt_message({"message_type": "committed_transcript", "text": "Aprinde lumina"})
    assert handler._response_task is not None
    await handler._response_task

    assert [r["text"] for r in tts_requests] == ["Am aprins lumina din sufragerie.", "Altceva?"]
    assert tts_requests[0]["language_code"] == "ro"
    items = _drain(handler)
    messages = [item.args[0] for item in items if isinstance(item, AdditionalOutputs)]
    assert {"role": "user", "content": "Aprinde lumina"} in messages
    assert {"role": "assistant", "content": "Am aprins lumina din sufragerie. Altceva?"} in messages
    assert sum(1 for item in items if isinstance(item, tuple)) == 2


@pytest.mark.asyncio
async def test_talking_over_the_robot_interrupts_it() -> None:
    """A user partial transcript during playback cancels the reply and clears queued audio."""
    handler = _handler()
    handler._speaking = True
    handler._playback_until = time.monotonic() + 5
    handler._spoken_text = "Am aprins lumina din sufragerie."
    handler._response_task = asyncio.create_task(asyncio.sleep(10))

    await handler._handle_stt_message({"message_type": "partial_transcript", "text": "lumina din sufragerie"})
    assert handler._speaking  # echo of its own words is ignored

    await handler._handle_stt_message({"message_type": "partial_transcript", "text": "stai stinge tot"})

    assert not handler._speaking
    handler._clear_queue.assert_called_once()  # type: ignore[attr-defined]
    assert handler._response_task is None


@pytest.mark.asyncio
async def test_wake_word_starts_a_conversation_and_buffers_audio(monkeypatch: Any) -> None:
    """While idle only the detector sees audio; after the wake word audio is buffered for STT."""
    handler = _handler()
    handler._active = False
    detector = MagicMock()
    detector.process.return_value = True
    handler._detector = detector
    monkeypatch.setattr(handler, "_run_stt", AsyncMock())

    await handler.receive((16000, np.zeros(1280, dtype=np.int16)))
    assert handler._active
    await handler.receive((16000, np.ones(320, dtype=np.int16)))

    assert sum(chunk.size for chunk in handler._pending_audio) == 320
    handler.deps.movement_manager.set_listening.assert_called_with(True)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_silence_timeout_returns_to_wake_word_mode(monkeypatch: Any) -> None:
    """With no speech for CONVERSATION_TIMEOUT_S the conversation closes."""
    monkeypatch.setattr(config, "CONVERSATION_TIMEOUT_S", 1.0)
    handler = _handler()
    handler._detector = MagicMock()
    handler._last_user_activity = time.monotonic() - 5

    await handler._supervise()

    assert not handler._active
    handler._detector.reset.assert_called_once()


def test_module_exposes_16khz_rate() -> None:
    """Playback frames use the 16 kHz rate the TTS output format produces."""
    assert hc_mod.SAMPLE_RATE == HermesCascadeHandler.SAMPLE_RATE == 16000


def test_transcript_cleanup_drops_wake_phrase_and_fillers() -> None:
    """The wake phrase is stripped and hesitation-only utterances are recognised."""
    assert hc_mod.strip_wake_phrase("Hey, Jarvis! Aprinde lumina.", "hey_jarvis") == "Aprinde lumina."
    assert hc_mod.strip_wake_phrase("Hey, Jarvis!", "hey_jarvis") == ""
    assert hc_mod.strip_wake_phrase("Hei, Richie, aprinde lumina!", "hey reachy|hei richie") == "aprinde lumina!"
    assert hc_mod.is_filler_only("Ăăă-")
    assert hc_mod.is_filler_only("Hmm...")
    assert not hc_mod.is_filler_only("mama")
    assert not hc_mod.is_filler_only("Hai")
    assert not hc_mod.is_filler_only("Ăăă, aprinde lumina")


def test_gain_amplifies_and_clips() -> None:
    """Gain boosts quiet TTS audio without wrapping loud samples around."""
    pcm = np.array([1000, -1000, 20000], dtype=np.int16)

    assert hc_mod.apply_gain(pcm, 4.0).tolist() == [4000, -4000, 32767]


class _FakeImpulseRunner:
    """Stand-in for the Edge Impulse runner binary: scores loud windows as the keyword."""

    def __init__(self, model_path: str) -> None:
        self.windows: list[int] = []

    def init(self) -> dict[str, Any]:
        params = {"frequency": 24000, "labels": ["hey_reachy", "noise", "unknown"], "input_features_count": 2400}
        return {"model_parameters": params}

    def classify(self, data: list[float]) -> dict[str, Any]:
        self.windows.append(len(data))
        loud = max(abs(x) for x in data) > 1000
        return {"result": {"classification": {"hey_reachy": 0.9 if loud else 0.1, "noise": 0.1}}}

    def stop(self) -> None:
        pass


def test_edge_impulse_detector_slides_a_window_and_fires_on_keyword(monkeypatch: Any) -> None:
    """The .eim detector classifies full windows every quarter window and fires above the threshold."""
    fake_runner_module = types.ModuleType("edge_impulse_linux.runner")
    fake_runner_module.ImpulseRunner = _FakeImpulseRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_impulse_linux", types.ModuleType("edge_impulse_linux"))
    monkeypatch.setitem(sys.modules, "edge_impulse_linux.runner", fake_runner_module)
    detector = hc_mod.make_wake_word_detector("/models/hey-reachy.eim", 0.7)
    assert isinstance(detector, hc_mod.EdgeImpulseDetector)

    assert not detector.process(np.zeros(1600, dtype=np.int16))
    assert not detector.process(np.zeros(400, dtype=np.int16))
    assert detector.process(np.full(400, 5000, dtype=np.int16))
    assert detector._runner.windows == [
        2400,
        2400,
        2400,
    ]  # 100 ms at 16 kHz, resampled to 24 kHz  # type: ignore[attr-defined]


def test_wake_phrase_comes_from_config_for_model_files(monkeypatch: Any) -> None:
    """A model file has no phrase of its own, so WAKE_PHRASE is used for transcript cleanup."""
    monkeypatch.setattr(config, "WAKE_WORD", "/models/hey-reachy.eim")
    monkeypatch.setattr(config, "WAKE_PHRASE", None)
    assert hc_mod.wake_phrase() == ""
    monkeypatch.setattr(config, "WAKE_PHRASE", "hey reachy")
    assert hc_mod.strip_wake_phrase("Hey, Reachy! Aprinde lumina.", hc_mod.wake_phrase()) == "Aprinde lumina."


def test_speakable_drops_emoji_and_leaked_tool_markup() -> None:
    """Emoji and tool-call markup that a small model leaks into text are never spoken."""
    leaked = "Să număr: <function=ha_list_entities><parameter=domain>light</parameter></function> Uau! 😄"

    assert hc_mod.speakable(leaked) == "Să număr: Uau!"
    assert hc_mod.speakable("Gata, lumină! 🤖") == "Gata, lumină!"


@pytest.mark.asyncio
async def test_sleep_request_turns_off_after_the_reply() -> None:
    """go_to_sleep waits until the goodbye has been spoken, then runs the app's sleep-and-stop routine."""
    handler = _handler()
    handler._detector = MagicMock()
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})
    handler.deps.go_to_sleep = go_to_sleep
    handler._playback_until = time.monotonic() + 60  # still speaking

    result = await handler._request_sleep({})
    assert "turning off" in result["status"]
    await handler._supervise()
    go_to_sleep.assert_not_called()

    handler._playback_until = 0.0
    await handler._supervise()
    go_to_sleep.assert_called_once()
    assert not handler._active


@pytest.mark.asyncio
async def test_listening_time_can_be_read_changed_and_is_persisted(tmp_path: Any, monkeypatch: Any) -> None:
    """set_listening_time reads the value, rejects out-of-range input, and saves valid changes to .env."""
    monkeypatch.setattr(config, "CONVERSATION_TIMEOUT_S", 5.0)
    monkeypatch.delenv("CONVERSATION_TIMEOUT_S", raising=False)
    handler = HermesCascadeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()), instance_path=str(tmp_path)
    )

    assert (await handler._set_listening_time({}))["listening_time_seconds"] == 5.0
    assert "error" in await handler._set_listening_time({"seconds": 45})
    assert "error" in await handler._set_listening_time({"seconds": 0})
    result = await handler._set_listening_time({"seconds": 12})

    assert result["listening_time_seconds"] == 12 and result["previous_seconds"] == 5.0
    assert config.CONVERSATION_TIMEOUT_S == 12.0
    assert "CONVERSATION_TIMEOUT_S=12" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_listening_time_is_clamped_to_the_allowed_range(monkeypatch: Any) -> None:
    """A hand-edited out-of-range .env value is kept within 1-30 s."""
    monkeypatch.setattr(config, "CONVERSATION_TIMEOUT_S", 300.0)
    assert HermesCascadeHandler._listen_time() == 30.0
    monkeypatch.setattr(config, "CONVERSATION_TIMEOUT_S", 0.0)
    assert HermesCascadeHandler._listen_time() == 1.0


def test_save_wake_clip_writes_audio_and_prunes(tmp_path: Any) -> None:
    """Each wake keeps a short WAV named with its score; only the newest clips are kept."""
    audio = np.ones(1600, dtype=np.int16)
    for _ in range(3):
        hc_mod.save_wake_clip(tmp_path, audio, 0.91, keep=2)
        time.sleep(1.05)  # distinct timestamps in the file names

    clips = sorted(tmp_path.glob("wake_*.wav"))
    assert len(clips) == 2 and clips[-1].name.endswith("_0.91.wav")


@pytest.mark.asyncio
async def test_wake_saves_the_triggering_audio(monkeypatch: Any, tmp_path: Any) -> None:
    """When the detector fires, the last seconds of idle audio and the score are saved before waking."""
    monkeypatch.setattr(config, "WAKE_WORD_SAVE_CLIPS", True)
    handler = HermesCascadeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()), instance_path=str(tmp_path)
    )
    handler._running = True
    detector = MagicMock()
    detector.process.side_effect = [False, True]
    detector.last_score = 0.93
    handler._detector = detector
    monkeypatch.setattr(handler, "_wake", AsyncMock())

    await handler.receive((16000, np.ones(1280, dtype=np.int16)))
    await handler.receive((16000, np.ones(1280, dtype=np.int16)))

    clips = list((tmp_path / "wake_clips").glob("wake_*_0.93.wav"))
    assert len(clips) == 1
    handler._wake.assert_awaited_once()  # type: ignore[attr-defined]


def test_waiting_for_the_user_is_not_busy_but_thinking_and_speaking_are() -> None:
    """An open conversation that is only listening does not block idle gestures; replies do."""
    handler = _handler()  # active conversation, nothing generating or playing
    assert not handler.is_busy()
    handler._speaking = True
    assert handler.is_busy()


@pytest.mark.asyncio
async def test_robot_events_go_ahead_of_the_next_message_in_a_daily_session() -> None:
    """Wake and other robot events reach Hermes before the user's words; the session id is per day."""
    handler = _handler()
    sent: list[tuple[str, dict[str, Any]]] = []

    def hermes(request: httpx.Request) -> httpx.Response:
        sent.append((request.headers["X-Hermes-Session-Id"], json.loads(request.content)))
        return httpx.Response(200, content=_sse(_delta("Bine.")), headers={"content-type": "text/event-stream"})

    _wire_clients(handler, b"", [])
    handler._hermes_http = httpx.AsyncClient(base_url="http://hermes/v1", transport=httpx.MockTransport(hermes))
    handler._session_id = handler._hermes_session_id()
    handler.note_event("a sunat alarma Trezirea")

    for text in ("Bună", "Mersi"):
        await handler._handle_stt_message({"message_type": "committed_transcript", "text": text})
        assert handler._response_task is not None
        await handler._response_task

    (first_id, first), (second_id, second) = sent
    assert first_id == second_id and first_id.startswith("reachy-20")
    first_text = first["messages"][0]["content"]
    assert first_text.startswith("[Reachy: ") and "a sunat alarma Trezirea]" in first_text
    assert first_text.endswith("\nBună")
    assert second["messages"][0]["content"] == "Mersi"


@pytest.mark.asyncio
async def test_ignored_noise_is_noted_but_a_bare_wake_phrase_is_not(monkeypatch: Any) -> None:
    """Filler that is dropped shows up as an event; hearing only the wake phrase does not."""
    monkeypatch.setattr(config, "WAKE_PHRASE", "hei reachy")
    handler = _handler()

    await handler._handle_stt_message({"message_type": "committed_transcript", "text": "Hei Reachy!"})
    assert not handler._events.pending()
    await handler._handle_stt_message({"message_type": "committed_transcript", "text": "Ăăă..."})
    assert "ignorat (zgomot/umplutură): «Ăăă...»" in handler._events.take_prefix()
