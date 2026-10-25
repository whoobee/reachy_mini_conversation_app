"""Wake-word voice loop: local wake word, ElevenLabs Scribe STT, a Hermes agent, ElevenLabs TTS.

While idle, microphone audio only feeds a local openWakeWord model, so nothing is sent
over the network. On the wake word the handler plays a short acknowledgement, opens an
ElevenLabs Scribe realtime session, and sends each committed utterance to a Hermes Agent
API server (OpenAI-compatible, streaming). The reply is split into sentences and spoken
with ElevenLabs streaming TTS. Speaking over the robot interrupts it. After
``CONVERSATION_TIMEOUT_S`` without speech the session closes and the robot waits for the
wake word again.
"""

from __future__ import annotations
import re
import sys
import json
import math
import time
import types
import base64
import asyncio
import logging
from typing import Any, Final, Optional
from pathlib import Path
from contextlib import AbstractAsyncContextManager
from urllib.parse import urlencode

import httpx
import numpy as np
from mcp import types as mcp_types
from numpy.typing import NDArray
from websockets.asyncio.client import ClientConnection, connect

from reachy_mini_conversation_app.config import config, set_custom_profile
from reachy_mini_conversation_app.robot_mcp import RobotMcpServer, parse_tool_names
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.elevenlabs_api import (
    ELEVENLABS_WS_BASE,
    voice_name_for,
    fetch_voice_ids,
    resolve_voice_id,
    elevenlabs_api_key,
    elevenlabs_http_client,
    raise_for_elevenlabs_status,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.conversation_handler import QueueItem, AudioFrame, ConversationHandler
from reachy_mini_conversation_app.tools.background_tool_manager import ToolNotification, BackgroundToolManager


logger = logging.getLogger(__name__)

SAMPLE_RATE: Final[int] = 16000
_WAKE_FRAME_SAMPLES: Final[int] = 1280  # openWakeWord expects 80 ms frames
_STT_CHUNK_SAMPLES: Final[int] = 1600  # send 100 ms chunks to Scribe
_PENDING_AUDIO_MAX_SAMPLES: Final[int] = SAMPLE_RATE * 5
_BARGE_IN_MIN_WORDS: Final[int] = 2
_ECHO_WORD_OVERLAP: Final[float] = 0.6
_MIN_SENTENCE_CHARS: Final[int] = 12
_MONITOR_INTERVAL_S: Final[float] = 0.2
_FILLER_ONLY = re.compile(
    r"^(?:[\W_]*(?:ă+m*|â+|a{2,}|e{2,}m*|m{2,}|hm+|mhm|eh+|ah+|uh+m*|um+))+[\W_]*$", re.IGNORECASE | re.UNICODE
)
_TOOL_MARKUP = re.compile(
    r"<(function|tool_call)[^>]*>.*?(</\1>|$)|</?(parameter|function|tool_call)[^>]*>", re.DOTALL
)
_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_WORD = re.compile(r"\w+", re.UNICODE)


def pop_sentences(buffer: str, final: bool = False) -> tuple[list[str], str]:
    """Split complete sentences off ``buffer``; return them and the unfinished rest.

    Very short fragments are merged into the next sentence so TTS gets natural chunks.
    With ``final`` the remaining text is returned as a last sentence.
    """
    sentences: list[str] = []
    pending = ""
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        piece = (
            (pending + " " + buffer[start : match.start()]).strip()
            if pending
            else buffer[start : match.start()].strip()
        )
        start = match.end()
        if len(piece) < _MIN_SENTENCE_CHARS:
            pending = piece
            continue
        sentences.append(piece)
        pending = ""
    rest = buffer[start:]
    if pending:
        rest = pending + " " + rest
    if final:
        if rest.strip():
            sentences.append(rest.strip())
        rest = ""
    return sentences, rest


def wake_phrase() -> str:
    """Return the spoken wake phrase: WAKE_PHRASE, or the model name for pretrained models."""
    if (config.WAKE_PHRASE or "").strip():
        return str(config.WAKE_PHRASE).strip()
    return "" if Path(config.WAKE_WORD).suffix else config.WAKE_WORD


def strip_wake_phrase(text: str, wake_word: str) -> str:
    """Remove the wake phrase (e.g. "Hey, Jarvis!") from a transcript."""
    words = [w for w in re.split(r"[-_\s]+", wake_word.strip()) if w]
    if not words:
        return text.strip()
    pattern = r"\b" + r"[\W_]*".join(re.escape(w) for w in words) + r"\b[\W_]*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def is_filler_only(text: str) -> bool:
    """Return whether ``text`` holds only hesitation sounds like "Ăăă" or "Hmm"."""
    return bool(_FILLER_ONLY.match(text.strip()))


def apply_gain(pcm: NDArray[np.int16], gain: float) -> NDArray[np.int16]:
    """Amplify int16 audio, clipping instead of wrapping around."""
    if gain == 1.0:
        return pcm
    amplified: NDArray[np.int16] = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
    return amplified


def speakable(text: str) -> str:
    """Drop text that must not be read aloud: leaked tool-call markup and emoji."""
    text = _EMOJI.sub("", _TOOL_MARKUP.sub("", text))
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def looks_like_echo(heard: str, spoken: str) -> bool:
    """Return whether ``heard`` is mostly words the robot itself is saying."""
    heard_words = [w.lower() for w in _WORD.findall(heard)]
    if not heard_words:
        return False
    spoken_words = {w.lower() for w in _WORD.findall(spoken)}
    overlap = sum(1 for w in heard_words if w in spoken_words) / len(heard_words)
    return overlap >= _ECHO_WORD_OVERLAP


def to_mono_16k(frame: AudioFrame) -> NDArray[np.int16]:
    """Return microphone audio as mono 16 kHz int16."""
    sample_rate, audio = frame
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio[:, 0]
    pcm = audio_to_int16(audio)
    if sample_rate == SAMPLE_RATE or pcm.size == 0:
        return pcm
    target_size = max(1, round(pcm.size * SAMPLE_RATE / sample_rate))
    positions = np.linspace(0, pcm.size - 1, target_size)
    return np.interp(positions, np.arange(pcm.size), pcm).astype(np.int16)


class WakeWordDetector:
    """openWakeWord model fed with 16 kHz int16 audio."""

    def __init__(self, model: str, threshold: float) -> None:
        """Load a pretrained model by name (downloaded on first use) or a model file path."""
        from openwakeword.model import Model
        from openwakeword.utils import download_models

        if Path(model).suffix in {".onnx", ".tflite"}:
            model_ref = model
        else:
            download_models([model])
            model_ref = model
        self._model = Model(wakeword_models=[model_ref], inference_framework="onnx")
        self._threshold = threshold
        self._buffer = np.zeros(0, dtype=np.int16)

    def process(self, pcm: NDArray[np.int16]) -> bool:
        """Feed audio; return True when the wake word was detected."""
        self._buffer = np.concatenate([self._buffer, pcm])
        detected = False
        while self._buffer.size >= _WAKE_FRAME_SAMPLES:
            frame, self._buffer = self._buffer[:_WAKE_FRAME_SAMPLES], self._buffer[_WAKE_FRAME_SAMPLES:]
            scores = self._model.predict(frame)
            if max(scores.values(), default=0.0) >= self._threshold:
                detected = True
        if detected:
            self.reset()
        return detected

    def reset(self) -> None:
        """Clear buffered audio and model state."""
        self._buffer = np.zeros(0, dtype=np.int16)
        self._model.reset()

    def close(self) -> None:
        """Release resources (nothing to do for openWakeWord)."""


class EdgeImpulseDetector:
    """Edge Impulse keyword-spotting model (``.eim``) fed with 16 kHz int16 audio.

    The ``.eim`` file is a standalone runner binary; it classifies a sliding window
    of raw samples. The keyword label defaults to the first label that is not noise.
    """

    _NON_KEYWORD_LABELS: Final[frozenset[str]] = frozenset({"noise", "unknown", "_noise", "_unknown", "background"})

    def __init__(self, model_path: str, threshold: float) -> None:
        """Start the runner binary and read the model window size."""
        # The SDK package imports its microphone helpers (pyaudio) at import time; only the runner is used here.
        sys.modules.setdefault("pyaudio", types.ModuleType("pyaudio"))
        from edge_impulse_linux.runner import ImpulseRunner

        self._runner = ImpulseRunner(str(Path(model_path).expanduser()))
        params = self._runner.init()["model_parameters"]
        self._model_rate = int(params.get("frequency", SAMPLE_RATE))
        labels = [str(label) for label in params.get("labels", [])]
        keywords = [label for label in labels if label.lower() not in self._NON_KEYWORD_LABELS]
        if not keywords:
            raise RuntimeError(f"Wake word model has no keyword label: {labels}")
        self._label = keywords[0]
        self._model_window = int(params["input_features_count"])
        # Buffer 16 kHz audio covering the model window; resample a whole window per classification.
        self._window = round(self._model_window * SAMPLE_RATE / self._model_rate)
        self._step = max(1, self._window // 4)
        self._threshold = threshold
        self._buffer = np.zeros(0, dtype=np.int16)
        self._since_last = 0
        logger.info(
            "Edge Impulse wake word model loaded: label=%s rate=%d Hz window=%.1f s",
            self._label,
            self._model_rate,
            self._model_window / self._model_rate,
        )

    def process(self, pcm: NDArray[np.int16]) -> bool:
        """Feed audio; return True when the keyword score crosses the threshold."""
        self._buffer = np.concatenate([self._buffer, pcm])[-self._window :]
        self._since_last += pcm.size
        if self._buffer.size < self._window or self._since_last < self._step:
            return False
        self._since_last = 0
        window = self._buffer.astype(np.float32)
        if self._model_rate != SAMPLE_RATE:
            from scipy.signal import resample_poly

            divisor = math.gcd(self._model_rate, SAMPLE_RATE)
            window = resample_poly(window, self._model_rate // divisor, SAMPLE_RATE // divisor)[: self._model_window]
        result = self._runner.classify(window.tolist())
        score = float(result.get("result", {}).get("classification", {}).get(self._label, 0.0))
        if score >= self._threshold:
            logger.debug("Wake word score %.2f", score)
            self.reset()
            return True
        return False

    def reset(self) -> None:
        """Clear buffered audio."""
        self._buffer = np.zeros(0, dtype=np.int16)
        self._since_last = 0

    def close(self) -> None:
        """Stop the runner binary."""
        self._runner.stop()


def make_wake_word_detector(model: str, threshold: float) -> WakeWordDetector | EdgeImpulseDetector:
    """Return an Edge Impulse detector for ``.eim`` files, otherwise openWakeWord."""
    if Path(model).suffix == ".eim":
        return EdgeImpulseDetector(model, threshold)
    return WakeWordDetector(model, threshold)


class HermesCascadeHandler(ConversationHandler):
    """Wake word + Scribe STT + Hermes agent + ElevenLabs TTS."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler; network clients and the wake word model load in start_up."""
        super().__init__()
        self.deps = deps
        self.instance_path = instance_path
        self.output_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.tool_manager = BackgroundToolManager()

        self._voice_choice: str | None = (startup_voice or "").strip() or None
        self._voice_ids_by_name: dict[str, str] = {}
        self._voice_id: str | None = None
        self._ack_pcm: NDArray[np.int16] | None = None

        self._tts_http: httpx.AsyncClient | None = None
        self._hermes_http: httpx.AsyncClient | None = None
        self._detector: WakeWordDetector | EdgeImpulseDetector | None = None
        self._running = False
        self._stopped = asyncio.Event()
        self._robot_mcp: RobotMcpServer | None = None
        self._sleep_requested = False
        self._robot_mcp_task: asyncio.Task[None] | None = None

        # Conversation state
        self._active = False
        self._session_id = ""
        self._stt_ws: ClientConnection | None = None
        self._stt_task: asyncio.Task[None] | None = None
        self._pending_audio: list[NDArray[np.int16]] = []
        self._send_buffer = np.zeros(0, dtype=np.int16)
        self._response_task: asyncio.Task[None] | None = None
        self._spoken_text = ""
        self._speaking = False
        self._playback_until = 0.0
        self._last_user_activity = 0.0
        self._turn_started_at: float | None = None

    # ---- lifecycle ----

    def _is_connected(self) -> bool:
        """Report ready once started; idle behaviors may run between conversations."""
        return self._running

    def _idle_behavior_ready(self) -> bool:
        """Only run idle behaviors while waiting for the wake word."""
        return not self._active and not self._speaking

    async def start_up(self) -> None:
        """Connect clients, load the wake word model, then supervise conversations until shutdown."""
        base_url = (config.HERMES_BASE_URL or "").strip()
        api_key = (config.HERMES_API_KEY or "").strip()
        if not base_url or not api_key:
            raise RuntimeError("HERMES_BASE_URL and HERMES_API_KEY must be set when REALTIME_BACKEND=hermes")
        self._hermes_http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(90.0, connect=5.0),
        )
        self._tts_http = elevenlabs_http_client(timeout=30.0)
        try:
            self._voice_ids_by_name = await fetch_voice_ids(self._tts_http)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.warning("Could not list ElevenLabs voices: %s", e)
        self._voice_id = resolve_voice_id(self._voice_choice, self._voice_ids_by_name)
        if self._voice_id is None:
            if not self._voice_ids_by_name:
                raise RuntimeError("Set ELEVENLABS_VOICE_ID: no ElevenLabs voice is available")
            self._voice_id = next(iter(self._voice_ids_by_name.values()))
        self._detector = await asyncio.to_thread(make_wake_word_detector, config.WAKE_WORD, config.WAKE_WORD_THRESHOLD)
        await self._cache_ack()
        self.tool_manager.start_up(tool_callbacks=[self._on_tool_done])
        self._start_robot_mcp()
        self._running = True
        logger.info(
            "Hermes voice loop ready: wake word=%s hermes=%s model=%s tts=%s voice=%s language=%s",
            config.WAKE_WORD,
            base_url,
            config.HERMES_MODEL,
            config.ELEVENLABS_TTS_MODEL,
            self.get_current_voice(),
            config.REALTIME_TRANSCRIPTION_LANGUAGE,
        )
        try:
            while not self._stopped.is_set():
                await self._supervise()
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=_MONITOR_INTERVAL_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._running = False
            await self._end_conversation("handler stopped")

    async def _supervise(self) -> None:
        """Track playback end, run a requested sleep, and close conversations after the silence timeout."""
        now = time.monotonic()
        if self._sleep_requested and not self._is_generating() and now >= self._playback_until:
            self._sleep_requested = False
            await self._go_to_sleep()
            return
        if self._speaking and now >= self._playback_until and not self._is_generating():
            self._speaking = False
            self.deps.movement_manager.set_speaking(False)
            self._mark_activity("assistant_transcript_done")
            if self._active:
                self.deps.movement_manager.set_listening(True)
        if self._active and not self._is_generating() and not self._speaking:
            quiet_since = max(self._last_user_activity, self._playback_until)
            if now - quiet_since > config.CONVERSATION_TIMEOUT_S:
                await self._end_conversation("silence timeout")

    def _start_robot_mcp(self) -> None:
        """Offer robot tools (volume, emotions, dances, head moves) to Hermes when a token is configured."""
        token = (config.ROBOT_MCP_TOKEN or "").strip()
        if not token:
            logger.info("ROBOT_MCP_TOKEN not set; Hermes gets no robot tools")
            return
        sleep_tool = mcp_types.Tool(
            name="go_to_sleep",
            description=(
                "Put Reachy to sleep and turn the conversation app off: it finishes speaking, moves to the sleep "
                "pose and stops listening completely (no wake word). Use only when the user explicitly asks Reachy "
                "to go to sleep or turn off; never for silence, tiredness or a sleepy mood."
            ),
            inputSchema={"type": "object", "properties": {}},
        )
        self._robot_mcp = RobotMcpServer(
            self.deps,
            parse_tool_names(config.ROBOT_MCP_TOOLS),
            config.ROBOT_MCP_HOST,
            config.ROBOT_MCP_PORT,
            token,
            extra_tools={"go_to_sleep": (sleep_tool, self._request_sleep)},
        )
        self._robot_mcp_task = asyncio.create_task(self._robot_mcp.serve(), name="robot-mcp")

    async def shutdown(self) -> None:
        """Stop the loop and close all connections."""
        self._stopped.set()
        await self._end_conversation("shutdown")
        if self._robot_mcp is not None:
            await self._robot_mcp.stop()
            if self._robot_mcp_task is not None:
                try:
                    await asyncio.wait_for(self._robot_mcp_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                    logger.debug("Robot MCP server stop: %s", e)
            self._robot_mcp = self._robot_mcp_task = None
        await self.tool_manager.shutdown()
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        for client in (self._tts_http, self._hermes_http):
            if client is not None:
                await client.aclose()
        self._tts_http = self._hermes_http = None
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _on_tool_done(self, completed_tool: ToolNotification) -> None:
        """Log idle-behavior tools; their results are not sent anywhere."""
        if completed_tool.error is not None:
            logger.warning("Idle tool '%s' failed: %s", completed_tool.tool_name, completed_tool.error)

    # ---- sleep ----

    async def _request_sleep(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        """Turn off once the current reply has been spoken, so the goodbye is heard."""
        if self.deps.go_to_sleep is None:
            return {"error": "go_to_sleep is unavailable in this runtime"}
        self._sleep_requested = True
        return {"status": "going to sleep and turning off after this reply"}

    async def _go_to_sleep(self) -> None:
        """Run the app's sleep-and-stop routine (sleep pose, then the app stops)."""
        await self._end_conversation("going to sleep")
        if self.deps.go_to_sleep is None:
            return
        logger.info("Going to sleep and stopping the app")
        try:
            await asyncio.to_thread(self.deps.go_to_sleep)
        except Exception as e:
            logger.error("go_to_sleep failed: %s", e)

    # ---- microphone ----

    async def receive(self, frame: AudioFrame) -> None:
        """Feed the wake word detector while idle, or stream audio to Scribe while active."""
        if not self._running or frame[1].size == 0:
            return
        pcm = to_mono_16k(frame)
        if not self._active:
            if self._detector is not None and self._detector.process(pcm):
                await self._wake()
            return
        if self._stt_ws is None:
            self._pending_audio.append(pcm)
            while sum(chunk.size for chunk in self._pending_audio) > _PENDING_AUDIO_MAX_SAMPLES:
                self._pending_audio.pop(0)
            return
        self._send_buffer = np.concatenate([self._send_buffer, pcm])
        if self._send_buffer.size >= _STT_CHUNK_SAMPLES:
            chunk, self._send_buffer = self._send_buffer, np.zeros(0, dtype=np.int16)
            await self._send_audio(chunk)

    async def _send_audio(self, pcm: NDArray[np.int16]) -> None:
        ws = self._stt_ws
        if ws is None:
            return
        message = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm.tobytes()).decode(),
            "commit": False,
            "sample_rate": SAMPLE_RATE,
        }
        try:
            await ws.send(json.dumps(message))
        except Exception as e:
            logger.debug("Dropping STT audio chunk: %s", e)

    # ---- conversation ----

    async def _wake(self) -> None:
        """Start a conversation: acknowledge, then open the Scribe session."""
        logger.info("Wake word detected")
        self._active = True
        self._session_id = f"reachy-{time.time_ns()}"
        self._last_user_activity = time.monotonic()
        self._pending_audio = []
        self._send_buffer = np.zeros(0, dtype=np.int16)
        self.deps.movement_manager.set_listening(True)
        self._mark_activity("user_speech_started")
        if self._ack_pcm is not None:
            self._spoken_text = config.WAKE_ACK_TEXT
            await self._play(self._ack_pcm)
        self._stt_task = asyncio.create_task(self._run_stt(), name="scribe-stt")

    async def _end_conversation(self, reason: str) -> None:
        """Close the STT session and go back to waiting for the wake word."""
        if not self._active:
            return
        logger.info("Conversation ended (%s)", reason)
        self._active = False
        await self._cancel_response()
        stt_task, self._stt_task = self._stt_task, None
        ws, self._stt_ws = self._stt_ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if stt_task is not None and stt_task is not asyncio.current_task():
            stt_task.cancel()
            try:
                await stt_task
            except (asyncio.CancelledError, Exception):
                pass
        self._pending_audio = []
        if self._detector is not None:
            self._detector.reset()
        self.deps.movement_manager.set_listening(False)

    def _stt_url(self) -> str:
        params = {
            "model_id": "scribe_v2_realtime",
            "audio_format": "pcm_16000",
            "language_code": config.REALTIME_TRANSCRIPTION_LANGUAGE,
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": config.STT_VAD_SILENCE_S,
            "no_verbatim": "true",
        }
        return f"{ELEVENLABS_WS_BASE}/v1/speech-to-text/realtime?{urlencode(params)}"

    async def _run_stt(self) -> None:
        """Run one Scribe realtime session for the current conversation."""
        try:
            async with connect(self._stt_url(), additional_headers={"xi-api-key": elevenlabs_api_key()}) as ws:
                self._stt_ws = ws
                pending, self._pending_audio = self._pending_audio, []
                if pending:
                    await self._send_audio(np.concatenate(pending))
                async for raw in ws:
                    if isinstance(raw, str):
                        await self._handle_stt_message(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._active:
                logger.debug("Scribe session closed: %s", e)
                return
            logger.error("Scribe session failed: %s", e)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] STT: {e}"}))
        finally:
            self._stt_ws = None
        if self._active:
            await self._end_conversation("speech-to-text session closed")

    async def _handle_stt_message(self, message: dict[str, Any]) -> None:
        """React to one Scribe server message."""
        message_type = message.get("message_type")
        text = strip_wake_phrase(str(message.get("text") or ""), wake_phrase())
        if is_filler_only(text):
            text = ""

        if message_type == "session_started":
            logger.debug("Scribe session %s started", message.get("session_id"))
        elif message_type == "partial_transcript":
            if not text or self._is_echo(text):
                return
            self._last_user_activity = time.monotonic()
            self._mark_activity("user_transcription_delta")
            await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": text}))
            if config.BARGE_IN and self._is_busy() and len(_WORD.findall(text)) >= _BARGE_IN_MIN_WORDS:
                logger.info("Barge-in: user started talking over the robot")
                await self._cancel_response()
        elif message_type == "committed_transcript":
            if not text or self._is_echo(text):
                return
            self._last_user_activity = time.monotonic()
            self._mark_activity("user_transcription_completed")
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
            self._emit_transcript("user", text, True)
            await self._cancel_response()
            self._response_task = asyncio.create_task(self._respond(text), name="hermes-response")
        elif message_type in {"warning", "committed_transcript_with_timestamps", "committed_transcript_entities"}:
            logger.debug("Scribe %s: %s", message_type, message)
        elif message_type:
            error = message.get("error") or message
            logger.error("Scribe %s: %s", message_type, error)

    def _is_echo(self, heard: str) -> bool:
        """Ignore transcripts of the robot's own voice while it is talking."""
        return self._is_busy() and looks_like_echo(heard, self._spoken_text)

    def _is_generating(self) -> bool:
        return self._response_task is not None and not self._response_task.done()

    def _is_busy(self) -> bool:
        return self._is_generating() or self._speaking

    async def _cancel_response(self) -> None:
        """Stop the current reply: cancel generation and drop queued audio."""
        task, self._response_task = self._response_task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._speaking or task is not None:
            if self._clear_queue:
                self._clear_queue()
            self._playback_until = time.monotonic()
            self._speaking = False
            self.deps.movement_manager.set_speaking(False)

    async def _respond(self, text: str) -> None:
        """Send ``text`` to Hermes and speak the streamed reply sentence by sentence."""
        assert self._hermes_http is not None
        self._turn_started_at = time.perf_counter()
        self._spoken_text = ""
        self.deps.movement_manager.set_listening(False)
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._speak_queue(sentences), name="tts")
        reply = ""
        try:
            async with self._hermes_http.stream(
                "POST",
                "/chat/completions",
                headers={"X-Hermes-Session-Id": self._session_id},
                json={
                    "model": config.HERMES_MODEL,
                    "stream": True,
                    "messages": [{"role": "user", "content": text}],
                },
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"Hermes returned {response.status_code}: {body[:300]}")
                buffer = ""
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    if "tool" in event and "status" in event:
                        logger.info("Hermes tool %s: %s", event.get("tool"), event.get("status"))
                        self._mark_activity("tool_call_received")
                        continue
                    choices = event.get("choices") or [{}]
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if not delta:
                        continue
                    if not reply:
                        delta = delta.lstrip()
                    reply += delta
                    buffer += delta
                    ready, buffer = pop_sentences(buffer)
                    for sentence in ready:
                        await sentences.put(sentence)
                for sentence in pop_sentences(buffer, final=True)[0]:
                    await sentences.put(sentence)
            await sentences.put(None)
            await tts_task
        except asyncio.CancelledError:
            tts_task.cancel()
            raise
        except Exception as e:
            tts_task.cancel()
            logger.error("Hermes reply failed: %s", e)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {e}"}))
            return
        finally:
            self._last_user_activity = time.monotonic()
        reply = reply.strip()
        if reply:
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": reply}))
            self._emit_transcript("assistant", reply, True)

    async def _speak_queue(self, sentences: asyncio.Queue[str | None]) -> None:
        """Synthesize queued sentences in order."""
        previous = ""
        while (sentence := await sentences.get()) is not None:
            sentence = speakable(sentence)
            if not _WORD.search(sentence):
                continue
            await self._speak(sentence, previous_text=previous)
            previous = f"{previous} {sentence}".strip()

    def _synthesize(self, text: str, previous_text: str = "") -> AbstractAsyncContextManager[httpx.Response]:
        """Open a streaming TTS request for ``text`` (caller iterates the response)."""
        assert self._tts_http is not None and self._voice_id is not None
        body: dict[str, Any] = {
            "text": text,
            "model_id": config.ELEVENLABS_TTS_MODEL,
            "language_code": config.REALTIME_TRANSCRIPTION_LANGUAGE,
        }
        if previous_text:
            body["previous_text"] = previous_text[-500:]
        return self._tts_http.stream(
            "POST",
            f"/v1/text-to-speech/{self._voice_id}/stream",
            params={"output_format": "pcm_16000"},
            json=body,
        )

    async def _speak(self, text: str, previous_text: str = "") -> None:
        """Stream one sentence of speech to the speaker."""
        self._spoken_text = f"{self._spoken_text} {text}".strip()
        remainder = b""
        async with self._synthesize(text, previous_text) as response:
            if response.is_error:
                await response.aread()
                raise_for_elevenlabs_status(response)
            async for chunk in response.aiter_bytes():
                data = remainder + chunk
                cut = len(data) - len(data) % 2
                remainder = data[cut:]
                if cut:
                    await self._play(np.frombuffer(data[:cut], dtype=np.int16))

    async def _play(self, pcm: NDArray[np.int16]) -> None:
        """Queue audio for playback and track when it will have finished."""
        if pcm.size == 0:
            return
        pcm = apply_gain(pcm, config.TTS_GAIN)
        now = time.monotonic()
        if self._turn_started_at is not None:
            logger.info(
                "Turn latency: first audio %.0f ms after user transcript",
                (time.perf_counter() - self._turn_started_at) * 1000,
            )
            self._turn_started_at = None
        if not self._speaking:
            self._speaking = True
            self.deps.movement_manager.set_speaking(True)
        self._playback_until = max(now, self._playback_until) + pcm.size / SAMPLE_RATE
        self._mark_activity("assistant_audio_delta")
        await self.output_queue.put((SAMPLE_RATE, pcm.reshape(1, -1)))

    async def _cache_ack(self) -> None:
        """Pre-synthesize the wake acknowledgement so it plays instantly."""
        text = (config.WAKE_ACK_TEXT or "").strip()
        self._ack_pcm = None
        if not text:
            return
        try:
            audio = b""
            async with self._synthesize(text) as response:
                if response.is_error:
                    await response.aread()
                    raise_for_elevenlabs_status(response)
                async for chunk in response.aiter_bytes():
                    audio += chunk
            self._ack_pcm = np.frombuffer(audio[: len(audio) - len(audio) % 2], dtype=np.int16).copy()
        except (httpx.HTTPError, RuntimeError) as e:
            logger.warning("Could not prepare the wake acknowledgement: %s", e)

    # ---- ConversationHandler controls ----

    async def apply_personality(self, profile: str | None) -> str:
        """Record the profile; the spoken personality lives in the Hermes profile's SOUL.md."""
        set_custom_profile(profile)
        return "Saved. With the Hermes backend the personality is set by the Hermes profile (SOUL.md)."

    async def get_available_voices(self) -> list[str]:
        """Return the voice names available in the ElevenLabs account."""
        if not self._voice_ids_by_name:
            async with elevenlabs_http_client() as http:
                self._voice_ids_by_name = await fetch_voice_ids(http)
        return sorted(self._voice_ids_by_name)

    def get_current_voice(self) -> str:
        """Return the active voice name (or ID when the name is unknown)."""
        voice_id = self._voice_id or resolve_voice_id(self._voice_choice, self._voice_ids_by_name)
        return voice_name_for(voice_id, self._voice_ids_by_name) or voice_id or self._voice_choice or ""

    async def change_voice(self, voice: str) -> str:
        """Switch the TTS voice; takes effect on the next sentence."""
        if not self._voice_ids_by_name:
            await self.get_available_voices()
        known = {name.lower() for name in self._voice_ids_by_name} | set(self._voice_ids_by_name.values())
        if voice.lower() not in known and voice not in known:
            return f"Unknown ElevenLabs voice {voice!r}."
        self._voice_choice = voice
        self._voice_id = resolve_voice_id(voice, self._voice_ids_by_name)
        if self._tts_http is not None:
            await self._cache_ack()
        return f"Voice changed to {self.get_current_voice()}."

    async def say(self, text: str) -> None:
        """Speak ``text`` verbatim through TTS."""
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if self._tts_http is None:
            raise RuntimeError("say: handler not started")
        self._mark_activity("say")
        await self._speak(text)
