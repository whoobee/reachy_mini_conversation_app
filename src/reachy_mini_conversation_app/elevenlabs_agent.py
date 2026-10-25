"""Conversation handler backed by an ElevenLabs Agent (Scribe STT, LLM, ElevenLabs TTS).

ElevenLabs runs speech recognition, turn-taking, barge-in, the LLM and speech
synthesis. The app streams microphone audio over the Agents WebSocket, plays the
returned audio, and runs robot tools locally as ElevenLabs "client tools".

Before each session the agent is created or updated from the active profile
(prompt, tools, language, voice), so the ElevenLabs dashboard needs no manual setup.
"""

import json
import time
import base64
import random
import asyncio
import logging
from typing import Any, Final, Optional
from pathlib import Path

import httpx
import numpy as np
import websockets
from numpy.typing import NDArray
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.config import config, set_custom_profile
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_greeting_prompt
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.tools.core_tools import ToolSpec, ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.conversation_handler import QueueItem, AudioFrame, ConversationHandler
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE: Final[str] = "https://api.elevenlabs.io"
ELEVENLABS_AUDIO_FORMAT: Final[str] = "pcm_16000"
ELEVENLABS_AGENT_NAME: Final[str] = "Reachy Mini conversation app"
AGENT_ID_FILENAME: Final[str] = "elevenlabs_agent_id"
# Client tool results are text only, so the camera image cannot reach the LLM.
UNSUPPORTED_TOOLS: Final[frozenset[str]] = frozenset({"camera"})
_TOOL_RESPONSE_TIMEOUT_S: Final[int] = 20
_VAD_SPEECH_THRESHOLD: Final[float] = 0.8


def _with_descriptions(schema: dict[str, Any], name: str = "value") -> dict[str, Any]:
    """Return ``schema`` with a description on every property, which ElevenLabs requires."""
    result = dict(schema)
    if "description" not in result:
        result["description"] = name.replace("_", " ")
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["properties"] = {
            key: _with_descriptions(value, key) if isinstance(value, dict) else value
            for key, value in properties.items()
        }
    items = result.get("items")
    if isinstance(items, dict):
        result["items"] = _with_descriptions(items, f"{name} item")
    return result


def to_client_tools(tool_specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert app tool specs to ElevenLabs client tool definitions."""
    tools = core_tools.get_tools()
    client_tools: list[dict[str, Any]] = []
    for spec in tool_specs:
        if spec["name"] in UNSUPPORTED_TOOLS:
            continue
        tool = tools.get(spec["name"])
        client_tools.append(
            {
                "type": "client",
                "name": spec["name"],
                "description": spec["description"],
                "parameters": _with_descriptions(spec["parameters"], spec["name"]),
                "expects_response": tool.needs_response if tool is not None else True,
                "response_timeout_secs": _TOOL_RESPONSE_TIMEOUT_S,
            }
        )
    return client_tools


def build_agent_body(instructions: str, tool_specs: list[ToolSpec], voice_id: str | None) -> dict[str, Any]:
    """Return the create/update agent payload for the active profile."""
    tts: dict[str, Any] = {
        "model_id": config.ELEVENLABS_TTS_MODEL,
        "agent_output_audio_format": ELEVENLABS_AUDIO_FORMAT,
    }
    if voice_id:
        tts["voice_id"] = voice_id
    return {
        "name": ELEVENLABS_AGENT_NAME,
        "conversation_config": {
            "asr": {"user_input_audio_format": ELEVENLABS_AUDIO_FORMAT},
            "tts": tts,
            "agent": {
                # The app sends the profile greeting prompt itself once connected.
                "first_message": "",
                "language": config.REALTIME_TRANSCRIPTION_LANGUAGE,
                "prompt": {
                    "prompt": instructions,
                    "llm": config.ELEVENLABS_LLM,
                    "tools": to_client_tools(tool_specs),
                },
            },
        },
    }


def _resample_to(audio: NDArray[np.int16], source_rate: int, target_rate: int) -> NDArray[np.int16]:
    """Linearly resample mono int16 audio."""
    if source_rate == target_rate or audio.size == 0:
        return audio
    target_size = max(1, round(audio.size * target_rate / source_rate))
    positions = np.linspace(0, audio.size - 1, target_size)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.int16)


class ElevenLabsAgentHandler(ConversationHandler):
    """Realtime stream handler for the ElevenLabs Agents WebSocket API."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__()
        self.deps = deps
        self.instance_path = instance_path
        self.output_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.tool_manager = BackgroundToolManager()
        self.connection: Any = None
        self._connected_event = asyncio.Event()
        self._session_task: asyncio.Task[None] | None = None

        # A voice name or ID; persisted HF voice names that ElevenLabs lacks fall back to the default.
        self._voice_choice: str | None = (startup_voice or "").strip() or None
        self._voice_ids_by_name: dict[str, str] = {}
        self._active_voice_id: str | None = None

        self._expects_response: dict[str, bool] = {}
        self._last_interrupt_event_id = 0
        self._agent_speaking = False
        self._user_speaking = False
        self._startup_greeting_sent = False
        self._turn_user_done_at: float | None = None
        self._turn_first_audio_at: float | None = None

    # ---- ElevenLabs REST helpers ----

    def _api_key(self) -> str:
        api_key = (config.ELEVENLABS_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY must be set when REALTIME_BACKEND=elevenlabs")
        return api_key

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=ELEVENLABS_API_BASE,
            headers={"xi-api-key": self._api_key()},
            timeout=15.0,
        )

    def _agent_id_path(self) -> Path:
        root = (
            Path(self.instance_path) if self.instance_path else Path.home() / ".cache" / "reachy_mini_conversation_app"
        )
        return root / AGENT_ID_FILENAME

    def _stored_agent_id(self) -> str | None:
        configured = (config.ELEVENLABS_AGENT_ID or "").strip()
        if configured:
            return configured
        try:
            return self._agent_id_path().read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise with the API error body, which names the rejected field."""
        if response.is_error:
            raise RuntimeError(
                f"ElevenLabs API {response.request.url.path} returned {response.status_code}: {response.text}"
            )

    async def _ensure_agent(self, http: httpx.AsyncClient, body: dict[str, Any]) -> str:
        """Update the configured agent, or create one and remember its ID."""
        agent_id = self._stored_agent_id()
        if agent_id:
            response = await http.patch(f"/v1/convai/agents/{agent_id}", json=body)
            if response.status_code != 404:
                self._raise_for_status(response)
                return agent_id
            logger.warning("ElevenLabs agent %s no longer exists; creating a new one", agent_id)

        response = await http.post("/v1/convai/agents/create", json=body)
        self._raise_for_status(response)
        agent_id = str(response.json()["agent_id"])
        path = self._agent_id_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(agent_id, encoding="utf-8")
        logger.info("Created ElevenLabs agent %s (ID saved to %s)", agent_id, path)
        return agent_id

    async def _load_voices(self, http: httpx.AsyncClient) -> dict[str, str]:
        """Fetch the account's voices as name -> voice ID."""
        response = await http.get("/v2/voices", params={"page_size": 100})
        self._raise_for_status(response)
        voices = response.json().get("voices", [])
        self._voice_ids_by_name = {str(v["name"]): str(v["voice_id"]) for v in voices if v.get("name")}
        return self._voice_ids_by_name

    def _resolve_voice_id(self, choice: str | None) -> str | None:
        """Map a voice name or ID to a voice ID, falling back to ELEVENLABS_VOICE_ID."""
        fallback = (config.ELEVENLABS_VOICE_ID or "").strip() or None
        if not choice:
            return fallback
        by_lowercase = {name.lower(): voice_id for name, voice_id in self._voice_ids_by_name.items()}
        if choice.lower() in by_lowercase:
            return by_lowercase[choice.lower()]
        if choice in self._voice_ids_by_name.values():
            return choice
        logger.warning("Ignoring unknown ElevenLabs voice %r; using the default voice", choice)
        return fallback

    # ---- ConversationHandler contract ----

    def _is_connected(self) -> bool:
        """Return whether the agent WebSocket is open."""
        return self.connection is not None

    def _idle_behavior_ready(self) -> bool:
        """Hold idle behavior while the agent is talking."""
        return not self._agent_speaking

    async def start_up(self) -> None:
        """Run agent sessions, retrying on abrupt WebSocket closure."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_session()
                return
            except ConnectionClosedError as e:
                logger.warning(
                    "ElevenLabs websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e
                )
                if attempt == max_attempts:
                    raise
                delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
                logger.info("Retrying in %.1f seconds...", delay)
                await asyncio.sleep(delay)

    async def _restart_session(self) -> None:
        """Close the current session and open a new one (re-syncing the agent config)."""
        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception:
                pass
            self.connection = None
        self._connected_event.clear()
        self._session_task = asyncio.create_task(self._run_session(), name="elevenlabs-session-restart")
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=10.0)
            logger.info("ElevenLabs session restarted and connected.")
        except asyncio.TimeoutError:
            logger.warning("ElevenLabs session restart timed out; continuing in background.")

    async def _run_session(self) -> None:
        """Sync the agent, then run one conversation over the Agents WebSocket."""
        tool_specs = get_tool_specs()
        instructions = get_session_instructions(self.instance_path)
        async with self._http_client() as http:
            try:
                await self._load_voices(http)
            except httpx.HTTPError as e:
                logger.warning("Could not list ElevenLabs voices: %s", e)
            self._active_voice_id = self._resolve_voice_id(self._voice_choice)
            agent_id = await self._ensure_agent(
                http, build_agent_body(instructions, tool_specs, self._active_voice_id)
            )
            response = await http.get("/v1/convai/conversation/get-signed-url", params={"agent_id": agent_id})
            self._raise_for_status(response)
            signed_url = str(response.json()["signed_url"])

        logger.info(
            "ElevenLabs agent %s synced: profile=%r llm=%s tts=%s language=%s voice=%s tools=%s",
            agent_id,
            getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
            config.ELEVENLABS_LLM,
            config.ELEVENLABS_TTS_MODEL,
            config.REALTIME_TRANSCRIPTION_LANGUAGE,
            self._active_voice_id or "<agent default>",
            [tool["name"] for tool in to_client_tools(tool_specs)],
        )

        async with websockets.connect(signed_url, max_size=None) as ws:
            await ws.send(json.dumps({"type": "conversation_initiation_client_data"}))
            self.connection = ws
            self._last_interrupt_event_id = 0
            self._expects_response.clear()
            self._connected_event.set()
            self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    await self._handle_event(json.loads(raw))
            finally:
                if self.connection is ws:
                    self.connection = None
                self._agent_speaking = False
                self.deps.movement_manager.set_speaking(False)
                await self.tool_manager.shutdown()

    async def _send(self, message: dict[str, Any]) -> None:
        if self.connection is None:
            raise RuntimeError("no active ElevenLabs session")
        await self.connection.send(json.dumps(message))

    async def _send_startup_greeting_prompt(self) -> None:
        """Ask the agent to open the conversation once, using the profile greeting prompt."""
        if self._startup_greeting_sent:
            return
        self._startup_greeting_sent = True
        greeting_prompt = get_session_greeting_prompt().strip()
        if greeting_prompt:
            await self._send({"type": "user_message", "text": greeting_prompt})
            self._mark_activity("startup_greeting_prompt")
            logger.info("Sent startup greeting prompt")

    async def _handle_event(self, event: dict[str, Any]) -> None:
        """Dispatch one server event."""
        event_type = event.get("type")

        if event_type == "ping":
            ping = event.get("ping_event", {})
            await self._send({"type": "pong", "event_id": ping.get("event_id")})

        elif event_type == "conversation_initiation_metadata":
            meta = event.get("conversation_initiation_metadata_event", {})
            logger.info(
                "ElevenLabs conversation %s started (in=%s out=%s)",
                meta.get("conversation_id"),
                meta.get("user_input_audio_format"),
                meta.get("agent_output_audio_format"),
            )
            if meta.get("agent_output_audio_format") != ELEVENLABS_AUDIO_FORMAT:
                logger.warning("Unexpected agent audio format %r", meta.get("agent_output_audio_format"))
            await self._send_startup_greeting_prompt()

        elif event_type == "audio":
            audio_event = event.get("audio_event", {})
            if int(audio_event.get("event_id") or 0) <= self._last_interrupt_event_id:
                return  # stale audio from an interrupted response
            pcm = np.frombuffer(base64.b64decode(audio_event.get("audio_base_64", "")), dtype=np.int16)
            if pcm.size == 0:
                return
            if not self._agent_speaking:
                self._agent_speaking = True
                self.deps.movement_manager.set_speaking(True)
            if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
                self._turn_first_audio_at = time.perf_counter()
                logger.info(
                    "Turn latency: first audio %.0f ms after user transcript",
                    (self._turn_first_audio_at - self._turn_user_done_at) * 1000,
                )
            self._mark_activity("assistant_audio_delta")
            await self.output_queue.put((self.SAMPLE_RATE, pcm.reshape(1, -1)))

        elif event_type == "agent_response":
            text = (event.get("agent_response_event", {}).get("agent_response") or "").strip()
            if text:
                await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
                self._emit_transcript("assistant", text, True)

        elif event_type == "agent_response_complete":
            self._agent_speaking = False
            self.deps.movement_manager.set_speaking(False)
            self._mark_activity("assistant_transcript_done")

        elif event_type == "agent_response_correction":
            correction = event.get("agent_response_correction_event", {})
            logger.debug("Agent response truncated to: %s", correction.get("corrected_agent_response"))

        elif event_type == "tentative_user_transcript":
            text = (event.get("tentative_user_transcription_event", {}).get("user_transcript") or "").strip()
            if text:
                self._mark_activity("user_transcription_delta")
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": text}))

        elif event_type == "user_transcript":
            text = (event.get("user_transcription_event", {}).get("user_transcript") or "").strip()
            self._user_speaking = False
            self.deps.movement_manager.set_listening(False)
            if not text:
                return
            self._mark_activity("user_transcription_completed")
            self._turn_user_done_at = time.perf_counter()
            self._turn_first_audio_at = None
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": text}))
            self._emit_transcript("user", text, True)

        elif event_type == "vad_score":
            score = float(event.get("vad_score_event", {}).get("vad_score") or 0.0)
            if score >= _VAD_SPEECH_THRESHOLD and not self._user_speaking:
                self._user_speaking = True
                self.deps.movement_manager.set_listening(True)
                self._mark_activity("user_speech_started")

        elif event_type == "interruption":
            self._last_interrupt_event_id = int(event.get("interruption_event", {}).get("event_id") or 0)
            if self._clear_queue:
                self._clear_queue()
            self._agent_speaking = False
            self.deps.movement_manager.set_speaking(False)
            self._mark_activity("user_speech_started")
            logger.debug("Agent interrupted by user")

        elif event_type == "client_tool_call":
            await self._start_tool_call(event.get("client_tool_call", {}))

        elif event_type == "client_error":
            error = event.get("error_event", {})
            message = error.get("message") or error.get("error_name") or "unknown error"
            logger.error("ElevenLabs error [%s]: %s", error.get("code"), message)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {message}"}))

        else:
            logger.debug("ElevenLabs event: %s", event_type)

    async def _start_tool_call(self, call: dict[str, Any]) -> None:
        """Run a client tool requested by the agent in the background."""
        tool_name = call.get("tool_name")
        call_id = str(call.get("tool_call_id") or "")
        if not isinstance(tool_name, str) or not call_id:
            logger.error("Invalid client tool call: %s", call)
            return
        args_json_str = json.dumps(call.get("parameters") or {})
        self._expects_response[call_id] = bool(call.get("expects_response", True))
        self._mark_activity("tool_call_received")
        logger.info("Tool call received — tool_name=%r, call_id=%s, args=%s", tool_name, call_id, args_json_str)
        background_tool = await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(tool_name=tool_name, args_json_str=args_json_str, deps=self.deps),
            is_idle_tool_call=False,
        )
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. Tool ID: {background_tool.tool_id}",
                }
            )
        )

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Report a finished tool back to the agent when it is waiting for the result."""
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed: %s", completed_tool.tool_name, completed_tool.id, completed_tool.error
            )
            result: Any = {"error": completed_tool.error}
        elif completed_tool.result is not None:
            result = completed_tool.result
            if isinstance(result, dict) and "b64_im" in result:
                result = {key: value for key, value in result.items() if key != "b64_im"}
            logger.info("Tool '%s' (id=%s) executed successfully.", completed_tool.tool_name, completed_tool.id)
        else:
            result = {"error": "No result returned from tool execution"}

        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": json.dumps(result)}))
        if completed_tool.is_idle_tool_call:
            return
        self._mark_activity("tool_result_ready")
        if not self._expects_response.pop(completed_tool.id, True):
            return
        try:
            await self._send(
                {
                    "type": "client_tool_result",
                    "tool_call_id": completed_tool.id,
                    "result": json.dumps(result),
                    "is_error": completed_tool.error is not None,
                }
            )
        except (RuntimeError, ConnectionClosed) as e:
            logger.warning("Could not send result of tool '%s': %s", completed_tool.tool_name, e)

    async def receive(self, frame: AudioFrame) -> None:
        """Stream one microphone frame to the agent as 16 kHz mono PCM."""
        if self.connection is None:
            return
        sample_rate, audio_frame = frame
        if audio_frame.size == 0:
            return
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            audio_frame = audio_frame[:, 0]
        pcm = _resample_to(audio_to_int16(audio_frame), sample_rate, self.SAMPLE_RATE)
        try:
            await self.connection.send(json.dumps({"user_audio_chunk": base64.b64encode(pcm.tobytes()).decode()}))
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)

    async def shutdown(self) -> None:
        """Close the session and stop background tools."""
        await self.tool_manager.shutdown()
        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("connection.close() ignored: %s", e)
            self.connection = None
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality; the agent is re-synced on the restarted session."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            get_session_instructions(self.instance_path)
            core_tools.initialize_tools(force=True)
        except Exception as exc:
            set_custom_profile(previous_profile)
            logger.error("Failed to resolve personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"
        if self.connection is None:
            return "Applied personality. Will take effect on next connection."
        await self._restart_session()
        return "Applied personality and restarted ElevenLabs session."

    async def get_available_voices(self) -> list[str]:
        """Return the voice names available in the ElevenLabs account."""
        if not self._voice_ids_by_name:
            async with self._http_client() as http:
                await self._load_voices(http)
        return sorted(self._voice_ids_by_name)

    def get_current_voice(self) -> str:
        """Return the active voice name (or ID when the name is unknown)."""
        voice_id = self._active_voice_id or self._resolve_voice_id(self._voice_choice)
        for name, candidate in self._voice_ids_by_name.items():
            if candidate == voice_id:
                return name
        return voice_id or self._voice_choice or ""

    async def change_voice(self, voice: str) -> str:
        """Switch the agent voice; takes effect by restarting the session."""
        if not self._voice_ids_by_name:
            await self.get_available_voices()
        known = {name.lower() for name in self._voice_ids_by_name} | set(self._voice_ids_by_name.values())
        if voice.lower() not in known and voice not in known:
            return f"Unknown ElevenLabs voice {voice!r}."
        self._voice_choice = voice
        if self.connection is None:
            return "Voice changed. Will take effect on next connection."
        await self._restart_session()
        return f"Voice changed to {self.get_current_voice()}."

    async def say(self, text: str) -> None:
        """Inject ``text`` as a user turn the agent answers now (not verbatim TTS)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        await self._send({"type": "user_message", "text": text})
        self._mark_activity("say")
