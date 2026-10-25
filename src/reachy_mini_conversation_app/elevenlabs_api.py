"""Small ElevenLabs REST helpers shared by the ElevenLabs-backed handlers."""

import logging
from typing import Final

import httpx

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE: Final[str] = "https://api.elevenlabs.io"
ELEVENLABS_WS_BASE: Final[str] = "wss://api.elevenlabs.io"


def elevenlabs_api_key() -> str:
    """Return the configured API key or raise a clear error."""
    api_key = (config.ELEVENLABS_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY must be set for the ElevenLabs-backed backends")
    return api_key


def elevenlabs_http_client(timeout: float = 15.0) -> httpx.AsyncClient:
    """Return an authenticated ElevenLabs REST client."""
    return httpx.AsyncClient(
        base_url=ELEVENLABS_API_BASE,
        headers={"xi-api-key": elevenlabs_api_key()},
        timeout=timeout,
    )


def raise_for_elevenlabs_status(response: httpx.Response) -> None:
    """Raise with the API error body, which names the rejected field."""
    if response.is_error:
        raise RuntimeError(
            f"ElevenLabs API {response.request.url.path} returned {response.status_code}: {response.text}"
        )


async def fetch_voice_ids(http: httpx.AsyncClient) -> dict[str, str]:
    """Fetch the account's voices as name -> voice ID."""
    response = await http.get("/v2/voices", params={"page_size": 100})
    raise_for_elevenlabs_status(response)
    voices = response.json().get("voices", [])
    return {str(v["name"]): str(v["voice_id"]) for v in voices if v.get("name")}


def resolve_voice_id(choice: str | None, voice_ids_by_name: dict[str, str]) -> str | None:
    """Map a voice name or ID to a voice ID, falling back to ELEVENLABS_VOICE_ID."""
    fallback = (config.ELEVENLABS_VOICE_ID or "").strip() or None
    if not choice:
        return fallback
    by_lowercase = {name.lower(): voice_id for name, voice_id in voice_ids_by_name.items()}
    if choice.lower() in by_lowercase:
        return by_lowercase[choice.lower()]
    if choice in voice_ids_by_name.values():
        return choice
    logger.warning("Ignoring unknown ElevenLabs voice %r; using the default voice", choice)
    return fallback


def voice_name_for(voice_id: str | None, voice_ids_by_name: dict[str, str]) -> str | None:
    """Return the name of ``voice_id`` when known."""
    for name, candidate in voice_ids_by_name.items():
        if candidate == voice_id:
            return name
    return None
