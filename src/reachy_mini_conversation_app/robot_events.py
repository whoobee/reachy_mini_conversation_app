"""Robot-side events (wake, sleep, alarms...) waiting to be told to the Hermes agent.

Things that happen on the robot without a Hermes turn would otherwise only reach the robot's
own log. They are kept here (on disk, so they survive the app stopping when Reachy sleeps) and
sent ahead of the next user message, so the Hermes session holds the whole story and the agent
can answer questions like "did my alarm ring?".
"""

from __future__ import annotations
import json
import logging
import datetime as dt
from typing import Final
from pathlib import Path
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

MAX_EVENTS: Final[int] = 30
PREFIX: Final[str] = "[Reachy"


class RobotEventLog:
    """Pending robot events, persisted to a small JSON file."""

    def __init__(self, path: Path, tz: ZoneInfo | dt.tzinfo | None = None) -> None:
        """Keep pending events in ``path``; times are shown in ``tz``."""
        self._path = path
        self._tz = tz
        self._events: list[tuple[str, str]] = self._load()

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self._tz)

    def _load(self) -> list[tuple[str, str]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [(str(when), str(text)) for when, text in data][-MAX_EVENTS:]
        except (OSError, ValueError, TypeError):
            return []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._events, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            logger.debug("Could not save robot events: %s", e)

    def note(self, text: str) -> None:
        """Record one event now."""
        self._events = [*self._events, (self._now().isoformat(timespec="seconds"), text)][-MAX_EVENTS:]
        self._save()

    def pending(self) -> bool:
        """Whether any event waits to be sent."""
        return bool(self._events)

    def take_prefix(self) -> str:
        """Return the events as one bracketed line for the next message and forget them."""
        if not self._events:
            return ""
        today = self._now().date()
        parts = []
        for when, text in self._events:
            try:
                stamp = dt.datetime.fromisoformat(when)
                if self._tz is not None:
                    stamp = stamp.astimezone(self._tz)
                shown = stamp.strftime("%H:%M") if stamp.date() == today else stamp.strftime("%d.%m %H:%M")
            except ValueError:
                shown = when
            parts.append(f"{shown} {text}")
        self._events = []
        self._save()
        return f"{PREFIX}: " + "; ".join(parts) + "]"
