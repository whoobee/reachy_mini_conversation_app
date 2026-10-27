"""Timers and alarms that ring on the robot, persisted across app restarts.

The clock only keeps state and answers tool calls; the handler polls ``pop_due`` and
plays the chime and spoken message, so alarms ring even when no conversation is open.
"""

from __future__ import annotations
import json
import uuid
import logging
import datetime as dt
from typing import Any, Final, Literal, cast
from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

Repeat = Literal["none", "daily", "weekdays"]
_REPEATS: Final[tuple[str, ...]] = ("none", "daily", "weekdays")
# Alarms that came due while the app was off are still rung if they are at most this late.
_MAX_LATE_S: Final[float] = 600.0


@dataclass
class Alarm:
    """One timer or alarm."""

    id: str
    kind: Literal["timer", "alarm"]
    due: float  # epoch seconds
    label: str = ""
    repeat: Repeat = "none"
    time_of_day: str = ""  # "HH:MM" for alarms

    def describe(self, tz: ZoneInfo, now: dt.datetime) -> dict[str, Any]:
        """Return a model-friendly summary."""
        when = dt.datetime.fromtimestamp(self.due, tz)
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "rings_at": when.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds_left": max(0, round(self.due - now.timestamp())),
            "repeat": self.repeat,
        }


def _next_occurrence(time_of_day: str, repeat: Repeat, tz: ZoneInfo, after: dt.datetime) -> dt.datetime:
    """Return the next local datetime matching ``time_of_day`` and ``repeat`` strictly after ``after``."""
    hour, minute = (int(part) for part in time_of_day.split(":"))
    candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= after:
        candidate += dt.timedelta(days=1)
    while repeat == "weekdays" and candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    return candidate


class AlarmClock:
    """Timers and alarms stored in a JSON file."""

    def __init__(self, path: Path, timezone: str | None = None) -> None:
        """Load saved alarms from ``path``; ``timezone`` is an IANA name (default: system local)."""
        self._path = path
        self.tz = ZoneInfo(timezone) if timezone else _local_zone()
        self._alarms: dict[str, Alarm] = {}
        self._load()

    def _load(self) -> None:
        try:
            items = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
            logger.warning("Could not read alarms from %s: %s", self._path, e)
            return
        for item in items:
            try:
                alarm = Alarm(**item)
            except TypeError:
                continue
            self._alarms[alarm.id] = alarm

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(a) for a in self._alarms.values()], indent=1), encoding="utf-8")
        tmp.replace(self._path)

    def now(self) -> dt.datetime:
        """Return the current time in the clock's timezone."""
        return dt.datetime.now(self.tz)

    # ---- tool handlers ----

    async def set_timer(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Start a countdown."""
        try:
            seconds = float(arguments.get("duration_seconds") or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds <= 0:
            return {"error": "duration_seconds is required: the total duration in seconds, e.g. 600 for 10 minutes"}
        alarm = Alarm(
            id=uuid.uuid4().hex[:6],
            kind="timer",
            due=self.now().timestamp() + seconds,
            label=str(arguments.get("label") or ""),
        )
        self._alarms[alarm.id] = alarm
        self._save()
        return {"status": "timer set", **alarm.describe(self.tz, self.now())}

    async def set_alarm(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Set an alarm at a local time of day."""
        time_of_day = str(arguments.get("time") or "").strip()
        repeat = str(arguments.get("repeat") or "none")
        try:
            hour, minute = (int(part) for part in time_of_day.split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError
        except ValueError:
            return {"error": "time must be HH:MM in 24-hour format, e.g. 07:30"}
        if repeat not in _REPEATS:
            return {"error": f"repeat must be one of {list(_REPEATS)}"}
        normalized = f"{hour:02d}:{minute:02d}"
        repeat_value = cast(Repeat, repeat)
        alarm = Alarm(
            id=uuid.uuid4().hex[:6],
            kind="alarm",
            due=_next_occurrence(normalized, repeat_value, self.tz, self.now()).timestamp(),
            label=str(arguments.get("label") or ""),
            repeat=repeat_value,
            time_of_day=normalized,
        )
        self._alarms[alarm.id] = alarm
        self._save()
        return {"status": "alarm set", **alarm.describe(self.tz, self.now())}

    async def list_alarms(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        """List active timers and alarms, soonest first."""
        alarms = sorted(self._alarms.values(), key=lambda a: a.due)
        now = self.now()
        return {"now": now.strftime("%Y-%m-%d %H:%M:%S"), "alarms": [a.describe(self.tz, now) for a in alarms]}

    async def cancel_alarm(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Cancel by id, by label, or all."""
        target = str(arguments.get("id_or_label") or "").strip().lower()
        if not target:
            return {"error": "give an id, a label, or 'all'"}
        matches = [
            a.id for a in self._alarms.values() if target == "all" or a.id == target or a.label.lower() == target
        ]
        for alarm_id in matches:
            del self._alarms[alarm_id]
        if matches:
            self._save()
        return {"cancelled": len(matches), "remaining": len(self._alarms)}

    # ---- ringing ----

    def pop_due(self) -> list[Alarm]:
        """Return alarms that are due now; repeating alarms are rescheduled, others removed."""
        now = self.now()
        due = [a for a in self._alarms.values() if a.due <= now.timestamp()]
        if not due:
            return []
        ring: list[Alarm] = []
        for alarm in due:
            if now.timestamp() - alarm.due <= _MAX_LATE_S:
                ring.append(alarm)
            else:
                logger.info("Skipping alarm %s missed while the app was off", alarm.id)
            if alarm.kind == "alarm" and alarm.repeat != "none":
                alarm.due = _next_occurrence(alarm.time_of_day, alarm.repeat, self.tz, now).timestamp()
            else:
                del self._alarms[alarm.id]
        self._save()
        return ring

    def announcement(self, alarm: Alarm) -> str:
        """Return the Romanian sentence spoken when ``alarm`` rings."""
        if alarm.kind == "timer":
            return f"Timpul a expirat: {alarm.label}!" if alarm.label else "Timpul a expirat!"
        return f"Este ora {alarm.time_of_day}. {alarm.label}!" if alarm.label else f"Este ora {alarm.time_of_day}!"


def _local_zone() -> ZoneInfo:
    """Return the system timezone as a ZoneInfo (UTC if it cannot be determined)."""
    try:
        name = Path("/etc/timezone").read_text(encoding="utf-8").strip()
        if name:
            return ZoneInfo(name)
    except (OSError, ValueError):
        pass
    try:
        link = Path("/etc/localtime").resolve()
        return ZoneInfo(str(link).split("zoneinfo/", 1)[1])
    except (OSError, IndexError, ValueError):
        return ZoneInfo("UTC")


def chime(sample_rate: int = 16000, repeats: int = 3) -> NDArray[np.int16]:
    """Return a short two-tone chime as int16 PCM."""

    def tone(freq: float, seconds: float) -> NDArray[np.float32]:
        t = np.arange(int(sample_rate * seconds)) / sample_rate
        envelope = np.minimum(1.0, np.minimum(t, seconds - t) * 40)
        wave: NDArray[np.float32] = (np.sin(2 * np.pi * freq * t) * envelope * 0.4).astype(np.float32)
        return wave

    gap = np.zeros(int(sample_rate * 0.08), dtype=np.float32)
    pattern = np.concatenate([tone(880, 0.18), gap, tone(1320, 0.28), np.zeros(int(sample_rate * 0.35), np.float32)])
    pcm: NDArray[np.int16] = (np.tile(pattern, repeats) * 32767).astype(np.int16)
    return pcm


TOOL_DEFINITIONS: Final[dict[str, tuple[str, dict[str, Any]]]] = {
    "set_timer": (
        "Start a countdown timer on Reachy. When it ends Reachy chimes and says the label out loud. "
        "Always pass duration_seconds as the total duration in seconds (10 minutes = 600, 1 hour = 3600).",
        {
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Total duration in seconds, e.g. 30, 600 for 10 minutes, 5400 for 1.5 hours.",
                },
                "label": {"type": "string", "description": "Short name spoken when it ends, e.g. 'paste'."},
            },
            "required": ["duration_seconds"],
        },
    ),
    "set_alarm": (
        "Set an alarm at a local time of day (24-hour HH:MM). Reachy chimes and speaks at that time. "
        "Use repeat 'daily' or 'weekdays' for recurring alarms.",
        {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "Local time HH:MM, e.g. '07:30'."},
                "label": {"type": "string", "description": "What the alarm is for, spoken when it rings."},
                "repeat": {"type": "string", "enum": list(_REPEATS), "description": "Repetition, default 'none'."},
            },
            "required": ["time"],
        },
    ),
    "list_alarms": (
        "List Reachy's active timers and alarms with the time left, soonest first. Also returns the current local time.",
        {"type": "object", "properties": {}},
    ),
    "cancel_alarm": (
        "Cancel a timer or alarm by its id or label, or everything with 'all'.",
        {
            "type": "object",
            "properties": {"id_or_label": {"type": "string", "description": "Alarm id, its label, or 'all'."}},
            "required": ["id_or_label"],
        },
    ),
}
