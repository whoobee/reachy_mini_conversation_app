import asyncio
import datetime as dt
from typing import Any
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.alarms as alarms_mod
from reachy_mini_conversation_app.alarms import AlarmClock, chime
from reachy_mini_conversation_app.hermes_cascade import HermesCascadeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


TZ = "Europe/Bucharest"


def _freeze(monkeypatch: Any, clock: AlarmClock, when: dt.datetime) -> None:
    monkeypatch.setattr(clock, "now", lambda: when)


@pytest.mark.asyncio
async def test_timer_rings_once_after_its_duration(tmp_path: Path, monkeypatch: Any) -> None:
    """A timer is due after its duration and is removed once it has rung."""
    clock = AlarmClock(tmp_path / "alarms.json", TZ)
    start = dt.datetime(2026, 10, 26, 9, 0, tzinfo=ZoneInfo(TZ))
    _freeze(monkeypatch, clock, start)

    result = await clock.set_timer({"duration_seconds": 600, "label": "paste"})
    assert result["seconds_left"] == 600

    _freeze(monkeypatch, clock, start + dt.timedelta(minutes=9))
    assert clock.pop_due() == []
    _freeze(monkeypatch, clock, start + dt.timedelta(minutes=10, seconds=1))
    rung = clock.pop_due()
    assert [a.label for a in rung] == ["paste"]
    assert clock.announcement(rung[0]) == "Timpul a expirat: paste!"
    assert (await clock.list_alarms({}))["alarms"] == []


@pytest.mark.asyncio
async def test_weekday_alarm_skips_the_weekend_and_reschedules(tmp_path: Path, monkeypatch: Any) -> None:
    """A weekdays alarm set on Friday evening rings Monday, then moves to Tuesday."""
    clock = AlarmClock(tmp_path / "alarms.json", TZ)
    friday_evening = dt.datetime(2026, 10, 23, 20, 0, tzinfo=ZoneInfo(TZ))
    _freeze(monkeypatch, clock, friday_evening)

    result = await clock.set_alarm({"time": "7:30", "label": "trezirea", "repeat": "weekdays"})
    assert result["rings_at"] == "2026-10-26 07:30:00"

    _freeze(monkeypatch, clock, dt.datetime(2026, 10, 26, 7, 30, 5, tzinfo=ZoneInfo(TZ)))
    assert [a.time_of_day for a in clock.pop_due()] == ["07:30"]
    assert (await clock.list_alarms({}))["alarms"][0]["rings_at"] == "2026-10-27 07:30:00"


@pytest.mark.asyncio
async def test_alarms_survive_restart_and_can_be_cancelled_by_label(tmp_path: Path) -> None:
    """Alarms are saved to disk, reloaded by a new clock, and cancelled by label."""
    first = AlarmClock(tmp_path / "alarms.json", TZ)
    await first.set_timer({"duration_seconds": 3600, "label": "Cuptor"})
    await first.set_alarm({"time": "06:45"})

    second = AlarmClock(tmp_path / "alarms.json", TZ)
    assert len((await second.list_alarms({}))["alarms"]) == 2
    assert await second.cancel_alarm({"id_or_label": "cuptor"}) == {"cancelled": 1, "remaining": 1}
    assert await second.cancel_alarm({"id_or_label": "all"}) == {"cancelled": 1, "remaining": 0}


@pytest.mark.asyncio
async def test_invalid_input_and_long_missed_alarms(tmp_path: Path, monkeypatch: Any) -> None:
    """Bad times are rejected; a one-off alarm missed by more than ten minutes is dropped silently."""
    clock = AlarmClock(tmp_path / "alarms.json", TZ)
    assert "error" in await clock.set_alarm({"time": "25:00"})
    assert "error" in await clock.set_timer({})

    start = dt.datetime(2026, 10, 26, 9, 0, tzinfo=ZoneInfo(TZ))
    _freeze(monkeypatch, clock, start)
    await clock.set_timer({"duration_seconds": 30})
    _freeze(monkeypatch, clock, start + dt.timedelta(hours=2))
    assert clock.pop_due() == []
    assert (await clock.list_alarms({}))["alarms"] == []


def test_chime_is_audible_int16() -> None:
    """The chime is a few seconds of non-silent 16 kHz PCM."""
    pcm = chime(16000, repeats=2)
    assert pcm.dtype.name == "int16" and 16000 < pcm.size < 16000 * 3
    assert abs(int(pcm.max())) > 5000


@pytest.mark.asyncio
async def test_handler_rings_due_alarms_from_its_supervisor(tmp_path: Path, monkeypatch: Any) -> None:
    """The handler's supervisor loop plays the chime and the spoken announcement for due alarms."""
    handler = HermesCascadeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()), instance_path=str(tmp_path)
    )
    handler._tts_http = MagicMock()
    play, speak = AsyncMock(), AsyncMock()
    monkeypatch.setattr(handler, "_play", play)
    monkeypatch.setattr(handler, "_speak", speak)
    await handler._alarm_clock.set_timer({"duration_seconds": 1, "label": "ceai"})
    monkeypatch.setattr(alarms_mod.dt, "datetime", _Later)

    await handler._supervise()
    for task in [t for t in asyncio.all_tasks() if t.get_name().startswith("alarm-")]:
        await task

    play.assert_awaited_once()
    speak.assert_awaited_once_with("Timpul a expirat: ceai!")


class _Later(dt.datetime):
    """datetime whose now() is ten seconds in the future."""

    @classmethod
    def now(cls, tz: Any = None) -> "_Later":
        return super().now(tz) + dt.timedelta(seconds=10)  # type: ignore[return-value]
