import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

from reachy_mini_conversation_app.robot_events import MAX_EVENTS, RobotEventLog


TZ = ZoneInfo("Europe/Berlin")


def test_events_survive_a_restart_and_are_sent_once(tmp_path: Path) -> None:
    """Events are kept on disk until taken, then cleared."""
    RobotEventLog(tmp_path / "e.json", TZ).note("a adormit prin apăsarea antenei")
    log = RobotEventLog(tmp_path / "e.json", TZ)  # e.g. the app restarted after sleeping

    prefix = log.take_prefix()
    assert prefix.startswith("[Reachy: ") and prefix.endswith("a adormit prin apăsarea antenei]")
    assert not log.pending()
    assert RobotEventLog(tmp_path / "e.json", TZ).take_prefix() == ""


def test_older_days_show_the_date_and_the_list_is_capped(tmp_path: Path) -> None:
    """Events from another day carry their date; only the newest MAX_EVENTS are kept."""
    yesterday = (dt.datetime.now(TZ) - dt.timedelta(days=1)).replace(hour=7, minute=0)
    (tmp_path / "e.json").write_text(f'[["{yesterday.isoformat()}", "a sunat alarma"]]', encoding="utf-8")
    log = RobotEventLog(tmp_path / "e.json", TZ)
    assert f"{yesterday:%d.%m} 07:00 a sunat alarma" in log.take_prefix()

    for i in range(MAX_EVENTS + 5):
        log.note(f"e{i}")
    prefix = log.take_prefix()
    assert "e4;" not in prefix and f"e{MAX_EVENTS + 4}]" in prefix
