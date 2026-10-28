"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)


@pytest.fixture(autouse=True)
def _isolated_robot_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep handlers built without instance_path from writing robot events into ~/.cache."""
    import reachy_mini_conversation_app.hermes_cascade as hermes_cascade

    real = hermes_cascade.RobotEventLog
    monkeypatch.setattr(
        hermes_cascade, "RobotEventLog", lambda _path, tz=None: real(tmp_path / "robot_events.json", tz)
    )
