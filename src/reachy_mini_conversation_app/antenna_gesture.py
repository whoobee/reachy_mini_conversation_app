"""Put Reachy to sleep when an antenna is pushed down by hand during the idle animation.

The gesture only arms while the idle breathing animation is the only motion and the
conversation is not busy, so emotions, dances or speech can never trigger it.

To avoid fighting the motor (holding an antenna against its servo overloads it), the
antenna *gives way*: once a push is felt, its command follows the measured position,
so the motor just holds wherever the hand puts it. When the antenna reaches the "down"
position and stays there briefly, ``on_trigger`` runs. If the hand lets go first, the
antenna is handed back to the animation after it has been still for a moment.

Measured on the robot: "down" is negative for antenna[0] and positive for antenna[1];
antenna[0] stops near -0.74 rad, and the breathing animation stays within about +-0.27 rad.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Any, Final
from collections.abc import Callable


logger = logging.getLogger(__name__)

# Sign that turns a position (or a measured-minus-commanded difference) into "down" for each antenna.
DOWN_SIGNS: Final[tuple[float, float]] = (-1.0, 1.0)


def down_amount(values: tuple[float, float]) -> tuple[float, float]:
    """Return each antenna's value in the "down" direction."""
    return DOWN_SIGNS[0] * values[0], DOWN_SIGNS[1] * values[1]


class AntennaSleepGesture:
    """Background watcher: push an antenna down (it gives way) and hold it there to sleep."""

    def __init__(
        self,
        robot: Any,
        movement_manager: Any,
        on_trigger: Callable[[], Any],
        is_busy: Callable[[], bool] = lambda: False,
        yield_rad: float = 0.15,
        down_position_rad: float = 0.6,
        hold_s: float = 0.3,
        release_after_s: float = 1.5,
        cooldown_s: float = 1.0,
        poll_s: float = 0.05,
        arm_after_s: float = 5.0,
    ) -> None:
        """Watch ``robot``'s antennas against ``movement_manager``'s commanded pose."""
        self._robot = robot
        self._movement_manager = movement_manager
        self._on_trigger = on_trigger
        self._is_busy = is_busy
        self.yield_rad = yield_rad
        self.down_position_rad = down_position_rad
        self.hold_s = hold_s
        self.release_after_s = release_after_s
        self.cooldown_s = cooldown_s
        self._cooldown_until = 0.0
        self._poll_s = poll_s
        self._arm_after_s = arm_after_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.following: int | None = None  # index of the antenna currently following the hand
        self._down_since: float | None = None
        self._last_position: float | None = None
        self._still_since: float | None = None
        self.triggered = False
        # Diagnostics, logged periodically: why the gesture did or did not engage.
        self._stats: dict[str, float] = {}
        self._stats_since = 0.0

    def _count(self, key: str, value: float = 1.0) -> None:
        self._stats[key] = self._stats.get(key, 0.0) + value

    def _log_stats(self, now: float, every_s: float = 10.0) -> None:
        if now - self._stats_since < every_s:
            return
        if self._stats.get("errors") or self._stats.get("max_push", 0.0) >= self.yield_rad:
            logger.info(
                "Antenna gesture last %.0fs: polls=%d idle_animation=%d busy=%d read_errors=%d max_push=%.2f rad",
                now - self._stats_since,
                self._stats.get("polls", 0),
                self._stats.get("idle", 0),
                self._stats.get("busy", 0),
                self._stats.get("errors", 0),
                self._stats.get("max_push", 0.0),
            )
        self._stats = {}
        self._stats_since = now

    def start(self) -> None:
        """Start watching in a daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="antenna-sleep-gesture", daemon=True)
        self._thread.start()
        logger.info(
            "Antenna sleep gesture armed: during idle, push an antenna down past %.2f rad for %.1f s",
            self.down_position_rad,
            self.hold_s,
        )

    def stop(self) -> None:
        """Stop watching and hand any followed antenna back to the animation."""
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._release()

    def _release(self) -> None:
        if self.following is not None:
            self._movement_manager.set_antenna_override(self.following, None)
        self.following = None
        self._down_since = self._last_position = self._still_since = None

    def _read(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        present = self._robot.get_present_antenna_joint_positions()
        status = self._movement_manager.get_status() or {}
        commanded = (status.get("last_commanded_pose") or {}).get("antennas")
        if present is None or commanded is None or len(present) != 2 or len(commanded) != 2:
            return None
        return (float(present[0]), float(present[1])), (float(commanded[0]), float(commanded[1]))

    def step(self, now: float) -> bool:
        """Advance the gesture once; return True when it fires (exposed for tests)."""
        if self.triggered:
            return False
        self._count("polls")
        try:
            reading = self._read()
        except Exception as e:
            self._count("errors")
            if self._stats.get("errors", 0) <= 1:
                logger.warning("Antenna read failed: %s", e)
            reading = None
        if reading is None:
            self._release()
            return False
        present, commanded = reading

        if self.following is None:
            pushed = [down_amount(present)[i] - down_amount(commanded)[i] for i in (0, 1)]
            self._stats["max_push"] = max(self._stats.get("max_push", 0.0), *pushed)
            busy = self._is_busy()
            idle = self._movement_manager.is_idle_animation()
            self._count("busy", float(busy))
            self._count("idle", float(idle))
            if now < self._cooldown_until or busy or not idle:
                return False
            index = 0 if pushed[0] >= pushed[1] else 1
            if pushed[index] < self.yield_rad:
                return False
            self.following = index  # give way: from now on the antenna follows the hand
            logger.info("Antenna %d pushed down: following the hand", index)

        index = self.following
        position = present[index]
        self._movement_manager.set_antenna_override(index, position)

        if down_amount(present)[index] >= self.down_position_rad:
            self._down_since = self._down_since if self._down_since is not None else now
            if now - self._down_since >= self.hold_s:
                self.triggered = True
                return True
        else:
            self._down_since = None

        if self._last_position is None or abs(position - self._last_position) > 0.02:
            self._still_since = now
        self._last_position = position
        if self._still_since is not None and now - self._still_since >= self.release_after_s:
            logger.info("Antenna %d released before reaching the sleep position", index)
            self._release()
            self._cooldown_until = now + self.cooldown_s  # let it swing back before watching again
        return False

    def _run(self) -> None:
        if self._stop.wait(self._arm_after_s):  # let the wake-up movement finish first
            return
        while not self._stop.is_set():
            now = time.monotonic()
            self._log_stats(now)
            try:
                fired = self.step(now)
            except Exception as e:
                logger.error("Antenna gesture check failed: %s", e, exc_info=True)
                fired = False
            if fired:
                logger.info("Antenna held down: going to sleep")
                try:
                    self._on_trigger()
                except Exception as e:
                    logger.error("Antenna sleep gesture action failed: %s", e)
                finally:
                    self._release()
                return
            self._stop.wait(self._poll_s)
