from typing import Any
from unittest.mock import MagicMock

from reachy_mini_conversation_app.antenna_gesture import AntennaSleepGesture


def _gesture(
    present: list[Any], commanded: Any = (0.0, 0.0), idle: bool = True, busy: bool = False
) -> AntennaSleepGesture:
    robot = MagicMock()
    robot.get_present_antenna_joint_positions.side_effect = present
    movement_manager = MagicMock()
    movement_manager.get_status.return_value = {"last_commanded_pose": {"antennas": list(commanded)}}
    movement_manager.is_idle_animation.return_value = idle
    return AntennaSleepGesture(robot, movement_manager, on_trigger=MagicMock(), is_busy=lambda: busy)


def test_push_down_gives_way_then_sleeps_at_the_down_position() -> None:
    """A downward push makes the antenna follow the hand; holding it at the down position fires."""
    gesture = _gesture([[-0.3, 0.0], [-0.5, 0.0], [-0.7, 0.0], [-0.72, 0.0], [-0.72, 0.0]])
    mm = gesture._movement_manager

    assert not gesture.step(0.0)  # pushed 0.3 rad: start following
    assert gesture.following == 0
    mm.set_antenna_override.assert_called_with(0, -0.3)
    assert not gesture.step(0.1)
    assert not gesture.step(0.2)  # reached the down position
    assert not gesture.step(0.4)
    assert gesture.step(0.55)  # held 0.35 s
    mm.set_antenna_override.assert_called_with(0, -0.72)


def test_right_antenna_down_is_positive_and_upward_pushes_are_ignored() -> None:
    """Antenna[1] goes down with positive values; pushing either antenna up never engages."""
    gesture = _gesture([[0.0, 0.7], [0.0, 0.7], [0.0, 0.7]])
    assert not gesture.step(0.0) and gesture.following == 1
    assert not gesture.step(0.1)
    assert gesture.step(0.45)

    upward = _gesture([[0.5, -0.5]] * 3)
    assert not any(upward.step(t) for t in (0.0, 0.5, 1.0)) and upward.following is None


def test_never_engages_outside_the_idle_animation_or_while_busy() -> None:
    """Emotions/dances (not idle) and an active conversation (busy) never start the gesture."""
    for gesture in (_gesture([[-0.8, 0.0]] * 5, idle=False), _gesture([[-0.8, 0.0]] * 5, busy=True)):
        assert not any(gesture.step(t * 0.2) for t in range(5))
        assert gesture.following is None
        gesture._movement_manager.set_antenna_override.assert_not_called()


def test_releasing_early_hands_the_antenna_back_to_the_animation() -> None:
    """If the hand lets go before the down position, the antenna returns to normal once still."""
    gesture = _gesture([[-0.35, 0.0]] + [[-0.4, 0.0]] * 6)
    mm = gesture._movement_manager

    for t in (0.0, 0.1, 0.5, 1.0, 1.5, 1.7):
        assert not gesture.step(t)
    assert gesture.following is None
    mm.set_antenna_override.assert_called_with(0, None)
    assert not gesture.step(1.8)  # still swinging back: the cooldown prevents grabbing it again
    assert gesture.following is None


def test_tracking_the_command_and_read_errors_do_nothing() -> None:
    """Antennas that follow their command (normal motion) and failed reads never engage."""
    moving = _gesture([[-0.25, 0.25]] * 3, commanded=(-0.25, 0.25))
    assert not any(moving.step(t) for t in (0.0, 0.5, 1.0)) and moving.following is None

    robot = MagicMock()
    robot.get_present_antenna_joint_positions.side_effect = RuntimeError("offline")
    mm = MagicMock()
    mm.get_status.return_value = {"last_commanded_pose": {"antennas": [0.0, 0.0]}}
    failing = AntennaSleepGesture(robot, mm, on_trigger=MagicMock())
    assert not failing.step(0.0)


def test_reads_commanded_antennas_from_the_real_movement_manager_status() -> None:
    """The commanded antennas come from MovementManager.get_status()'s real shape (regression)."""
    from reachy_mini_conversation_app.moves import MovementManager

    status = MovementManager(current_robot=MagicMock()).get_status()
    assert "antennas" in status["last_commanded_pose"]
    robot = MagicMock()
    robot.get_present_antenna_joint_positions.return_value = [0.0, 0.0]
    mm = MagicMock()
    mm.get_status.return_value = status
    assert AntennaSleepGesture(robot, mm, on_trigger=MagicMock())._read() is not None
