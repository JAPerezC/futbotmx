"""Tests de detectores rule-based de eventos."""

from __future__ import annotations

import numpy as np

from src.events.rules import (
    FIELD_LENGTH_MM,
    FIELD_WIDTH_MM,
    KICK_DV_MM_S,
    detect_kick,
    is_damaged_robot,
    is_in_goal_roi,
    is_kick,
    is_no_progress,
    is_retention,
)


def test_kick_detected_above_threshold():
    prev = np.array([0.0, 0.0])
    curr = np.array([100.0, 0.0])  # 100 mm en 1/30 s = 3000 mm/s = 3 m/s
    dt = 1.0 / 30
    v = detect_kick(prev, curr, dt)
    assert v == 3000.0
    assert is_kick(v)


def test_slow_movement_not_kick():
    prev = np.array([0.0, 0.0])
    curr = np.array([1.0, 0.0])  # 1 mm en 1/30 s = 30 mm/s
    v = detect_kick(prev, curr, 1.0 / 30)
    assert not is_kick(v)


def test_goal_left_roi():
    ball_center = np.array([FIELD_WIDTH_MM / 2 * 0 - 50, FIELD_WIDTH_MM / 2])
    # x = -50 está dentro de la portería izquierda (x ∈ [-100, 0])
    assert is_in_goal_roi(np.array([-50, FIELD_WIDTH_MM / 2]), "left")


def test_goal_right_roi():
    assert is_in_goal_roi(np.array([FIELD_LENGTH_MM + 50, FIELD_WIDTH_MM / 2]), "right")


def test_no_goal_outside_y_range():
    # Balón fuera del ancho de la portería (60 cm centrada)
    assert not is_in_goal_roi(np.array([-50, 0]), "left")
    assert not is_in_goal_roi(np.array([-50, FIELD_WIDTH_MM]), "left")


def test_retention_triggers():
    ball = np.array([100.0, 100.0])
    robot = np.array([120.0, 105.0])  # ~21 mm
    assert is_retention(ball, robot, time_in_contact_s=2.0)


def test_retention_too_far():
    ball = np.array([100.0, 100.0])
    robot = np.array([300.0, 105.0])  # 200 mm
    assert not is_retention(ball, robot, time_in_contact_s=10.0)


def test_no_progress_stuck_ball():
    # Balón quieto por 6 segundos a 30 fps
    n = int(6 * 30)
    pos = np.tile(np.array([1000.0, 800.0]), (n, 1))
    assert is_no_progress(pos, dt_s=1 / 30, window_s=5.0)


def test_no_progress_moving_ball():
    # Balón viajando a 1 m/s durante 6 s
    n = int(6 * 30)
    t = np.arange(n) / 30
    pos = np.stack([t * 1000, np.zeros(n)], axis=1)
    assert not is_no_progress(pos, dt_s=1 / 30, window_s=5.0)


def test_damaged_robot_idle():
    # 61 s de velocidad casi cero
    n = int(61 * 30)
    v = np.full(n, 5.0)  # 5 mm/s < umbral 20
    assert is_damaged_robot(v, dt_s=1 / 30)


def test_healthy_robot():
    n = int(61 * 30)
    v = np.full(n, 200.0)  # 200 mm/s muy por encima
    assert not is_damaged_robot(v, dt_s=1 / 30)


def test_kick_threshold_value():
    """El umbral default debe ser 500 mm/s (0.5 m/s, golf ball)."""
    assert KICK_DV_MM_S == 500.0
