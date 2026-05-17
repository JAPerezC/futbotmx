"""Tests de detección de pase, intercepción y colisión."""

from __future__ import annotations

import numpy as np

from src.events.rules import detect_collisions, detect_pass_or_interception


def test_pass_same_team_with_translation():
    result = detect_pass_or_interception(
        prev_owner_track=1,
        prev_owner_team="A",
        curr_owner_track=2,
        curr_owner_team="A",
        ball_xy_prev_mm=np.array([100.0, 100.0]),
        ball_xy_curr_mm=np.array([500.0, 100.0]),  # 400 mm
    )
    assert result == "pass"


def test_interception_opposite_team():
    result = detect_pass_or_interception(
        prev_owner_track=1,
        prev_owner_team="A",
        curr_owner_track=2,
        curr_owner_team="B",
        ball_xy_prev_mm=np.array([100.0, 100.0]),
        ball_xy_curr_mm=np.array([600.0, 100.0]),
    )
    assert result == "interception"


def test_no_event_if_same_track():
    result = detect_pass_or_interception(
        prev_owner_track=1,
        prev_owner_team="A",
        curr_owner_track=1,
        curr_owner_team="A",
        ball_xy_prev_mm=np.array([100.0, 100.0]),
        ball_xy_curr_mm=np.array([500.0, 100.0]),
    )
    assert result is None


def test_no_event_if_translation_below_threshold():
    result = detect_pass_or_interception(
        prev_owner_track=1,
        prev_owner_team="A",
        curr_owner_track=2,
        curr_owner_team="A",
        ball_xy_prev_mm=np.array([100.0, 100.0]),
        ball_xy_curr_mm=np.array([120.0, 100.0]),  # 20 mm <300
    )
    assert result is None


def test_no_event_if_missing_owner():
    assert (
        detect_pass_or_interception(
            None,
            "A",
            2,
            "A",
            ball_xy_prev_mm=np.array([0.0, 0.0]),
            ball_xy_curr_mm=np.array([500.0, 0.0]),
        )
        is None
    )


def test_collision_detected_when_robots_close():
    robots = {1: np.array([100.0, 100.0]), 2: np.array([130.0, 110.0])}
    cols = detect_collisions(robots, dist_threshold_mm=50.0)
    assert len(cols) == 1
    a, b, d = cols[0]
    assert (a, b) == (1, 2)
    assert d < 50.0


def test_no_collision_when_apart():
    robots = {1: np.array([0.0, 0.0]), 2: np.array([1000.0, 1000.0])}
    cols = detect_collisions(robots)
    assert cols == []


def test_collision_multiple_pairs():
    robots = {
        1: np.array([0.0, 0.0]),
        2: np.array([20.0, 0.0]),  # cerca de 1
        3: np.array([5.0, 5.0]),  # cerca de 1 y 2
    }
    cols = detect_collisions(robots, dist_threshold_mm=30.0)
    # 3 pares: (1,2), (1,3), (2,3)
    assert len(cols) == 3


def test_collision_returns_sorted_pairs():
    robots = {3: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
    cols = detect_collisions(robots, dist_threshold_mm=50.0)
    assert cols[0][0] < cols[0][1]
