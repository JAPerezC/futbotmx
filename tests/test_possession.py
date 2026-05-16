"""Tests de asignación de posesión."""

from __future__ import annotations

import numpy as np

from src.events.possession import closest_robot_possession


def test_assigns_to_closest_robot():
    ball = np.array([1000.0, 800.0])
    robots = {
        1: np.array([1050.0, 810.0]),  # ~51 mm
        2: np.array([1500.0, 800.0]),  # 500 mm
    }
    teams = {1: "A", 2: "B"}
    pos = closest_robot_possession(ball, robots, teams)
    assert pos.track_id == 1
    assert pos.team == "A"


def test_no_possession_when_too_far():
    ball = np.array([100.0, 100.0])
    robots = {1: np.array([2000.0, 1000.0])}
    teams = {1: "A"}
    pos = closest_robot_possession(ball, robots, teams, radius_mm=150.0)
    assert pos.track_id is None
    assert pos.team is None


def test_empty_robots_returns_none():
    ball = np.array([100.0, 100.0])
    pos = closest_robot_possession(ball, {}, {})
    assert pos.track_id is None
    assert pos.method == "closest"
