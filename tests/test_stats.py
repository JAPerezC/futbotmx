"""Tests del agregador MatchStats."""

from __future__ import annotations

import numpy as np

from src.metrics.stats import MatchStats


def test_score_and_event_counts():
    s = MatchStats()
    s.register_event("kick")
    s.register_event("kick")
    s.register_event("goal", team="A")
    s.register_event("goal", team="B")
    s.register_event("goal", team="A")
    assert s.score_a == 2
    assert s.score_b == 1
    assert s.event_counts["kick"] == 2
    assert s.event_counts["goal"] == 3


def test_possession_accumulation():
    s = MatchStats()
    s.update_possession("A", 1.0)
    s.update_possession("A", 0.5)
    s.update_possession("B", 0.3)
    s.update_possession(None, 0.2)
    assert s.possession_time_a == 1.5
    assert s.possession_time_b == 0.3
    assert s.possession_time_none == 0.2
    # pct = A/(A+B) = 1.5/1.8 ≈ 83.3
    assert 80 < s.possession_pct_a < 86
    assert 14 < s.possession_pct_b < 20


def test_distance_per_robot_increments():
    s = MatchStats()
    s.update_robot_position(1, np.array([0.0, 0.0]), "A", 0.0)
    s.end_frame(0.0)
    s.update_robot_position(1, np.array([100.0, 0.0]), "A", 1.0)
    s.end_frame(1.0)
    s.update_robot_position(1, np.array([100.0, 100.0]), "A", 2.0)
    s.end_frame(2.0)
    assert abs(s.distance_per_track[1] - 200.0) < 1e-6
    # velocidad: 100 mm/s en ambos saltos, max=100
    assert abs(s.max_speed_per_track[1] - 100.0) < 1e-6


def test_ball_distance_and_speed():
    s = MatchStats()
    s.update_ball_position(np.array([0.0, 0.0]), 0.0)
    s.end_frame(0.0)
    s.update_ball_position(np.array([500.0, 0.0]), 1.0)  # 500 mm/s
    s.end_frame(1.0)
    s.update_ball_position(np.array([500.0, 0.0]), 2.0)  # 0 mm/s
    s.end_frame(2.0)
    assert abs(s.ball_distance_mm - 500.0) < 1e-6
    assert abs(s.ball_max_speed_mm_s - 500.0) < 1e-6
    assert abs(s.ball_avg_speed_mm_s - 250.0) < 1e-6


def test_to_dict_serializable():
    import json

    s = MatchStats()
    s.update_robot_position(1, np.array([0.0, 0.0]), "A", 0.0)
    s.end_frame(0.0)
    s.register_event("kick")
    d = s.to_dict()
    json.dumps(d)  # debe ser JSON-serializable


def test_positions_by_team_accumulated():
    s = MatchStats()
    s.update_robot_position(1, np.array([100.0, 100.0]), "A", 0.0)
    s.update_robot_position(2, np.array([200.0, 200.0]), "B", 0.0)
    s.update_robot_position(3, np.array([300.0, 300.0]), None, 0.0)
    assert len(s.positions_by_team_mm["A"]) == 1
    assert len(s.positions_by_team_mm["B"]) == 1
    assert "None" not in s.positions_by_team_mm  # None no se guarda
