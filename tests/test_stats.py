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


def test_robot_jump_filtered():
    """Salto mayor a MAX_ROBOT_JUMP_MM se descarta del acumulado."""
    s = MatchStats()
    s.update_robot_position(1, np.array([0.0, 0.0]), "A", 0.0)
    s.end_frame(0.0)
    # Movimiento creíble: 100 mm en 1 s
    s.update_robot_position(1, np.array([100.0, 0.0]), "A", 1.0)
    s.end_frame(1.0)
    # Salto absurdo: 2000 mm en 1 s (artefacto). Debe descartarse.
    s.update_robot_position(1, np.array([2100.0, 0.0]), "A", 2.0)
    s.end_frame(2.0)
    # Movimiento creíble desde la nueva posición: 50 mm en 1 s.
    s.update_robot_position(1, np.array([2150.0, 0.0]), "A", 3.0)
    s.end_frame(3.0)
    # Distancia = 100 (válido) + 50 (válido). El salto de 2000 se descarta.
    assert abs(s.distance_per_track[1] - 150.0) < 1e-6
    # Velocidad máxima = 100 mm/s (no 2000 que fue el salto descartado).
    assert abs(s.max_speed_per_track[1] - 100.0) < 1e-6
    assert s.n_jumps_discarded_robots == 1


def test_ball_jump_filtered():
    """Salto mayor a MAX_BALL_JUMP_MM se descarta del acumulado."""
    s = MatchStats()
    s.update_ball_position(np.array([0.0, 0.0]), 0.0)
    s.end_frame(0.0)
    # Golpe creíble: 1000 mm en 1 s (1 m/s, dentro del umbral 1500).
    s.update_ball_position(np.array([1000.0, 0.0]), 1.0)
    s.end_frame(1.0)
    # Teletransporte: 5000 mm en 1 s (artefacto HSV o fallback). Descartar.
    s.update_ball_position(np.array([6000.0, 0.0]), 2.0)
    s.end_frame(2.0)
    # Continúa desde la nueva posición: 500 mm en 1 s.
    s.update_ball_position(np.array([6500.0, 0.0]), 3.0)
    s.end_frame(3.0)
    # Distancia = 1000 + 500 (el salto de 5000 se descarta).
    assert abs(s.ball_distance_mm - 1500.0) < 1e-6
    assert abs(s.ball_max_speed_mm_s - 1000.0) < 1e-6
    assert s.n_jumps_discarded_ball == 1
    # Solo 2 muestras de velocidad válidas (la del salto no se registra).
    assert len(s.ball_speed_samples) == 2


def test_to_dict_includes_tracking_artifacts():
    s = MatchStats()
    s.update_ball_position(np.array([0.0, 0.0]), 0.0)
    s.end_frame(0.0)
    s.update_ball_position(np.array([5000.0, 0.0]), 1.0)  # salto descartado
    s.end_frame(1.0)
    d = s.to_dict()
    assert "tracking_artifacts" in d
    assert d["tracking_artifacts"]["n_jumps_discarded_ball"] == 1
    assert d["tracking_artifacts"]["max_ball_jump_mm"] == 1500.0
    assert d["tracking_artifacts"]["max_robot_jump_mm"] == 500.0
