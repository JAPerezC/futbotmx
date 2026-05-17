"""Tests del Kalman ball tracker."""

from __future__ import annotations

import numpy as np

from src.tracking.ball import BallState, BallTracker


def test_init_without_detection_returns_lost():
    t = BallTracker()
    s = t.update(None)
    assert not s.found
    assert s.source == "lost"
    assert not t.is_initialized


def test_first_detection_initializes():
    t = BallTracker()
    s = t.update(np.array([100.0, 200.0]))
    assert s.found
    assert s.source == "init"
    assert s.cx == 100.0
    assert s.cy == 200.0
    assert t.is_initialized


def test_constant_velocity_recovered():
    """Si el balón se mueve linealmente, Kalman debe estimar la velocidad."""
    t = BallTracker(dt=1 / 30)
    # Trayectoria: balón en x=100, moviéndose 30 px/frame en x (= 900 px/s).
    for i in range(30):
        t.update(np.array([100.0 + 30 * i, 200.0]))
    s = t._last_state
    assert s is not None
    # Velocidad estimada cerca de 900 px/s (puede haber sesgo Kalman)
    assert 700 < s.vx < 1100, f"vx={s.vx}"
    assert abs(s.vy) < 30, f"vy={s.vy}"


def test_short_occlusion_predicts():
    """Sin observación por algunos frames, debe predecir con confianza decreciente."""
    t = BallTracker(max_missing_frames=10)
    # Inicializar con trayectoria conocida
    for i in range(20):
        t.update(np.array([100.0 + 10 * i, 200.0]))
    # Ahora 5 frames sin observación
    states = [t.update(None) for _ in range(5)]
    for i, s in enumerate(states, start=1):
        assert s.found
        assert s.source == "predicted"
        assert s.missing_frames == i
        assert 0 < s.confidence < 1


def test_long_occlusion_marks_lost():
    """Si pasan más de max_missing_frames sin observación, se marca lost."""
    t = BallTracker(max_missing_frames=5)
    t.update(np.array([100.0, 200.0]))
    for _ in range(10):
        s = t.update(None)
    assert not s.found
    assert s.source == "lost"
    assert s.missing_frames == 10


def test_observation_after_prediction_resets_missing():
    t = BallTracker(max_missing_frames=10)
    for i in range(5):
        t.update(np.array([100.0 + 10 * i, 200.0]))
    for _ in range(3):
        t.update(None)
    s = t.update(np.array([200.0, 200.0]))
    assert s.source == "observed"
    assert s.missing_frames == 0


def test_reset_clears_state():
    t = BallTracker()
    t.update(np.array([100.0, 200.0]))
    assert t.is_initialized
    t.reset()
    assert not t.is_initialized
    s = t.update(None)
    assert s.source == "lost"


def test_ball_state_fields():
    s = BallState(
        found=True,
        cx=1,
        cy=2,
        vx=0,
        vy=0,
        source="observed",
        confidence=1.0,
        missing_frames=0,
    )
    assert s.cx == 1
