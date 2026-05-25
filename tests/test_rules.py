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


# Tests del detector de gol por línea (NO bbox-contains)

from src.events.rules import ball_inside_goal, detect_goal_crossing, goal_line_endpoints


def _make_simple_field():
    """Cuadrilátero del campo en imagen (rectángulo 1000×500, origen 0,0)."""
    return np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)


def test_ball_outside_goal_when_in_field():
    """Balón en el centro del campo no está en ninguna portería."""
    field = _make_simple_field()
    # Portería izquierda: bbox a la izquierda del campo (x: -100..0)
    goal_left = np.array([-100, 200, 0, 300], dtype=np.int32)
    ball_center = np.array([500, 250])
    assert ball_inside_goal(ball_center, goal_left, field) is False


def test_ball_inside_goal_when_behind_line():
    """Balón ya cruzó la línea: está al lado lejos del centro del campo."""
    field = _make_simple_field()
    goal_left = np.array([-100, 200, 0, 300], dtype=np.int32)
    ball_inside = np.array([-50, 250])  # adentro del bbox de portería
    assert ball_inside_goal(ball_inside, goal_left, field) is True


def test_ball_at_line_boundary_is_not_yet_inside():
    """En la línea misma cuenta como afuera (delta == 0, requiere < -deadband)."""
    field = _make_simple_field()
    goal_left = np.array([-100, 200, 0, 300], dtype=np.int32)
    ball_on_line = np.array([0, 250])  # justo en x=0 (la línea de gol)
    assert ball_inside_goal(ball_on_line, goal_left, field, prev_inside=False) is False


def test_deadband_hysteresis():
    """Una vez adentro, no sale hasta cruzar el deadband al lado campo."""
    field = _make_simple_field()
    goal_left = np.array([-100, 200, 0, 300], dtype=np.int32)
    ball_near_line_field_side = np.array([10, 250])  # 10 px del lado campo
    # Si estaba adentro y ahora a 10 px del lado campo (< deadband 25): sigue adentro.
    assert (
        ball_inside_goal(ball_near_line_field_side, goal_left, field, prev_inside=True)
        is True
    )
    # Si estaba afuera, sigue afuera (10 px > -deadband).
    assert (
        ball_inside_goal(ball_near_line_field_side, goal_left, field, prev_inside=False)
        is False
    )


def test_goal_crossing_transition():
    """Solo dispara afuera→adentro, no en estado persistente."""
    assert detect_goal_crossing(False, True) is True
    assert detect_goal_crossing(True, True) is False  # ya estaba dentro
    assert detect_goal_crossing(False, False) is False
    assert detect_goal_crossing(True, False) is False  # salió


def test_goal_line_endpoints_returns_inner_edge():
    """Línea de gol = 2 puntos del bbox más cercanos al campo."""
    field = _make_simple_field()
    # Portería a la derecha (x: 1000..1100)
    goal_right = np.array([1000, 200, 1100, 300], dtype=np.int32)
    a, b = goal_line_endpoints(goal_right, field)
    # Los puntos deben tener x = 1000 (lado interior)
    assert abs(a[0] - 1000) < 1e-6
    assert abs(b[0] - 1000) < 1e-6


def test_ball_inside_goal_right_side():
    """Portería en el lado derecho del campo."""
    field = _make_simple_field()
    goal_right = np.array([1000, 200, 1100, 300], dtype=np.int32)
    ball_in_field = np.array([800, 250])
    ball_behind_line = np.array([1050, 250])
    assert ball_inside_goal(ball_in_field, goal_right, field) is False
    assert ball_inside_goal(ball_behind_line, goal_right, field) is True


# Tests fix B — detección de gol por coordenadas mundo (mm)

from src.events.rules import (
    FIELD_LENGTH_MM as L,
    FIELD_WIDTH_MM as W,
    GOAL_HALF_WIDTH_MM,
    ball_inside_goal_mm,
)


def test_mm_ball_center_field_no_goal():
    """Balón en el centro del campo: no está en ninguna portería."""
    center = np.array([L / 2, W / 2])
    assert ball_inside_goal_mm(center, "yellow", prev_inside=False) is False
    assert ball_inside_goal_mm(center, "blue", prev_inside=False) is False


def test_mm_ball_behind_yellow_line_is_goal():
    """Balón detrás de x=0 (yellow) con y dentro del ancho de portería: gol."""
    ball = np.array([-100, W / 2])  # 100 mm detrás de la línea
    assert ball_inside_goal_mm(ball, "yellow", prev_inside=False) is True


def test_mm_ball_behind_blue_line_is_goal():
    """Balón pasado x=L (blue) con y centrado: gol."""
    ball = np.array([L + 100, W / 2])
    assert ball_inside_goal_mm(ball, "blue", prev_inside=False) is True


def test_mm_ball_at_line_no_goal_without_deadband():
    """En la línea misma (x=0) no es gol todavía (deadband)."""
    ball = np.array([0.0, W / 2])
    assert ball_inside_goal_mm(ball, "yellow", prev_inside=False) is False


def test_mm_ball_outside_goal_height_no_goal():
    """Balón detrás de la línea pero fuera del ancho de portería: no gol."""
    # y = 100 mm (lejos del centro W/2=790)
    ball = np.array([-100, 100])
    half = GOAL_HALF_WIDTH_MM
    # Solo es gol si |y - W/2| <= half. W/2-100 = 690, no <= 300.
    assert abs(100 - W / 2) > half  # sanity de la prueba
    assert ball_inside_goal_mm(ball, "yellow", prev_inside=False) is False


def test_mm_hysteresis_keeps_inside():
    """Si estaba adentro, sigue adentro hasta delta > +deadband."""
    # ball.x = 10 mm (lado campo, cerca de la línea)
    ball = np.array([10.0, W / 2])
    # Si estaba adentro: sigue adentro (10 < deadband=30).
    assert ball_inside_goal_mm(ball, "yellow", prev_inside=True) is True
    # Si estaba afuera: sigue afuera (10 > -deadband).
    assert ball_inside_goal_mm(ball, "yellow", prev_inside=False) is False


def test_mm_unknown_color_returns_false():
    ball = np.array([-100, W / 2])
    assert ball_inside_goal_mm(ball, "red", prev_inside=False) is False
