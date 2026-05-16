"""Detección rule-based de eventos sobre trayectorias top-down.

Inspirado en los AutoRefs de RoboCup SSL (ER-Force, TIGERs Mannheim).
Ver docs/literature-review.md § 5 para los umbrales y su justificación.

Convenciones:
- Coordenadas en mm, eje X largo, eje Y ancho. Origen en esquina
  superior-izquierda del campo.
- Tiempos en segundos.
- Velocidades en mm/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Umbrales (refinables tras observar partidos reales)
KICK_DV_MM_S = 500.0  # 0.5 m/s en un frame, golf ball
RETENTION_DIST_MM = 90.0  # SSL AutoRefs
RETENTION_TIME_S = 1.5
NO_PROGRESS_SIGMA_MM = 50.0
NO_PROGRESS_WINDOW_S = 5.0
DAMAGED_VELOCITY_MM_S = 20.0
DAMAGED_TIME_S = 60.0
GOAL_DEPTH_MM = 100  # paredes traseras de las porterías (10 cm)
FIELD_LENGTH_MM = 2190
FIELD_WIDTH_MM = 1580
GOAL_HALF_WIDTH_MM = 300  # portería 60 cm centrada

EventType = Literal["goal", "kick", "retention", "no_progress", "damaged", "kickoff"]


@dataclass(frozen=True)
class Event:
    t: float  # segundos desde inicio del video
    type: EventType
    actors: list[int]  # track_ids involucrados
    position_mm: tuple[float, float]
    confidence: float
    meta: dict


def detect_kick(
    ball_xy_prev: np.ndarray,
    ball_xy_curr: np.ndarray,
    dt: float,
) -> float:
    """Magnitud de velocidad del balón en mm/s. Compara contra KICK_DV_MM_S."""
    if dt <= 0:
        return 0.0
    return float(np.linalg.norm(ball_xy_curr - ball_xy_prev) / dt)


def is_kick(velocity_mm_s: float) -> bool:
    return velocity_mm_s > KICK_DV_MM_S


def is_in_goal_roi(ball_xy: np.ndarray, side: Literal["left", "right"]) -> bool:
    """¿El balón está dentro del rectángulo ROI de la portería?

    Portería izquierda: x ∈ [-GOAL_DEPTH, 0], y ∈ [centro ± half_width].
    Portería derecha: x ∈ [LENGTH, LENGTH + GOAL_DEPTH], y ∈ misma.
    """
    cx, cy = float(ball_xy[0]), float(ball_xy[1])
    cy_center = FIELD_WIDTH_MM / 2
    if abs(cy - cy_center) > GOAL_HALF_WIDTH_MM:
        return False
    if side == "left":
        return -GOAL_DEPTH_MM <= cx <= 0
    return FIELD_LENGTH_MM <= cx <= FIELD_LENGTH_MM + GOAL_DEPTH_MM


def is_retention(
    ball_xy: np.ndarray,
    robot_xy: np.ndarray,
    time_in_contact_s: float,
    dist_threshold_mm: float = RETENTION_DIST_MM,
    time_threshold_s: float = RETENTION_TIME_S,
) -> bool:
    """Retención del balón (prohibida, reglamento § 4.4.1)."""
    dist = float(np.linalg.norm(np.asarray(ball_xy) - np.asarray(robot_xy)))
    return dist < dist_threshold_mm and time_in_contact_s > time_threshold_s


def is_no_progress(
    ball_positions_mm: np.ndarray,
    dt_s: float,
    window_s: float = NO_PROGRESS_WINDOW_S,
    sigma_threshold_mm: float = NO_PROGRESS_SIGMA_MM,
) -> bool:
    """Falta de progreso (§ 4.4.3): el balón quedó estancado.

    Args:
        ball_positions_mm: array (N, 2) de posiciones recientes.
        dt_s: paso temporal entre posiciones.
        window_s: ventana de tiempo a evaluar.
    """
    if len(ball_positions_mm) < 2:
        return False
    n = int(np.ceil(window_s / dt_s))
    recent = ball_positions_mm[-n:]
    if len(recent) < n // 2:
        return False
    return float(recent.std(axis=0).max()) < sigma_threshold_mm


def is_damaged_robot(
    robot_velocities_mm_s: np.ndarray,
    dt_s: float,
    time_threshold_s: float = DAMAGED_TIME_S,
    vel_threshold_mm_s: float = DAMAGED_VELOCITY_MM_S,
) -> bool:
    """Robot dañado (§ 4.4.10.7): sin movimiento durante T segundos."""
    n = int(np.ceil(time_threshold_s / dt_s))
    if len(robot_velocities_mm_s) < n:
        return False
    return bool(np.all(robot_velocities_mm_s[-n:] < vel_threshold_mm_s))
