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

EventType = Literal[
    "goal",
    "kick",
    "retention",
    "no_progress",
    "damaged",
    "kickoff",
    "pass",
    "interception",
    "collision",
]

# Eventos nuevos
PASS_MIN_TRANSLATION_MM = 300.0  # balón se mueve >30 cm para considerar pase
COLLISION_DIST_MM = 50.0  # 2 robots en contacto (cuerpo a cuerpo)


def ball_inside_goal(
    ball_px: np.ndarray,
    goal_bbox_xyxy: np.ndarray,
    field_corners: np.ndarray,
    deadband_px: float = 25.0,
    prev_inside: bool = False,
    lateral_margin_px: float = 20.0,
) -> bool:
    """Determina si el balón ya cruzó la línea de gol de una portería.

    Línea de gol = arista del bbox de la portería más cercana al centro del
    campo. Calculamos la distancia signada del balón al punto medio de esa
    arista (positivo = lado campo, negativo = dentro de la portería).

    Histeresis con deadband: para evitar oscilaciones cuando el balón está
    pegado a la línea, una vez "adentro" se sigue contando adentro hasta
    que delta > +deadband_px; una vez "afuera" se cuenta afuera hasta que
    delta < -deadband_px. Esto suprime el flicker del centroide del balón
    (que vibra ±10 px por frame) cuando rueda sobre la línea.

    Filtro lateral (`lateral_margin_px`): el balón también debe estar
    entre los 2 endpoints de la línea de gol (proyección a lo largo de la
    línea, con margen). Sin este filtro, un balón que pasa por delante de
    la portería pero arriba/abajo del arco dispara "gol" porque la proyección
    perpendicular cae del lado portería. Pasa por defecto a 20 px (~5% del
    ancho típico de un bbox de portería robótica).
    """
    bbox = np.asarray(goal_bbox_xyxy, dtype=np.float64).reshape(4)
    x1, y1, x2, y2 = bbox
    goal_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)
    field_center = (
        np.asarray(field_corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    )
    direction = field_center - goal_center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return False
    direction /= norm
    bbox_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    projections = (bbox_corners - goal_center) @ direction
    order = np.argsort(-projections)
    line_a = bbox_corners[order[0]]
    line_b = bbox_corners[order[1]]
    edge_midpoint = (line_a + line_b) / 2
    ball = np.asarray(ball_px, dtype=np.float64).reshape(2)
    along = line_b - line_a
    along_len = float(np.linalg.norm(along))
    if along_len > 1e-3:
        along_unit = along / along_len
        s = float((ball - line_a) @ along_unit)
        if s < -lateral_margin_px or s > along_len + lateral_margin_px:
            return False
    delta = float((ball - edge_midpoint) @ direction)
    if prev_inside:
        return delta < deadband_px
    return delta < -deadband_px


def goal_line_endpoints(
    goal_bbox_xyxy: np.ndarray, field_corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve los 2 puntos en imagen que forman la línea de gol.

    Útil para visualización (dibujar la línea proyectada sobre el campo) y
    para diagnóstico.
    """
    bbox = np.asarray(goal_bbox_xyxy, dtype=np.float64).reshape(4)
    x1, y1, x2, y2 = bbox
    goal_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)
    field_center = (
        np.asarray(field_corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    )
    direction = field_center - goal_center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return np.array([x1, y1]), np.array([x2, y2])
    direction /= norm
    bbox_corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    projections = (bbox_corners - goal_center) @ direction
    order = np.argsort(-projections)
    return bbox_corners[order[0]], bbox_corners[order[1]]


def detect_goal_crossing(was_inside: bool, is_inside_now: bool) -> bool:
    """Detecta el momento exacto en que el balón cruza la línea de gol.

    Solo dispara en la transición afuera→adentro. Evita la sobre-detección
    típica de "balón dentro de la portería" durante varios frames seguidos.
    """
    return is_inside_now and not was_inside


def goal_line_from_field_edge(
    field_corners: np.ndarray,
    goal_centroid_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints de la arista del cuadrilátero del CAMPO más cercana a la portería.

    Más robusta que `goal_line_endpoints` (que usa el bbox HSV de portería) porque:
    - El cuadrilátero del campo se calibra UNA vez al inicio y no driftea con la
      cámara como sí lo hace el bbox HSV recalibrado cada N frames.
    - El HSV puede sobre-detectar la portería (reflejos, sombras del mismo color)
      expandiendo el bbox hacia adentro del campo. La arista del campo es la línea
      geométrica REAL definida por las líneas blancas del reglamento.

    Args:
        field_corners: (4, 2) esquinas del campo en orden TL, TR, BR, BL.
        goal_centroid_img: (2,) centroide de la portería detectada en imagen.

    Returns:
        Tupla (a, b) con los 2 endpoints de la arista en imagen.
    """
    corners = np.asarray(field_corners, dtype=np.float64).reshape(-1, 2)
    if len(corners) < 4:
        raise ValueError("se requieren al menos 4 esquinas del campo")
    g = np.asarray(goal_centroid_img, dtype=np.float64).reshape(2)
    n = len(corners)
    best_idx = 0
    best_dist = float("inf")
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        ab = b - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-6:
            continue
        ab_unit = ab / ab_len
        t = float((g - a) @ ab_unit)
        t_c = max(0.0, min(ab_len, t))
        nearest = a + t_c * ab_unit
        d = float(np.linalg.norm(g - nearest))
        if d < best_dist:
            best_dist = d
            best_idx = i
    return corners[best_idx].copy(), corners[(best_idx + 1) % n].copy()


def ball_inside_goal_field(
    ball_px: np.ndarray,
    goal_line: tuple[np.ndarray, np.ndarray],
    field_corners: np.ndarray,
    deadband_px: float = 25.0,
    lateral_margin_px: float = 20.0,
    prev_inside: bool = False,
    goal_bbox_xyxy: np.ndarray | None = None,
) -> bool:
    """Versión de `ball_inside_goal` con línea de gol arbitraria (no del bbox).

    Pensada para recibir la arista del campo (via `goal_line_from_field_edge`)
    en vez de la arista del bbox HSV de portería. Mantiene la misma lógica
    de histéresis y filtro lateral.

    Si se pasa `goal_bbox_xyxy`, el rango lateral aceptado se restringe a la
    proyección del bbox sobre la línea (con margen). Eso evita FP cuando el
    balón aparece detrás de la línea pero LATERALMENTE LEJOS del arco real
    (artefacto típico cuando el tracker del balón salta a posiciones espurias
    en perspectiva oblicua).

    Convención de signos: positivo = lado campo, negativo = lado portería
    (igual que `ball_inside_goal` para consistencia).
    """
    a = np.asarray(goal_line[0], dtype=np.float64).reshape(2)
    b = np.asarray(goal_line[1], dtype=np.float64).reshape(2)
    midpoint = (a + b) / 2
    along = b - a
    along_len = float(np.linalg.norm(along))
    if along_len < 1e-3:
        return False
    along_unit = along / along_len
    field_center = (
        np.asarray(field_corners, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    )
    direction = field_center - midpoint
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return False
    direction /= norm
    ball = np.asarray(ball_px, dtype=np.float64).reshape(2)
    if goal_bbox_xyxy is not None:
        bbox = np.asarray(goal_bbox_xyxy, dtype=np.float64).reshape(4)
        x1, y1, x2, y2 = bbox
        bbox_corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        )
        projections_bbox = (bbox_corners - a) @ along_unit
        s_min = float(projections_bbox.min()) - lateral_margin_px
        s_max = float(projections_bbox.max()) + lateral_margin_px
    else:
        s_min = -lateral_margin_px
        s_max = along_len + lateral_margin_px
    s = float((ball - a) @ along_unit)
    if s < s_min or s > s_max:
        return False
    delta = float((ball - midpoint) @ direction)
    if prev_inside:
        return delta < deadband_px
    return delta < -deadband_px


# === Detección de gol por coordenadas mundo (mm) — fix B ===
#
# Más robusto que la versión por bbox de portería en imagen porque NO depende
# de la perspectiva del frame ni de qué tan bien quedó el bbox de la portería:
# usa la homografía validada (que sí sabemos correcta tras audit_homography) y
# las dimensiones fijas del reglamento.
#
# Convención (ver src/utils/calib.py):
#   x ∈ [0, FIELD_LENGTH_MM] — lado largo, portería YELLOW en x≈0, BLUE en x≈L
#   y ∈ [0, FIELD_WIDTH_MM]  — lado corto, portería centrada en y≈W/2

GOAL_LINE_TOLERANCE_MM = 30.0  # zona muerta antes de la línea (perfecciona FP)


def ball_inside_goal_mm(
    ball_mm: np.ndarray,
    goal_color: str,
    deadband_mm: float = GOAL_LINE_TOLERANCE_MM,
    prev_inside: bool = False,
    field_length_mm: float = FIELD_LENGTH_MM,
    field_width_mm: float = FIELD_WIDTH_MM,
    goal_half_width_mm: float = GOAL_HALF_WIDTH_MM,
    goal_y_range_mm: tuple[float, float] | None = None,
) -> bool:
    """Determina si el balón cruzó la línea de gol usando coordenadas mundo.

    - "yellow": línea de gol en x = 0. Gol si ball.x < -deadband Y la y del
      balón está dentro del ancho de la portería (W/2 ± goal_half_width).
    - "blue": línea de gol en x = field_length. Gol si ball.x > L+deadband.

    Histeresis: una vez adentro, sigue adentro hasta delta > +deadband.

    Esta función NO usa el bbox de la portería en imagen — solo coordenadas
    mundo derivadas de la homografía, así no le afecta el flicker del balón
    cuando pasa cerca de un robot que oculta la portería en perspectiva.
    """
    ball = np.asarray(ball_mm, dtype=np.float64).reshape(2)
    bx, by = float(ball[0]), float(ball[1])
    if goal_y_range_mm is not None:
        y_min, y_max = goal_y_range_mm
        in_goal_height = y_min - deadband_mm <= by <= y_max + deadband_mm
    else:
        y_center = field_width_mm / 2
        in_goal_height = abs(by - y_center) <= goal_half_width_mm
    if goal_color == "yellow":
        delta = bx
        if prev_inside:
            inside = delta < deadband_mm
        else:
            inside = delta < -deadband_mm
    elif goal_color == "blue":
        delta = field_length_mm - bx
        if prev_inside:
            inside = delta < deadband_mm
        else:
            inside = delta < -deadband_mm
    else:
        return False
    return bool(inside and in_goal_height)


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


# ============================================================================
# Eventos nuevos: pase, intercepción, colisión
# ============================================================================


def detect_pass_or_interception(
    prev_owner_track: int | None,
    prev_owner_team: str | None,
    curr_owner_track: int | None,
    curr_owner_team: str | None,
    ball_xy_prev_mm: np.ndarray | None,
    ball_xy_curr_mm: np.ndarray,
    min_translation_mm: float = PASS_MIN_TRANSLATION_MM,
) -> str | None:
    """Devuelve 'pass', 'interception' o None tras un cambio de posesión.

    Args:
        prev_owner_track: track_id del dueño del balón en frame anterior.
        prev_owner_team: equipo del dueño anterior.
        curr_owner_track: track_id actual.
        curr_owner_team: equipo actual.
        ball_xy_prev_mm: posición del balón cuando lo perdió el anterior.
        ball_xy_curr_mm: posición del balón ahora.
        min_translation_mm: distancia mínima que debe recorrer el balón.

    Returns:
        'pass' si mismo equipo recibe y balón se desplazó.
        'interception' si equipo contrario recibe y balón se desplazó.
        None si no hubo cambio significativo.
    """
    if (
        prev_owner_track is None
        or curr_owner_track is None
        or prev_owner_team is None
        or curr_owner_team is None
    ):
        return None
    if prev_owner_track == curr_owner_track:
        return None  # mismo dueño, no es pase ni intercepción
    if ball_xy_prev_mm is None:
        return None
    translation = float(
        np.linalg.norm(np.asarray(ball_xy_curr_mm) - np.asarray(ball_xy_prev_mm))
    )
    if translation < min_translation_mm:
        return None
    if prev_owner_team == curr_owner_team:
        return "pass"
    return "interception"


def detect_collisions(
    robots_xy_mm: dict[int, np.ndarray],
    dist_threshold_mm: float = COLLISION_DIST_MM,
) -> list[tuple[int, int, float]]:
    """Lista pares de robots en colisión.

    Args:
        robots_xy_mm: dict {track_id: posición mm}.
        dist_threshold_mm: distancia centroidal para considerar colisión.

    Returns:
        Lista de tuplas (track_id_a, track_id_b, distancia_mm), una por colisión
        única (a < b para evitar duplicados).
    """
    ids = sorted(robots_xy_mm.keys())
    collisions = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            d = float(np.linalg.norm(robots_xy_mm[a] - robots_xy_mm[b]))
            if d < dist_threshold_mm:
                collisions.append((a, b, d))
    return collisions
