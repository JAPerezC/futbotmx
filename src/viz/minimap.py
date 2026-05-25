"""Render top-down (cenital) animado de la cancha FutBotMX.

Aprovecha la homografía ya calculada por el pipeline: para cada frame
procesado tenemos posiciones de robots y balón en coordenadas mundo (mm)
sobre una cancha de 2190 × 1580 mm. Este módulo dibuja esa información
sobre un canvas con la cancha vista desde arriba, igual a la del paper
"From Broadcast to Minimap" (arXiv:2504.06357) y a los broadcasts de
RoboCup SSL.

Es la base de la animación de minimap que se renderiza con
`scripts/render_minimap_video.py` y se incrusta en el dashboard.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from src.utils.calib import FIELD_LENGTH_MM, FIELD_WIDTH_MM

# Estética (BGR)
COLOR_FIELD = (40, 90, 30)  # verde oscuro del fieltro
COLOR_LINE = (240, 240, 240)  # blanco de líneas
COLOR_BG = (28, 28, 28)  # gris muy oscuro fondo (banner)
COLOR_GOAL_YELLOW = (40, 210, 240)
COLOR_GOAL_BLUE = (220, 90, 0)
COLOR_BALL = (50, 165, 255)  # naranja BGR
COLOR_TEAM_A = (200, 30, 130)  # morado/púrpura
COLOR_TEAM_B = (245, 245, 245)  # blanco
COLOR_TEAM_UNK = (130, 130, 130)
COLOR_TRAIL = (50, 165, 255)
COLOR_TEXT = (240, 240, 240)
COLOR_HIGHLIGHT = (0, 215, 255)  # amarillo dorado

# Geometría real (reglamento § 7)
GOAL_WIDTH_MM = 600  # 60 cm
GOAL_DEPTH_MM = 100  # 10 cm caja
PENALTY_BOX_LEN_MM = 350  # área cercana a la portería
PENALTY_BOX_WIDTH_MM = 900

# Escala canvas
DEFAULT_PX_PER_MM = 0.35  # 766×553 px del campo


def _to_canvas(
    xy_mm: np.ndarray, scale: float, margin_x: int, margin_y: int
) -> tuple[int, int]:
    """Convierte coordenadas mundo (mm) a píxeles en el canvas."""
    x = int(round(xy_mm[0] * scale)) + margin_x
    y = int(round(xy_mm[1] * scale)) + margin_y
    return x, y


def _team_color(team: str | None) -> tuple[int, int, int]:
    if team == "A":
        return COLOR_TEAM_A
    if team == "B":
        return COLOR_TEAM_B
    return COLOR_TEAM_UNK


def render_minimap_frame(
    robots_mm: dict[int, np.ndarray],
    teams: dict[int, str | None],
    ball_mm: np.ndarray | None = None,
    ball_trail: Iterable[np.ndarray] | None = None,
    score_a: int = 0,
    score_b: int = 0,
    pos_pct_a: float = 0.0,
    pos_pct_b: float = 0.0,
    t_s: float = 0.0,
    duration_s: float = 0.0,
    event_label: str | None = None,
    scale: float = DEFAULT_PX_PER_MM,
) -> np.ndarray:
    """Dibuja un frame de la cancha cenital con todos los actores.

    Args:
        robots_mm: track_id → (x, y) en mm.
        teams: track_id → "A" | "B" | None.
        ball_mm: (x, y) del balón en mm, o None si no se ve.
        ball_trail: iterable de posiciones recientes del balón (mm).
        score_a, score_b: marcador.
        pos_pct_a, pos_pct_b: posesión acumulada en %.
        t_s, duration_s: tiempo actual y total del video (s).
        event_label: si se pasa, banner inferior con el evento reciente.
        scale: píxeles por mm (default 0.35).

    Returns:
        Imagen BGR uint8 lista para escribir al video.
    """
    margin = 40
    banner_h = 90  # banner superior con score + tiempo
    field_w_px = int(round(FIELD_LENGTH_MM * scale))
    field_h_px = int(round(FIELD_WIDTH_MM * scale))
    canvas_w = field_w_px + 2 * margin
    canvas_h = field_h_px + 2 * margin + banner_h
    canvas = np.full((canvas_h, canvas_w, 3), COLOR_BG, dtype=np.uint8)

    field_y0 = banner_h + margin
    field_x0 = margin
    # Fieltro
    cv2.rectangle(
        canvas,
        (field_x0, field_y0),
        (field_x0 + field_w_px, field_y0 + field_h_px),
        COLOR_FIELD,
        -1,
    )
    # Líneas perimetrales blancas
    cv2.rectangle(
        canvas,
        (field_x0, field_y0),
        (field_x0 + field_w_px, field_y0 + field_h_px),
        COLOR_LINE,
        2,
    )
    # Línea central
    cx = field_x0 + field_w_px // 2
    cv2.line(canvas, (cx, field_y0), (cx, field_y0 + field_h_px), COLOR_LINE, 1)
    # Círculo central
    cv2.circle(
        canvas,
        (cx, field_y0 + field_h_px // 2),
        int(round(250 * scale)),
        COLOR_LINE,
        1,
    )
    # Áreas de portería (rectangular)
    pa_w = int(round(PENALTY_BOX_LEN_MM * scale))
    pa_h = int(round(PENALTY_BOX_WIDTH_MM * scale))
    pa_y0 = field_y0 + (field_h_px - pa_h) // 2
    cv2.rectangle(
        canvas,
        (field_x0, pa_y0),
        (field_x0 + pa_w, pa_y0 + pa_h),
        COLOR_LINE,
        1,
    )
    cv2.rectangle(
        canvas,
        (field_x0 + field_w_px - pa_w, pa_y0),
        (field_x0 + field_w_px, pa_y0 + pa_h),
        COLOR_LINE,
        1,
    )
    # Porterías de color (afuera del campo, pegadas a las líneas cortas)
    g_h = int(round(GOAL_WIDTH_MM * scale))
    g_w = int(round(GOAL_DEPTH_MM * scale))
    g_y0 = field_y0 + (field_h_px - g_h) // 2
    # Amarilla a la izquierda (lado x=0)
    cv2.rectangle(
        canvas,
        (field_x0 - g_w, g_y0),
        (field_x0, g_y0 + g_h),
        COLOR_GOAL_YELLOW,
        -1,
    )
    cv2.rectangle(
        canvas,
        (field_x0 - g_w, g_y0),
        (field_x0, g_y0 + g_h),
        (0, 0, 0),
        1,
    )
    # Azul a la derecha (lado x=FIELD_LENGTH_MM)
    cv2.rectangle(
        canvas,
        (field_x0 + field_w_px, g_y0),
        (field_x0 + field_w_px + g_w, g_y0 + g_h),
        COLOR_GOAL_BLUE,
        -1,
    )
    cv2.rectangle(
        canvas,
        (field_x0 + field_w_px, g_y0),
        (field_x0 + field_w_px + g_w, g_y0 + g_h),
        (0, 0, 0),
        1,
    )

    # Trail del balón (cola desvanecida)
    if ball_trail is not None:
        prev = None
        trail_list = list(ball_trail)
        for i, pos in enumerate(trail_list):
            if pos is None:
                prev = None
                continue
            px = _to_canvas(np.asarray(pos), scale, field_x0, field_y0)
            if prev is not None:
                # Alpha simulado por grosor: más reciente → más grueso
                thickness = max(1, 1 + i * 2 // max(1, len(trail_list)))
                cv2.line(canvas, prev, px, COLOR_TRAIL, thickness, cv2.LINE_AA)
            prev = px

    # Robots
    robot_radius_px = int(round(80 * scale))  # robots ~80 mm radio en cancha
    for tid, pos in robots_mm.items():
        cx_px, cy_px = _to_canvas(np.asarray(pos), scale, field_x0, field_y0)
        color = _team_color(teams.get(tid))
        cv2.circle(canvas, (cx_px, cy_px), robot_radius_px, color, -1)
        cv2.circle(canvas, (cx_px, cy_px), robot_radius_px, (0, 0, 0), 1)
        label = str(tid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_color = (0, 0, 0) if teams.get(tid) == "B" else (255, 255, 255)
        cv2.putText(
            canvas,
            label,
            (cx_px - tw // 2, cy_px + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Balón
    if ball_mm is not None:
        bx, by = _to_canvas(np.asarray(ball_mm), scale, field_x0, field_y0)
        cv2.circle(canvas, (bx, by), 6, COLOR_BALL, -1)
        cv2.circle(canvas, (bx, by), 6, (0, 0, 0), 1)

    # Banner superior con score + tiempo + posesión
    cv2.rectangle(canvas, (0, 0), (canvas_w, banner_h), (15, 15, 15), -1)
    cv2.line(canvas, (0, banner_h), (canvas_w, banner_h), COLOR_HIGHLIGHT, 2)
    # Equipo A (izquierda)
    cv2.putText(
        canvas,
        "A",
        (margin, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        COLOR_TEAM_A,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        str(score_a),
        (margin + 30, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        1.4,
        COLOR_TEXT,
        3,
        cv2.LINE_AA,
    )
    # Equipo B (derecha)
    cv2.putText(
        canvas,
        "B",
        (canvas_w - margin - 65, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        COLOR_TEAM_B,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        str(score_b),
        (canvas_w - margin - 35, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        1.4,
        COLOR_TEXT,
        3,
        cv2.LINE_AA,
    )
    # Tiempo (centro)
    time_text = f"{t_s:5.1f} / {duration_s:5.1f} s"
    (tw, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        canvas,
        time_text,
        ((canvas_w - tw) // 2, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_HIGHLIGHT,
        2,
        cv2.LINE_AA,
    )
    # Barra de posesión
    bar_y = 55
    bar_x0 = 100
    bar_x1 = canvas_w - 100
    bar_w = bar_x1 - bar_x0
    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + 18), (60, 60, 60), -1)
    a_w = int(round(bar_w * pos_pct_a / 100))
    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x0 + a_w, bar_y + 18), COLOR_TEAM_A, -1)
    cv2.rectangle(canvas, (bar_x0 + a_w, bar_y), (bar_x1, bar_y + 18), COLOR_TEAM_B, -1)
    pos_text = f"posesion  A {pos_pct_a:.0f}%   B {pos_pct_b:.0f}%"
    (ptw, _), _ = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        canvas,
        pos_text,
        ((canvas_w - ptw) // 2, bar_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    # Banner inferior con evento reciente (opcional)
    if event_label:
        evt_y0 = canvas_h - 30
        cv2.rectangle(canvas, (0, evt_y0), (canvas_w, canvas_h), (15, 15, 15), -1)
        cv2.line(canvas, (0, evt_y0), (canvas_w, evt_y0), COLOR_HIGHLIGHT, 2)
        (etw, _), _ = cv2.getTextSize(event_label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.putText(
            canvas,
            event_label,
            ((canvas_w - etw) // 2, canvas_h - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            COLOR_HIGHLIGHT,
            2,
            cv2.LINE_AA,
        )

    return canvas


def topdown_canvas_size(scale: float = DEFAULT_PX_PER_MM) -> tuple[int, int]:
    """Devuelve (width, height) del canvas que produce render_minimap_frame."""
    margin = 40
    banner_h = 90
    canvas_w = int(round(FIELD_LENGTH_MM * scale)) + 2 * margin
    canvas_h = int(round(FIELD_WIDTH_MM * scale)) + 2 * margin + banner_h
    return canvas_w, canvas_h
