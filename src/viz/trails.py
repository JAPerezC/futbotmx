"""Trayectorias (trails) sobre el campo top-down."""

from __future__ import annotations

import numpy as np
import cv2

from src.utils.calib import FIELD_LENGTH_MM, FIELD_WIDTH_MM


TRACK_COLORS = [
    (200, 30, 130),  # morado
    (50, 200, 240),  # amarillo-naranja
    (60, 200, 60),  # verde
    (240, 100, 100),  # azul claro
    (180, 180, 240),  # rosa
]
BALL_COLOR = (0, 165, 255)


def render_trails(
    trajectories: dict[int, np.ndarray],
    ball_trajectory: np.ndarray | None = None,
    px_per_mm: float = 0.4,
    thickness: int = 2,
) -> np.ndarray:
    """Dibuja todas las trayectorias sobre vista top-down del campo.

    Args:
        trajectories: dict {track_id: (N, 2) posiciones mm}.
        ball_trajectory: (N, 2) posiciones del balón en mm.
        px_per_mm: resolución.
        thickness: grosor de líneas.
    """
    w = int(round(FIELD_LENGTH_MM * px_per_mm))
    h = int(round(FIELD_WIDTH_MM * px_per_mm))
    canvas = np.full((h, w, 3), (45, 90, 35), dtype=np.uint8)  # fieltro verde

    # Líneas y círculo central
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    cx = w // 2
    cv2.line(canvas, (cx, 0), (cx, h - 1), (255, 255, 255), 1)
    r = int(round(300 * px_per_mm))
    cv2.circle(canvas, (cx, h // 2), r, (255, 255, 255), 1)
    # porterías (60 cm centradas en cada extremo)
    goal_h = int(round(600 * px_per_mm))
    cv2.rectangle(
        canvas, (0, h // 2 - goal_h // 2), (5, h // 2 + goal_h // 2), (0, 255, 255), -1
    )  # amarillo
    cv2.rectangle(
        canvas,
        (w - 5, h // 2 - goal_h // 2),
        (w - 1, h // 2 + goal_h // 2),
        (255, 80, 0),
        -1,
    )  # azul

    def _to_px(xy: np.ndarray) -> tuple[int, int]:
        return int(round(xy[0] * px_per_mm)), int(round(xy[1] * px_per_mm))

    for i, (tid, traj) in enumerate(sorted(trajectories.items())):
        if len(traj) < 2:
            continue
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        pts = np.array([_to_px(p) for p in traj], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
        # marcador final
        cv2.circle(canvas, tuple(pts[-1]), 6, color, -1)
        cv2.putText(
            canvas,
            f"id{tid}",
            (pts[-1][0] + 8, pts[-1][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if ball_trajectory is not None and len(ball_trajectory) >= 2:
        pts = np.array([_to_px(p) for p in ball_trajectory], dtype=np.int32)
        cv2.polylines(
            canvas, [pts], isClosed=False, color=BALL_COLOR, thickness=thickness + 1
        )
        cv2.circle(canvas, tuple(pts[-1]), 5, BALL_COLOR, -1)

    return canvas
