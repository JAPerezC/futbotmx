"""Diagrama de Voronoi sobre el campo top-down."""

from __future__ import annotations

import numpy as np
import cv2

from src.utils.calib import FIELD_LENGTH_MM, FIELD_WIDTH_MM


def render_voronoi(
    robots_mm: dict[int, np.ndarray],
    teams: dict[int, str | None],
    ball_mm: np.ndarray | None = None,
    px_per_mm: float = 0.4,
) -> np.ndarray:
    """Voronoi de control de espacio (top-down).

    Args:
        robots_mm: dict {track_id: posición mm}.
        teams: dict {track_id: "A"|"B"|None}.
        ball_mm: posición del balón (opcional, se dibuja).

    Returns:
        Imagen BGR con el Voronoi coloreado por equipo + landmarks.
    """
    w = int(round(FIELD_LENGTH_MM * px_per_mm))
    h = int(round(FIELD_WIDTH_MM * px_per_mm))
    canvas = np.full((h, w, 3), (45, 90, 35), dtype=np.uint8)

    if len(robots_mm) >= 2:
        # Construir mapa de distancia "asignar cada píxel al robot más cercano"
        sites_xy = np.array(
            [[r[0] * px_per_mm, r[1] * px_per_mm] for r in robots_mm.values()]
        )
        site_ids = list(robots_mm.keys())
        team_colors = {"A": (200, 30, 130), "B": (240, 240, 240), None: (130, 130, 130)}

        ys, xs = np.indices((h, w))
        coords = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        dists = np.linalg.norm(coords[:, None, :] - sites_xy[None, :, :], axis=2)
        nearest = dists.argmin(axis=1).reshape(h, w)

        colored = np.zeros_like(canvas)
        for i, tid in enumerate(site_ids):
            color = team_colors.get(teams.get(tid), (130, 130, 130))
            colored[nearest == i] = color
        canvas = cv2.addWeighted(colored, 0.4, canvas, 0.6, 0)

        # Sitios + IDs
        for tid, xy in robots_mm.items():
            ix, iy = int(round(xy[0] * px_per_mm)), int(round(xy[1] * px_per_mm))
            color = team_colors.get(teams.get(tid), (130, 130, 130))
            cv2.circle(canvas, (ix, iy), 8, color, -1)
            cv2.circle(canvas, (ix, iy), 8, (0, 0, 0), 2)
            cv2.putText(
                canvas,
                f"{tid}",
                (ix - 5, iy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    # Líneas del campo encima
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    cv2.line(canvas, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1)
    cv2.circle(
        canvas, (w // 2, h // 2), int(round(300 * px_per_mm)), (255, 255, 255), 1
    )

    if ball_mm is not None:
        ix, iy = int(round(ball_mm[0] * px_per_mm)), int(round(ball_mm[1] * px_per_mm))
        cv2.circle(canvas, (ix, iy), 6, (0, 165, 255), -1)

    return canvas
