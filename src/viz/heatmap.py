"""Heatmaps de actividad sobre el campo top-down."""

from __future__ import annotations

import numpy as np
import cv2

from src.utils.calib import FIELD_LENGTH_MM, FIELD_WIDTH_MM


def render_heatmap(
    positions_mm: np.ndarray,
    px_per_mm: float = 0.4,
    blur_sigma_px: int = 21,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Heatmap de densidad de posiciones sobre el plano del campo.

    Args:
        positions_mm: (N, 2) array de posiciones (x, y) en mm.
        px_per_mm: resolución del output. 0.4 → 876x632 px.
        blur_sigma_px: sigma del filtro Gaussiano (más alto = más suave).
        colormap: colormap de OpenCV (default JET).

    Returns:
        Imagen BGR (H, W, 3) con el heatmap.
    """
    w = int(round(FIELD_LENGTH_MM * px_per_mm))
    h = int(round(FIELD_WIDTH_MM * px_per_mm))
    canvas = np.zeros((h, w), dtype=np.float32)

    if len(positions_mm) > 0:
        xy = np.asarray(positions_mm, dtype=np.float64)
        # Filtrar puntos dentro del campo
        mask = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] <= FIELD_LENGTH_MM)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= FIELD_WIDTH_MM)
        )
        xy = xy[mask]
        for x, y in xy:
            ix = int(round(x * px_per_mm))
            iy = int(round(y * px_per_mm))
            if 0 <= ix < w and 0 <= iy < h:
                canvas[iy, ix] += 1.0

    if blur_sigma_px > 0 and canvas.max() > 0:
        k = blur_sigma_px | 1  # impar
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    if canvas.max() > 0:
        norm = (canvas / canvas.max() * 255).astype(np.uint8)
    else:
        norm = canvas.astype(np.uint8)

    colored = cv2.applyColorMap(norm, colormap)
    return _draw_field_overlay(colored, px_per_mm)


def _draw_field_overlay(img: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Dibuja líneas del campo (bordes + círculo central) sobre el heatmap."""
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    # línea central
    cx = w // 2
    cv2.line(out, (cx, 0), (cx, h - 1), (255, 255, 255), 1)
    # círculo central 60 cm = 600 mm de diámetro → radio 300 mm
    r = int(round(300 * px_per_mm))
    cv2.circle(out, (cx, h // 2), r, (255, 255, 255), 1)
    return out


# Colormaps por equipo
COLORMAP_TEAM = {
    "A": cv2.COLORMAP_PINK,  # tonalidad morada/rosa
    "B": cv2.COLORMAP_BONE,  # tonalidad gris/blanca
}


def render_heatmap_by_team(
    positions_by_team_mm: dict[str, list],
    px_per_mm: float = 0.4,
    blur_sigma_px: int = 21,
) -> dict[str, np.ndarray]:
    """Renderiza un heatmap por cada equipo presente en el dict.

    Args:
        positions_by_team_mm: dict {team_label: [posiciones mm]}.

    Returns:
        dict {team_label: imagen BGR (H, W, 3)}.
    """
    out = {}
    for team, positions in positions_by_team_mm.items():
        if not team or not positions:
            continue
        arr = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        cmap = COLORMAP_TEAM.get(team, cv2.COLORMAP_JET)
        out[team] = render_heatmap(
            arr, px_per_mm=px_per_mm, blur_sigma_px=blur_sigma_px, colormap=cmap
        )
    return out
