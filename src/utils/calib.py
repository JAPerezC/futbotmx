"""Calibración de cámara y homografía para el campo de fútbol robótico.

La cancha oficial de la Copa FutBotMX (reglamento § 7) tiene dimensiones
internas (zona de juego) de 219 cm × 158 cm, con paredes negras de altura
≥ 22 cm. Como las dimensiones son conocidas y constantes, la rectificación
a vista top-down se resuelve con una sola homografía calculada al inicio
del video, opcionalmente actualizada con KLT para compensar paneos suaves.

Convención de coordenadas mundo (mm):
    Origen (0, 0): esquina interior superior-izquierda del campo.
    Eje X positivo: hacia la derecha (largo de 2190 mm).
    Eje Y positivo: hacia abajo (ancho de 1580 mm).

Ver `docs/literature-review.md` § 4 para la justificación de elegir
homografía clásica sobre PnLCalib.
"""

from __future__ import annotations

import numpy as np
import cv2

FIELD_LENGTH_MM = 2190
FIELD_WIDTH_MM = 1580
FIELD_CORNERS_WORLD_MM = np.array(
    [
        [0, 0],
        [FIELD_LENGTH_MM, 0],
        [FIELD_LENGTH_MM, FIELD_WIDTH_MM],
        [0, FIELD_WIDTH_MM],
    ],
    dtype=np.float64,
)


def compute_homography(image_corners: np.ndarray) -> np.ndarray:
    """Calcula la homografía imagen -> mundo desde las 4 esquinas del campo.

    Args:
        image_corners: array (4, 2) con las esquinas del campo en píxeles,
            ordenadas: top-left, top-right, bottom-right, bottom-left.

    Returns:
        Matriz H (3, 3) tal que `world_pt = H @ image_pt_h`.

    Raises:
        ValueError: si `image_corners` no es (4, 2) o tiene NaN.
    """
    pts = np.asarray(image_corners, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"image_corners debe ser (4, 2), recibido {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("image_corners contiene NaN o inf")
    H, mask = cv2.findHomography(pts, FIELD_CORNERS_WORLD_MM, method=0)
    if H is None:
        raise RuntimeError("findHomography devolvió None — esquinas degeneradas")
    return H


def project_points(points_img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Proyecta puntos en coordenadas imagen a coordenadas mundo (mm).

    Args:
        points_img: array (N, 2) en píxeles.
        H: matriz de homografía (3, 3).

    Returns:
        array (N, 2) en coordenadas mundo (mm).
    """
    pts = np.asarray(points_img, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2)


def inverse_project(points_world_mm: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Proyecta puntos en coordenadas mundo (mm) de vuelta a píxeles.

    Útil para dibujar landmarks del campo sobre el frame original.
    """
    Hinv = np.linalg.inv(H)
    pts = np.asarray(points_world_mm, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)


def warp_to_topdown(image: np.ndarray, H: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """Rectifica un frame a vista top-down del campo.

    Args:
        image: frame BGR.
        H: homografía imagen -> mundo.
        scale: píxeles por mm (default 0.3 → 657×474 px).

    Returns:
        Imagen top-down con el campo cubriendo todo el lienzo.
    """
    out_w = int(round(FIELD_LENGTH_MM * scale))
    out_h = int(round(FIELD_WIDTH_MM * scale))
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    M = S @ H
    return cv2.warpPerspective(image, M, (out_w, out_h))
