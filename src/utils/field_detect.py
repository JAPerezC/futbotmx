"""Detección automática de las 4 esquinas del campo (fieltro verde).

Estrategia (más simple y robusta que Hough sobre las líneas blancas):
1. Segmentar píxeles verdes con HSV.
2. Limpiar con morfología (close + open).
3. Componente conexo más grande (presumiblemente el campo).
4. Contorno externo.
5. Aproximar contorno a polígono con `cv2.approxPolyDP`; si tiene
   4 vértices, esas son las esquinas.
6. Ordenar TL, TR, BR, BL.

Fundamento: el campo verde domina el centro de cada frame, su color
HSV está bien delimitado, y la forma observada es siempre un
cuadrilátero (oblicuo o no). Hough sobre líneas blancas falla cuando
las líneas del área de penalti tienen contraste similar a las del
borde — segmentar verde es más discriminativo.

Como fallback: el usuario puede anotar las 4 esquinas a mano y
guardar en `data/calibration/<video>.json`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Rango HSV para fieltro verde de la cancha (saturado, brillante).
# Frame de muestra: H ~ 60-75, S ~ 100-220, V ~ 80-200.
GREEN_HSV_LOWER = np.array([35, 70, 50], dtype=np.uint8)
GREEN_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class FieldDetectionResult:
    success: bool
    corners: np.ndarray  # (4, 2) en orden TL, TR, BR, BL — vacío si !success
    mask: np.ndarray  # uint8, máscara binaria del campo
    contour_area_ratio: float  # área del contorno / área del frame


def segment_field_mask(
    image_bgr: np.ndarray,
    hsv_lower: np.ndarray = GREEN_HSV_LOWER,
    hsv_upper: np.ndarray = GREEN_HSV_UPPER,
    kernel_size: int = 9,
) -> np.ndarray:
    """Máscara binaria del fieltro verde."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos a TL, TR, BR, BL por geometría."""
    pts = pts.reshape(4, 2).astype(np.float64)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()  # x - y? cv2 returns (y, x)? aclaro:
    # diff = pts[:,1] - pts[:,0] ... pero esto no es lo que queremos.
    # Mejor: ordenar por suma (TL min, BR max) y por resta (TR max, BL min).
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.stack([tl, tr, br, bl])


def detect_field_corners(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.15,
    approx_epsilon_frac: float = 0.02,
) -> FieldDetectionResult:
    """Detecta las 4 esquinas del campo aproximando el contorno verde.

    Args:
        image_bgr: frame en BGR.
        min_area_ratio: área mínima del campo como fracción del frame.
        approx_epsilon_frac: tolerancia de aproximación poligonal
            como fracción del perímetro (típico 0.01-0.05).

    Returns:
        FieldDetectionResult con success=False si no se pudo
        determinar un cuadrilátero confiable.
    """
    h, w = image_bgr.shape[:2]
    frame_area = h * w
    mask = segment_field_mask(image_bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, 0.0)

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    area_ratio = area / frame_area

    if area_ratio < min_area_ratio:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)

    def _is_valid(c: np.ndarray) -> bool:
        """Las 4 esquinas deben estar separadas por al menos 5 px."""
        pw = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(pw, np.inf)
        return bool(pw.min() >= 5.0)

    # Aproximación poligonal — si no da 4, probar con epsilons distintos.
    for eps_frac in (approx_epsilon_frac, 0.01, 0.03, 0.05, 0.08):
        epsilon = eps_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            corners = _order_corners_tl_tr_br_bl(approx)
            if _is_valid(corners):
                return FieldDetectionResult(True, corners, mask, area_ratio)

    # Fallback: bounding box rotado.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    corners = _order_corners_tl_tr_br_bl(box)
    if _is_valid(corners):
        return FieldDetectionResult(True, corners, mask, area_ratio)
    return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)


def draw_corners(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Dibuja las 4 esquinas + el cuadrilátero conectándolas."""
    out = image_bgr.copy()
    if corners.size == 0:
        return out
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip(corners, labels):
        cv2.circle(out, (int(x), int(y)), 12, (0, 0, 255), -1)
        cv2.putText(
            out,
            label,
            (int(x) + 16, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out
