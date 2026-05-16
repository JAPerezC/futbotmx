"""Baselines de detección sin redes neuronales.

El balón oficial de la Categoría Abierta es una pelota de golf naranja
brillante de 42 mm (reglamento § 1.2). Contra el fieltro verde del campo,
el contraste es máximo en el plano matiz-saturación.

Fundamento de literatura (`docs/literature-review.md` § 3.2):
- Trackers Kalman-IoU fallan con balones de <20 px por error acumulado
  en oclusiones.
- Detectores como HSV ofrecen una segunda línea robusta y barata que
  complementa SAM 3.1 o TOTNet.
- Reglamento § 3.7 prohíbe colores naranja, amarillo y azul en robots,
  asegurando que el balón sea único en el rango naranja.

Este módulo expone `detect_orange_ball` para máscara binaria y
`find_ball_centroid` para coordenada en píxeles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

# Rangos HSV para naranja brillante (golf ball oficial).
# OpenCV usa H en [0, 179]. Naranja saturado ~ H en [5, 25].
ORANGE_HSV_LOWER = np.array([5, 140, 140], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([22, 255, 255], dtype=np.uint8)

# Área mínima en píxeles para considerar la detección válida.
# Balón a distancia: ~10-20 px de diámetro → área ~80-300 px².
# Cercano: hasta ~400 px² (limite superior generoso).
MIN_BALL_AREA_PX = 20
MAX_BALL_AREA_PX = 4000


@dataclass(frozen=True)
class BallDetection:
    """Resultado de la detección del balón en un frame."""

    found: bool
    cx: float
    cy: float
    radius: float
    area: int
    confidence: float  # 0-1, basada en circularidad


def detect_orange_ball_mask(
    image_bgr: np.ndarray,
    hsv_lower: np.ndarray = ORANGE_HSV_LOWER,
    hsv_upper: np.ndarray = ORANGE_HSV_UPPER,
) -> np.ndarray:
    """Devuelve una máscara binaria del balón naranja.

    Args:
        image_bgr: frame en formato BGR (OpenCV default).
        hsv_lower, hsv_upper: rangos HSV de tolerancia. Ajustables si la
            iluminación del torneo difiere de la del sample.

    Returns:
        Máscara uint8 (0/255) del mismo tamaño que `image_bgr`.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    # Morfología: cierre para tapar huecos, apertura para quitar ruido.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def find_ball_centroid(
    image_bgr: np.ndarray,
    min_area: int = MIN_BALL_AREA_PX,
    max_area: int = MAX_BALL_AREA_PX,
) -> BallDetection:
    """Detecta el balón naranja y devuelve su centro + métricas.

    Estrategia: máscara HSV → componentes conexos → seleccionar el más
    circular dentro del rango de área válido.

    Returns:
        BallDetection con `found=False` si no encuentra candidato válido.
    """
    mask = detect_orange_ball_mask(image_bgr)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    best = BallDetection(False, -1.0, -1.0, 0.0, 0, 0.0)
    best_conf = -1.0

    for i in range(1, num_labels):  # 0 es el fondo
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        # Circularidad: 4*pi*area / perimeter^2, aproximada por aspect ratio
        # más relación área/bbox.
        aspect = min(w, h) / max(w, h)  # 1.0 ideal
        bbox_fill = area / (w * h) if w * h > 0 else 0.0  # ~pi/4 ideal
        # Confidence: combinación normalizada.
        circ = 0.6 * aspect + 0.4 * min(bbox_fill / (np.pi / 4), 1.0)
        if circ > best_conf:
            best_conf = float(circ)
            cx, cy = centroids[i]
            radius = float(max(w, h) / 2)
            best = BallDetection(True, float(cx), float(cy), radius, area, circ)
    return best


def draw_detection(image_bgr: np.ndarray, detection: BallDetection) -> np.ndarray:
    """Dibuja un círculo y texto sobre el frame con la detección.

    Devuelve una copia anotada (no modifica el original).
    """
    out = image_bgr.copy()
    if not detection.found:
        return out
    center = (int(round(detection.cx)), int(round(detection.cy)))
    r = max(int(round(detection.radius)) + 6, 12)
    cv2.circle(out, center, r, (0, 255, 0), 2)
    label = f"ball conf={detection.confidence:.2f} area={detection.area}"
    cv2.putText(
        out,
        label,
        (center[0] + r + 4, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out
