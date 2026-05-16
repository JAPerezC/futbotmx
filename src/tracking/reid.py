"""Re-identificación de robots por color de bandera (HSV).

Justificación: las banderas verticales sobre los robots son la firma de
identidad de equipo (CLAUDE.md, dataset-inspection.md). Un clasificador
HSV ligero resuelve los ID switches de OC-SORT cuando la máscara del
robot se pierde brevemente.

Convención: equipo A = morada/violeta, equipo B = blanca/verde-lima.
Configurable si el partido tiene otros colores de bandera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

# Rangos HSV. OpenCV: H en [0, 179], S y V en [0, 255].
TEAM_A_HSV_RANGES = [
    (np.array([125, 60, 60]), np.array([155, 255, 255])),  # morado/violeta
]
TEAM_B_HSV_RANGES = [
    (np.array([0, 0, 200]), np.array([179, 40, 255])),  # blanco brillante
    (np.array([35, 80, 80]), np.array([75, 255, 255])),  # verde-lima accent
]

TeamLabel = Literal["A", "B", None]


@dataclass(frozen=True)
class TeamScore:
    label: TeamLabel
    score_a: float
    score_b: float


def _fraction_in_ranges(
    hsv_patch: np.ndarray, ranges: list[tuple[np.ndarray, np.ndarray]]
) -> float:
    if hsv_patch.size == 0:
        return 0.0
    mask = np.zeros(hsv_patch.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_patch, lo, hi))
    return float(mask.sum()) / (mask.size * 255)


def classify_team(image_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> TeamScore:
    """Clasifica al robot dentro de bbox_xyxy como equipo A, B o None.

    Args:
        image_bgr: frame completo en BGR.
        bbox_xyxy: caja del robot (x1, y1, x2, y2) en píxeles.

    Returns:
        TeamScore con label y los scores de cada equipo.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    # Tomar la mitad superior del bbox (donde está la bandera).
    yh = y1 + (y2 - y1) // 2
    patch = image_bgr[y1:yh, x1:x2]
    if patch.size == 0:
        return TeamScore(None, 0.0, 0.0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    score_a = _fraction_in_ranges(hsv, TEAM_A_HSV_RANGES)
    score_b = _fraction_in_ranges(hsv, TEAM_B_HSV_RANGES)
    # Umbral mínimo + ventaja relativa
    if max(score_a, score_b) < 0.02:
        return TeamScore(None, score_a, score_b)
    if score_a >= score_b * 1.3:
        return TeamScore("A", score_a, score_b)
    if score_b >= score_a * 1.3:
        return TeamScore("B", score_a, score_b)
    return TeamScore(None, score_a, score_b)
