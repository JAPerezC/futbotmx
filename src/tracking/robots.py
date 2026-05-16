"""Tracking de robots con OC-SORT (vía BoxMOT).

Justificación (docs/literature-review.md § 3.1):
- SportsMOT 2023: OC-SORT ofrece HOTA=63.2 con menos ID switches que
  ByteTrack para escenas con ≤10 jugadores.
- Nuestros 2-4 robots tienen identidad estable por bandera → re-ID HSV
  resuelve los pocos switches que OC-SORT no atrape.

Pendiente: instalar `boxmot` y conectar con la salida de SAM 3.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotTrack:
    """Estado del tracking de un robot en un frame."""

    track_id: int
    bbox_xyxy: np.ndarray  # (4,)
    centroid_img: np.ndarray  # (2,)
    team: str | None  # "A", "B" o None si no se ha identificado
    confidence: float


class RobotTracker:
    """OC-SORT envuelto para producir RobotTrack por frame."""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        # self._tracker = boxmot.OcSort(...)  # implementar
        raise NotImplementedError("Implementar tras instalar boxmot")

    def update(
        self, detections_xyxy: np.ndarray, frame_bgr: np.ndarray
    ) -> list[RobotTrack]:
        """Actualiza el tracker con detecciones del frame actual."""
        raise NotImplementedError
