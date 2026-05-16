"""Tracking del balón con TOTNet/TrackNetV3 + Kalman fallback.

Justificación (docs/literature-review.md § 3.2):
- TOTNet (arXiv:2508.09650, 2025): reduce RMSE de 37.30 a 7.19 en
  deportes con oclusiones, usa ventana de 5 frames.
- Para oclusiones cortas (<5 frames): Kalman 2D sobre la centroide
  detectada por SAM 3.1 o HSV.

Pipeline:
1. Detectar balón en frame N (SAM 3.1 + HSV fallback).
2. Si se detectó, alimentar Kalman.
3. Si no, predecir con Kalman; si pasan >5 frames sin detección,
   activar TOTNet sobre la ventana retenida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallState:
    """Estado del balón en un frame: posición + velocidad estimadas."""

    found: bool
    cx: float
    cy: float
    vx: float
    vy: float
    source: str  # "sam3" | "hsv" | "kalman" | "totnet"
    confidence: float


class BallTracker:
    """Tracker del balón con cascada de estrategias."""

    def __init__(self, dt: float = 1.0 / 30, max_missing_frames: int = 5):
        self.dt = dt
        self.max_missing_frames = max_missing_frames
        # self._kalman = cv2.KalmanFilter(...)  # implementar
        raise NotImplementedError(
            "Implementar Kalman 2D constante-velocidad en Fase 1; TOTNet en Fase 2"
        )

    def update(self, detection_xy: np.ndarray | None) -> BallState:
        """Actualiza el estado con la detección del frame actual (o None)."""
        raise NotImplementedError
