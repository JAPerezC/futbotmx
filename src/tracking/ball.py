"""Tracking del balón con Kalman 2D constante-velocidad + cascada de fuentes.

Modelo de estado: [x, y, vx, vy]ᵀ. Medición: [x, y]ᵀ.

Pipeline (docs/literature-review.md § 3.2):
1. Detectar balón en frame N (SAM 3.1 → HSV fallback).
2. Si hay detección, corregir Kalman con la observación.
3. Si no, predecir con Kalman; si pasan > max_missing_frames sin
   observación, marcar como lost (source="lost").
4. (Fase 2) Activar TOTNet sobre ventana retenida cuando lost > N.

Justificación del modelo:
- Constante velocidad es la baseline de literatura para balones en
  deportes (paper AutoRefs SSL, SoccerNet).
- dt configurable según fps del video; default 1/30 s.
- Process noise (Q) bajo: el balón cambia poco entre frames.
- Measurement noise (R) moderado: la detección HSV puede oscilar 1-2 px.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

DetectionSource = Literal["observed", "predicted", "lost", "init"]


@dataclass
class BallState:
    """Estado del balón en un frame."""

    found: bool
    cx: float
    cy: float
    vx: float
    vy: float
    source: DetectionSource
    confidence: float
    missing_frames: int  # cuántos frames seguidos sin observación


class BallTracker:
    """Kalman 2D para el balón con manejo de oclusiones cortas.

    Uso:
        tracker = BallTracker(dt=1/30)
        for frame in video:
            xy = detector(frame)   # ndarray(2,) o None
            state = tracker.update(xy)
            ... usar state.cx, state.cy ...
    """

    def __init__(
        self,
        dt: float = 1.0 / 30,
        max_missing_frames: int = 15,
        process_noise: float = 1.0,
        measurement_noise: float = 3.0,
        init_covariance: float = 100.0,
    ):
        self.dt = float(dt)
        self.max_missing_frames = int(max_missing_frames)

        self._kf = cv2.KalmanFilter(dynamParams=4, measureParams=2, controlParams=0)
        self._kf.transitionMatrix = np.array(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(
            measurement_noise
        )
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * float(init_covariance)

        self._initialized = False
        self._missing = 0
        self._last_state: BallState | None = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self._initialized = False
        self._missing = 0
        self._last_state = None
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        self._kf.statePost = np.zeros((4, 1), dtype=np.float32)

    def update(self, detection_xy: np.ndarray | None) -> BallState:
        """Avanza el filtro con la observación del frame (o None)."""
        if not self._initialized:
            if detection_xy is None:
                return BallState(
                    found=False,
                    cx=-1.0,
                    cy=-1.0,
                    vx=0.0,
                    vy=0.0,
                    source="lost",
                    confidence=0.0,
                    missing_frames=self._missing,
                )
            x, y = float(detection_xy[0]), float(detection_xy[1])
            self._kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self._initialized = True
            self._missing = 0
            state = BallState(
                found=True,
                cx=x,
                cy=y,
                vx=0.0,
                vy=0.0,
                source="init",
                confidence=1.0,
                missing_frames=0,
            )
            self._last_state = state
            return state

        # Paso 1: predicción
        predicted = self._kf.predict().ravel()
        px, py = float(predicted[0]), float(predicted[1])
        pvx, pvy = float(predicted[2]), float(predicted[3])

        if detection_xy is not None:
            # Paso 2a: corrección con observación
            x, y = float(detection_xy[0]), float(detection_xy[1])
            measurement = np.array([[x], [y]], dtype=np.float32)
            corrected = self._kf.correct(measurement).ravel()
            cx, cy = float(corrected[0]), float(corrected[1])
            vx, vy = float(corrected[2]), float(corrected[3])
            self._missing = 0
            state = BallState(
                found=True,
                cx=cx,
                cy=cy,
                vx=vx,
                vy=vy,
                source="observed",
                confidence=1.0,
                missing_frames=0,
            )
        else:
            # Paso 2b: solo predicción
            self._missing += 1
            if self._missing > self.max_missing_frames:
                state = BallState(
                    found=False,
                    cx=px,
                    cy=py,
                    vx=pvx,
                    vy=pvy,
                    source="lost",
                    confidence=0.0,
                    missing_frames=self._missing,
                )
            else:
                # confianza decrece linealmente con frames perdidos
                conf = max(0.0, 1.0 - self._missing / self.max_missing_frames)
                state = BallState(
                    found=True,
                    cx=px,
                    cy=py,
                    vx=pvx,
                    vy=pvy,
                    source="predicted",
                    confidence=conf,
                    missing_frames=self._missing,
                )

        self._last_state = state
        return state
