"""Tracking de robots con OC-SORT (vía BoxMOT 18).

API verificada el 2026-05-16 contra `boxmot==18.0.0`:
- ``boxmot.trackers.OcSort(min_conf=0.1, delta_t=3, ...)``.
- ``tracker.update(dets, img)`` con ``dets`` de shape (N, 6) =
  ``[x1, y1, x2, y2, conf, class_id]``.
- Devuelve ndarray (M, 8) = ``[x1, y1, x2, y2, track_id, conf, class, ?]``.

Justificación (docs/literature-review.md § 3.1):
- SportsMOT 2023: OC-SORT ofrece HOTA=63.2 con menos ID switches que
  ByteTrack para escenas con ≤10 jugadores.
- Nuestros 2-4 robots tienen identidad estable por bandera → re-ID HSV
  resuelve los pocos switches que OC-SORT no atrape.
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
    confidence: float
    class_id: int


class RobotTracker:
    """OC-SORT envuelto para producir RobotTrack por frame."""

    def __init__(
        self,
        det_thresh: float = 0.3,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        delta_t: int = 3,
    ):
        # Import perezoso para no romper tests cuando boxmot no esté instalado.
        from boxmot.trackers import OcSort

        self._tracker = OcSort(
            det_thresh=det_thresh,
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
            delta_t=delta_t,
        )

    def update(
        self, detections_xyxy_conf: np.ndarray, frame_bgr: np.ndarray
    ) -> list[RobotTrack]:
        """Actualiza el tracker con detecciones del frame actual.

        Args:
            detections_xyxy_conf: array (N, 6) con
                ``[x1, y1, x2, y2, conf, class_id]``. Si solo tienes
                (N, 4) o (N, 5), se completan los campos faltantes.
            frame_bgr: frame BGR (necesario para algunos trackers
                aunque OC-SORT no lo use directamente).

        Returns:
            Lista de RobotTrack (puede estar vacía).
        """
        if detections_xyxy_conf.size == 0:
            dets = np.empty((0, 6), dtype=np.float64)
        else:
            dets = np.asarray(detections_xyxy_conf, dtype=np.float64)
            if dets.shape[1] == 4:
                dets = np.hstack(
                    [dets, np.ones((len(dets), 1)), np.zeros((len(dets), 1))]
                )
            elif dets.shape[1] == 5:
                dets = np.hstack([dets, np.zeros((len(dets), 1))])

        result = self._tracker.update(dets, frame_bgr)
        if hasattr(result, "shape"):
            arr = np.asarray(result)
        else:
            arr = np.asarray(result)
        if arr.size == 0:
            return []

        tracks: list[RobotTrack] = []
        for row in arr:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            tid = int(row[4])
            conf = float(row[5]) if arr.shape[1] > 5 else 1.0
            cls = int(row[6]) if arr.shape[1] > 6 else 0
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tracks.append(
                RobotTrack(
                    track_id=tid,
                    bbox_xyxy=np.array([x1, y1, x2, y2]),
                    centroid_img=np.array([cx, cy]),
                    confidence=conf,
                    class_id=cls,
                )
            )
        return tracks
