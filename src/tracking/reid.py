"""Re-identificación de robots por color dominante (HSV adaptativo).

Reemplaza el enfoque hardcoded (morado vs blanco) por uno adaptativo
que aprende los 2 colores dominantes del partido a partir de los robots
detectados en los primeros frames. Más robusto a torneos donde los
equipos usan colores arbitrarios.

API pública mantenida (`classify_team`, `TeamScore`) por compatibilidad.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

TeamLabel = Literal["A", "B", None]


@dataclass(frozen=True)
class TeamScore:
    label: TeamLabel
    score_a: float
    score_b: float


def _dominant_hue(
    image_bgr: np.ndarray, bbox_xyxy, top_fraction: float = 0.5
) -> int | None:
    """Devuelve el matiz HSV dominante (0-179) de la mitad superior del bbox.

    Filtra pixeles oscuros (V<60) y desaturados (S<50) que no aportan
    información cromática. Devuelve None si no hay suficientes pixeles
    válidos.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    yh = y1 + int(round((y2 - y1) * top_fraction))
    patch = image_bgr[y1:yh, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    valid = (hsv[:, :, 2] > 60) & (hsv[:, :, 1] > 50)
    if int(valid.sum()) < 30:
        return None
    hues = hsv[valid][:, 0]
    # histograma circular de 18 bins (cada bin = 10°)
    counts = np.bincount(hues // 10, minlength=18)
    return int(np.argmax(counts) * 10)


def _hue_distance(h1: int, h2: int) -> int:
    """Distancia circular entre 2 matices (0-90)."""
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


class AdaptiveTeamClassifier:
    """Clasificador adaptativo de equipos a partir de los matices observados.

    Estrategia:
    - Acumula muestras de matiz por track_id en los primeros N frames.
    - Calcula los 2 centros de cluster dominantes (k-means simple 1D).
    - Asigna track_id → equipo según el cluster más cercano.

    Si solo se observa un cluster (todos los robots de un color similar),
    todos quedan asignados al mismo equipo.
    """

    def __init__(self, warmup_frames: int = 8, hue_separation_min: int = 15):
        self.warmup_frames = warmup_frames
        self.hue_separation_min = hue_separation_min
        self._track_hues: dict[int, list[int]] = {}
        self._centers: tuple[int | None, int | None] = (None, None)
        self._track_assignment: dict[int, TeamLabel] = {}
        self._n_frames_seen = 0

    def observe(self, track_id: int, hue: int | None) -> None:
        if hue is None:
            return
        self._track_hues.setdefault(track_id, []).append(hue)

    def end_frame(self) -> None:
        self._n_frames_seen += 1
        if self._n_frames_seen >= self.warmup_frames and self._centers == (None, None):
            self._compute_centers()
        if self._centers != (None, None):
            self._assign_tracks()

    def _compute_centers(self) -> None:
        """Encuentra los 2 hues más frecuentes con separación mínima."""
        all_hues = [h for hs in self._track_hues.values() for h in hs]
        if not all_hues:
            return
        counter = Counter(h // 10 * 10 for h in all_hues)
        sorted_hues = [h for h, _ in counter.most_common()]
        if not sorted_hues:
            return
        c1 = sorted_hues[0]
        c2 = None
        for cand in sorted_hues[1:]:
            if _hue_distance(c1, cand) >= self.hue_separation_min:
                c2 = cand
                break
        self._centers = (c1, c2)

    def _assign_tracks(self) -> None:
        c1, c2 = self._centers
        for tid, hues in self._track_hues.items():
            if not hues:
                continue
            counter = Counter(h // 10 * 10 for h in hues)
            mode = counter.most_common(1)[0][0]
            if c2 is None:
                self._track_assignment[tid] = "A"
                continue
            d1 = _hue_distance(mode, c1)
            d2 = _hue_distance(mode, c2)
            if d1 <= d2:
                self._track_assignment[tid] = "A"
            else:
                self._track_assignment[tid] = "B"

    def assign(self, track_id: int) -> TeamLabel:
        return self._track_assignment.get(track_id)

    @property
    def centers(self) -> tuple[int | None, int | None]:
        return self._centers


def classify_team(image_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> TeamScore:
    """Compatibilidad: clasificador no-adaptativo (sin estado, por frame).

    Devuelve A si el matiz dominante está en el rango morado/violeta,
    B si está en blanco/verde-lima, None si no hay info clara.

    Para clasificación robusta usar `AdaptiveTeamClassifier`.
    """
    hue = _dominant_hue(image_bgr, bbox_xyxy)
    if hue is None:
        return TeamScore(None, 0.0, 0.0)
    # A: morado/violeta (125-155)
    # B: verde-lima (40-80) o blanco (saturación baja, no entra aquí)
    if 125 <= hue <= 165:
        return TeamScore("A", 1.0, 0.0)
    if 40 <= hue <= 90:
        return TeamScore("B", 0.0, 1.0)
    return TeamScore(None, 0.0, 0.0)
