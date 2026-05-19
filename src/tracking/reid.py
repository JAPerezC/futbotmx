"""Re-identificación de robots por color dominante (HSV adaptativo).

Reemplaza el enfoque hardcoded (morado vs blanco) por uno adaptativo
que aprende los 2 colores dominantes del partido a partir de los robots
detectados en los primeros frames. Más robusto a torneos donde los
equipos usan colores arbitrarios.

API pública mantenida (`classify_team`, `TeamScore`) por compatibilidad.

v2 (mayo 2026): aplica las recomendaciones del paper STRIKER (IJRIAS 2026)
y del survey de team-aware tracking con SAM (arXiv:2512.08467):
- warmup más largo (30 frames vs 8) para observar ambos equipos antes
  de fijar centros.
- recomputación periódica de centros (cada `recompute_every` frames)
  para corregir si al inicio solo se vio un equipo.
- votación temporal por tracklet: la asignación se basa en la moda de
  los ÚLTIMOS N hues observados, no de toda la historia.
- considera también la saturación (S) para discriminar colores apagados
  vs saturados (típico cuando un equipo es blanco/plata).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal, Union

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
    feat = _dominant_feature(image_bgr, bbox_xyxy, top_fraction)
    return None if feat is None else feat[0]


def _dominant_feature(
    image_bgr: np.ndarray, bbox_xyxy, top_fraction: float = 0.5
) -> tuple[int, int] | None:
    """Devuelve (hue, saturation) dominantes (0-179, 0-255) del bbox.

    Usa la mitad superior del bbox (donde típicamente está la bandera/
    casaca del robot, no el suelo verde). Filtra pixeles oscuros y
    desaturados.
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
    valid_pixels = hsv[valid]
    hues = valid_pixels[:, 0]
    sats = valid_pixels[:, 1]
    # histograma circular de 18 bins (cada bin = 10°)
    hue_counts = np.bincount(hues // 10, minlength=18)
    dom_bin = int(np.argmax(hue_counts))
    dom_hue = dom_bin * 10
    # saturación mediana de los pixeles en el bin dominante
    in_bin = (hues // 10) == dom_bin
    dom_sat = int(np.median(sats[in_bin])) if in_bin.any() else int(np.median(sats))
    return dom_hue, dom_sat


def _hue_distance(h1: int, h2: int) -> int:
    """Distancia circular entre 2 matices (0-90)."""
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


def _feature_distance(
    f1: tuple[int, int], f2: tuple[int, int], sat_weight: float = 0.5
) -> float:
    """Distancia entre features (h, s).

    Hue ponderado al 100% (escala 0-90), saturación ponderada por
    sat_weight (escala 0-255 → 0-90 al normalizar). Default 0.5 favorece
    hue (que es más estable bajo iluminación variable) pero permite que
    saturación rompa empates cuando los hues son similares.
    """
    dh = _hue_distance(f1[0], f2[0])
    ds = abs(int(f1[1]) - int(f2[1])) * (90 / 255)
    return dh + sat_weight * ds


class AdaptiveTeamClassifier:
    """Clasificador adaptativo de equipos a partir de los matices observados.

    Estrategia v2:
    - Acumula muestras (hue, sat) por track_id.
    - Calcula centros con k-means de 2 clusters sobre todas las muestras.
    - **Recompute online**: recalcula centros cada `recompute_every` frames
      mientras el partido avanza (resuelve el problema de "solo se ve un
      equipo al inicio").
    - **Votación temporal**: la asignación final se hace con la moda de
      las últimas `vote_window` muestras del tracklet (no toda la historia).
    """

    def __init__(
        self,
        warmup_frames: int = 30,
        recompute_every: int = 15,
        vote_window: int = 20,
        hue_separation_min: int = 15,
        sat_weight: float = 0.5,
    ):
        self.warmup_frames = warmup_frames
        self.recompute_every = recompute_every
        self.vote_window = vote_window
        self.hue_separation_min = hue_separation_min
        self.sat_weight = sat_weight
        self._track_features: dict[int, deque] = {}
        self._centers: tuple[tuple[int, int] | None, tuple[int, int] | None] = (
            None,
            None,
        )
        self._track_assignment: dict[int, TeamLabel] = {}
        self._n_frames_seen = 0
        self._last_recompute_frame = -(10**9)

    def observe(
        self, track_id: int, feature: Union[int, tuple[int, int], None]
    ) -> None:
        """Registra una observación de color para el track.

        `feature` puede ser:
        - None → ignorada
        - int → hue solo (asume saturación media 150 para compat. con tests)
        - tuple (hue, sat)
        """
        if feature is None:
            return
        if isinstance(feature, (int, np.integer)):
            h, s = int(feature), 150
        else:
            h, s = int(feature[0]), int(feature[1])
        buf = self._track_features.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.vote_window)
            self._track_features[track_id] = buf
        buf.append((h, s))

    def end_frame(self) -> None:
        self._n_frames_seen += 1
        first_compute = (
            self._centers == (None, None) and self._n_frames_seen >= self.warmup_frames
        )
        recompute_due = (
            self._centers != (None, None)
            and self._n_frames_seen - self._last_recompute_frame >= self.recompute_every
        )
        if first_compute or recompute_due:
            self._compute_centers()
            self._last_recompute_frame = self._n_frames_seen
        if self._centers != (None, None):
            self._assign_tracks()

    def _compute_centers(self) -> None:
        """Encuentra los 2 features dominantes con separación mínima en hue.

        Usa todas las muestras acumuladas. Agrupa hues en bins de 10° y
        para cada bin promedia la saturación. Toma el bin más frecuente
        como c1; busca c2 como el primer bin (en orden de frecuencia)
        que está separado en hue >= `hue_separation_min`.
        """
        all_feats: list[tuple[int, int]] = [
            f for buf in self._track_features.values() for f in buf
        ]
        if not all_feats:
            return
        hue_bin_counter: Counter = Counter()
        hue_bin_sat: dict[int, list[int]] = {}
        for h, s in all_feats:
            b = (h // 10) * 10
            hue_bin_counter[b] += 1
            hue_bin_sat.setdefault(b, []).append(s)
        sorted_bins = [b for b, _ in hue_bin_counter.most_common()]
        if not sorted_bins:
            return
        c1_h = sorted_bins[0]
        c1_s = int(np.median(hue_bin_sat[c1_h]))
        c2 = None
        for cand_h in sorted_bins[1:]:
            if _hue_distance(c1_h, cand_h) >= self.hue_separation_min:
                cand_s = int(np.median(hue_bin_sat[cand_h]))
                c2 = (cand_h, cand_s)
                break
        self._centers = ((c1_h, c1_s), c2)

    def _assign_tracks(self) -> None:
        c1, c2 = self._centers
        for tid, buf in self._track_features.items():
            if not buf:
                continue
            # Votación temporal: cada muestra reciente vota por A o B
            if c2 is None:
                self._track_assignment[tid] = "A"
                continue
            votes_a = votes_b = 0
            for f in buf:
                d1 = _feature_distance(f, c1, self.sat_weight)
                d2 = _feature_distance(f, c2, self.sat_weight)
                if d1 <= d2:
                    votes_a += 1
                else:
                    votes_b += 1
            self._track_assignment[tid] = "A" if votes_a >= votes_b else "B"

    def assign(self, track_id: int) -> TeamLabel:
        return self._track_assignment.get(track_id)

    @property
    def centers(self) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
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
    # A: morado/violeta (125-165)
    # B: verde-lima (40-80) o blanco (saturación baja, no entra aquí)
    if 125 <= hue <= 165:
        return TeamScore("A", 1.0, 0.0)
    if 40 <= hue <= 90:
        return TeamScore("B", 0.0, 1.0)
    return TeamScore(None, 0.0, 0.0)
