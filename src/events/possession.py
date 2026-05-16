"""Asignación de posesión del balón.

Estrategias:
1. **Closest robot**: el robot más cercano al balón dentro de un radio
   (R = 150 mm por defecto) tiene posesión. Cambio de robot = pase /
   intercepción dependiendo del equipo.
2. **PathCRF fallback** (Kim et al. 2025, arXiv:2602.12080): infiere
   posesión sin posición del balón cuando éste se oclude largo tiempo.
   Activar solo si TOTNet + Kalman no recuperan en N frames.

Pendiente: implementar PathCRF tras tener pipeline de trayectorias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSSESSION_RADIUS_MM = 150.0


@dataclass(frozen=True)
class PossessionState:
    track_id: int | None
    team: str | None  # "A" | "B" | None
    distance_mm: float
    method: str  # "closest" | "pathcrf"


def closest_robot_possession(
    ball_xy_mm: np.ndarray,
    robots_xy_mm: dict[int, np.ndarray],
    robot_teams: dict[int, str],
    radius_mm: float = POSSESSION_RADIUS_MM,
) -> PossessionState:
    """Devuelve la posesión basada en distancia mínima.

    Args:
        ball_xy_mm: posición del balón en mm.
        robots_xy_mm: dict {track_id: posición mm}.
        robot_teams: dict {track_id: "A"|"B"}.
        radius_mm: distancia máxima para considerar posesión.
    """
    best_id, best_dist = None, np.inf
    ball = np.asarray(ball_xy_mm)
    for tid, pos in robots_xy_mm.items():
        d = float(np.linalg.norm(ball - np.asarray(pos)))
        if d < best_dist:
            best_dist = d
            best_id = tid
    if best_id is None or best_dist > radius_mm:
        return PossessionState(None, None, best_dist, "closest")
    return PossessionState(best_id, robot_teams.get(best_id), best_dist, "closest")
