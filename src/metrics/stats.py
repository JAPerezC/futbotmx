"""Estadísticas agregadas del partido para banner en vivo + summary.json.

Acumula durante el loop principal del pipeline:
- Goles por equipo (score)
- Tiempo de posesión por equipo + sin posesión
- Distancia recorrida por cada robot (integral de velocidades)
- Velocidad máxima y promedio del balón y de cada robot
- Conteo de eventos por tipo
- Posiciones por equipo (para heatmap por equipo)

Soporta consulta en cualquier momento para renderizar banner persistente.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchStats:
    # Estado acumulado
    score_a: int = 0
    score_b: int = 0
    possession_time_a: float = 0.0
    possession_time_b: float = 0.0
    possession_time_none: float = 0.0
    event_counts: Counter = field(default_factory=Counter)
    # Por track
    distance_per_track: dict[int, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    max_speed_per_track: dict[int, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    team_per_track: dict[int, str | None] = field(default_factory=dict)
    positions_by_team_mm: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Balón
    ball_distance_mm: float = 0.0
    ball_max_speed_mm_s: float = 0.0
    ball_speed_samples: list[float] = field(default_factory=list)
    # Último estado conocido (para deltas)
    _last_ball_xy: np.ndarray | None = None
    _last_robot_xy: dict[int, np.ndarray] = field(default_factory=dict)
    _last_t: float | None = None

    def update_robot_position(
        self, track_id: int, xy_mm: np.ndarray, team: str | None, t_s: float
    ) -> None:
        """Acumula distancia + velocidad de un robot."""
        if team:
            self.positions_by_team_mm[team].append(xy_mm.copy())
        self.team_per_track[track_id] = team
        if track_id in self._last_robot_xy and self._last_t is not None:
            dt = t_s - self._last_t
            if dt > 0:
                d = float(np.linalg.norm(xy_mm - self._last_robot_xy[track_id]))
                self.distance_per_track[track_id] += d
                speed = d / dt
                if speed > self.max_speed_per_track[track_id]:
                    self.max_speed_per_track[track_id] = speed
        self._last_robot_xy[track_id] = xy_mm.copy()

    def update_ball_position(self, xy_mm: np.ndarray, t_s: float) -> None:
        """Acumula distancia + velocidad del balón."""
        if self._last_ball_xy is not None and self._last_t is not None:
            dt = t_s - self._last_t
            if dt > 0:
                d = float(np.linalg.norm(xy_mm - self._last_ball_xy))
                self.ball_distance_mm += d
                speed = d / dt
                self.ball_speed_samples.append(speed)
                if speed > self.ball_max_speed_mm_s:
                    self.ball_max_speed_mm_s = speed
        self._last_ball_xy = xy_mm.copy()

    def update_possession(self, owner_team: str | None, dt_s: float) -> None:
        """Acumula tiempo de posesión por equipo."""
        if owner_team == "A":
            self.possession_time_a += dt_s
        elif owner_team == "B":
            self.possession_time_b += dt_s
        else:
            self.possession_time_none += dt_s

    def end_frame(self, t_s: float) -> None:
        self._last_t = t_s

    def register_event(self, event_type: str, team: str | None = None) -> None:
        self.event_counts[event_type] += 1
        if event_type == "goal":
            if team == "A":
                self.score_a += 1
            elif team == "B":
                self.score_b += 1

    # ---- consultas para banner / dashboard ----

    @property
    def total_possession_time(self) -> float:
        return (
            self.possession_time_a + self.possession_time_b + self.possession_time_none
        )

    @property
    def possession_pct_a(self) -> float:
        tot = self.possession_time_a + self.possession_time_b
        return 100 * self.possession_time_a / tot if tot > 0 else 0.0

    @property
    def possession_pct_b(self) -> float:
        tot = self.possession_time_a + self.possession_time_b
        return 100 * self.possession_time_b / tot if tot > 0 else 0.0

    @property
    def ball_avg_speed_mm_s(self) -> float:
        if not self.ball_speed_samples:
            return 0.0
        return float(np.mean(self.ball_speed_samples))

    def to_dict(self) -> dict:
        return {
            "score": {"A": self.score_a, "B": self.score_b},
            "possession_time_s": {
                "A": self.possession_time_a,
                "B": self.possession_time_b,
                "none": self.possession_time_none,
            },
            "possession_pct": {
                "A": round(self.possession_pct_a, 1),
                "B": round(self.possession_pct_b, 1),
            },
            "ball": {
                "distance_mm": round(self.ball_distance_mm, 1),
                "max_speed_mm_s": round(self.ball_max_speed_mm_s, 1),
                "avg_speed_mm_s": round(self.ball_avg_speed_mm_s, 1),
                "n_speed_samples": len(self.ball_speed_samples),
            },
            "robots": {
                str(tid): {
                    "team": self.team_per_track.get(tid),
                    "distance_mm": round(d, 1),
                    "max_speed_mm_s": round(self.max_speed_per_track.get(tid, 0.0), 1),
                }
                for tid, d in self.distance_per_track.items()
            },
            "events_by_type": dict(self.event_counts),
            "tracks_seen": len(self.team_per_track),
        }
