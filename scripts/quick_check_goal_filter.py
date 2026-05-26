"""Quick check: simula el detector nuevo (arista del campo + filtro lateral)
sobre los goles detectados por el run anterior.

Usa events.json (que guarda ball_px + bbox de portería en meta del gol) y
tracks.json (corners_img). Re-evalúa con `ball_inside_goal_field` +
`goal_line_from_field_edge` sin re-procesar el video.

Limitaciones:
- Solo aplica el filtro de un solo frame por gol (el del momento detectado).
- No simula la histéresis temporal (un FP previo podría haberse "limpiado"
  con la histéresis, pero aquí lo evaluamos como afuera→adentro single-shot).
- Para validación rigurosa, re-procesar el video completo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.events.rules import (  # noqa: E402
    ball_inside_goal_field,
    goal_line_from_field_edge,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()

    events_path = args.run / "events.json"
    tracks_path = args.run / "tracks.json"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    corners = np.array(tracks["corners_img"], dtype=np.float64)

    goals = [e for e in events if e.get("type") == "goal"]
    if not goals:
        print("Sin goles en este run.")
        return 0

    print(f"Goles detectados (run antiguo): {len(goals)}")
    print(f"Corners del campo: {corners.tolist()}")
    print()

    # Estado simulado de histéresis para el quick check (single-shot por frame).
    # Asumimos prev_inside=False para reportar si CADA gol antiguo sigue siendo
    # detectado bajo la nueva lógica como "afuera->adentro" en su frame.
    kept = 0
    dropped = 0
    for i, g in enumerate(goals, 1):
        meta = g.get("meta", {})
        ball_px = meta.get("ball_px")
        old_line = meta.get("goal_line_px")
        if ball_px is None or old_line is None:
            print(f"[{i}] t={g['t']:.2f}s sin meta — saltado")
            continue
        # Centroide de la portería: aproximamos por el midpoint de la línea
        # antigua (que es la arista del bbox HSV). Esto es razonable porque
        # la arista del bbox no está LEJOS del centro real de la portería.
        old_mid = np.array(old_line, dtype=np.float64).mean(axis=0)
        # Para mayor fidelidad, desplazamos el centroide hacia el LADO PORTERÍA
        # (opuesto al campo) por una distancia pequeña. Pero como solo lo usamos
        # para elegir la arista del campo más cercana, basta con el midpoint.
        goal_line = goal_line_from_field_edge(corners, old_mid)
        is_inside_now = ball_inside_goal_field(
            np.array(ball_px, dtype=np.float64),
            goal_line,
            corners,
            prev_inside=False,
        )
        verdict = "QUEDA" if is_inside_now else "FILTRADO"
        old_a, old_b = old_line[0], old_line[1]
        new_a, new_b = goal_line
        print(
            f"[{i}] t={g['t']:5.2f}s {meta.get('goal_color', '?'):6s} "
            f"ball=({ball_px[0]:6.0f},{ball_px[1]:6.0f}) "
            f"old=({old_a[0]:5.0f},{old_a[1]:5.0f})-({old_b[0]:5.0f},{old_b[1]:5.0f}) "
            f"new=({new_a[0]:5.0f},{new_a[1]:5.0f})-({new_b[0]:5.0f},{new_b[1]:5.0f}) "
            f"-> {verdict}"
        )
        if is_inside_now:
            kept += 1
        else:
            dropped += 1

    print()
    print(f"Resumen: quedan {kept} / filtrados {dropped} de {len(goals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
