"""Diagnóstico: línea de gol vs cámara móvil + clasificador de equipos.

Saca 4 frames distribuidos a lo largo del video (t=2, 20, 40, 55 s),
dibuja sobre cada uno:
  - El cuadrilátero del campo según detección en ESE frame
  - El cuadrilátero del campo según calibración INICIAL (frame 0)
  - Las porterías (HSV) en ese frame
  - La línea de gol según las dos versiones
  - Los robots detectados con su clasificación de equipo + matiz HSV

Compara visualmente si la recalibración online captura el movimiento
de cámara y si el classifier separa los equipos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.events.rules import goal_line_from_field_edge
from src.utils.field_detect_v2 import detect_field_geometry, detect_goals_by_color
from src.utils.io import probe, read_frames
from src.utils.calib import compute_homography  # noqa: F401

VIDEO = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras" / "IMG_9821.MOV"
RUN_DIR = ROOT / "data" / "processed" / "runs" / "IMG_9821_v4"
OUT_DIR = ROOT / "data" / "processed" / "experiments" / "diagnose_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TS = [2.0, 20.0, 40.0, 55.0]


def main() -> int:
    meta = probe(VIDEO)
    print(
        f"Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.duration_s:.1f}s"
    )

    # Cargar tracks.json del run v4
    tracks = json.loads((RUN_DIR / "tracks.json").read_text(encoding="utf-8"))
    initial_corners = np.array(tracks["corners_img"], dtype=np.float64)
    print(f"Calibración INICIAL del run v4: {initial_corners.tolist()}")

    # Sacar los frames objetivo
    frames_to_test: dict[float, np.ndarray] = {}
    for idx, frame in read_frames(VIDEO, stride=1):
        t = idx / meta.fps
        for tgt in TARGET_TS:
            if tgt not in frames_to_test and t >= tgt:
                frames_to_test[tgt] = (idx, frame.copy())
                break
        if len(frames_to_test) == len(TARGET_TS):
            break

    print(f"\nFrames extraídos en t = {sorted(frames_to_test.keys())}")
    print()

    for t_target in sorted(frames_to_test.keys()):
        idx, frame = frames_to_test[t_target]
        print(f"=== t={t_target}s (idx={idx}) ===")

        # Detección en ESTE frame específico
        new_geom = detect_field_geometry(frame)
        new_goals = detect_goals_by_color(frame)

        print(
            f"  detect_field_geometry: success={new_geom.success} "
            f"method={new_geom.debug.get('method', '?')} "
            f"goals={[g.color for g in new_goals]}"
        )
        if new_geom.success:
            jump = float(
                np.linalg.norm(new_geom.corners_img - initial_corners, axis=1).max()
            )
            print(f"  Esquinas nuevas: {new_geom.corners_img.astype(int).tolist()}")
            print(f"  Salto vs inicial (max corner): {jump:.0f} px")

        # Frame del run procesado más cercano a este t (para teams)
        run_frame = min(tracks["frames"], key=lambda f: abs(f["t_s"] - t_target))
        team_counts = {"A": 0, "B": 0, None: 0}
        hues_by_team: dict[str, list[float]] = {"A": [], "B": [], "none": []}
        for r in run_frame["robots"]:
            t_label = r["team"]
            team_counts[t_label] = team_counts.get(t_label, 0) + 1
            hue = r.get("team_hue")
            if hue is not None:
                hues_by_team[t_label or "none"].append(hue)
        print(
            f"  Robots detectados (frame {run_frame['frame_idx']}): {len(run_frame['robots'])}"
        )
        print(
            f"    team A: {team_counts.get('A', 0)}, team B: {team_counts.get('B', 0)}, none: {team_counts.get(None, 0)}"
        )
        for t_label, hues in hues_by_team.items():
            if hues:
                hues_arr = np.array(hues)
                print(
                    f"    {t_label}: hue mean={hues_arr.mean():.0f} std={hues_arr.std():.0f} "
                    f"min={hues_arr.min():.0f} max={hues_arr.max():.0f}"
                )

        # Visualización
        viz = frame.copy()

        # 1. Cuadrilátero INICIAL (azul claro - hipótesis del fix de hoy)
        pts_init = initial_corners.astype(int)
        for i in range(4):
            cv2.line(
                viz,
                tuple(pts_init[i]),
                tuple(pts_init[(i + 1) % 4]),
                (255, 200, 0),
                3,
            )

        # 2. Cuadrilátero NUEVO en este frame (verde brillante)
        if new_geom.success:
            pts_new = new_geom.corners_img.astype(int)
            for i in range(4):
                cv2.line(
                    viz,
                    tuple(pts_new[i]),
                    tuple(pts_new[(i + 1) % 4]),
                    (0, 255, 0),
                    3,
                )

        # 3. Porterías (HSV) en este frame
        for g in new_goals:
            x1, y1, x2, y2 = [int(v) for v in g.bbox_xyxy]
            color = (0, 200, 255) if g.color == "yellow" else (255, 100, 0)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 4)
            cv2.putText(
                viz,
                f"GOAL {g.color}",
                (x1, max(40, y1 - 12)),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

            # 4. Línea de gol según las DOS calibraciones
            if new_geom.success:
                # Linea según calibración nueva (verde brillante)
                gl_a, gl_b = goal_line_from_field_edge(
                    new_geom.corners_img, g.centroid_img
                )
                cv2.line(
                    viz,
                    (int(gl_a[0]), int(gl_a[1])),
                    (int(gl_b[0]), int(gl_b[1])),
                    (0, 255, 0),
                    5,
                    cv2.LINE_AA,
                )
            # Linea según calibración inicial (cyan)
            gl_ia, gl_ib = goal_line_from_field_edge(initial_corners, g.centroid_img)
            cv2.line(
                viz,
                (int(gl_ia[0]), int(gl_ia[1])),
                (int(gl_ib[0]), int(gl_ib[1])),
                (255, 200, 0),
                3,
                cv2.LINE_AA,
            )

        # 5. Robots con su team de este frame
        team_color = {"A": (200, 30, 130), "B": (240, 240, 240), None: (180, 180, 180)}
        for r in run_frame["robots"]:
            x1, y1, x2, y2 = [int(v) for v in r["bbox_xyxy"]]
            c = team_color.get(r["team"], (180, 180, 180))
            cv2.rectangle(viz, (x1, y1), (x2, y2), c, 2)
            hue = r.get("team_hue")
            label = f"id={r['track_id']} t={r['team'] or '?'} h={int(hue) if hue is not None else '?'}"
            cv2.putText(
                viz,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                2,
                cv2.LINE_AA,
            )

        # Leyenda
        cv2.putText(
            viz,
            "Verde brillante = recalibracion AHORA / Cyan = inicial",
            (20, viz.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out_path = OUT_DIR / f"diagnose_t{int(t_target):03d}.jpg"
        cv2.imwrite(str(out_path), viz)
        print(f"  -> {out_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
