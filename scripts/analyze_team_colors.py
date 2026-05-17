"""Diagnóstico: extrae los colores HSV dominantes de cada robot detectado.

Lee data/processed/runs/<videoname>/tracks.json + el video original,
extrae el parche de cada robot por frame, calcula histograma HSV y
reporta los rangos dominantes por track_id.

Útil para calibrar src/tracking/reid.py con datos reales.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument(
        "--tracks", type=Path, required=True, help="tracks.json del pipeline"
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--top-fraction",
        type=float,
        default=0.5,
        help="fracción superior del bbox a muestrear (bandera)",
    )
    return p.parse_args()


def color_name_from_hue(h):
    """Mapea matiz dominante a nombre legible."""
    if h < 10 or h > 170:
        return "rojo"
    if h < 25:
        return "naranja"
    if h < 35:
        return "amarillo"
    if h < 85:
        return "verde"
    if h < 100:
        return "cian"
    if h < 130:
        return "azul"
    if h < 160:
        return "morado/violeta"
    return "magenta"


def main():
    args = parse_args()
    if args.out is None:
        args.out = args.tracks.parent / "team_colors.json"

    with open(args.tracks, "r", encoding="utf-8") as f:
        data = json.load(f)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir {args.video}", file=sys.stderr)
        return 1

    # Indexar frames por idx para lectura aleatoria
    frames_by_idx = {}
    for fr in data["frames"]:
        frames_by_idx[fr["frame_idx"]] = fr

    per_track_hsv = defaultdict(list)
    for idx, fr in frames_by_idx.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        for r in fr["robots"]:
            x1, y1, x2, y2 = [int(round(v)) for v in r["bbox_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            yh = y1 + int(round((y2 - y1) * args.top_fraction))
            patch = frame[y1:yh, x1:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            # Filtrar pixeles oscuros / desaturados
            valid = (hsv[:, :, 2] > 60) & (hsv[:, :, 1] > 50)
            if valid.sum() < 30:
                continue
            per_track_hsv[r["track_id"]].append(hsv[valid])

    cap.release()

    report = {}
    print("\n=== Análisis de colores por track ===\n")
    for tid, hsv_list in sorted(per_track_hsv.items()):
        if not hsv_list:
            continue
        all_hsv = np.vstack(hsv_list)
        h_med = int(np.median(all_hsv[:, 0]))
        s_med = int(np.median(all_hsv[:, 1]))
        v_med = int(np.median(all_hsv[:, 2]))
        h_hist = np.bincount(all_hsv[:, 0], minlength=180).astype(float)
        h_hist /= h_hist.sum()
        top_hues = np.argsort(h_hist)[-3:][::-1]
        color_name = color_name_from_hue(h_med)
        info = {
            "n_samples": int(all_hsv.shape[0]),
            "n_frames": len(hsv_list),
            "H_median": h_med,
            "S_median": s_med,
            "V_median": v_med,
            "color_dominante": color_name,
            "top_hues": [int(x) for x in top_hues],
            "hue_fractions_top": [float(h_hist[i]) for i in top_hues],
        }
        report[str(tid)] = info
        print(
            f"track {tid}: H={h_med:3d} S={s_med:3d} V={v_med:3d}  "
            f"[{color_name}]  n_frames={info['n_frames']}  "
            f"top_hues={info['top_hues']} ({[f'{f * 100:.0f}%' for f in info['hue_fractions_top']]})"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
