"""Valida los baselines (HSV ball + homografía 4-puntos) sobre 1 frame real.

Uso:
    python scripts/baseline_check.py --frame data/processed/sample_frames/frame_24s.jpg

Genera:
    data/processed/baseline_check/
        ball_detection.jpg     frame con círculo verde sobre el balón
        ball_mask.jpg          máscara HSV binaria
        topdown.jpg            rectificación con esquinas hardcoded de prueba
        report.json            métricas de la corrida
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.baselines import (
    detect_orange_ball_mask,
    draw_detection,
    find_ball_centroid,
)
from src.utils.calib import compute_homography, warp_to_topdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "data" / "processed" / "sample_frames" / "frame_24s.jpg",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "baseline_check",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.frame.exists():
        print(f"FRAME NO ENCONTRADO: {args.frame}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.frame))
    if img is None:
        print(f"NO SE PUDO LEER {args.frame}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]

    # ----- 1. Ball detection -----
    det = find_ball_centroid(img)
    mask = detect_orange_ball_mask(img)
    annotated = draw_detection(img, det)

    cv2.imwrite(str(args.out / "ball_detection.jpg"), annotated)
    cv2.imwrite(str(args.out / "ball_mask.jpg"), mask)

    # ----- 2. Homografía con esquinas estimadas visualmente del frame -----
    # NOTA: estas esquinas son aproximadas para una prueba. Se reemplazan
    # con detección automática o anotación manual en Fase 1.
    # Para frame_24s.jpg de IMG_9915.MOV (Meta Glasses iPhone), las 4
    # esquinas interiores del campo visibles aproximadas en píxeles:
    # (cancha vista oblicuamente desde una esquina izquierda-cercana)
    estimated_corners = np.array(
        [
            [340, 60],  # TL (lejano)
            [1400, 65],  # TR (lejano)
            [1450, 720],  # BR (cercano)
            [40, 700],  # BL (cercano)
        ],
        dtype=np.float64,
    )
    try:
        H = compute_homography(estimated_corners)
        topdown = warp_to_topdown(img, H, scale=0.3)
        cv2.imwrite(str(args.out / "topdown.jpg"), topdown)
        topdown_shape = list(topdown.shape)
        homo_ok = True
    except Exception as e:  # noqa: BLE001
        H = None
        topdown_shape = None
        homo_ok = False
        homo_err = str(e)

    # ----- 3. Reporte -----
    report = {
        "frame": str(args.frame.name),
        "frame_size": [int(w), int(h)],
        "ball_detection": {
            "found": bool(det.found),
            "cx": det.cx,
            "cy": det.cy,
            "radius": det.radius,
            "area": det.area,
            "confidence": det.confidence,
            "mask_pixel_count": int((mask > 0).sum()),
        },
        "homography": {
            "ok": homo_ok,
            "estimated_corners": estimated_corners.tolist(),
            "topdown_shape": topdown_shape,
        },
    }
    if not homo_ok:
        report["homography"]["error"] = homo_err

    with open(args.out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
