"""Experimento: ¿SAM 3 detecta porterías robóticas mejor que HSV?

Carga el modelo, prueba 3 prompts distintos sobre 3 frames (inicio, medio,
final) de IMG_9821 (cámara móvil) y compara con el HSV actual.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.network import enable_system_ssl

enable_system_ssl()

import cv2
import numpy as np

from src.segmentation.sam3 import load_model, segment_with_text
from src.utils.field_detect_v2 import detect_goals_by_color
from src.utils.io import probe, read_frames

VIDEO = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras" / "IMG_9821.MOV"
OUT_DIR = ROOT / "data" / "processed" / "experiments" / "sam_goals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "yellow goal box",
    "blue goal box",
    "robot soccer goal",
]


def main() -> int:
    meta = probe(VIDEO)
    print(
        f"Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.duration_s:.1f}s"
    )

    # Sacar 3 frames: inicio, medio, final
    target_ts = [2.0, 30.0, 55.0]
    frames_to_test: dict[float, np.ndarray] = {}
    for idx, frame in read_frames(VIDEO, stride=1):
        t = idx / meta.fps
        for tgt in target_ts:
            if tgt not in frames_to_test and t >= tgt:
                frames_to_test[tgt] = frame.copy()
                break
        if len(frames_to_test) == len(target_ts):
            break

    print(f"Frames extraídos en t = {sorted(frames_to_test.keys())}")

    print("Cargando SAM 3.1...")
    processor, model = load_model("facebook/sam3", device=None)
    print("OK")

    for t_target in sorted(frames_to_test.keys()):
        frame = frames_to_test[t_target]
        print(f"\n=== Frame t={t_target}s ===")

        # HSV (lo actual)
        hsv_goals = detect_goals_by_color(frame)
        print(f"HSV: {len(hsv_goals)} porterías")
        for g in hsv_goals:
            print(
                f"  {g.color:6s} bbox={g.bbox_xyxy.tolist()} centroid=({g.centroid_img[0]:.0f},{g.centroid_img[1]:.0f}) area={g.area_px}"
            )

        # SAM 3 con cada prompt
        seg = segment_with_text(frame, PROMPTS, processor, model, threshold=0.2)
        for prompt in PROMPTS:
            masks = seg.get(prompt, [])
            print(f"SAM '{prompt}': {len(masks)} detecciones")
            for m in masks[:3]:
                ys, xs = np.where(m.mask)
                if xs.size == 0:
                    continue
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                w, h = x2 - x1, y2 - y1
                print(
                    f"  score={m.score:.3f} bbox=({x1},{y1})-({x2},{y2}) "
                    f"w={w} h={h} area={m.mask.sum()}px²"
                )

        # Visualización
        viz = frame.copy()
        # HSV en magenta
        for g in hsv_goals:
            x1, y1, x2, y2 = [int(v) for v in g.bbox_xyxy]
            cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.putText(
                viz,
                f"HSV {g.color}",
                (x1, max(40, y1 - 12)),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        # SAM en verde (top-1 por prompt)
        sam_colors = {
            "yellow goal box": (0, 200, 255),
            "blue goal box": (255, 200, 0),
            "robot soccer goal": (0, 255, 0),
        }
        for prompt in PROMPTS:
            masks = seg.get(prompt, [])
            if not masks:
                continue
            best = max(masks, key=lambda m: m.score)
            ys, xs = np.where(best.mask)
            if xs.size == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            color = sam_colors[prompt]
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                viz,
                f"SAM '{prompt}' s={best.score:.2f}",
                (x1, max(70, y1 - 50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        out_path = OUT_DIR / f"IMG_9821_t{int(t_target):03d}_comparison.jpg"
        cv2.imwrite(str(out_path), viz)
        print(f"  -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
