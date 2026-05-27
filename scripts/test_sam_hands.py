"""Experimento: ¿SAM 3 detecta manos humanas para identificar interrupciones?

Reglamento Copa FutBotMX § 4.3.1 (saque inicial), § 4.4.10.7 (robot dañado),
§ 4.4.3 (falta de progreso) — todos requieren que el árbitro intervenga
físicamente. Detectar manos permite pausar el pipeline durante esos eventos
y evitar que la línea de gol se desestabilice por features espurios.

Probamos 5 prompts distintos sobre 6 frames de IMG_9852 (inicio = saque,
medio = juego normal, final = post-gol con reposicionamiento).
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
from src.utils.io import probe, read_frames

VIDEO = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras" / "IMG_9852.MOV"
OUT_DIR = ROOT / "data" / "processed" / "experiments" / "sam_hands"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "human hand",
    "hand reaching into field",
    "person arm",
    "hand",
    "human reaching",
]

TARGET_TS = [0.5, 2.0, 10.0, 15.0, 18.5, 20.5]


def main() -> int:
    meta = probe(VIDEO)
    print(
        f"Video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.duration_s:.1f}s"
    )

    frames_to_test: dict[float, tuple[int, np.ndarray]] = {}
    for idx, frame in read_frames(VIDEO, stride=1):
        t = idx / meta.fps
        for tgt in TARGET_TS:
            if tgt not in frames_to_test and t >= tgt:
                frames_to_test[tgt] = (idx, frame.copy())
                break
        if len(frames_to_test) == len(TARGET_TS):
            break

    print(f"Frames extraídos en t = {sorted(frames_to_test.keys())}")

    print(
        "Cargando SAM 3.1 en CPU (no compite con GPU si hay otro pipeline corriendo)..."
    )
    processor, model = load_model("facebook/sam3", device="cpu")
    print("OK")

    for t_target in sorted(frames_to_test.keys()):
        idx, frame = frames_to_test[t_target]
        print(f"\n=== t={t_target}s (idx={idx}) ===")

        seg = segment_with_text(frame, PROMPTS, processor, model, threshold=0.2)
        viz = frame.copy()
        colors = {
            "human hand": (0, 255, 255),
            "hand reaching into field": (0, 200, 255),
            "person arm": (0, 150, 255),
            "hand": (255, 200, 0),
            "human reaching": (255, 150, 0),
        }
        any_hand = False
        for prompt in PROMPTS:
            masks = seg.get(prompt, [])
            print(f"  '{prompt}': {len(masks)} detecciones")
            for m in masks[:3]:
                ys, xs = np.where(m.mask)
                if xs.size == 0:
                    continue
                x1, y1, x2, y2 = (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                )
                w, h = x2 - x1, y2 - y1
                area_frac = m.mask.sum() / (frame.shape[0] * frame.shape[1])
                print(
                    f"    score={m.score:.3f} bbox=({x1},{y1})-({x2},{y2}) "
                    f"w={w} h={h} area_frac={area_frac:.3f}"
                )
                color = colors[prompt]
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    viz,
                    f"{prompt[:15]} s={m.score:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                if m.score > 0.3:
                    any_hand = True

        cv2.putText(
            viz,
            f"t={t_target:.1f}s {'HAND DETECTED' if any_hand else 'no hand'}",
            (20, viz.shape[0] - 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 0, 255) if any_hand else (0, 200, 0),
            3,
            cv2.LINE_AA,
        )
        out_path = OUT_DIR / f"IMG_9852_t{int(t_target * 10):04d}_hands.jpg"
        cv2.imwrite(str(out_path), viz)
        print(f"  -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
