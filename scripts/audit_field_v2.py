"""Auditoría visual del detector v2.

Para cada frame de prueba muestra un panel 2x4:
1. ORIGINAL
2. Máscara verde estricta
3. Máscara blanca dentro del fieltro
4. Líneas Hough detectadas
5. Esquinas REALES detectadas (líneas blancas)
6. Portería AMARILLA (bbox + centroide)
7. Portería AZUL (bbox + centroide)
8. RESULTADO FINAL combinado

Útil para validar el detector v2 antes de tocar el pipeline.
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

from src.utils import field_detect_v2 as v2
from src.utils.io import probe, read_frames

CAMARAS = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras"
TARGETS = [
    CAMARAS / "IMG_9821.MOV",
    CAMARAS / "IMG_9811.MOV",
    CAMARAS / "IMG_9808.MOV",
    CAMARAS / "IMG_9800.MOV",
]
OUT = ROOT / "data" / "processed" / "audit_v2"
OUT.mkdir(parents=True, exist_ok=True)


def label(img: np.ndarray, text: str, color=(0, 255, 255)) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        img, text, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 4, cv2.LINE_AA
    )
    return img


def panel_2x4(images_with_titles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    while len(images_with_titles) < 8:
        images_with_titles.append(("", np.zeros_like(images_with_titles[0][1])))
    labeled = [label(img.copy(), t) for t, img in images_with_titles]
    rows = []
    for r in range(2):
        rows.append(np.hstack(labeled[r * 4 : (r + 1) * 4]))
    return np.vstack(rows)


def draw_lines_hough(frame: np.ndarray, debug: dict) -> np.ndarray:
    vis = frame.copy()
    edges = debug.get("edges")
    if edges is not None:
        vis[edges > 0] = (0, 255, 255)
    segs = debug.get("picked_segments")
    if segs:
        for s in segs:
            x1, y1, x2, y2 = [int(v) for v in s]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 5, cv2.LINE_AA)
    return vis


def draw_corners(frame: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    vis = frame.copy()
    h_img, w_img = vis.shape[:2]
    if corners is None or len(corners) == 0:
        cv2.putText(
            vis,
            "FAIL",
            (20, 140),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )
        return vis
    pts = corners.astype(np.int32)
    for i in range(4):
        cv2.line(
            vis,
            tuple(pts[i]),
            tuple(pts[(i + 1) % 4]),
            (0, 255, 0),
            5,
            cv2.LINE_AA,
        )
    inside = 0
    for p in pts:
        in_img = 0 <= p[0] < w_img and 0 <= p[1] < h_img
        inside += int(in_img)
        cv2.circle(vis, tuple(p), 22, (0, 255, 0) if in_img else (0, 165, 255), -1)
        cv2.circle(vis, tuple(p), 22, (0, 0, 0), 3)
    cv2.putText(
        vis,
        f"{inside}/4 dentro",
        (20, 140),
        cv2.FONT_HERSHEY_DUPLEX,
        1.6,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    return vis


def draw_goal(frame: np.ndarray, goals, color_filter: str) -> np.ndarray:
    vis = frame.copy()
    found = False
    for g in goals:
        if g.color != color_filter:
            continue
        found = True
        x1, y1, x2, y2 = g.bbox_xyxy
        box_color = (0, 255, 255) if color_filter == "yellow" else (255, 100, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 6)
        cx, cy = g.centroid_img.astype(int)
        cv2.circle(vis, (cx, cy), 18, (255, 255, 255), -1)
        cv2.circle(vis, (cx, cy), 18, (0, 0, 0), 3)
        cv2.putText(
            vis,
            f"{color_filter} {g.area_px}px",
            (x1, max(40, y1 - 12)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            box_color,
            3,
            cv2.LINE_AA,
        )
    if not found:
        cv2.putText(
            vis,
            f"{color_filter}: no detectada",
            (20, 140),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return vis


def draw_final(frame: np.ndarray, geom: v2.FieldGeometryV2) -> np.ndarray:
    vis = draw_corners(frame, geom.corners_img if geom.success else None)
    for g in geom.goals:
        x1, y1, x2, y2 = g.bbox_xyxy
        box_color = (0, 255, 255) if g.color == "yellow" else (255, 100, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 6)
    return vis


def audit(video: Path) -> None:
    meta = probe(video)
    total = int(meta.duration_s * meta.fps)
    samples = np.linspace(0, total - 1, 3, dtype=int).tolist()
    sample_set = set(samples)
    print(f"\nVideo: {video.name}, dur={meta.duration_s:.1f}s, fps={meta.fps:.1f}")
    print(f"  sampling frames: {samples}")
    for idx, frame in read_frames(video, stride=1):
        if idx not in sample_set:
            continue
        sample_set.discard(idx)
        geom = v2.detect_field_geometry(frame)
        panel_img = panel_2x4(
            [
                ("1. ORIGINAL", frame.copy()),
                ("2. FIELTRO VERDE", geom.green_mask),
                ("3. BLANCO en CAMPO", geom.white_mask),
                ("4. LINEAS HOUGH", draw_lines_hough(frame, geom.debug)),
                (
                    "5. ESQUINAS REALES",
                    draw_corners(frame, geom.corners_img if geom.success else None),
                ),
                ("6. PORTERIA AMARILLA", draw_goal(frame, geom.goals, "yellow")),
                ("7. PORTERIA AZUL", draw_goal(frame, geom.goals, "blue")),
                ("8. RESULTADO FINAL", draw_final(frame, geom)),
            ]
        )
        out = OUT / f"{video.stem}_v2_frame{idx:05d}.jpg"
        cv2.imwrite(str(out), panel_img)
        print(
            f"  guardado {out.name}  "
            f"corners={'OK' if geom.success else 'FAIL'}  "
            f"goals={[g.color for g in geom.goals]}"
        )
        if not sample_set:
            break


def main() -> int:
    for v_path in TARGETS:
        if v_path.exists():
            audit(v_path)
        else:
            print(f"WARN: no existe {v_path}")
    print(f"\nPaneles en: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
