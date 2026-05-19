"""Debug visual del detector de esquinas sobre un video.

Extrae 6 frames equidistantes y para cada uno guarda:
- frame original
- máscara verde
- contorno seleccionado
- polígono aproximado con esquinas marcadas

Sirve para entender por qué falla y diseñar uno mejor.
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

from src.utils.field_detect import detect_field_corners, segment_field_mask
from src.utils import field_detect as _fd  # noqa: F401  (acceso dinámico a _hough/_hull)
from src.utils.io import probe, read_frames

CAMARAS = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras"
TARGETS = [
    CAMARAS / "IMG_9821.MOV",
    CAMARAS / "IMG_9811.MOV",
    CAMARAS / "IMG_9808.MOV",
    CAMARAS / "IMG_9800.MOV",
]
OUT = ROOT / "data" / "processed" / "calib_debug"
OUT.mkdir(parents=True, exist_ok=True)


def panel(images_with_titles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Composito 2x3 de imágenes etiquetadas para auditoría visual."""
    h, w = images_with_titles[0][1].shape[:2]
    # Convertir grayscales a BGR
    norm = []
    for title, img in images_with_titles:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            img,
            title,
            (20, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            1.8,
            (0, 255, 255),
            4,
            cv2.LINE_AA,
        )
        norm.append(img)
    # Construir 2x3 grid
    while len(norm) < 6:
        norm.append(np.zeros_like(norm[0]))
    rows = []
    for r in range(2):
        rows.append(np.hstack(norm[r * 3 : (r + 1) * 3]))
    return np.vstack(rows)


def debug_video(VIDEO: Path) -> None:
    meta = probe(VIDEO)
    total = int(meta.duration_s * meta.fps)
    samples = np.linspace(0, total - 1, 3, dtype=int).tolist()
    sample_set = set(samples)
    print(f"\nVideo: {VIDEO.name}, {meta.duration_s:.1f}s, {meta.fps:.1f}fps")
    print(f"Sampling frames: {samples}")

    for idx, frame in read_frames(VIDEO, stride=1):
        if idx not in sample_set:
            continue
        sample_set.discard(idx)
        h_img, w_img = frame.shape[:2]

        # 1. Máscara verde
        mask = segment_field_mask(frame)
        green_overlay = frame.copy()
        green_overlay[mask > 0] = (0, 255, 0)

        # 2. Contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_vis = frame.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_vis, [largest], -1, (255, 0, 255), 4)
            area_ratio = cv2.contourArea(largest) / (h_img * w_img)
            cv2.putText(
                contour_vis,
                f"area={area_ratio * 100:.1f}%",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        # 3. approxPolyDP detalle (mostrar n vértices encontrados)
        approx_vis = frame.copy()
        n_vert = 0
        if contours:
            perim = cv2.arcLength(largest, True)
            for eps_frac in (0.02, 0.01, 0.03, 0.05, 0.08):
                approx = cv2.approxPolyDP(largest, eps_frac * perim, True)
                if len(approx) == 4:
                    break
            n_vert = len(approx)
            cv2.polylines(approx_vis, [approx], True, (0, 165, 255), 4)
            for p in approx.reshape(-1, 2):
                inside = 0 <= p[0] < w_img and 0 <= p[1] < h_img
                cv2.circle(
                    approx_vis,
                    tuple(p),
                    14,
                    (0, 0, 255) if inside else (0, 165, 255),
                    -1,
                )
            cv2.putText(
                approx_vis,
                f"approx vertices={n_vert}",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        # 4. Resultado actual detect_field_corners
        res = detect_field_corners(frame)
        result_vis = frame.copy()
        if res.success:
            pts = res.corners.astype(np.int32)
            inside_count = 0
            for i in range(4):
                cv2.line(
                    result_vis,
                    tuple(pts[i]),
                    tuple(pts[(i + 1) % 4]),
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA,
                )
            for p in pts:
                inside = 0 <= p[0] < w_img and 0 <= p[1] < h_img
                inside_count += int(inside)
                cv2.circle(
                    result_vis,
                    tuple(p),
                    14,
                    (0, 0, 255) if inside else (0, 165, 255),
                    -1,
                )
            cv2.putText(
                result_vis,
                f"DETECTOR ACTUAL: {inside_count}/4 esquinas dentro",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                result_vis,
                "DETECTOR ACTUAL: FAIL",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        # 5. Líneas blancas (Hough) — para futuro detector mejorado
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 60, 255], dtype=np.uint8),
        )
        # Restringir a zona del campo (verde dilatado)
        green_dilated = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        white_mask = cv2.bitwise_and(white_mask, green_dilated)
        # Detección Canny + HoughP
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=20
        )
        white_vis = frame.copy()
        white_vis[white_mask > 0] = (255, 255, 255)
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(white_vis, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(
                white_vis,
                f"Hough lines blancas: {len(lines)}",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )

        # 7. NUEVO detector Hough
        res_hough = _fd.detect_field_corners_hull(frame)
        hough_vis = frame.copy()
        if res_hough.success:
            pts = res_hough.corners.astype(np.int32)
            inside_count = sum(
                1 for x, y in res_hough.corners if 0 <= x < w_img and 0 <= y < h_img
            )
            for i in range(4):
                cv2.line(
                    hough_vis,
                    tuple(pts[i]),
                    tuple(pts[(i + 1) % 4]),
                    (0, 255, 0),
                    5,
                    cv2.LINE_AA,
                )
            for p in pts:
                inside = 0 <= p[0] < w_img and 0 <= p[1] < h_img
                cv2.circle(
                    hough_vis,
                    tuple(p),
                    18,
                    (0, 255, 0) if inside else (0, 165, 255),
                    -1,
                )
            cv2.putText(
                hough_vis,
                f"DETECTOR HULL: {inside_count}/4",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                hough_vis,
                "DETECTOR HULL: FAIL",
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        panel_img = panel(
            [
                ("1. ORIGINAL", frame.copy()),
                ("2. MASCARA VERDE", green_overlay),
                ("3. CONTORNO MAYOR", contour_vis),
                ("4. LINEAS BLANCAS (Hough)", white_vis),
                ("5. ACTUAL (verde+approx)", result_vis),
                ("6. NUEVO (convex hull)", hough_vis),
            ]
        )
        out_path = OUT / f"{VIDEO.stem}_debug_frame{idx:05d}.jpg"
        cv2.imwrite(str(out_path), panel_img)
        print(f"  guardado {out_path.name}")
        if not sample_set:
            break


def main() -> int:
    for v in TARGETS:
        if v.exists():
            debug_video(v)
        else:
            print(f"WARN: no existe {v}")
    print(f"\nPaneles en: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
