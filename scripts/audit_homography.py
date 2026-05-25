"""Auditoría visual del mapeo de franjas blancas con la homografía.

Para un frame dado:
1. Detecta el cuadrilátero del campo (cascade actual) y las porterías.
2. Calcula la homografía imagen → mundo (mm).
3. Extrae la máscara de líneas blancas y la proyecta al espacio top-down.
4. Dibuja un campo de referencia con las dimensiones del reglamento
   (2190×1580 mm, círculo central, áreas de portería) y superpone la
   máscara blanca proyectada en magenta.

Si la calibración es correcta, las franjas blancas reales del campo
deberían superponerse aproximadamente sobre las líneas del campo de
referencia.

Uso:
    python scripts/audit_homography.py --video <ruta.MOV> [--frame 0]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.utils.calib import (
    FIELD_LENGTH_MM,
    FIELD_WIDTH_MM,
    compute_homography,
)
from src.utils.field_detect_v2 import (
    WHITE_HSV_LOWER,
    WHITE_HSV_UPPER,
    _make_field_roi,
    detect_field_geometry,
)

logger = logging.getLogger("audit_homography")

PX_PER_MM = 0.3  # 657×474 px de campo
TOPDOWN_W = int(FIELD_LENGTH_MM * PX_PER_MM)
TOPDOWN_H = int(FIELD_WIDTH_MM * PX_PER_MM)


def reference_field(canvas_w: int, canvas_h: int) -> np.ndarray:
    """Dibuja el campo de referencia con dimensiones del reglamento (verde + líneas).

    Líneas dibujadas según convención conocida:
    - Rectángulo perimetral.
    - Línea de medio campo (vertical en x=length/2).
    - Círculo central radio ~300 mm.
    - Áreas de portería (rectángulos en los lados, ~600×1100 mm).
    """
    img = np.full((canvas_h, canvas_w, 3), (50, 110, 50), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (canvas_w - 1, canvas_h - 1), (255, 255, 255), 3)
    cv2.line(
        img,
        (canvas_w // 2, 0),
        (canvas_w // 2, canvas_h - 1),
        (255, 255, 255),
        2,
    )
    cv2.circle(
        img,
        (canvas_w // 2, canvas_h // 2),
        int(300 * PX_PER_MM),
        (255, 255, 255),
        2,
    )
    box_w = int(600 * PX_PER_MM)
    box_h = int(1100 * PX_PER_MM)
    box_y0 = (canvas_h - box_h) // 2
    cv2.rectangle(img, (0, box_y0), (box_w, box_y0 + box_h), (255, 255, 255), 2)
    cv2.rectangle(
        img,
        (canvas_w - box_w, box_y0),
        (canvas_w - 1, box_y0 + box_h),
        (255, 255, 255),
        2,
    )
    return img


def warp_white_mask_to_topdown(
    frame_bgr: np.ndarray, H: np.ndarray, out_w: int, out_h: int
) -> np.ndarray:
    """Aplica la homografía a la máscara blanca dentro del ROI verde."""
    wide_green, strict_green = _make_field_roi(frame_bgr)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    field_roi = cv2.dilate(strict_green, kernel_roi)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white = cv2.bitwise_and(white, field_roi)
    S = np.array([[PX_PER_MM, 0, 0], [0, PX_PER_MM, 0], [0, 0, 1]], dtype=np.float64)
    M = S @ H
    return cv2.warpPerspective(white, M, (out_w, out_h))


def render_audit_panel(
    frame_bgr: np.ndarray, video_name: str, frame_idx: int
) -> tuple[np.ndarray, dict]:
    """Devuelve panel 1x2: (annotated_left, topdown_audit_right) + métricas."""
    res = detect_field_geometry(frame_bgr)
    if not res.success:
        raise RuntimeError("Detector de campo falló — sin esquinas")
    H = compute_homography(res.corners_img)

    # Panel izquierdo: frame original con cuadrilátero + porterías + máscara blanca superpuesta
    h_img, w_img = frame_bgr.shape[:2]
    left = frame_bgr.copy()
    pts = res.corners_img.astype(int)
    for i in range(4):
        cv2.line(
            left, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 5, cv2.LINE_AA
        )
    for p in pts:
        cv2.circle(left, tuple(p), 18, (0, 0, 255), -1)
    for g in res.goals:
        x1, y1, x2, y2 = g.bbox_xyxy
        color = (0, 255, 255) if g.color == "yellow" else (255, 80, 0)
        cv2.rectangle(left, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            left,
            g.color.upper(),
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            color,
            3,
        )

    # Líneas blancas en magenta dentro del cuadrilátero
    wide_green, strict_green = _make_field_roi(frame_bgr)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    field_roi = cv2.dilate(strict_green, kernel_roi)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_left = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white_left = cv2.bitwise_and(white_left, field_roi)
    left[white_left > 0] = (255, 0, 255)  # magenta

    # Panel derecho: top-down con campo de referencia + máscara blanca proyectada
    right = reference_field(TOPDOWN_W, TOPDOWN_H)
    warped_white = warp_white_mask_to_topdown(frame_bgr, H, TOPDOWN_W, TOPDOWN_H)
    right[warped_white > 0] = (255, 0, 255)  # magenta sobre referencia
    # Marcar el contorno del rectángulo del campo de referencia en amarillo para
    # contraste con el blanco real.
    cv2.rectangle(right, (0, 0), (TOPDOWN_W - 1, TOPDOWN_H - 1), (0, 255, 255), 2)

    # Métricas de cobertura: % de píxeles blancos proyectados dentro del campo
    inside = int((warped_white > 0).sum())
    metrics = {
        "video": video_name,
        "frame_idx": frame_idx,
        "method": res.debug.get("method", "?"),
        "goals_detected": [g.color for g in res.goals],
        "corners_px": res.corners_img.astype(int).tolist(),
        "white_pixels_topdown": inside,
        "topdown_canvas_px": TOPDOWN_W * TOPDOWN_H,
        "white_coverage_pct": round(100 * inside / (TOPDOWN_W * TOPDOWN_H), 2),
    }

    # Componer panel: ambos lados a la misma altura. El frame original es
    # 1920x1080, la top-down es 657x474 — escalamos top-down ×4 y ponemos
    # margen para que se vea claro.
    scale = h_img / TOPDOWN_H
    new_w = int(TOPDOWN_W * scale)
    right_scaled = cv2.resize(right, (new_w, h_img), interpolation=cv2.INTER_NEAREST)

    # Banner superior con métricas
    panel = np.hstack([left, right_scaled])
    banner_h = 100
    banner = np.zeros((banner_h, panel.shape[1], 3), dtype=np.uint8)
    txt = (
        f"{video_name}  frame={frame_idx}  metodo={metrics['method']}  "
        f"porterias={metrics['goals_detected']}  "
        f"blanco_topdown={metrics['white_coverage_pct']}%"
    )
    cv2.putText(
        banner,
        txt,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([banner, panel]), metrics


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0, help="Frame index (default 0)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JPG (default: data/processed/calib_debug/<stem>_audit.jpg)",
    )
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error("No se pudo abrir %s", args.video)
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        logger.error("No se leyó frame %d", args.frame)
        return 1

    out_path = args.out or (
        ROOT / "data" / "processed" / "calib_debug" / f"{args.video.stem}_audit.jpg"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel, metrics = render_audit_panel(frame, args.video.name, args.frame)
    cv2.imwrite(str(out_path), panel)
    logger.info("Panel guardado: %s", out_path)
    for k, v in metrics.items():
        logger.info("  %s = %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
