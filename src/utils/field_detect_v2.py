"""Detector v2 — basado en evidencia visual real del dataset oficial.

Reemplaza las suposiciones erradas de v1:
- El borde del fieltro verde NO es el límite del campo de juego.
  El campo de juego está delimitado por LINEAS BLANCAS interiores con
  ESCUADRAS BLANCAS L-SHAPED en las 4 esquinas reales.
- Las porterías NO son ROI virtual en mundo mm. Son CAJAS DE COLOR
  visibles: AMARILLA a un lado, AZUL al otro.

Pipeline v2:
1. `detect_white_field_lines(frame)` → 4 esquinas reales por Hough sobre
   líneas blancas RESTRINGIDAS al fieltro verde.
2. `detect_goals_by_color(frame)` → bbox + centro de cada portería por
   máscara HSV amarilla y azul.
3. `detect_field_geometry(frame)` → combina ambos en `FieldGeometryV2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.field_detect import (
    GREEN_HSV_LOWER,
    GREEN_HSV_UPPER,
    _order_corners_tl_tr_br_bl,
    segment_field_mask,
)

WHITE_HSV_LOWER = np.array([0, 0, 170], dtype=np.uint8)
WHITE_HSV_UPPER = np.array([180, 70, 255], dtype=np.uint8)

# Amarillo portería (saturado, brillante).
YELLOW_HSV_LOWER = np.array([18, 110, 110], dtype=np.uint8)
YELLOW_HSV_UPPER = np.array([38, 255, 255], dtype=np.uint8)

# Azul portería (acepta franja amplia, incluso azul muy oscuro / casi negro).
BLUE_HSV_LOWER = np.array([95, 50, 30], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class GoalDetection:
    color: str  # "yellow" | "blue"
    bbox_xyxy: np.ndarray  # (4,) — x1,y1,x2,y2 en imagen
    centroid_img: np.ndarray  # (2,)
    area_px: int


@dataclass(frozen=True)
class FieldGeometryV2:
    """Resultado completo de la detección v2."""

    success: bool
    corners_img: np.ndarray  # (4,2) en orden TL,TR,BR,BL — líneas blancas
    green_mask: np.ndarray  # uint8, máscara verde del fieltro
    white_mask: np.ndarray  # uint8, máscara blanca dentro del fieltro
    goals: list[GoalDetection]  # 0..2 porterías detectadas
    debug: dict  # números intermedios para auditoría


# ---------- helpers ----------


def _line_angle_deg(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if ang <= -90:
        ang += 180
    if ang > 90:
        ang -= 180
    return ang


def _line_length(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_homog(seg: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = seg
    return np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)


def _intersect(L1: np.ndarray, L2: np.ndarray) -> np.ndarray | None:
    cross = np.cross(L1, L2)
    if abs(cross[2]) < 1e-9:
        return None
    return np.array([cross[0] / cross[2], cross[1] / cross[2]], dtype=np.float64)


def _segment_distance(seg_ref: np.ndarray, seg_other: np.ndarray) -> float:
    """Distancia normal del punto medio de `seg_other` a la recta de `seg_ref`."""
    L = _line_homog(seg_ref)
    norm = np.hypot(L[0], L[1])
    if norm < 1e-9:
        return 0.0
    mid = np.array(
        [(seg_other[0] + seg_other[2]) / 2, (seg_other[1] + seg_other[3]) / 2],
        dtype=np.float64,
    )
    return abs(L[0] * mid[0] + L[1] * mid[1] + L[2]) / norm


# ---------- detectores ----------


def _make_field_roi(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (mascara_verde_amplia, mascara_fieltro_estricto)."""
    h_img, w_img = frame_bgr.shape[:2]
    # Verde amplio para ROI (incluye perímetro con líneas blancas).
    wide = segment_field_mask(frame_bgr, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    # Estricto: saturación alta para fieltro interior.
    strict = segment_field_mask(
        frame_bgr,
        np.array([35, 110, 60], dtype=np.uint8),
        GREEN_HSV_UPPER,
    )
    return wide, strict


def detect_white_field_lines(
    frame_bgr: np.ndarray,
    canny_low: int = 60,
    canny_high: int = 180,
    hough_threshold: int = 80,
    min_line_length_frac: float = 0.12,
    max_line_gap_frac: float = 0.03,
    dilate_roi_px: int = 35,
) -> tuple[np.ndarray | None, dict]:
    """Detecta las 4 esquinas reales del rectángulo de juego usando Hough sobre
    líneas blancas restringidas al fieltro verde.

    Returns:
        (corners (4,2) en orden TL,TR,BR,BL, debug_dict) ó (None, debug_dict).
    """
    h_img, w_img = frame_bgr.shape[:2]
    diag = float(np.hypot(h_img, w_img))
    min_len = int(min_line_length_frac * max(h_img, w_img))
    max_gap = int(max_line_gap_frac * max(h_img, w_img))

    wide_green, strict_green = _make_field_roi(frame_bgr)
    # ROI = fieltro estricto dilatado (incluye un margen para capturar
    # las líneas blancas que pueden estar justo en el borde).
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_roi_px, dilate_roi_px)
    )
    field_roi = cv2.dilate(strict_green, kernel)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white = cv2.bitwise_and(white, field_roi)
    # Pequeña apertura para limpiar ruido (~1 px).
    white = cv2.morphologyEx(
        white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    edges = cv2.Canny(white, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    debug = {
        "white_mask": white,
        "edges": edges,
        "n_lines_raw": 0 if lines is None else int(len(lines)),
        "field_roi": field_roi,
    }
    if lines is None or len(lines) < 4:
        return None, debug

    all_segs = lines.reshape(-1, 4).astype(np.float64)
    all_lengths = np.array([_line_length(s) for s in all_segs], dtype=np.float64)
    # Quedarnos con las top 40% más largas para excluir líneas de áreas de penalti
    # y otros marcadores cortos.
    keep_n = max(8, int(0.4 * len(all_segs)))
    top_idx = np.argsort(-all_lengths)[:keep_n]
    segs = all_segs[top_idx]
    angles = np.array([_line_angle_deg(s) for s in segs], dtype=np.float64)
    lengths = all_lengths[top_idx]
    debug["n_lines_kept"] = int(len(segs))

    # Clustering por (cos2θ, sin2θ) para evitar wrap-around 0/180.
    theta = np.radians(angles)
    feats = np.stack([np.cos(2 * theta), np.sin(2 * theta)], axis=1).astype(np.float32)
    _, labels, _ = cv2.kmeans(
        feats,
        2,
        None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        attempts=8,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    debug["n_group0"] = int((labels == 0).sum())
    debug["n_group1"] = int((labels == 1).sum())

    # Por grupo, escoger 2 líneas más largas que estén SEPARADAS por al menos
    # `min_sep_frac` del lado relevante (típicamente las dos sidelines opuestas).
    min_sep = 0.15 * diag
    picks_per_group: list[list[int]] = []
    for g in (0, 1):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            return None, {**debug, "fail": f"grupo {g} con <2 líneas"}
        order = idx[np.argsort(-lengths[idx])]
        first = order[0]
        picks = [first]
        for j in order[1:]:
            if _segment_distance(segs[first], segs[j]) > min_sep:
                picks.append(int(j))
                break
        if len(picks) < 2:
            return None, {**debug, "fail": f"grupo {g} sin par separado"}
        picks_per_group.append(picks)

    L = [_line_homog(segs[i]) for picks in picks_per_group for i in picks]
    inters: list[np.ndarray] = []
    for i in (0, 1):
        for k in (2, 3):
            pt = _intersect(L[i], L[k])
            if pt is None:
                return None, {**debug, "fail": "líneas paralelas"}
            inters.append(pt)
    corners = np.stack(inters)

    # Sanity: 4 esquinas no degeneradas + dentro de margen razonable
    margin = 0.5 * max(h_img, w_img)
    if (
        (corners[:, 0] < -margin).any()
        or (corners[:, 0] > w_img + margin).any()
        or (corners[:, 1] < -margin).any()
        or (corners[:, 1] > h_img + margin).any()
    ):
        return None, {**debug, "fail": "esquinas fuera de margen"}
    ordered = _order_corners_tl_tr_br_bl(corners)
    # Sanity: las esquinas deben estar mayormente DENTRO de la dilatación del
    # fieltro estricto (cerca del campo, no aleatorias).
    inside_roi = 0
    for x, y in ordered:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w_img and 0 <= yi < h_img and field_roi[yi, xi] > 0:
            inside_roi += 1
    debug["inside_roi"] = inside_roi
    if inside_roi < 3:
        return None, {**debug, "fail": f"{inside_roi}/4 esquinas dentro del ROI"}

    # Sanity geométrica: área del cuadrilátero razonable + aspect ratio plausible.
    quad_area = cv2.contourArea(ordered.astype(np.float32))
    area_frac = quad_area / (h_img * w_img)
    debug["quad_area_frac"] = float(area_frac)
    if area_frac < 0.20:
        return None, {**debug, "fail": f"cuadrilátero pequeño ({area_frac:.2f})"}
    top = float(np.linalg.norm(ordered[0] - ordered[1]))
    bot = float(np.linalg.norm(ordered[3] - ordered[2]))
    left = float(np.linalg.norm(ordered[0] - ordered[3]))
    right = float(np.linalg.norm(ordered[1] - ordered[2]))
    long_side = max(top, bot, left, right)
    short_side = max(1.0, min(top, bot, left, right))
    aspect = long_side / short_side
    debug["aspect"] = aspect
    if aspect > 4.0:
        return None, {**debug, "fail": f"aspect ratio degenerado ({aspect:.2f})"}

    debug["picked_segments"] = [
        segs[i].tolist() for picks in picks_per_group for i in picks
    ]
    return ordered, debug


def detect_goals_by_color(
    frame_bgr: np.ndarray, green_mask: np.ndarray | None = None
) -> list[GoalDetection]:
    """Encuentra portería AMARILLA y AZUL como componentes conexos de mayor área
    pegados al fieltro verde.

    La portería azul suele estar parcialmente ocluida (manos del público, sombras)
    y su HSV cae en rangos cercanos al negro: aplicamos dilatación generosa del
    ROI, cierre morfológico para reconectar fragmentos, y área mínima más baja
    para no descartar por ruido cuando solo se ve un trozo de la caja.
    """
    if green_mask is None:
        green_mask = segment_field_mask(frame_bgr)
    h_img, w_img = frame_bgr.shape[:2]
    # ROI extendido generoso para capturar porterías ocluidas en el lateral.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))
    near_field = cv2.dilate(green_mask, kernel)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    goals: list[GoalDetection] = []
    color_params = (
        ("yellow", YELLOW_HSV_LOWER, YELLOW_HSV_UPPER, 0.002),
        ("blue", BLUE_HSV_LOWER, BLUE_HSV_UPPER, 0.0008),
    )
    for color, lo, hi, area_min_frac in color_params:
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.bitwise_and(m, near_field)
        m = cv2.morphologyEx(
            m,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        m = cv2.morphologyEx(
            m,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < area_min_frac * h_img * w_img:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bbox = np.array([x, y, x + w, y + h], dtype=np.int32)
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        goals.append(
            GoalDetection(
                color=color,
                bbox_xyxy=bbox,
                centroid_img=np.array([cx, cy], dtype=np.float64),
                area_px=int(area),
            )
        )
    return goals


def _sector_corners(
    hull_points: np.ndarray, h_img: int, w_img: int
) -> np.ndarray | None:
    """4 vértices únicos del hull tomando el más lejano al centroide por cuadrante.

    Garantiza 4 puntos únicos (no colapsados) siempre que el hull cubra los
    4 cuadrantes alrededor de su centroide. Más robusto que argmin/argmax
    de x±y cuando el campo se extiende hasta el borde de la imagen (en cuyo
    caso varios vértices del hull comparten el mismo x extremo).
    """
    pts = hull_points.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    quadrant_filters = [
        (lambda p, cx=cx, cy=cy: p[0] <= cx and p[1] <= cy),  # TL
        (lambda p, cx=cx, cy=cy: p[0] >= cx and p[1] <= cy),  # TR
        (lambda p, cx=cx, cy=cy: p[0] >= cx and p[1] >= cy),  # BR
        (lambda p, cx=cx, cy=cy: p[0] <= cx and p[1] >= cy),  # BL
    ]
    out: list[np.ndarray] = []
    for f in quadrant_filters:
        cands = pts[[i for i, p in enumerate(pts) if f(p)]]
        if len(cands) == 0:
            return None
        dist = np.hypot(cands[:, 0] - cx, cands[:, 1] - cy)
        out.append(cands[int(np.argmax(dist))])
    return np.array(out, dtype=np.float64)


def detect_field_corners_from_white_mask(
    frame_bgr: np.ndarray,
    dilate_px: int = 25,
    close_px: int = 71,
    min_area_frac: float = 0.10,
    min_diag_separation_frac: float = 0.08,
) -> tuple[np.ndarray | None, dict]:
    """Detecta el cuadrilátero del campo usando la máscara blanca como evidencia.

    Estrategia: las líneas blancas perimetrales (sidelines) ya forman el
    cuadrilátero. Dilatamos para conectar segmentos, cerramos huecos,
    concatenamos TODOS los contornos no triviales en un solo nube de puntos
    y aplicamos convex hull + selección por cuadrantes.

    Returns:
        (corners (4,2) TL,TR,BR,BL, debug_dict) ó (None, debug_dict).
    """
    h_img, w_img = frame_bgr.shape[:2]
    diag_img = float(np.hypot(h_img, w_img))

    wide_green, strict_green = _make_field_roi(frame_bgr)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    field_roi = cv2.dilate(strict_green, kernel_roi)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white = cv2.bitwise_and(white, field_roi)

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    white_d = cv2.dilate(white, dilate_k)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px))
    white_closed = cv2.morphologyEx(white_d, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(
        white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    debug: dict = {"n_contours": len(contours)}
    if not contours:
        return None, {**debug, "fail": "sin contornos"}

    # Concatenar puntos de todos los contornos no triviales para que la
    # fragmentación no excluya zonas del campo (las líneas blancas suelen
    # fracturarse al pasar bajo robots o cerca de sombras).
    min_contour_area = 0.005 * h_img * w_img
    big_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    debug["n_contours_kept"] = len(big_contours)
    if not big_contours:
        big_contours = [max(contours, key=cv2.contourArea)]

    all_pts = np.vstack([c.reshape(-1, 2) for c in big_contours])
    combined_area = float(
        cv2.contourArea(cv2.convexHull(all_pts.reshape(-1, 1, 2).astype(np.int32)))
    )
    area_frac = combined_area / (h_img * w_img)
    debug["area_frac"] = area_frac
    if area_frac < min_area_frac:
        return None, {**debug, "fail": f"hull pequeño ({area_frac:.2f})"}

    hull = cv2.convexHull(all_pts.reshape(-1, 1, 2).astype(np.int32))
    corners = _sector_corners(hull, h_img, w_img)
    if corners is None:
        return None, {**debug, "fail": "hull no cubre los 4 cuadrantes"}

    min_sep = min_diag_separation_frac * diag_img
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(corners[i] - corners[j])) < min_sep:
                return None, {
                    **debug,
                    "fail": f"esquinas {i},{j} colapsadas",
                }

    return corners, {**debug, "method_inner": "white_mask_hull_sectors"}


def detect_field_geometry(frame_bgr: np.ndarray) -> FieldGeometryV2:
    """Detecta esquinas blancas + porterías en un solo pase.

    Cascada de métodos (de más preciso a más robusto):
    1. Hough sobre líneas blancas + clustering por ángulo (esquinas exactas
       cuando las sidelines se ven limpias y completas).
    2. Convex hull de la máscara blanca dilatada (extremos diagonales).
       Robusto cuando las líneas están fragmentadas u oclusas en un lado.
    3. Convex hull del fieltro verde (último recurso, menos preciso porque
       el verde se "corta" al perímetro de la mesa con sombras).
    """
    wide_green, strict_green = _make_field_roi(frame_bgr)
    corners, debug = detect_white_field_lines(frame_bgr)
    method = "white_lines"
    if corners is None:
        corners2, debug2 = detect_field_corners_from_white_mask(frame_bgr)
        debug.update({f"v3_{k}": v for k, v in debug2.items()})
        if corners2 is not None:
            corners = corners2
            method = "white_mask_hull"
    if corners is None:
        from src.utils.field_detect import detect_field_corners_hull

        hull_res = detect_field_corners_hull(frame_bgr)
        if hull_res.success:
            corners = hull_res.corners
            method = "hull_fallback"
            debug["fallback_used"] = "hull"
    debug["method"] = method
    goals = detect_goals_by_color(frame_bgr, green_mask=wide_green)
    return FieldGeometryV2(
        success=corners is not None,
        corners_img=corners if corners is not None else np.empty((0, 2)),
        green_mask=strict_green,
        white_mask=debug.get("white_mask", np.zeros_like(strict_green)),
        goals=goals,
        debug=debug,
    )
