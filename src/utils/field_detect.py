"""Detección automática de las 4 esquinas del campo (fieltro verde).

Estrategia (más simple y robusta que Hough sobre las líneas blancas):
1. Segmentar píxeles verdes con HSV.
2. Limpiar con morfología (close + open).
3. Componente conexo más grande (presumiblemente el campo).
4. Contorno externo.
5. Aproximar contorno a polígono con `cv2.approxPolyDP`; si tiene
   4 vértices, esas son las esquinas.
6. Ordenar TL, TR, BR, BL.

Fundamento: el campo verde domina el centro de cada frame, su color
HSV está bien delimitado, y la forma observada es siempre un
cuadrilátero (oblicuo o no). Hough sobre líneas blancas falla cuando
las líneas del área de penalti tienen contraste similar a las del
borde — segmentar verde es más discriminativo.

Como fallback: el usuario puede anotar las 4 esquinas a mano y
guardar en `data/calibration/<video>.json`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Rango HSV para fieltro verde de la cancha (saturado, brillante).
# Frame de muestra: H ~ 60-75, S ~ 100-220, V ~ 80-200.
GREEN_HSV_LOWER = np.array([35, 70, 50], dtype=np.uint8)
GREEN_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class FieldDetectionResult:
    success: bool
    corners: np.ndarray  # (4, 2) en orden TL, TR, BR, BL — vacío si !success
    mask: np.ndarray  # uint8, máscara binaria del campo
    contour_area_ratio: float  # área del contorno / área del frame


def segment_field_mask(
    image_bgr: np.ndarray,
    hsv_lower: np.ndarray = GREEN_HSV_LOWER,
    hsv_upper: np.ndarray = GREEN_HSV_UPPER,
    kernel_size: int = 9,
) -> np.ndarray:
    """Máscara binaria del fieltro verde."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos a TL, TR, BR, BL por geometría."""
    pts = pts.reshape(4, 2).astype(np.float64)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()  # x - y? cv2 returns (y, x)? aclaro:
    # diff = pts[:,1] - pts[:,0] ... pero esto no es lo que queremos.
    # Mejor: ordenar por suma (TL min, BR max) y por resta (TR max, BL min).
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.stack([tl, tr, br, bl])


def detect_field_corners(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.15,
    approx_epsilon_frac: float = 0.02,
) -> FieldDetectionResult:
    """Detecta las 4 esquinas del campo aproximando el contorno verde.

    Args:
        image_bgr: frame en BGR.
        min_area_ratio: área mínima del campo como fracción del frame.
        approx_epsilon_frac: tolerancia de aproximación poligonal
            como fracción del perímetro (típico 0.01-0.05).

    Returns:
        FieldDetectionResult con success=False si no se pudo
        determinar un cuadrilátero confiable.
    """
    h, w = image_bgr.shape[:2]
    frame_area = h * w
    mask = segment_field_mask(image_bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, 0.0)

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    area_ratio = area / frame_area

    if area_ratio < min_area_ratio:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)

    def _is_valid(c: np.ndarray) -> bool:
        """Las 4 esquinas deben estar separadas por al menos 5 px."""
        pw = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(pw, np.inf)
        return bool(pw.min() >= 5.0)

    # Aproximación poligonal — si no da 4, probar con epsilons distintos.
    for eps_frac in (approx_epsilon_frac, 0.01, 0.03, 0.05, 0.08):
        epsilon = eps_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            corners = _order_corners_tl_tr_br_bl(approx)
            if _is_valid(corners):
                return FieldDetectionResult(True, corners, mask, area_ratio)

    # Fallback: bounding box rotado.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    corners = _order_corners_tl_tr_br_bl(box)
    if _is_valid(corners):
        return FieldDetectionResult(True, corners, mask, area_ratio)
    return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)


WHITE_HSV_LOWER = np.array([0, 0, 180], dtype=np.uint8)
WHITE_HSV_UPPER = np.array([180, 60, 255], dtype=np.uint8)


def detect_field_corners_hull(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.15,
    erode_px: int = 25,
) -> FieldDetectionResult:
    """Detección por envolvente convexa del contorno verde.

    Más robusta que approxPolyDP directo cuando hay manos/personas/objetos
    no-verdes sobre el campo que fragmentan el contorno:

    1. Máscara verde HSV restrictiva (saturación alta).
    2. Erosión agresiva para desconectar el campo de otras zonas verdes
       (césped exterior, sillas, fondo).
    3. Componente conexo mayor por reflejo del area.
    4. Re-dilatación para recuperar bordes perdidos por la erosión.
    5. Convex hull del contorno (elimina concavidades por manos/personas).
    6. approxPolyDP con epsilon adaptativo sobre la HULL hasta obtener 4 vértices.
    7. Ordenar TL, TR, BR, BL.

    Idea: la silueta del campo desde cualquier ángulo es un cuadrilátero
    convexo. Las "mordidas" en el contorno (manos arriba del campo)
    desaparecen al pasar al hull. La erosión asegura que el campo se
    aísle de otras zonas verdes externas.
    """
    h_img, w_img = image_bgr.shape[:2]
    frame_area = h_img * w_img
    # Máscara verde MÁS estricta (saturación alta) para excluir verdes pastel
    # del entorno (sillas, alfombras claras, césped exterior).
    strict_lower = np.array([35, 110, 60], dtype=np.uint8)
    strict_upper = GREEN_HSV_UPPER
    raw_mask = segment_field_mask(image_bgr, strict_lower, strict_upper)

    # Erosión agresiva para desconectar el campo de zonas verdes contiguas.
    er_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    eroded = cv2.erode(raw_mask, er_kernel)

    # Componente conexo más grande (el campo es la masa central dominante).
    n_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
    if n_labels <= 1:
        return FieldDetectionResult(False, np.empty((0, 2)), raw_mask, 0.0)
    # Saltar fondo (label 0).
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    isolated = ((labels_im == largest_label).astype(np.uint8)) * 255

    # Re-dilatación para recuperar el borde recortado por la erosión.
    isolated = cv2.dilate(isolated, er_kernel)
    mask = isolated

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, 0.0)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    area_ratio = area / frame_area
    if area_ratio < min_area_ratio:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < area:
        hull_area = area
    perim = cv2.arcLength(hull, True)
    if perim < 1.0:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)

    # Búsqueda binaria del epsilon que produce exactamente 4 vértices.
    # epsilon más grande => menos vértices.
    best = None
    lo, hi = 0.001, 0.20
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(hull, mid * perim, True)
        n = len(approx)
        if n == 4:
            best = approx
            break
        if n > 4:
            lo = mid
        else:
            hi = mid
    if best is None:
        # Barrido grueso de respaldo.
        for eps in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.10):
            approx = cv2.approxPolyDP(hull, eps * perim, True)
            if len(approx) == 4:
                best = approx
                break
    if best is None:
        return FieldDetectionResult(False, np.empty((0, 2)), mask, area_ratio)

    corners = _order_corners_tl_tr_br_bl(best)
    return FieldDetectionResult(True, corners, mask, area_ratio)


def _line_angle_deg(x1, y1, x2, y2) -> float:
    """Ángulo en (-90, 90] grados."""
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if ang <= -90:
        ang += 180
    if ang > 90:
        ang -= 180
    return ang


def _line_length(x1, y1, x2, y2) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_homog(x1, y1, x2, y2) -> np.ndarray:
    """Coeficientes (a, b, c) de la recta ax + by + c = 0."""
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return np.array([a, b, c], dtype=np.float64)


def _intersect(L1: np.ndarray, L2: np.ndarray) -> np.ndarray | None:
    cross = np.cross(L1, L2)
    if abs(cross[2]) < 1e-9:
        return None
    return np.array([cross[0] / cross[2], cross[1] / cross[2]], dtype=np.float64)


def detect_field_corners_hough(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.15,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 60,
    min_line_length_frac: float = 0.10,
    max_line_gap: int = 25,
    dilate_field_px: int = 35,
) -> FieldDetectionResult:
    """Detección por líneas blancas — más robusta cuando el contorno verde se fragmenta.

    Estrategia:
    1. Máscara verde del campo (para limitar la región de interés).
    2. Máscara blanca dentro de la máscara verde dilatada.
    3. Canny + HoughLinesP sobre la máscara blanca.
    4. Clustering de líneas en 2 grupos por orientación (k-means 1D sobre ángulos).
    5. Por grupo, escoger las 2 líneas más largas → 4 líneas del campo.
    6. Calcular las 4 intersecciones de líneas cruzadas.
    7. Ordenar TL, TR, BR, BL.

    Devuelve `success=False` si no logra reunir 4 esquinas plausibles.
    """
    h_img, w_img = image_bgr.shape[:2]
    frame_area = h_img * w_img
    field_mask = segment_field_mask(image_bgr)

    contours, _ = cv2.findContours(
        field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return FieldDetectionResult(False, np.empty((0, 2)), field_mask, 0.0)
    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / frame_area
    if area_ratio < min_area_ratio:
        return FieldDetectionResult(False, np.empty((0, 2)), field_mask, area_ratio)

    # Máscara blanca limitada al ROI del campo (verde dilatado).
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_field_px, dilate_field_px)
    )
    field_roi = cv2.dilate(field_mask, kernel)
    white = cv2.bitwise_and(white, field_roi)

    edges = cv2.Canny(white, canny_low, canny_high, apertureSize=3)
    min_line_length = int(min_line_length_frac * max(h_img, w_img))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None or len(lines) < 4:
        return FieldDetectionResult(False, np.empty((0, 2)), field_mask, area_ratio)

    segs = lines.reshape(-1, 4)
    angles = np.array(
        [_line_angle_deg(*s) for s in segs], dtype=np.float64
    )  # (-90, 90]
    lengths = np.array([_line_length(*s) for s in segs], dtype=np.float64)

    # Clustering por seno y coseno del 2*theta para evitar wrap-around.
    theta_rad = np.radians(angles)
    feats = np.stack([np.cos(2 * theta_rad), np.sin(2 * theta_rad)], axis=1).astype(
        np.float32
    )
    _, labels, _ = cv2.kmeans(
        feats,
        2,
        None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()

    # Para cada grupo, escoger las 2 líneas más largas que sean SEPARADAS
    # (no del mismo borde) — separación medida por distancia normal entre rectas.
    picked_per_group = []
    for g in (0, 1):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            return FieldDetectionResult(False, np.empty((0, 2)), field_mask, area_ratio)
        # Ordenar por largo descendente, escoger las top que cubran rangos
        # ortogonales (distancia normal entre rectas > 5% del lado relevante).
        order = idx[np.argsort(-lengths[idx])]
        picks = [order[0]]
        ref_seg = segs[order[0]]
        ref_norm_axis = np.array(
            [ref_seg[1] - ref_seg[3], ref_seg[2] - ref_seg[0]], dtype=np.float64
        )
        norm_len = np.linalg.norm(ref_norm_axis)
        if norm_len < 1e-9:
            return FieldDetectionResult(False, np.empty((0, 2)), field_mask, area_ratio)
        ref_norm_axis /= norm_len
        ref_point = np.array([ref_seg[0], ref_seg[1]], dtype=np.float64)
        min_sep_px = 0.05 * max(h_img, w_img)
        for j in order[1:]:
            seg = segs[j]
            p = np.array([seg[0], seg[1]], dtype=np.float64)
            d = abs(np.dot(p - ref_point, ref_norm_axis))
            if d > min_sep_px:
                picks.append(j)
                break
        if len(picks) < 2:
            return FieldDetectionResult(False, np.empty((0, 2)), field_mask, area_ratio)
        picked_per_group.append(picks)

    # 4 líneas en homogéneas
    L = []
    for picks in picked_per_group:
        for j in picks:
            L.append(_line_homog(*segs[j]))

    # 4 intersecciones: grupo0[0]xgrupo1[0], grupo0[0]xgrupo1[1],
    # grupo0[1]xgrupo1[0], grupo0[1]xgrupo1[1]
    inters = []
    for i in (0, 1):
        for k in (2, 3):
            pt = _intersect(L[i], L[k])
            if pt is None:
                return FieldDetectionResult(
                    False, np.empty((0, 2)), field_mask, area_ratio
                )
            inters.append(pt)
    corners_arr = np.stack(inters)

    # Sanity: deben estar todas dentro de un margen razonable del frame.
    margin = 0.5 * max(h_img, w_img)
    if (
        (corners_arr[:, 0] < -margin).any()
        or (corners_arr[:, 0] > w_img + margin).any()
        or (corners_arr[:, 1] < -margin).any()
        or (corners_arr[:, 1] > h_img + margin).any()
    ):
        return FieldDetectionResult(False, np.empty((0, 2)), field_mask, area_ratio)

    ordered = _order_corners_tl_tr_br_bl(corners_arr)
    return FieldDetectionResult(True, ordered, field_mask, area_ratio)


def draw_corners(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Dibuja las 4 esquinas + el cuadrilátero conectándolas."""
    out = image_bgr.copy()
    if corners.size == 0:
        return out
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip(corners, labels):
        cv2.circle(out, (int(x), int(y)), 12, (0, 0, 255), -1)
        cv2.putText(
            out,
            label,
            (int(x) + 16, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out
