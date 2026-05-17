"""Tests del detector de esquinas del campo."""

from __future__ import annotations

import numpy as np
import cv2

from src.utils.field_detect import (
    detect_field_corners,
    draw_corners,
    segment_field_mask,
)


def _make_synthetic_field(
    corners_img: np.ndarray | None = None,
    size: tuple[int, int] = (1080, 1920),
    field_bgr: tuple[int, int, int] = (60, 140, 50),
    bg_bgr: tuple[int, int, int] = (40, 40, 40),
) -> np.ndarray:
    """Frame sintético: fondo oscuro + cuadrilátero verde."""
    img = np.full((size[0], size[1], 3), bg_bgr, dtype=np.uint8)
    if corners_img is None:
        # Trapecio centrado simulando vista oblicua
        corners_img = np.array(
            [[400, 200], [1500, 200], [1750, 900], [200, 900]],
            dtype=np.int32,
        )
    pts = corners_img.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(img, [pts], field_bgr)
    return img


def test_mask_segments_green():
    img = _make_synthetic_field()
    mask = segment_field_mask(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_mask_ignores_dark_background():
    img = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
    mask = segment_field_mask(img)
    assert mask.sum() == 0


def test_detect_finds_4_corners():
    gt = np.array([[400, 200], [1500, 200], [1750, 900], [200, 900]])
    img = _make_synthetic_field(gt)
    res = detect_field_corners(img)
    assert res.success
    assert res.corners.shape == (4, 2)
    # Cada esquina detectada debe estar a < 30 px de su contraparte (tolerancia
    # por la aproximación poligonal + morfología).
    # Ordeno tanto gt como detected con la misma rutina.
    from src.utils.field_detect import _order_corners_tl_tr_br_bl

    gt_ordered = _order_corners_tl_tr_br_bl(gt)
    for det_pt, gt_pt in zip(res.corners, gt_ordered):
        assert np.linalg.norm(det_pt - gt_pt) < 30, f"det={det_pt}, gt={gt_pt}"


def test_detect_fails_when_no_green():
    img = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
    res = detect_field_corners(img)
    assert not res.success
    assert res.corners.shape == (0, 2)
    assert res.contour_area_ratio == 0.0


def test_detect_rejects_tiny_blob():
    img = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 50), (60, 140, 50), -1)
    res = detect_field_corners(img, min_area_ratio=0.15)
    assert not res.success


def test_draw_corners_returns_copy():
    img = _make_synthetic_field()
    corners = np.array(
        [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float64
    )
    out = draw_corners(img, corners)
    assert out.shape == img.shape
    # original no fue mutado
    assert (out != img).any()


def test_draw_corners_with_empty_returns_copy():
    img = np.full((100, 100, 3), 0, dtype=np.uint8)
    out = draw_corners(img, np.empty((0, 2)))
    np.testing.assert_array_equal(out, img)
