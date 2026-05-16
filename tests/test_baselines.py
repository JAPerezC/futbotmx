"""Tests del detector HSV del balón naranja.

Estrategia: generar imágenes sintéticas con un círculo naranja conocido
sobre fondo verde y verificar que el detector encuentra el centro con
error subpíxel.
"""

from __future__ import annotations

import numpy as np
import cv2

from src.segmentation.baselines import (
    detect_orange_ball_mask,
    draw_detection,
    find_ball_centroid,
    ORANGE_HSV_LOWER,
    ORANGE_HSV_UPPER,
)


def _make_synthetic_frame(
    cx: int = 960,
    cy: int = 540,
    radius: int = 12,
    ball_bgr: tuple[int, int, int] = (40, 130, 255),  # naranja brillante en BGR
    field_bgr: tuple[int, int, int] = (60, 140, 50),  # verde fieltro en BGR
    size: tuple[int, int] = (1080, 1920),
) -> np.ndarray:
    """Frame sintético: fondo verde + círculo naranja en (cx, cy)."""
    img = np.full((size[0], size[1], 3), field_bgr, dtype=np.uint8)
    cv2.circle(img, (cx, cy), radius, ball_bgr, thickness=-1)
    return img


def test_mask_detects_orange_pixels():
    img = _make_synthetic_frame()
    mask = detect_orange_ball_mask(img)
    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]
    assert mask.sum() > 0, "la máscara debe activar al menos algunos píxeles"


def test_mask_ignores_pure_green_field():
    """Un frame solo verde no debe activar la máscara."""
    img = np.full((720, 1280, 3), (60, 140, 50), dtype=np.uint8)
    mask = detect_orange_ball_mask(img)
    assert mask.sum() == 0


def test_find_ball_returns_correct_center():
    """El centroide detectado debe estar dentro de 2 px del centro real."""
    cx, cy = 800, 450
    img = _make_synthetic_frame(cx=cx, cy=cy, radius=15)
    det = find_ball_centroid(img)
    assert det.found
    assert abs(det.cx - cx) < 2.0
    assert abs(det.cy - cy) < 2.0
    assert det.area > 100  # radio 15 → área ~707


def test_find_ball_handles_no_ball():
    """Si no hay balón, debe retornar found=False sin crashear."""
    img = np.full((720, 1280, 3), (60, 140, 50), dtype=np.uint8)
    det = find_ball_centroid(img)
    assert not det.found


def test_find_ball_rejects_tiny_blob():
    """Blobs menores al área mínima deben rechazarse."""
    img = _make_synthetic_frame(cx=400, cy=400, radius=2)  # área ~12
    det = find_ball_centroid(img, min_area=20)
    assert not det.found


def test_find_ball_rejects_huge_blob():
    """Blobs mayores al área máxima (probablemente camiseta) se rechazan."""
    img = _make_synthetic_frame(cx=960, cy=540, radius=80)  # área ~20000
    det = find_ball_centroid(img, max_area=4000)
    assert not det.found


def test_find_ball_picks_most_circular():
    """Entre un blob circular y uno alargado, debe elegir el circular."""
    img = np.full((720, 1280, 3), (60, 140, 50), dtype=np.uint8)
    # blob 1: rectangular naranja (no es balón)
    cv2.rectangle(img, (100, 100), (200, 130), (40, 130, 255), -1)
    # blob 2: círculo naranja (es balón)
    cv2.circle(img, (700, 400), 12, (40, 130, 255), -1)
    det = find_ball_centroid(img)
    assert det.found
    # debe elegir el círculo (centro ~700, 400) en vez del rectángulo
    assert abs(det.cx - 700) < 5


def test_draw_detection_does_not_mutate_input():
    img = _make_synthetic_frame()
    det = find_ball_centroid(img)
    assert det.found
    snapshot = img.copy()
    _ = draw_detection(img, det)
    np.testing.assert_array_equal(img, snapshot)


def test_hsv_range_sanity():
    """Los rangos HSV deben tener forma y orden correctos."""
    assert ORANGE_HSV_LOWER.shape == (3,)
    assert ORANGE_HSV_UPPER.shape == (3,)
    assert (ORANGE_HSV_LOWER <= ORANGE_HSV_UPPER).all()
