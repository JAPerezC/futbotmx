"""Tests de homografía y proyección.

Estrategia: usar puntos sintéticos con homografía conocida (warp inverso
de las 4 esquinas mundo) y verificar que la composición forward+inverse
recupere los puntos originales dentro de tolerancia subpíxel.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.calib import (
    FIELD_CORNERS_WORLD_MM,
    FIELD_LENGTH_MM,
    FIELD_WIDTH_MM,
    compute_homography,
    inverse_project,
    project_points,
    warp_to_topdown,
)


@pytest.fixture
def synthetic_image_corners() -> np.ndarray:
    """Esquinas en una imagen ficticia 1920x1080 con perspectiva oblicua.

    Simula la vista del iPhone desde la esquina inferior-izquierda:
    el campo aparece como trapecio.
    """
    return np.array(
        [
            [300, 200],  # top-left (lejos, alto)
            [1700, 220],  # top-right
            [1850, 950],  # bottom-right (cerca, abajo)
            [80, 900],  # bottom-left
        ],
        dtype=np.float64,
    )


def test_compute_homography_returns_3x3(synthetic_image_corners):
    H = compute_homography(synthetic_image_corners)
    assert H.shape == (3, 3)
    assert np.isfinite(H).all()


def test_compute_homography_rejects_bad_shape():
    with pytest.raises(ValueError):
        compute_homography(np.zeros((3, 2)))


def test_compute_homography_rejects_nan():
    bad = np.array([[0, 0], [1, 0], [1, 1], [np.nan, 0]], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_homography(bad)


def test_corners_map_to_world(synthetic_image_corners):
    """Las 4 esquinas imagen deben mapear EXACTAMENTE a las 4 esquinas mundo."""
    H = compute_homography(synthetic_image_corners)
    projected = project_points(synthetic_image_corners, H)
    np.testing.assert_allclose(projected, FIELD_CORNERS_WORLD_MM, atol=1e-6)


def test_roundtrip_world_to_image_to_world(synthetic_image_corners):
    """Proyectar mundo -> imagen -> mundo debe recuperar el punto original."""
    H = compute_homography(synthetic_image_corners)
    world_pts = np.array(
        [
            [FIELD_LENGTH_MM / 2, FIELD_WIDTH_MM / 2],  # centro
            [600, 790],
            [1500, 300],
        ],
        dtype=np.float64,
    )
    img_pts = inverse_project(world_pts, H)
    world_again = project_points(img_pts, H)
    np.testing.assert_allclose(world_again, world_pts, atol=1e-3)


def test_warp_to_topdown_shape(synthetic_image_corners):
    """warp_to_topdown debe producir una imagen del tamaño esperado."""
    H = compute_homography(synthetic_image_corners)
    fake_img = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    scale = 0.3
    top = warp_to_topdown(fake_img, H, scale=scale)
    expected_w = int(round(FIELD_LENGTH_MM * scale))
    expected_h = int(round(FIELD_WIDTH_MM * scale))
    assert top.shape == (expected_h, expected_w, 3)


def test_field_dimensions_match_rules():
    """Las constantes deben respetar el reglamento § 7.1 (219x158 cm)."""
    assert FIELD_LENGTH_MM == 2190
    assert FIELD_WIDTH_MM == 1580
