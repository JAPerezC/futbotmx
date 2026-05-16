"""Tests del clasificador HSV de bandera por equipo."""

from __future__ import annotations

import numpy as np
import cv2

from src.tracking.reid import TeamScore, classify_team


def _bbox_image(team_color_bgr: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Crea un frame con un parche de color en una bbox conocida."""
    img = np.full((400, 400, 3), (60, 140, 50), dtype=np.uint8)  # campo verde
    cv2.rectangle(img, (100, 100), (200, 250), team_color_bgr, -1)
    bbox = np.array([100, 100, 200, 250], dtype=np.float64)
    return img, bbox


def test_team_a_purple_classified_correctly():
    img, bbox = _bbox_image((200, 30, 130))  # morado en BGR
    score = classify_team(img, bbox)
    assert score.label == "A"
    assert score.score_a > score.score_b


def test_team_b_white_classified_correctly():
    img, bbox = _bbox_image((250, 250, 250))  # blanco
    score = classify_team(img, bbox)
    assert score.label == "B"


def test_empty_bbox_returns_none():
    img = np.full((100, 100, 3), 0, dtype=np.uint8)
    bbox = np.array([50, 50, 50, 50], dtype=np.float64)
    score = classify_team(img, bbox)
    assert score.label is None


def test_ambiguous_returns_none():
    """Si A y B están en proporciones similares en la mitad superior, no decide."""
    img = np.full((400, 400, 3), (60, 140, 50), dtype=np.uint8)
    # bbox: (100,100)-(200,250). La mitad superior va de y=100 a y=175.
    # Pongo morado (A) y blanco (B) lado a lado, ambos en la mitad superior.
    cv2.rectangle(img, (100, 100), (150, 175), (200, 30, 130), -1)  # A
    cv2.rectangle(img, (150, 100), (200, 175), (250, 250, 250), -1)  # B
    bbox = np.array([100, 100, 200, 250], dtype=np.float64)
    score = classify_team(img, bbox)
    assert score.score_a > 0
    assert score.score_b > 0
    # ninguno debería ganar 1.3x el otro
    assert score.label is None


def test_score_dataclass_immutable():
    s = TeamScore(label="A", score_a=0.5, score_b=0.1)
    assert s.label == "A"
