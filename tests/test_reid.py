"""Tests del clasificador de equipo (legacy + adaptativo)."""

from __future__ import annotations

import numpy as np
import cv2

from src.tracking.reid import (
    AdaptiveTeamClassifier,
    TeamScore,
    _dominant_hue,
    _hue_distance,
    classify_team,
)


def _bbox_image(team_color_bgr: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((400, 400, 3), (60, 140, 50), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 250), team_color_bgr, -1)
    bbox = np.array([100, 100, 200, 250], dtype=np.float64)
    return img, bbox


# -------- API legacy (classify_team) --------


def test_team_a_purple_classified_correctly():
    img, bbox = _bbox_image((200, 30, 130))  # morado BGR
    score = classify_team(img, bbox)
    assert score.label == "A"


def test_team_b_green_lime_classified_correctly():
    """Verde-lima sí entra como B en la nueva semántica."""
    img, bbox = _bbox_image((80, 240, 60))  # verde-lima BGR
    score = classify_team(img, bbox)
    assert score.label == "B"


def test_legacy_white_not_classified():
    """Blanco puro ya no se clasifica (S baja → descartado).

    Esto es intencional: blanco/gris no es discriminativo en torneos
    donde puede aparecer en líneas, balones blancos, parches de luz.
    Para clasificación con blanco usar AdaptiveTeamClassifier.
    """
    img, bbox = _bbox_image((250, 250, 250))
    score = classify_team(img, bbox)
    assert score.label is None


def test_empty_bbox_returns_none():
    img = np.full((100, 100, 3), 0, dtype=np.uint8)
    bbox = np.array([50, 50, 50, 50], dtype=np.float64)
    score = classify_team(img, bbox)
    assert score.label is None


def test_score_dataclass_immutable():
    s = TeamScore(label="A", score_a=0.5, score_b=0.1)
    assert s.label == "A"


# -------- helpers de bajo nivel --------


def test_dominant_hue_finds_purple():
    img, bbox = _bbox_image((200, 30, 130))
    hue = _dominant_hue(img, bbox)
    assert hue is not None
    # morado en OpenCV HSV: H ~ 145-150
    assert 130 <= hue <= 165


def test_dominant_hue_handles_dark():
    img = np.full((200, 200, 3), 5, dtype=np.uint8)
    bbox = np.array([0, 0, 200, 200], dtype=np.float64)
    assert _dominant_hue(img, bbox) is None


def test_hue_distance_circular():
    assert _hue_distance(10, 170) == 20  # se cierra el círculo
    assert _hue_distance(50, 60) == 10
    assert _hue_distance(0, 0) == 0


# -------- AdaptiveTeamClassifier --------


def test_adaptive_assigns_after_warmup():
    clf = AdaptiveTeamClassifier(warmup_frames=3, hue_separation_min=20)
    # Track 1: morado consistente (H ~ 145)
    # Track 2: verde-lima consistente (H ~ 60)
    for _ in range(5):
        clf.observe(1, 145)
        clf.observe(2, 60)
        clf.end_frame()
    a1 = clf.assign(1)
    a2 = clf.assign(2)
    assert {a1, a2} == {"A", "B"}, f"got {a1=} {a2=}"


def test_adaptive_no_assignment_before_warmup():
    clf = AdaptiveTeamClassifier(warmup_frames=10)
    for _ in range(3):
        clf.observe(1, 145)
        clf.end_frame()
    # antes del warmup no asigna
    assert clf.assign(1) is None


def test_adaptive_single_color_falls_to_one_team():
    """Si solo se ve un cluster, todos quedan asignados al mismo."""
    clf = AdaptiveTeamClassifier(warmup_frames=3, hue_separation_min=20)
    for _ in range(5):
        clf.observe(1, 145)
        clf.observe(2, 148)  # casi igual al 1
        clf.end_frame()
    a1 = clf.assign(1)
    a2 = clf.assign(2)
    assert a1 == a2 == "A"


def test_adaptive_ignores_none_observations():
    clf = AdaptiveTeamClassifier(warmup_frames=3)
    for _ in range(5):
        clf.observe(1, None)  # no debería romper
        clf.observe(2, 60)
        clf.end_frame()
    # 2 fue observado → asignado; 1 nunca observado → None
    assert clf.assign(2) is not None
    assert clf.assign(1) is None
