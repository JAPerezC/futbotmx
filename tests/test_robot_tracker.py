"""Tests del wrapper OC-SORT (RobotTracker)."""

from __future__ import annotations

import numpy as np
import pytest


def test_can_instantiate():
    from src.tracking.robots import RobotTracker

    t = RobotTracker()
    assert t is not None


def test_update_empty_detections_returns_empty():
    from src.tracking.robots import RobotTracker

    t = RobotTracker(min_hits=1)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = t.update(np.empty((0, 6), dtype=np.float64), img)
    assert out == []


def test_assigns_track_id_over_frames():
    """Una caja que se mueve consistentemente debe mantener track_id."""
    from src.tracking.robots import RobotTracker

    t = RobotTracker(min_hits=1)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    track_ids = []
    for i in range(5):
        dx = i * 5
        dets = np.array([[100 + dx, 100, 200 + dx, 200, 0.9, 0]], dtype=np.float64)
        tracks = t.update(dets, img)
        if tracks:
            track_ids.append(tracks[0].track_id)
    # Después de min_hits frames debe haber al menos 1 track
    assert len(track_ids) >= 1
    # Y todos los track_ids deben ser iguales (mismo robot)
    assert len(set(track_ids)) == 1


def test_handles_4col_detections():
    """Si paso solo xyxy sin conf/class, debe completarse."""
    from src.tracking.robots import RobotTracker

    t = RobotTracker(min_hits=1)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets4 = np.array([[100, 100, 200, 200]], dtype=np.float64)
    out = t.update(dets4, img)
    assert isinstance(out, list)


def test_track_centroid_matches_bbox():
    from src.tracking.robots import RobotTracker

    t = RobotTracker(min_hits=1)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Inicialmente sin tracks (min_hits=1 → activa al primer frame)
    for _ in range(2):
        dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float64)
        tracks = t.update(dets, img)
    if tracks:
        tr = tracks[0]
        assert tr.centroid_img.tolist() == pytest.approx([150, 150], abs=2)
