"""Tests de utilidades de IO de video."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import VideoMeta, probe


SAMPLE_VIDEO = Path(__file__).resolve().parents[1] / "data" / "raw" / "IMG_9915.MOV"


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="sample video no disponible")
def test_probe_returns_meta():
    meta = probe(SAMPLE_VIDEO)
    assert isinstance(meta, VideoMeta)
    assert meta.width == 1920
    assert meta.height == 1080
    assert 29.5 < meta.fps < 30.5  # ~29.97
    assert meta.n_frames > 2000


def test_probe_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        probe(Path("inexistente.mp4"))
