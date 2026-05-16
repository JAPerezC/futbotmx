"""Lectura y escritura de video con OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def probe(path: Path) -> VideoMeta:
    """Devuelve metadatos del video sin leer todos los frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir {path}")
    try:
        return VideoMeta(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(path: Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Generador de (frame_idx, frame_bgr).

    Args:
        path: ruta al video.
        stride: leer 1 de cada N frames (1 = todos, 3 = uno de cada 3).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Wrapper simple sobre cv2.VideoWriter con codec mp4v."""

    def __init__(
        self, path: Path, fps: float, width: int, height: int, fourcc: str = "mp4v"
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fcc = cv2.VideoWriter_fourcc(*fourcc)
        self._w = cv2.VideoWriter(str(self.path), fcc, fps, (width, height))
        if not self._w.isOpened():
            raise RuntimeError(f"No se pudo abrir el writer en {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._w.write(frame_bgr)

    def close(self) -> None:
        self._w.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
