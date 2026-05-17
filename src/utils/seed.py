"""Fija seeds para reproducibilidad (convocatoria § 3.2.2).

Llamar `set_global_seed(42)` al inicio de cualquier script que produzca
salidas finales. No afecta velocidad significativamente.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fija RNGs de Python, NumPy, PyTorch (CPU+CUDA si está disponible)
    y OpenCV. Idempotente.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except Exception:
        pass
