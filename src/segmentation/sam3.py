"""Wrapper de SAM 3.1 (Meta, mar 2026) para segmentación por prompts.

Pendiente de implementación — bloqueado por:
1. Descarga de pesos (gated en huggingface.co/facebook/sam3, ya aceptados).
2. Validación de la API de transformers.Sam3Processor / Sam3Model.

La firma de las funciones públicas refleja el diseño final del pipeline.
Ver docs/architecture.md § 5.2 y docs/literature-review.md § 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SegMask:
    """Máscara binaria de un objeto con metadatos."""

    mask: np.ndarray  # uint8, mismo tamaño que el frame
    label: str
    score: float


def load_model(
    checkpoint: str = "facebook/sam3",
    device: str = "cuda",
):
    """Carga SAM 3.1 desde HuggingFace.

    Returns:
        Tupla (processor, model) lista para inferencia.
    """
    raise NotImplementedError(
        "Implementar en Fase 1 (25 may). Requiere `transformers >= 4.45` y "
        "pesos descargados de huggingface.co/facebook/sam3."
    )


def segment_with_text(
    image_bgr: np.ndarray,
    prompts: Sequence[str],
    processor,
    model,
    threshold: float = 0.5,
) -> list[SegMask]:
    """Segmenta una imagen con una lista de prompts de texto.

    Args:
        image_bgr: frame en BGR.
        prompts: lista de strings (ver src.segmentation.prompts).
        processor, model: salida de `load_model`.
        threshold: umbral de confianza para retener máscaras.

    Returns:
        Lista de SegMask, una por prompt que superó el umbral.
    """
    raise NotImplementedError("Implementar en Fase 1")


def segment_with_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    processor,
    model,
) -> list[SegMask]:
    """Segmenta usando bounding boxes como prompts (re-prompting adaptativo).

    Útil cuando el tracker tiene una caja confiable y se quiere refinar
    la máscara para el frame actual.
    """
    raise NotImplementedError("Implementar en Fase 1")


def save_checkpoint(model, out_dir: Path) -> Path:
    """Guarda los pesos (incluyendo adapters LoRA si los hay)."""
    raise NotImplementedError("Implementar en Fase 2 (fine-tuning)")
