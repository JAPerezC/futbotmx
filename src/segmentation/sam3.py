"""Wrapper de SAM 3.1 (Meta, mar 2026) para segmentación por prompts.

API verificada el 2026-05-16 contra `transformers 5.8.1` y `facebook/sam3`
(HuggingFace gated, descarga ~5 GB). Inferencia en RTX 5080 16 GB:
~0.5-1.5 s por prompt sobre frame 1080p (primera inferencia más lenta).

Hallazgo importante:
- Prompts "robot with purple flag" vs "robot with white flag" NO
  discriminan por color de bandera. SAM 3 entiende "robot" pero los
  modificadores son débiles. Para identificar equipo, usar la cascada:
    SAM 3 → src.tracking.reid.classify_team (HSV)

Ver docs/literature-review.md § 1-2 y docs/architecture.md § 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# IMPORTANTE: enable_system_ssl() debe llamarse ANTES de from_pretrained
# si el entorno tiene SSL inspection (Norton, Kaspersky, etc.).
from src.utils.network import enable_system_ssl


@dataclass(frozen=True)
class SegMask:
    """Máscara binaria de un objeto detectado."""

    mask: np.ndarray  # bool, mismo (H, W) que el frame original
    label: str
    score: float


def load_model(
    checkpoint: str = "facebook/sam3",
    device: str | None = None,
    half_precision: bool = True,
    lora_ckpt: str | None = None,
) -> tuple["object", "object"]:
    """Carga procesador y modelo SAM 3.1 desde HuggingFace.

    Args:
        checkpoint: model id en HF (default ``facebook/sam3``).
        device: ``cuda`` / ``cpu`` / None (autodetecta).
        half_precision: usar fp16 en GPU (default True). ~40% speedup y
            ~50% menos VRAM con calidad equivalente para inferencia.
        lora_ckpt: ruta opcional a adaptador LoRA fine-tuned. Si se pasa,
            monta el adaptador sobre el modelo base con
            ``PeftModel.from_pretrained``. Permite usar pesos fine-tuned
            con § 3.7.3 sin re-descargar el modelo base completo.

    Returns:
        (processor, model). El model está en eval() y movido al device.
    """
    enable_system_ssl()
    from transformers import Sam3Model, Sam3Processor  # import perezoso

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(checkpoint)
    dtype = torch.float16 if (half_precision and device == "cuda") else torch.float32
    model = Sam3Model.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    if lora_ckpt:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_ckpt).to(device).eval()
    return processor, model


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convierte BGR (OpenCV) a PIL RGB."""
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def segment_with_text(
    image_bgr: np.ndarray,
    prompts: Sequence[str],
    processor,
    model,
    threshold: float = 0.2,
    mask_threshold: float = 0.5,
) -> dict[str, list[SegMask]]:
    """Segmenta un frame BGR con una secuencia de prompts de texto.

    Args:
        image_bgr: frame (H, W, 3) en BGR.
        prompts: lista de strings descriptivos.
        processor, model: salida de ``load_model``.
        threshold: confianza mínima para retener instancias.
        mask_threshold: umbral binarización máscara (0-1).

    Returns:
        Diccionario {prompt: [SegMask, ...]}. Lista vacía si SAM 3 no
        detectó nada para ese prompt.
    """
    h, w = image_bgr.shape[:2]
    pil = _to_pil(image_bgr)
    device = next(model.parameters()).device

    out: dict[str, list[SegMask]] = {}
    model_dtype = next(model.parameters()).dtype
    for prompt in prompts:
        inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
        # Si el modelo está en fp16, los pixel_values deben coincidir
        if "pixel_values" in inputs and model_dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(h, w)],
        )
        r0 = results[0]
        masks_raw = r0.get("masks")
        scores_raw = r0.get("scores")
        if masks_raw is None or len(masks_raw) == 0:
            out[prompt] = []
            continue
        masks_np = (
            masks_raw.cpu().numpy()
            if hasattr(masks_raw, "cpu")
            else np.asarray(masks_raw)
        )
        scores_np = (
            scores_raw.cpu().numpy()
            if hasattr(scores_raw, "cpu")
            else np.asarray(scores_raw)
        )
        out[prompt] = [
            SegMask(mask=m.astype(bool), label=prompt, score=float(s))
            for m, s in zip(masks_np, scores_np)
        ]
    return out


def masks_to_bboxes(masks: list[SegMask]) -> np.ndarray:
    """Convierte una lista de máscaras a bboxes xyxy.

    Returns:
        Array (N, 4) con columnas x1, y1, x2, y2. Vacío si no hay máscaras.
    """
    if not masks:
        return np.empty((0, 4), dtype=np.float64)
    boxes = []
    for sm in masks:
        ys, xs = np.where(sm.mask)
        if xs.size == 0:
            boxes.append([0, 0, 0, 0])
            continue
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return np.asarray(boxes, dtype=np.float64)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Centroide (cx, cy) de una máscara binaria. None si vacía."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())
