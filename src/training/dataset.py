"""Dataset PyTorch para fine-tuning SAM 3.1 sobre máscaras pseudo-supervisadas.

Lee del directorio generado por `scripts/generate_pseudo_annotations.py`:

    data/processed/pseudo_dataset/
        images/<video>_<frame>.jpg
        masks/<video>_<frame>_<category>_<idx>.png
        metadata.jsonl

Cada entrada del jsonl es una máscara pseudo-anotada con su bbox y categoría.
El Dataset retorna tuplas (image_bgr, bbox_prompt, mask_gt) listas para
pasar al Sam3Processor con `input_boxes=[[bbox]]` como prompt.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PseudoSample:
    image_path: Path
    mask_path: Path
    category: str
    prompt: str
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    source_video: str
    source_frame: int


class SAM3PseudoDataset(Dataset):
    """Dataset de máscaras pseudo-anotadas para fine-tuning SAM 3.

    Cada elemento es un dict compatible con `Sam3Processor`:
        image:        np.ndarray HxWx3 BGR
        mask_gt:      np.ndarray HxW {0, 1}
        bbox_prompt:  list[int] [x1, y1, x2, y2]  (prompt para el modelo)
        category:     str       ('robot' | 'ball')
        prompt_text:  str       ('soccer robot' | 'ball')
    """

    def __init__(
        self,
        root: Path,
        category_filter: str | None = None,
        min_score: float = 0.6,
        bbox_jitter_px: int = 5,
    ):
        """Args:
        root: directorio que contiene `metadata.jsonl`, `images/`, `masks/`.
        category_filter: si se pasa ('robot' | 'ball'), solo retorna esas.
        min_score: filtra entradas con score SAM base < min_score.
        bbox_jitter_px: ruido uniforme al bbox prompt (data augmentation).
        """
        self.root = Path(root)
        self.bbox_jitter_px = bbox_jitter_px
        meta_path = self.root / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No existe {meta_path}")
        self.samples: list[PseudoSample] = []
        with open(meta_path, encoding="utf-8") as fp:
            for line in fp:
                d = json.loads(line)
                if category_filter and d["category"] != category_filter:
                    continue
                if d["score"] < min_score:
                    continue
                self.samples.append(
                    PseudoSample(
                        image_path=self.root / d["image"],
                        mask_path=self.root / d["mask"],
                        category=d["category"],
                        prompt=d["prompt"],
                        bbox_xyxy=tuple(d["bbox_xyxy"]),
                        score=d["score"],
                        source_video=d["source_video"],
                        source_frame=d["source_frame"],
                    )
                )
        if not self.samples:
            raise RuntimeError(
                f"Dataset vacío tras filtrar (category={category_filter}, "
                f"min_score={min_score})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = cv2.imread(str(s.image_path))
        if image is None:
            raise RuntimeError(f"No se pudo leer {s.image_path}")
        mask = cv2.imread(str(s.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"No se pudo leer {s.mask_path}")
        mask_bin = (mask > 127).astype(np.uint8)
        x1, y1, x2, y2 = s.bbox_xyxy
        if self.bbox_jitter_px > 0:
            j = self.bbox_jitter_px
            x1 = max(0, x1 + random.randint(-j, j))
            y1 = max(0, y1 + random.randint(-j, j))
            x2 = min(image.shape[1], x2 + random.randint(-j, j))
            y2 = min(image.shape[0], y2 + random.randint(-j, j))
        return {
            "image": image,
            "mask_gt": mask_bin,
            "bbox_prompt": [int(x1), int(y1), int(x2), int(y2)],
            "category": s.category,
            "prompt_text": s.prompt,
            "source": f"{s.source_video}@{s.source_frame}",
        }


def collate_keep_lists(batch: list[dict]) -> dict:
    """Collate que NO apila tensores (SAM 3 procesa de a uno por batch).

    Mantiene listas por sample porque cada imagen puede tener distinto HxW.
    El training loop se encarga de iterar por el batch.
    """
    return {
        "images": [b["image"] for b in batch],
        "masks_gt": [b["mask_gt"] for b in batch],
        "bbox_prompts": [b["bbox_prompt"] for b in batch],
        "categories": [b["category"] for b in batch],
        "prompt_texts": [b["prompt_text"] for b in batch],
        "sources": [b["source"] for b in batch],
    }


def split_train_val(
    dataset: SAM3PseudoDataset, val_fraction: float = 0.15, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split por VIDEO (no por frame) para evitar leakage."""
    rng = random.Random(seed)
    videos = sorted({s.source_video for s in dataset.samples})
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_fraction))
    val_videos = set(videos[:n_val])
    train_idx = [
        i for i, s in enumerate(dataset.samples) if s.source_video not in val_videos
    ]
    val_idx = [i for i, s in enumerate(dataset.samples) if s.source_video in val_videos]
    return train_idx, val_idx
