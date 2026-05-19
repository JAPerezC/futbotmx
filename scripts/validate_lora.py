"""Compara mIoU del modelo SAM 3.1 BASE vs FINE-TUNED LoRA.

Usa el mismo val split (por video, no por frame) para evitar leakage.
Reporta IoU promedio y por categoría (robot / ball). Salida JSON
para auditoría posterior.

Uso:
    python scripts/validate_lora.py --ckpt data/processed/lora_checkpoints/lora_final
    python scripts/validate_lora.py --ckpt ... --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.network import enable_system_ssl

enable_system_ssl()
from src.utils.seed import set_global_seed

set_global_seed(42)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.training.dataset import (
    SAM3PseudoDataset,
    collate_keep_lists,
    split_train_val,
)

# Reutilizo helpers privados del training script
from scripts.train_sam3_lora import _prepare_inputs, _select_best_query_mask


DATASET_DIR = ROOT / "data" / "processed" / "pseudo_dataset"


@torch.no_grad()
def eval_dataset(model, processor, loader, device, label: str):
    """mIoU + breakdown por categoría sobre el loader."""
    model.eval()
    ious_by_cat: dict[str, list[float]] = {}
    n = 0
    t0 = time.time()
    model_dtype = next(model.parameters()).dtype
    for batch in loader:
        for img, box, gt, cat in zip(
            batch["images"],
            batch["bbox_prompts"],
            batch["masks_gt"],
            batch["categories"],
        ):
            inputs = _prepare_inputs(processor, img, box, device, model_dtype)
            outputs = model(**inputs)
            H_img, W_img = img.shape[:2]
            pred_logit = _select_best_query_mask(outputs, box, (H_img, W_img))
            pred = (torch.sigmoid(pred_logit.float()) > 0.5).cpu().numpy()
            H_p, W_p = pred.shape
            gt_resized = cv2.resize(gt, (W_p, H_p), interpolation=cv2.INTER_NEAREST) > 0
            inter = np.logical_and(pred, gt_resized).sum()
            union = np.logical_or(pred, gt_resized).sum()
            iou = inter / max(1, union)
            ious_by_cat.setdefault(cat, []).append(float(iou))
            n += 1
    elapsed = time.time() - t0
    means = {c: float(np.mean(v)) for c, v in ious_by_cat.items()}
    overall = float(np.mean([i for v in ious_by_cat.values() for i in v]))
    print(
        f"[{label}] n={n} miou={overall:.3f} "
        + " ".join(f"{c}={m:.3f}" for c, m in means.items())
        + f"  ({elapsed:.1f}s)"
    )
    return {
        "label": label,
        "n_samples": n,
        "miou": overall,
        "miou_by_category": means,
        "elapsed_s": elapsed,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="ruta al checkpoint LoRA")
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--min-score", type=float, default=0.6)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, ckpt: {args.ckpt}")

    print("Cargando dataset...")
    ds = SAM3PseudoDataset(DATASET_DIR, min_score=args.min_score)
    _, val_idx = split_train_val(ds, val_fraction=0.15, seed=42)
    val_idx = val_idx[: args.max_samples]
    print(f"  val samples: {len(val_idx)}")
    loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_keep_lists,
        num_workers=0,
    )

    print("\n[1/2] Evaluando BASE...")
    from src.segmentation.sam3 import load_model

    proc, base_model = load_model(half_precision=True)
    base_metrics = eval_dataset(base_model, proc, loader, device, "BASE")
    del base_model
    torch.cuda.empty_cache()

    print("\n[2/2] Evaluando LoRA fine-tuned...")
    _, lora_model = load_model(half_precision=True, lora_ckpt=args.ckpt)
    lora_metrics = eval_dataset(lora_model, proc, loader, device, "LORA")

    report = {
        "ckpt": str(args.ckpt),
        "n_val_samples": len(val_idx),
        "min_score_filter": args.min_score,
        "base": base_metrics,
        "lora": lora_metrics,
        "improvement_miou": lora_metrics["miou"] - base_metrics["miou"],
        "improvement_pct": (
            100
            * (lora_metrics["miou"] - base_metrics["miou"])
            / max(0.001, base_metrics["miou"])
        ),
    }
    print(f"\n{'=' * 60}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = args.out or (
        Path(args.ckpt).parent / f"validation_{Path(args.ckpt).name}.json"
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
