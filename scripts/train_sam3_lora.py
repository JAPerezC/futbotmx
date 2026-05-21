"""Fine-tuning LoRA de SAM 3.1 sobre dataset pseudo-supervisado.

Implementa la innovación obligatoria § 3.7.3 de la convocatoria
(Categoría Profesional): adaptación del modelo base SAM 3 al dominio
específico de fútbol robótico vía adaptadores LoRA.

Configuración (basada en SAM3_LoRA Sompote — github.com/Sompote/SAM3_LoRA
y survey arXiv:2512.08467):
- LoRA rank=8 (12 GB VRAM) o 16 (16 GB, justo en RTX 5080)
- target_modules: q/k/v/o_proj de vision_encoder + mask_decoder
  (preservamos text_encoder y geometry_encoder)
- batch=2, gradient_accumulation_steps=8 (batch efectivo 16)
- lr=5e-5, warmup_steps=50, scheduler=cosine
- loss: BCE-with-logits + Dice (alpha=0.5)
- 50 epochs, eval cada 5 epochs

Salida: checkpoint LoRA-only en `data/processed/lora_checkpoints/`,
~30-50 MB en lugar de los ~5 GB del modelo base.

ESTADO: skeleton funcional + NOTAS. Requiere validar la API de Sam3Model
con bbox prompts (input_boxes) cuando el dataset esté listo. Si
processor.post_process devuelve logits crudos sin sigmoid, usar
BCEWithLogitsLoss; si ya están en (0,1), usar BCELoss + cuidar overflow.

Uso:
    python scripts/train_sam3_lora.py --epochs 50 --rank 8
    python scripts/train_sam3_lora.py --dry-run  # 1 epoch sobre 10 samples
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

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.training.dataset import (
    SAM3PseudoDataset,
    collate_keep_lists,
    split_train_val,
)


DATASET_DIR = ROOT / "data" / "processed" / "pseudo_dataset"
CKPT_DIR = ROOT / "data" / "processed" / "lora_checkpoints"


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Dice loss simétrica: 1 - 2|P∩T| / (|P|+|T|)."""
    pred = pred.flatten()
    target = target.flatten().float()
    inter = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def build_target_modules() -> list[str]:
    """Target modules para LoRA basado en `scripts/inspect_sam3_modules.py`.

    Vision encoder (32 capas con q/k/v/o_proj) + mask decoder (q/k/v/o_proj).
    NO incluye fc1/fc2 para mantener parámetros bajos (rank 8 sobre QKVO
    de 32 capas = ~512K params nuevos, ~0.1% del modelo).
    """
    targets = []
    for i in range(32):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            targets.append(f"vision_encoder.layers.{i}.attention.{proj}")
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        targets.append(f"mask_decoder.attention.{proj}")
    return targets


def apply_lora(model, rank: int, alpha: int, dropout: float):
    """Inyecta adaptadores LoRA en los target_modules.

    Devuelve el modelo envuelto en `PeftModel`. Los parámetros base
    quedan congelados; solo los LoRA son entrenables.
    """
    from peft import LoraConfig, get_peft_model

    targets = build_target_modules()
    # PEFT acepta substring matching: si los nombres exactos no aparecen,
    # caer a substrings comunes.
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=None,
    )
    peft_model = get_peft_model(model, config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    print(
        f"  LoRA aplicado: {n_trainable / 1e6:.2f}M entrenables de "
        f"{n_total / 1e6:.1f}M totales ({100 * n_trainable / n_total:.3f}%)"
    )
    return peft_model


def _prepare_inputs(processor, image_bgr, bbox_prompt, device, model_dtype):
    """Prepara inputs SAM 3 con cast COMPLETO de floats al dtype del modelo.

    Verificado contra `transformers 5.8.1`: si solo se castea pixel_values,
    falla con 'mat1 and mat2 must have the same dtype' al combinar boxes.
    """
    import cv2
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, input_boxes=[[bbox_prompt]], return_tensors="pt").to(
        device
    )
    if model_dtype == torch.float16:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                inputs[k] = v.to(torch.float16)
    return inputs


def _select_best_query_mask(outputs, bbox_prompt, image_hw):
    """Selecciona la query DETR mejor alineada con el bbox prompt.

    SAM 3 es DETR-style: pred_masks shape (1, 200, H', W'), pred_boxes
    shape (1, 200, 4) en cxcywh normalizado. Para training supervisado,
    queremos la query cuya bbox predicha más se parece al prompt.
    """
    pred_masks = outputs.pred_masks  # (1, 200, H', W')
    pred_boxes = outputs.pred_boxes  # (1, 200, 4) cxcywh normalizado
    H_img, W_img = image_hw
    x1, y1, x2, y2 = bbox_prompt
    cx = (x1 + x2) / 2 / W_img
    cy = (y1 + y2) / 2 / H_img
    w = (x2 - x1) / W_img
    h = (y2 - y1) / H_img
    target = torch.tensor(
        [cx, cy, w, h], device=pred_boxes.device, dtype=pred_boxes.dtype
    )
    dists = (pred_boxes[0] - target).abs().sum(dim=-1)  # (200,)
    best_idx = int(dists.argmin().item())
    return pred_masks[0, best_idx]  # (H', W')


def forward_one_sample(model, processor, image_bgr, bbox_prompt, mask_gt, device):
    """Forward + loss para un sample con bbox prompt (DETR-style).

    Selecciona la query mejor alineada con el bbox, redimensiona GT a
    la resolución interna (288x288 verificado), y combina BCE + Dice.
    """
    import cv2

    model_dtype = next(model.parameters()).dtype
    inputs = _prepare_inputs(processor, image_bgr, bbox_prompt, device, model_dtype)
    outputs = model(**inputs)
    H_img, W_img = image_bgr.shape[:2]
    pred = _select_best_query_mask(outputs, bbox_prompt, (H_img, W_img))  # (H', W')
    H_p, W_p = pred.shape
    gt_resized = cv2.resize(mask_gt, (W_p, H_p), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.from_numpy(gt_resized).to(device).to(pred.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, gt_t)
    pred_prob = torch.sigmoid(pred.float())
    dice = dice_loss(pred_prob, gt_t.float())
    return bce + dice


def train_epoch(
    model, processor, loader, optimizer, scheduler, device, accum_steps, log_every=20
):
    """Training de 1 epoch con limpieza periódica de VRAM.

    SAM 3 con backward y fp16 fragmenta memoria. Llamamos
    `torch.cuda.empty_cache()` cada 20 samples para evitar OOM
    acumulativo en datasets grandes con imágenes 1080p+.
    """
    import time

    model.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad()
    t0 = time.time()
    for batch in loader:
        for img, box, gt in zip(
            batch["images"], batch["bbox_prompts"], batch["masks_gt"]
        ):
            try:
                loss = forward_one_sample(model, processor, img, box, gt, device)
                (loss / accum_steps).backward()
                total_loss += loss.item()
                n += 1
                if n % accum_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                if n % log_every == 0:
                    rate = n / max(0.001, time.time() - t0)
                    print(
                        f"    sample {n}  loss={loss.item():.3f}  rate={rate:.2f}/s",
                        flush=True,
                    )
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                print(f"    OOM en sample {n + 1}: {e}", flush=True)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
    if n > 0 and (n % accum_steps != 0):
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / max(1, n)


@torch.no_grad()
def eval_iou(model, processor, loader, device, max_samples: int = 50) -> float:
    import cv2

    model.eval()
    ious = []
    seen = 0
    model_dtype = next(model.parameters()).dtype
    for batch in loader:
        for img, box, gt in zip(
            batch["images"], batch["bbox_prompts"], batch["masks_gt"]
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
            ious.append(inter / max(1, union))
            seen += 1
            if seen >= max_samples:
                return float(np.mean(ious))
    return float(np.mean(ious)) if ious else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--category", default=None, help="robot | ball | None (todas)")
    p.add_argument("--min-score", type=float, default=0.6)
    p.add_argument("--dry-run", action="store_true", help="1 epoch sobre 10 samples")
    p.add_argument(
        "--max-samples-per-epoch",
        type=int,
        default=0,
        help="si >0, en cada epoch muestrea aleatoriamente este número de "
        "samples del train set (limita acumulación de VRAM en datasets "
        "grandes con imágenes 1080p+). 0 = usar todo el train set.",
    )
    args = p.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Cargando dataset desde {DATASET_DIR}...")
    ds = SAM3PseudoDataset(
        DATASET_DIR, category_filter=args.category, min_score=args.min_score
    )
    print(f"  total samples: {len(ds)}")
    train_idx, val_idx = split_train_val(ds, val_fraction=0.15, seed=42)
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}")
    if args.dry_run:
        train_idx = train_idx[:8]
        val_idx = val_idx[:2]
        args.epochs = 1
        print(f"  DRY-RUN: train={len(train_idx)} val={len(val_idx)}")
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_keep_lists,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_keep_lists,
        num_workers=0,
    )

    print("\nCargando SAM 3.1 base...")
    from src.segmentation.sam3 import load_model

    processor, model = load_model(half_precision=True)
    print(f"\nAplicando LoRA (rank={args.rank}, alpha={args.alpha})...")
    model = apply_lora(model, args.rank, args.alpha, args.dropout)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    total_steps = max(1, args.epochs * (len(train_idx) // args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    best_iou = -1.0
    t_start = time.time()
    import random as _rnd

    rnd = _rnd.Random(42)
    for epoch in range(1, args.epochs + 1):
        # Si max_samples_per_epoch > 0, muestrea aleatoriamente del train set
        # un subset reducido para evitar acumulación de VRAM por epoch.
        if args.max_samples_per_epoch and args.max_samples_per_epoch > 0:
            epoch_idx = rnd.sample(
                train_idx, min(args.max_samples_per_epoch, len(train_idx))
            )
            ep_loader = DataLoader(
                Subset(ds, epoch_idx),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_keep_lists,
                num_workers=0,
            )
        else:
            ep_loader = train_loader
        ep_loss = train_epoch(
            model,
            processor,
            ep_loader,
            optimizer,
            scheduler,
            device,
            args.accum_steps,
        )
        log = {
            "epoch": epoch,
            "train_loss": ep_loss,
            "elapsed_min": (time.time() - t_start) / 60,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            iou = eval_iou(model, processor, val_loader, device)
            log["val_iou"] = iou
            if iou > best_iou:
                best_iou = iou
                ckpt_path = CKPT_DIR / f"lora_best_iou{iou:.3f}_ep{epoch}"
                model.save_pretrained(str(ckpt_path))
                log["saved"] = str(ckpt_path)
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

    # Final checkpoint + historia
    final_ckpt = CKPT_DIR / "lora_final"
    model.save_pretrained(str(final_ckpt))
    (CKPT_DIR / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTraining completo. Best IoU: {best_iou:.3f}")
    print(f"Checkpoint final: {final_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
