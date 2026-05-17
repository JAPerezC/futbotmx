"""Primera prueba de SAM 3.1: descarga pesos, infiere sobre un frame.

Descarga ~5-10 GB la primera vez (cacheado en ~/.cache/huggingface). Puede
tardar varios minutos según red. Las descargas siguientes son instantáneas.

Uso:
    python scripts/sam3_smoke_test.py --frame data/processed/sample_frames/frame_24s.jpg

Genera:
    data/processed/sam3_smoke/<prompt>_mask.png   máscara binaria por prompt
    data/processed/sam3_smoke/<prompt>_overlay.jpg  overlay verde sobre frame
    data/processed/sam3_smoke/report.json         métricas (tiempos, scores)
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

# IMPORTANTE: enable system SSL ANTES de importar transformers/huggingface_hub
from src.utils.network import enable_system_ssl

enable_system_ssl()

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from src.segmentation.prompts import ALL_PROMPTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame",
        type=Path,
        default=ROOT / "data" / "processed" / "sample_frames" / "frame_24s.jpg",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "sam3_smoke",
    )
    p.add_argument("--model", default="facebook/sam3", help="HF model id")
    p.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Subset de prompts (keys de ALL_PROMPTS). Default: todos.",
    )
    return p.parse_args()


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4):
    out = image_bgr.copy()
    if mask.dtype != bool:
        mask = mask > 0
    overlay = out.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Cargando frame: {args.frame}")
    img_bgr = cv2.imread(str(args.frame))
    if img_bgr is None:
        print("ERROR: frame no se pudo leer", file=sys.stderr)
        return 1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] Device: {device}")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Cargando processor y model: {args.model}")
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model).to(device).eval()
    t_load = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Modelo cargado en {t_load:.1f}s")

    prompt_keys = args.prompts if args.prompts else list(ALL_PROMPTS.keys())
    report = {
        "frame": args.frame.name,
        "device": device,
        "model": args.model,
        "load_time_s": t_load,
        "results": {},
    }

    for key in prompt_keys:
        prompt = ALL_PROMPTS[key]
        print(f"\n[{time.strftime('%H:%M:%S')}] Prompt '{key}': {prompt!r}")
        t1 = time.time()
        inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        t_inf = time.time() - t1
        print(f"  inferencia: {t_inf * 1000:.1f} ms")

        # post-procesar: usar el helper del processor si existe
        try:
            h_img, w_img = img_bgr.shape[:2]
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.2,
                mask_threshold=0.5,
                target_sizes=[(h_img, w_img)],
            )
            r0 = results[0]
            masks = r0.get("masks")
            scores = r0.get("scores")
            n_inst = 0 if masks is None else len(masks)
            print(f"  instancias detectadas: {n_inst}")

            if n_inst > 0:
                # combinar todas las máscaras en una sola binaria
                if hasattr(masks, "cpu"):
                    masks_np = masks.cpu().numpy()
                else:
                    masks_np = np.asarray(masks)
                combined = masks_np.any(axis=0).astype(np.uint8) * 255
                cv2.imwrite(str(args.out / f"{key}_mask.png"), combined)
                overlay = overlay_mask(img_bgr, combined > 0)
                cv2.imwrite(str(args.out / f"{key}_overlay.jpg"), overlay)

                top_scores = (
                    scores.cpu().numpy().tolist()
                    if hasattr(scores, "cpu")
                    else list(scores)
                )
            else:
                top_scores = []

            report["results"][key] = {
                "prompt": prompt,
                "inference_ms": t_inf * 1000,
                "n_instances": int(n_inst),
                "scores": top_scores[:10],
            }
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR post-process: {e}")
            report["results"][key] = {
                "prompt": prompt,
                "inference_ms": t_inf * 1000,
                "error": str(e),
            }

    with open(args.out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n===== RESUMEN =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
