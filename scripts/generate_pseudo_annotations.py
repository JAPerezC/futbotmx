"""Genera dataset pseudo-supervisado para fine-tuning LoRA de SAM 3.1.

Estrategia: extraer N frames de cada video del Drive oficial, correr
SAM 3.1 base con prompts robustos, filtrar máscaras de alta confianza
(score >= --score-min, área plausible, aspect razonable), y guardarlas
como (image, mask_bin, bbox) para entrenamiento.

Pseudo-anotación NO es ground truth real, pero ya con la curaduría por
score+filtros geométricos produce un dataset razonable para que LoRA
afine la cabeza del decoder sobre el dominio "robots de fútbol" (tamaño
~30x30 px, vista oblicua, motion blur).

Salida en `data/processed/pseudo_dataset/`:
  images/      <video>_<frame>.jpg
  masks/       <video>_<frame>_<idx>.png      (uint8, 0/255)
  metadata.jsonl                              # un JSON por máscara

Uso:
    python scripts/generate_pseudo_annotations.py --frames-per-video 20 \\
        --score-min 0.6 --prompt-robot "soccer robot" --prompt-ball "ball"
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

from src.segmentation.sam3 import load_model, segment_with_text
from src.utils.io import probe, read_frames

DRIVE_DIR = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras"
SAMPLES_DIR = ROOT / "data" / "raw" / "drive_samples"
OUT_DIR = ROOT / "data" / "processed" / "pseudo_dataset"


def list_long_videos(min_duration_s: float = 30.0) -> list[Path]:
    candidates = sorted(list(DRIVE_DIR.glob("*.MOV"))) + sorted(
        SAMPLES_DIR.glob("video-*.mov")
    )
    longs: list[tuple[float, Path]] = []
    for v in candidates:
        if "clip45-65" in v.name:
            continue
        try:
            d = probe(v).duration_s
        except Exception:
            continue
        if d >= min_duration_s:
            longs.append((d, v))
    longs.sort(reverse=True)
    return [v for _, v in longs]


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= 0:
        return []
    if n >= total_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n, dtype=int).tolist()


def passes_quality_filter(
    mask: np.ndarray,
    score: float,
    score_min: float,
    area_min_frac: float,
    area_max_frac: float,
    aspect_min: float,
    aspect_max: float,
) -> bool:
    h, w = mask.shape[:2]
    if score < score_min:
        return False
    frame_area = h * w
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return False
    bw = xs.max() - xs.min()
    bh = ys.max() - ys.min()
    area = bw * bh
    frac = area / frame_area
    if frac < area_min_frac or frac > area_max_frac:
        return False
    aspect = bh / max(1, bw)
    if aspect < aspect_min or aspect > aspect_max:
        return False
    return True


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-per-video", type=int, default=20)
    p.add_argument("--score-min", type=float, default=0.6)
    p.add_argument("--prompt-robot", default="soccer robot")
    p.add_argument("--prompt-ball", default="ball")
    p.add_argument(
        "--robot-area-min-frac",
        type=float,
        default=0.001,
        help="bbox >= 0.1%% del frame",
    )
    p.add_argument(
        "--robot-area-max-frac",
        type=float,
        default=0.05,
        help="bbox <= 5%% del frame",
    )
    p.add_argument(
        "--ball-area-min-frac",
        type=float,
        default=0.0001,
        help="balón puede ser muy pequeño",
    )
    p.add_argument("--ball-area-max-frac", type=float, default=0.005)
    p.add_argument("--max-videos", type=int, default=None)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)
    (OUT_DIR / "masks").mkdir(exist_ok=True)
    metadata_path = OUT_DIR / "metadata.jsonl"

    videos = list_long_videos()
    if args.max_videos:
        videos = videos[: args.max_videos]
    print(f"Procesando {len(videos)} videos con {args.frames_per_video} frames c/u")

    print("Cargando SAM 3.1...")
    t0 = time.time()
    processor, model = load_model(half_precision=True)
    print(f"  modelo cargado en {time.time() - t0:.1f}s")

    n_masks_saved = 0
    n_frames_processed = 0
    t_start = time.time()
    with open(metadata_path, "w", encoding="utf-8") as meta_fp:
        for vi, video in enumerate(videos, 1):
            meta = probe(video)
            total = int(meta.duration_s * meta.fps)
            sample_idx = set(sample_frame_indices(total, args.frames_per_video))
            print(
                f"\n[{vi}/{len(videos)}] {video.name}  dur={meta.duration_s:.1f}s "
                f"sample={len(sample_idx)} frames"
            )
            for idx, frame in read_frames(video, stride=1):
                if idx not in sample_idx:
                    continue
                sample_idx.discard(idx)
                seg = segment_with_text(
                    frame,
                    [args.prompt_robot, args.prompt_ball],
                    processor,
                    model,
                    threshold=0.2,
                )
                img_name = f"{video.stem}_f{idx:06d}.jpg"
                img_path = OUT_DIR / "images" / img_name
                image_written = False
                for prompt_name, prompt_text, area_min, area_max, aspect_range in (
                    (
                        "robot",
                        args.prompt_robot,
                        args.robot_area_min_frac,
                        args.robot_area_max_frac,
                        (0.4, 4.0),
                    ),
                    (
                        "ball",
                        args.prompt_ball,
                        args.ball_area_min_frac,
                        args.ball_area_max_frac,
                        (0.4, 2.5),
                    ),
                ):
                    masks = seg.get(prompt_text, [])
                    for mi, m in enumerate(masks):
                        if not passes_quality_filter(
                            m.mask,
                            m.score,
                            args.score_min,
                            area_min,
                            area_max,
                            aspect_range[0],
                            aspect_range[1],
                        ):
                            continue
                        if not image_written:
                            cv2.imwrite(str(img_path), frame)
                            image_written = True
                        mask_name = f"{video.stem}_f{idx:06d}_{prompt_name}_{mi}.png"
                        mask_path = OUT_DIR / "masks" / mask_name
                        cv2.imwrite(str(mask_path), (m.mask > 0).astype(np.uint8) * 255)
                        x1, y1, x2, y2 = mask_to_bbox(m.mask)
                        meta_fp.write(
                            json.dumps(
                                {
                                    "image": f"images/{img_name}",
                                    "mask": f"masks/{mask_name}",
                                    "category": prompt_name,
                                    "prompt": prompt_text,
                                    "score": float(m.score),
                                    "bbox_xyxy": [x1, y1, x2, y2],
                                    "frame_size_hw": list(frame.shape[:2]),
                                    "source_video": video.name,
                                    "source_frame": idx,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_masks_saved += 1
                n_frames_processed += 1
                if n_frames_processed % 20 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"  frames={n_frames_processed} masks={n_masks_saved} "
                        f"({n_frames_processed / elapsed:.2f} fps)"
                    )
                if not sample_idx:
                    break

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {n_masks_saved} máscaras en {n_frames_processed} frames")
    print(
        f"  tiempo: {total_time / 60:.1f} min ({n_frames_processed / total_time:.2f} fps)"
    )
    print(f"  dataset en: {OUT_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
