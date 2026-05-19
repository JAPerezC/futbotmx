"""Evalúa qué clip tiene mejor visibilidad del campo para el pipeline.

Para cada video largo, muestrea ~10 frames y mide:
- success rate de detect_field_corners
- area_ratio promedio del cuadrilátero detectado
- estabilidad (std de área entre frames)

Imprime ranking y guarda los frames de muestra con las esquinas pintadas
en data/processed/calib_scoring/ para inspección visual.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.network import enable_system_ssl

enable_system_ssl()

import cv2
import numpy as np

from src.utils.field_detect import detect_field_corners
from src.utils.io import probe, read_frames

OUT_DIR = ROOT / "data" / "processed" / "calib_scoring"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def score(video: Path, n_samples: int = 12) -> dict:
    meta = probe(video)
    total_frames = int(meta.duration_s * meta.fps)
    # Muestra equidistante
    sample_idx = np.linspace(
        0, max(1, total_frames - 10), n_samples, dtype=int
    ).tolist()
    sample_set = set(sample_idx)

    areas = []
    successes = 0
    all_inside_count = 0
    saved_first = False
    last_corners = None
    for idx, frame in read_frames(video, stride=1):
        if idx not in sample_set:
            continue
        sample_set.discard(idx)
        h_img, w_img = frame.shape[:2]
        res = detect_field_corners(frame)
        if res.success:
            successes += 1
            areas.append(res.contour_area_ratio)
            last_corners = res.corners
            inside = sum(1 for x, y in res.corners if 0 <= x < w_img and 0 <= y < h_img)
            if inside == 4:
                all_inside_count += 1
            if not saved_first:
                vis = frame.copy()
                pts = res.corners.astype(np.int32)
                for i in range(4):
                    cv2.line(
                        vis,
                        tuple(pts[i]),
                        tuple(pts[(i + 1) % 4]),
                        (0, 255, 255),
                        4,
                        cv2.LINE_AA,
                    )
                for x, y in pts:
                    is_in = 0 <= x < w_img and 0 <= y < h_img
                    color = (0, 0, 255) if is_in else (0, 165, 255)
                    cv2.circle(vis, (int(x), int(y)), 14, color, -1)
                cv2.putText(
                    vis,
                    f"{inside}/4 esquinas dentro del frame",
                    (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.6,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(OUT_DIR / f"{video.stem}_calib.jpg"), vis)
                saved_first = True
        if not sample_set:
            break

    area_mean = float(np.mean(areas)) if areas else 0.0
    area_std = float(np.std(areas)) if areas else 0.0
    return {
        "video": video.name,
        "duration_s": meta.duration_s,
        "n_sampled": n_samples,
        "success_rate": successes / n_samples,
        "all_inside_rate": all_inside_count / n_samples,
        "area_mean": area_mean,
        "area_std": area_std,
        "last_corners": last_corners.tolist() if last_corners is not None else None,
    }


def main() -> int:
    candidates = sorted(
        list(
            (ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras").glob(
                "*.MOV"
            )
        )
        + list((ROOT / "data" / "raw" / "drive_samples").glob("video-*.mov"))
    )
    # Filtrar largos (>40s) y excluir el clip45-65
    long_videos = []
    for v in candidates:
        if "clip45-65" in v.name:
            continue
        try:
            d = probe(v).duration_s
        except Exception:
            continue
        if d >= 40:
            long_videos.append((d, v))
    long_videos.sort(reverse=True)
    print(f"Evaluando {len(long_videos)} videos con duración >= 40s...\n")

    results = []
    for dur, v in long_videos:
        try:
            r = score(v)
            results.append(r)
            print(
                f"  {v.name:30s}  dur={dur:5.1f}s  success={r['success_rate'] * 100:5.1f}%  "
                f"4-inside={r['all_inside_rate'] * 100:5.1f}%  "
                f"area={r['area_mean']:.3f}±{r['area_std']:.3f}"
            )
        except Exception as e:
            print(f"  {v.name}: ERROR {e}")

    # Ranking: privilegiar 4 esquinas DENTRO del frame
    def rank_key(r):
        return (
            r["all_inside_rate"] * r["area_mean"]
            - 0.5 * r["area_std"]
            + 0.05 * r["success_rate"]
        )

    results.sort(key=rank_key, reverse=True)
    print("\n=== TOP-5 (privilegiando 4 esquinas dentro del frame) ===")
    for r in results[:5]:
        print(
            f"  {r['video']:30s}  rank={rank_key(r):.3f}  "
            f"4-inside={r['all_inside_rate'] * 100:.0f}%  "
            f"area={r['area_mean']:.2f}±{r['area_std']:.2f}"
        )
    print(f"\nFrames de calibración pintados en: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
