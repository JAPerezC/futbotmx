"""Corre el pipeline completo sobre una lista de videos en serie.

Útil para procesar los 4 clips validados (IMG_9821, 9811, 9808, 9800) sin
tener que lanzar uno por uno. Imprime un resumen consolidado al final.

Uso:
    python scripts/batch_pipeline.py
    python scripts/batch_pipeline.py --stride 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMARAS = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras"

DEFAULT_VIDEOS = [
    CAMARAS / "IMG_9863.MOV",
    CAMARAS / "IMG_9865.MOV",
    CAMARAS / "IMG_9851.MOV",
    CAMARAS / "IMG_9855.MOV",
]


def run_one(video: Path, stride: int) -> dict:
    out_dir = ROOT / "data" / "processed" / "runs" / video.stem
    print(f"\n{'=' * 70}")
    print(f"  {video.name}  (stride={stride})")
    print(f"  destino: {out_dir}")
    print(f"{'=' * 70}")
    t0 = time.time()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--video",
        str(video),
        "--stride",
        str(stride),
    ]
    proc = subprocess.run(
        cmd,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "video": video.name,
        "ok": proc.returncode == 0,
        "elapsed_s": elapsed,
        "summary": summary,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=5)
    p.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Rutas absolutas o relativas a videos. Si se omite, usa DEFAULT_VIDEOS.",
    )
    args = p.parse_args()

    videos = [Path(v) for v in args.videos] if args.videos else DEFAULT_VIDEOS

    print(f"Procesando {len(videos)} videos con stride={args.stride}")
    for v in videos:
        if not v.exists():
            print(f"  ERROR: no existe {v}")
            return 1

    results = []
    t0 = time.time()
    for v in videos:
        results.append(run_one(v, args.stride))

    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  BATCH COMPLETADO en {total / 60:.1f} min")
    print(f"{'=' * 70}")
    for r in results:
        s = r["summary"]
        events = s.get("events_by_type", {})
        score = s.get("score", {})
        possession = s.get("possession_pct", {})
        print(
            f"  {r['video']:20s}  {'OK' if r['ok'] else 'FAIL'}  "
            f"{r['elapsed_s'] / 60:5.1f} min  "
            f"score A={score.get('A', 0)}-B={score.get('B', 0)}  "
            f"pos A={possession.get('A', 0):.0f}%-B={possession.get('B', 0):.0f}%  "
            f"events: {dict(events)}"
        )

    # Resumen agregado JSON
    agg = ROOT / "data" / "processed" / "runs" / "_batch_summary.json"
    agg.write_text(
        json.dumps(
            {
                "stride": args.stride,
                "total_time_s": total,
                "results": [
                    {
                        "video": r["video"],
                        "ok": r["ok"],
                        "elapsed_s": r["elapsed_s"],
                        "events_by_type": r["summary"].get("events_by_type", {}),
                        "score": r["summary"].get("score", {}),
                        "possession_pct": r["summary"].get("possession_pct", {}),
                        "frames_processed": r["summary"].get("frames_processed", 0),
                    }
                    for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResumen agregado: {agg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
