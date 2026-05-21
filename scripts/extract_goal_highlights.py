"""Extrae GIFs/MP4 cortos de cada gol detectado por el pipeline.

Útil para el reel Instagram + video demo de 2 min (§ 3.5.3). Lee
`events.json` de un run y para cada evento `goal` extrae una ventana
de [t-2s, t+2s] del video annotated y la guarda como clip independiente.

Uso:
    python scripts/extract_goal_highlights.py --run data/processed/runs/IMG_9811
    python scripts/extract_goal_highlights.py --run ... --window 3.0 --format gif
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_clip(video: Path, t_start: float, t_end: float, out: Path, fmt: str):
    """Usa ffmpeg para extraer el clip; gif o mp4 según fmt."""
    duration = t_end - t_start
    if fmt == "gif":
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t_start:.2f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.2f}",
            "-vf",
            "fps=12,scale=720:-1:flags=lanczos",
            "-loop",
            "0",
            str(out),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t_start:.2f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.2f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-an",
            str(out),
        ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ERROR ffmpeg: {res.stderr[-300:]}")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=Path, required=True, help="carpeta data/processed/runs/<video>"
    )
    p.add_argument("--window", type=float, default=2.0, help="segundos antes y después")
    p.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    p.add_argument(
        "--source",
        choices=("annotated", "original"),
        default="annotated",
        help="usar video annotated (con bbox/banner) u original",
    )
    args = p.parse_args()

    events_path = args.run / "events.json"
    summary_path = args.run / "summary.json"
    if not events_path.exists():
        print(f"ERROR: {events_path} no existe")
        return 1
    events = json.loads(events_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    goals = [e for e in events if e["type"] == "goal"]
    print(f"Run: {args.run.name}")
    print(f"  goles detectados: {len(goals)}")
    if not goals:
        print("  nada para extraer")
        return 0

    if args.source == "annotated":
        video = args.run / "annotated.mp4"
    else:
        # Buscar el video original por nombre
        video_name = summary["video"]
        for candidate in (
            ROOT
            / "data"
            / "raw"
            / "drive_oficial"
            / "17Abril"
            / "Cámaras"
            / video_name,
            ROOT / "data" / "raw" / "drive_samples" / video_name,
        ):
            if candidate.exists():
                video = candidate
                break
        else:
            print(f"ERROR: video original {video_name} no encontrado")
            return 1
    if not video.exists():
        print(f"ERROR: {video} no existe")
        return 1

    out_dir = args.run / "highlights"
    out_dir.mkdir(exist_ok=True)
    duration_total = summary.get("duration_s", 9999)
    for i, ev in enumerate(goals):
        t0 = max(0, ev["t"] - args.window)
        t1 = min(duration_total, ev["t"] + args.window)
        team = ev["meta"].get("scoring_team") or "NA"
        color = ev["meta"].get("goal_color", "NA")
        # Sanitizar para nombre de archivo (Windows no permite ? : * etc)
        safe_team = "".join(c if c.isalnum() else "_" for c in str(team))
        safe_color = "".join(c if c.isalnum() else "_" for c in str(color))
        out_name = (
            f"goal_{i + 1:02d}_t{ev['t']:.1f}_team{safe_team}_{safe_color}."
            f"{args.format}"
        )
        out_path = out_dir / out_name
        ok = extract_clip(video, t0, t1, out_path, args.format)
        if ok:
            print(f"  [{i + 1}/{len(goals)}] {out_name}  ({t0:.1f}s-{t1:.1f}s)")
    print(f"\nHighlights en: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
