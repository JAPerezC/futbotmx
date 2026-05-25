"""Renderiza el video del minimap top-down a partir de un run del pipeline.

Lee `tracks.json` + `events.json` + `summary.json` de
`data/processed/runs/<video>/` y produce `minimap.mp4`: una animación de
la cancha vista desde arriba con robots, balón con trail, marcador,
posesión acumulada y eventos recientes.

Aprovecha la homografía ya calculada por el pipeline (no recomputa nada,
solo dibuja). Cero llamadas a GPU.

Uso:
    python scripts/render_minimap_video.py --run data/processed/runs/IMG_9821
    python scripts/render_minimap_video.py --run ... --side-by-side
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.viz.minimap import render_minimap_frame, topdown_canvas_size


TRAIL_FRAMES = 12  # ~2 s a 6 fps


def _accumulate_state(frames: list[dict], events: list[dict]) -> list[dict]:
    """Pre-calcula score, posesión acumulada y evento reciente por frame.

    El pipeline guarda eventos globales (events.json) pero no per-frame
    score. Aquí los reconstruimos en streaming.
    """
    state = []
    score_a = score_b = 0
    pos_a_acc = pos_b_acc = 0.0
    last_t = 0.0
    recent_event_window_s = 0.6
    event_iter = iter(sorted(events, key=lambda e: e["t"]))
    next_event = next(event_iter, None)
    for fr in frames:
        t = float(fr["t_s"])
        # Aplicar eventos hasta este t
        while next_event is not None and next_event["t"] <= t:
            etype = next_event["type"]
            if etype == "goal":
                team = (next_event.get("meta") or {}).get("scoring_team")
                if team == "A":
                    score_a += 1
                elif team == "B":
                    score_b += 1
            next_event = next(event_iter, None)
        # Posesión acumulada (sumar dt al equipo con posesión actual)
        dt = max(0.0, t - last_t)
        last_t = t
        poss = fr.get("possession")
        if poss is not None:
            team = poss.get("team")
            if team == "A":
                pos_a_acc += dt
            elif team == "B":
                pos_b_acc += dt
        total = pos_a_acc + pos_b_acc
        pa = 100.0 * pos_a_acc / total if total > 0 else 0.0
        pb = 100.0 * pos_b_acc / total if total > 0 else 0.0
        # Evento reciente
        recent_label = None
        for ev in events:
            if t - recent_event_window_s <= ev["t"] <= t:
                recent_label = f"{ev['type'].upper()} t={ev['t']:.1f}s"
                break
        state.append(
            {
                "score_a": score_a,
                "score_b": score_b,
                "pos_pct_a": pa,
                "pos_pct_b": pb,
                "event_label": recent_label,
            }
        )
    return state


def render(run_dir: Path, side_by_side: bool = False) -> int:
    tracks_path = run_dir / "tracks.json"
    events_path = run_dir / "events.json"
    summary_path = run_dir / "summary.json"
    annotated_path = run_dir / "annotated.mp4"
    if not tracks_path.exists():
        print(f"ERROR: {tracks_path} no existe")
        return 1
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    events = (
        json.loads(events_path.read_text(encoding="utf-8"))
        if events_path.exists()
        else []
    )
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )

    frames = tracks.get("frames", [])
    if not frames:
        print("ERROR: tracks.json no tiene frames")
        return 1
    duration_s = float(summary.get("duration_s") or tracks.get("duration_s", 0))
    fps_out = float(tracks.get("fps_out", 6.0))

    print(f"Run: {run_dir.name}")
    print(f"  frames: {len(frames)}  duration: {duration_s:.1f}s  fps: {fps_out:.2f}")

    state = _accumulate_state(frames, events)
    canvas_w, canvas_h = topdown_canvas_size()
    out_path = run_dir / "minimap.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"ERROR: no se pudo abrir writer para {out_path}")
        return 1

    ball_trail: deque = deque(maxlen=TRAIL_FRAMES)
    t0 = time.time()
    for i, fr in enumerate(frames):
        # Construir mapas
        robots_mm: dict[int, np.ndarray] = {}
        teams: dict[int, str | None] = {}
        for r in fr.get("robots", []):
            cm = r.get("centroid_mm")
            if cm is None:
                continue
            tid = int(r.get("track_id", -1))
            robots_mm[tid] = np.asarray(cm, dtype=np.float64)
            teams[tid] = r.get("team")
        ball = fr.get("ball") or {}
        ball_mm = ball.get("world_mm")
        ball_xy = np.asarray(ball_mm, dtype=np.float64) if ball_mm else None
        ball_trail.append(ball_xy)
        st = state[i]
        img = render_minimap_frame(
            robots_mm=robots_mm,
            teams=teams,
            ball_mm=ball_xy,
            ball_trail=list(ball_trail),
            score_a=st["score_a"],
            score_b=st["score_b"],
            pos_pct_a=st["pos_pct_a"],
            pos_pct_b=st["pos_pct_b"],
            t_s=float(fr["t_s"]),
            duration_s=duration_s,
            event_label=st["event_label"],
        )
        writer.write(img)
        if (i + 1) % 50 == 0:
            print(f"  frame {i + 1}/{len(frames)}", flush=True)
    writer.release()
    print(f"  OK minimap: {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"  tiempo: {time.time() - t0:.1f}s")

    if side_by_side:
        if not annotated_path.exists():
            print(f"WARN: no se puede hacer side-by-side, falta {annotated_path}")
        else:
            sbs_out = run_dir / "annotated_with_minimap.mp4"
            _ffmpeg_side_by_side(annotated_path, out_path, sbs_out)
            if sbs_out.exists():
                print(f"  OK side-by-side: {sbs_out}")
    return 0


def _ffmpeg_side_by_side(left: Path, right: Path, out: Path) -> None:
    """Pega dos videos lado a lado, alineando alturas."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v]scale=-1:720[L];[1:v]scale=-1:720[R];[L][R]hstack=inputs=2",
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
        print(f"WARN ffmpeg: {res.stderr[-300:]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--side-by-side",
        action="store_true",
        help="además genera annotated_with_minimap.mp4 (cámara | minimap)",
    )
    args = p.parse_args()
    return render(args.run, side_by_side=args.side_by_side)


if __name__ == "__main__":
    sys.exit(main())
