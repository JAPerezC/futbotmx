"""Pipeline end-to-end MÍNIMO: SAM 3.1 + OC-SORT + Kalman + HSV re-ID.

Uso:
    python scripts/run_pipeline.py \\
        --video data/raw/IMG_9915.MOV \\
        --duration 5 \\
        --stride 3 \\
        --out data/processed/runs/mvp

Componentes:
    1. SAM 3.1 segmenta campo, balón y robots por frame (stride).
    2. Kalman 2D trackea el balón (SAM 3 + HSV fallback).
    3. OC-SORT trackea los robots con identidades persistentes.
    4. HSV re-ID asigna equipo (A morado, B blanco) por bbox.
    5. Output: video MP4 anotado + JSON con trayectorias.

NOTA: este es el MVP de Fase 1. NO incluye aún:
    - Homografía a top-down (próximo paso).
    - Detección de eventos (próximo paso).
    - Visualizaciones extra (Fase 2).
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

import cv2
import numpy as np

from src.segmentation.baselines import find_ball_centroid
from src.segmentation.prompts import (
    BALL,
    PROMPT_ALL_ROBOTS,
)
from src.segmentation.sam3 import (
    SegMask,
    load_model,
    mask_centroid,
    masks_to_bboxes,
    segment_with_text,
)
from src.tracking.ball import BallTracker
from src.tracking.reid import classify_team
from src.tracking.robots import RobotTracker
from src.utils.io import VideoWriter, probe, read_frames


# -------- helpers de visualización --------

TEAM_COLOR = {"A": (200, 30, 130), "B": (240, 240, 240), None: (180, 180, 180)}
BALL_COLOR = (0, 165, 255)  # naranja BGR


def overlay_mask(image: np.ndarray, mask: np.ndarray, color, alpha=0.35) -> np.ndarray:
    out = image.copy()
    if mask.dtype != bool:
        mask = mask > 0
    overlay = out.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def draw_track(image: np.ndarray, bbox, track_id, team, conf):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    color = TEAM_COLOR.get(team, TEAM_COLOR[None])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"id={track_id} team={team or '?'} c={conf:.2f}"
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_ball(image: np.ndarray, state):
    if not state.found:
        return
    cx, cy = int(round(state.cx)), int(round(state.cy))
    cv2.circle(image, (cx, cy), 14, BALL_COLOR, 2)
    label = f"ball ({state.source} c={state.confidence:.2f})"
    cv2.putText(
        image,
        label,
        (cx + 18, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BALL_COLOR,
        2,
        cv2.LINE_AA,
    )


# -------- pipeline --------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=ROOT / "data" / "raw" / "IMG_9915.MOV")
    p.add_argument(
        "--out", type=Path, default=ROOT / "data" / "processed" / "runs" / "mvp"
    )
    p.add_argument("--duration", type=float, default=5.0, help="segundos a procesar")
    p.add_argument("--stride", type=int, default=3, help="leer 1 de cada N frames")
    p.add_argument("--model", default="facebook/sam3")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        print(f"ERROR: video no existe: {args.video}", file=sys.stderr)
        return 1

    meta = probe(args.video)
    print(
        f"Video: {meta.width}x{meta.height} @ {meta.fps:.2f} fps, {meta.duration_s:.1f}s"
    )
    max_frames_to_read = int(args.duration * meta.fps)
    print(
        f"Procesando {args.duration}s x {meta.fps:.1f} fps / stride {args.stride} ~ {max_frames_to_read // args.stride} frames"
    )

    print(f"Cargando SAM 3.1 ({args.model}) ...")
    t0 = time.time()
    processor, model = load_model(args.model, device=args.device)
    print(f"  cargado en {time.time() - t0:.1f}s")

    robot_tracker = RobotTracker(min_hits=1, det_thresh=0.2, iou_threshold=0.3)
    ball_tracker = BallTracker(dt=args.stride / meta.fps, max_missing_frames=15)

    out_video = args.out / "annotated.mp4"
    out_json = args.out / "tracks.json"
    fps_out = meta.fps / args.stride

    writer = VideoWriter(out_video, fps=fps_out, width=meta.width, height=meta.height)
    record: dict = {
        "video": str(args.video.name),
        "duration_s": args.duration,
        "stride": args.stride,
        "fps_in": meta.fps,
        "fps_out": fps_out,
        "frames": [],
    }

    try:
        n_processed = 0
        t_pipeline_start = time.time()
        for idx, frame in read_frames(args.video, stride=args.stride):
            if idx >= max_frames_to_read:
                break
            t_frame = time.time()
            # 1. SAM 3.1: detectar balón, campo, todos los robots
            seg = segment_with_text(
                frame,
                [BALL, PROMPT_ALL_ROBOTS],
                processor,
                model,
                threshold=0.2,
            )
            ball_masks: list[SegMask] = seg.get(BALL, [])
            robot_masks: list[SegMask] = seg.get(PROMPT_ALL_ROBOTS, [])

            # 2. Balón: SAM 3 si lo detectó (mejor score), sino HSV fallback
            ball_xy = None
            ball_source = "lost"
            if ball_masks:
                best = max(ball_masks, key=lambda m: m.score)
                c = mask_centroid(best.mask)
                if c is not None:
                    ball_xy = np.array(c)
                    ball_source = "sam3"
            if ball_xy is None:
                det = find_ball_centroid(frame)
                if det.found:
                    ball_xy = np.array([det.cx, det.cy])
                    ball_source = "hsv"
            ball_state = ball_tracker.update(ball_xy)

            # 3. Robots: bboxes de SAM 3 → OC-SORT
            robot_bboxes = masks_to_bboxes(robot_masks)
            dets = (
                np.hstack(
                    [
                        robot_bboxes,
                        np.array([[m.score] for m in robot_masks]),
                        np.zeros((len(robot_masks), 1)),
                    ]
                )
                if len(robot_masks) > 0
                else np.empty((0, 6))
            )
            tracks = robot_tracker.update(dets, frame)

            # 4. Re-ID por equipo
            annotated = frame.copy()
            tracks_record = []
            for tr in tracks:
                ts = classify_team(frame, tr.bbox_xyxy)
                draw_track(
                    annotated, tr.bbox_xyxy, tr.track_id, ts.label, tr.confidence
                )
                tracks_record.append(
                    {
                        "track_id": tr.track_id,
                        "bbox_xyxy": tr.bbox_xyxy.tolist(),
                        "centroid": tr.centroid_img.tolist(),
                        "team": ts.label,
                        "team_scores": {"A": ts.score_a, "B": ts.score_b},
                        "confidence": tr.confidence,
                    }
                )

            draw_ball(annotated, ball_state)
            writer.write(annotated)
            n_processed += 1

            record["frames"].append(
                {
                    "frame_idx": idx,
                    "t_s": idx / meta.fps,
                    "ball": {
                        "found": bool(ball_state.found),
                        "cx": ball_state.cx,
                        "cy": ball_state.cy,
                        "source": ball_source
                        if ball_xy is not None
                        else ball_state.source,
                        "confidence": ball_state.confidence,
                    },
                    "robots": tracks_record,
                    "infer_ms": (time.time() - t_frame) * 1000,
                }
            )

            if n_processed % 5 == 0:
                elapsed = time.time() - t_pipeline_start
                fps = n_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  frame {idx:4d}  robots={len(tracks)}  ball={ball_state.source}  "
                    f"({n_processed} procesados, {fps:.2f} fps efectivo)"
                )
    finally:
        writer.close()

    record["summary"] = {
        "frames_processed": n_processed,
        "pipeline_time_s": time.time() - t_pipeline_start,
        "avg_inference_ms": (
            float(np.mean([f["infer_ms"] for f in record["frames"]]))
            if record["frames"]
            else 0.0
        ),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"\nVideo anotado: {out_video}")
    print(f"JSON tracks:   {out_json}")
    print(f"Frames procesados: {n_processed}")
    print(f"Inferencia promedio: {record['summary']['avg_inference_ms']:.1f} ms/frame")
    print(
        f"Pipeline FPS efectivo: {n_processed / record['summary']['pipeline_time_s']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
