"""Pipeline end-to-end con homografía, eventos y visualizaciones.

Usa SAM 3.1 (Meta) + OC-SORT (BoxMOT) + Kalman ball + HSV re-ID + AutoRefs-style
event detection sobre coordenadas top-down rectificadas con homografía 4-puntos.

Uso:
    python scripts/run_pipeline.py --video data/raw/IMG_9915.MOV
    python scripts/run_pipeline.py --video data/raw/drive_samples/video-943.mov \\
        --duration 14 --stride 2

Genera (en --out):
    annotated.mp4          video con segmentación + tracks + balón + banners de eventos
    topdown.mp4            video de vista cenital reconstruida (trails + posición)
    heatmap_robots.png     densidad de actividad
    heatmap_ball.png       densidad del balón
    trails.png             trayectorias completas top-down
    voronoi_final.png      control de espacio (último frame)
    tracks.json            trayectorias en imagen y mundo
    events.json            lista de eventos con timestamps
    summary.json           resumen agregado (FPS, evento counts, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.network import enable_system_ssl

enable_system_ssl()

import cv2
import numpy as np

from src.events.possession import POSSESSION_RADIUS_MM, closest_robot_possession
from src.events.rules import (
    DAMAGED_TIME_S,
    Event,
    detect_kick,
    is_damaged_robot,
    is_in_goal_roi,
    is_kick,
    is_no_progress,
)
from src.segmentation.baselines import find_ball_centroid
from src.segmentation.prompts import BALL, PROMPT_ALL_ROBOTS
from src.segmentation.sam3 import (
    load_model,
    mask_centroid,
    masks_to_bboxes,
    segment_with_text,
)
from src.tracking.ball import BallTracker
from src.tracking.reid import AdaptiveTeamClassifier, _dominant_hue
from src.tracking.robots import RobotTracker
from src.utils.calib import (
    compute_homography,
    project_points,
)
from src.utils.field_detect import detect_field_corners
from src.utils.io import VideoWriter, probe, read_frames
from src.viz.heatmap import render_heatmap
from src.viz.trails import render_trails
from src.viz.voronoi import render_voronoi

# -------- helpers de visualización --------

TEAM_COLOR = {"A": (200, 30, 130), "B": (240, 240, 240), None: (180, 180, 180)}
BALL_COLOR = (0, 165, 255)  # naranja BGR
EVENT_COLOR = {
    "kick": (0, 220, 255),
    "goal": (0, 255, 0),
    "retention": (50, 50, 255),
    "no_progress": (180, 180, 0),
    "damaged": (255, 100, 0),
    "possession": (255, 255, 255),
}


def overlay_polygon(image, corners, color=(0, 200, 255)):
    if corners is None or len(corners) < 4:
        return
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)


def draw_track(image, bbox, track_id, team, conf):
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


def draw_ball(image, state):
    if not state.found:
        return
    cx, cy = int(round(state.cx)), int(round(state.cy))
    cv2.circle(image, (cx, cy), 14, BALL_COLOR, 2)
    cv2.putText(
        image,
        f"ball ({state.source} c={state.confidence:.2f})",
        (cx + 18, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BALL_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_event_banner(image, recent_events):
    """Banner superior con eventos recientes (últimos 0.6s)."""
    if not recent_events:
        return
    h, w = image.shape[:2]
    banner_h = 50
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    x = 12
    for ev in recent_events:
        color = EVENT_COLOR.get(ev.type, (255, 255, 255))
        text = f"{ev.type.upper()} t={ev.t:.2f}s"
        cv2.putText(
            image, text, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
        x += int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) + 28


def draw_possession_info(image, pos):
    if pos is None or pos.track_id is None:
        return
    h, w = image.shape[:2]
    text = (
        f"posession id={pos.track_id} team={pos.team or '?'} d={pos.distance_mm:.0f}mm"
    )
    cv2.putText(
        image,
        text,
        (12, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# -------- calibración --------


def calibrate_homography(video_path, max_frames_to_try=60):
    """Busca el primer frame donde se detecten 4 esquinas y devuelve H + corners."""
    print("  buscando frame para calibrar...")
    tried = 0
    for idx, frame in read_frames(video_path, stride=5):
        tried += 1
        res = detect_field_corners(frame)
        if res.success:
            H = compute_homography(res.corners)
            print(
                f"  ✓ esquinas detectadas en frame idx={idx} (area_ratio={res.contour_area_ratio:.2f})"
            )
            return H, res.corners, idx
        if tried >= max_frames_to_try:
            break
    print("  WARN: no se pudo calibrar automáticamente; usando esquinas por defecto")
    h, w = frame.shape[:2]
    fallback = np.array(
        [
            [w * 0.1, h * 0.2],
            [w * 0.9, h * 0.2],
            [w * 0.95, h * 0.9],
            [w * 0.05, h * 0.9],
        ],
        dtype=np.float64,
    )
    return compute_homography(fallback), fallback, 0


# -------- pipeline --------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=ROOT / "data" / "raw" / "IMG_9915.MOV")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="carpeta destino (default data/processed/runs/<videoname>)",
    )
    p.add_argument(
        "--duration", type=float, default=None, help="segundos a procesar (None=todo)"
    )
    p.add_argument("--stride", type=int, default=3, help="leer 1 de cada N frames")
    p.add_argument("--model", default="facebook/sam3")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--robot-score-min",
        type=float,
        default=0.5,
        help="filtro de score SAM 3 para detecciones de robots (defecto 0.5)",
    )
    p.add_argument(
        "--robot-area-min-frac",
        type=float,
        default=0.001,
        help="área mínima del bbox como fracción del frame (defecto 0.1%)",
    )
    p.add_argument(
        "--robot-area-max-frac",
        type=float,
        default=0.05,
        help="área máxima del bbox como fracción del frame (defecto 5%)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        args.out = ROOT / "data" / "processed" / "runs" / args.video.stem
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        print(f"ERROR: video no existe: {args.video}", file=sys.stderr)
        return 1

    meta = probe(args.video)
    duration = args.duration if args.duration else meta.duration_s
    max_frames_to_read = int(duration * meta.fps)
    print(
        f"Video: {meta.width}x{meta.height} @ {meta.fps:.2f} fps, {meta.duration_s:.1f}s"
    )
    print(
        f"Procesando {duration:.1f}s / stride {args.stride} ~ {max_frames_to_read // args.stride} frames"
    )

    # Calibrar homografía
    print("Calibrando homografía...")
    H, corners, _ = calibrate_homography(args.video)

    # Cargar SAM 3
    print(f"Cargando SAM 3.1 ({args.model})...")
    t0 = time.time()
    processor, model = load_model(args.model, device=args.device)
    print(f"  cargado en {time.time() - t0:.1f}s")

    robot_tracker = RobotTracker(min_hits=1, det_thresh=0.2, iou_threshold=0.3)
    ball_tracker = BallTracker(dt=args.stride / meta.fps, max_missing_frames=20)
    team_clf = AdaptiveTeamClassifier(warmup_frames=8, hue_separation_min=20)

    out_video = args.out / "annotated.mp4"
    fps_out = meta.fps / args.stride
    writer = VideoWriter(out_video, fps=fps_out, width=meta.width, height=meta.height)

    # Estado del partido
    ball_positions_mm = []  # lista de (t, np.array([x,y]))
    robot_positions_mm = defaultdict(list)  # track_id -> [(t, xy)]
    robot_team_history = defaultdict(list)  # track_id -> [team_label]
    events: list[Event] = []
    last_event_id = 0

    # Buffers para detecciones de eventos
    ball_retention_start = None
    ball_retention_robot = None

    record = {
        "video": args.video.name,
        "duration_s": duration,
        "stride": args.stride,
        "fps_in": meta.fps,
        "fps_out": fps_out,
        "homography": H.tolist(),
        "corners_img": corners.tolist(),
        "frames": [],
    }

    try:
        n_processed = 0
        t_start = time.time()
        for idx, frame in read_frames(args.video, stride=args.stride):
            if idx >= max_frames_to_read:
                break
            t_now = idx / meta.fps

            # 1. SAM 3.1: balón + robots
            seg = segment_with_text(
                frame, [BALL, PROMPT_ALL_ROBOTS], processor, model, threshold=0.2
            )
            ball_masks = seg.get(BALL, [])
            robot_masks_raw = seg.get(PROMPT_ALL_ROBOTS, [])

            # Filtrar robots: score alto + área plausible (descarta balón,
            # cajas amarillas, parches grandes que SAM 3 confunde).
            frame_area = frame.shape[0] * frame.shape[1]
            area_min = args.robot_area_min_frac * frame_area
            area_max = args.robot_area_max_frac * frame_area
            robot_masks = []
            for m in robot_masks_raw:
                if m.score < args.robot_score_min:
                    continue
                ys, xs = np.where(m.mask)
                if xs.size == 0:
                    continue
                w = xs.max() - xs.min()
                h = ys.max() - ys.min()
                area = w * h
                if area < area_min or area > area_max:
                    continue
                # Robots con bandera suelen ser ligeramente más altos que anchos.
                # Pero ser permisivo (0.4 a 4.0) por perspectiva.
                aspect = h / max(1, w)
                if aspect < 0.4 or aspect > 4.0:
                    continue
                robot_masks.append(m)

            # 2. Balón: SAM 3 → HSV fallback
            ball_xy_img = None
            ball_source = None
            if ball_masks:
                best = max(ball_masks, key=lambda m: m.score)
                c = mask_centroid(best.mask)
                if c is not None:
                    ball_xy_img = np.array(c)
                    ball_source = "sam3"
            if ball_xy_img is None:
                det = find_ball_centroid(frame)
                if det.found:
                    ball_xy_img = np.array([det.cx, det.cy])
                    ball_source = "hsv"
            ball_state = ball_tracker.update(ball_xy_img)

            # 3. Robots: SAM 3 → OC-SORT
            robot_bboxes = masks_to_bboxes(robot_masks)
            if len(robot_masks) > 0:
                dets = np.hstack(
                    [
                        robot_bboxes,
                        np.array([[m.score] for m in robot_masks]),
                        np.zeros((len(robot_masks), 1)),
                    ]
                )
            else:
                dets = np.empty((0, 6))
            tracks = robot_tracker.update(dets, frame)

            # 4. Re-ID por bandera + proyección a mundo (clasificador adaptativo)
            robots_mm_this_frame = {}
            teams_this_frame = {}
            tracks_record = []
            for tr in tracks:
                hue = _dominant_hue(frame, tr.bbox_xyxy)
                team_clf.observe(tr.track_id, hue)
                team_label = team_clf.assign(tr.track_id)
                # proyectar centroide a mundo
                world_xy = project_points(tr.centroid_img.reshape(1, 2), H)[0]
                robots_mm_this_frame[tr.track_id] = world_xy
                teams_this_frame[tr.track_id] = team_label
                robot_positions_mm[tr.track_id].append((t_now, world_xy))
                robot_team_history[tr.track_id].append(team_label)
                tracks_record.append(
                    {
                        "track_id": tr.track_id,
                        "bbox_xyxy": tr.bbox_xyxy.tolist(),
                        "centroid_img": tr.centroid_img.tolist(),
                        "centroid_mm": world_xy.tolist(),
                        "team": team_label,
                        "team_hue": hue,
                        "confidence": tr.confidence,
                    }
                )
            team_clf.end_frame()

            ball_mm = None
            if ball_state.found:
                xy_img = np.array([[ball_state.cx, ball_state.cy]])
                ball_mm = project_points(xy_img, H)[0]
                ball_positions_mm.append((t_now, ball_mm))

            # 5. Detección de eventos
            current_events = []

            # Kick: cambio velocidad balón
            if len(ball_positions_mm) >= 2:
                tp, xy_p = ball_positions_mm[-2]
                tc, xy_c = ball_positions_mm[-1]
                dt = tc - tp
                v = detect_kick(xy_p, xy_c, dt)
                if is_kick(v):
                    last_event_id += 1
                    ev = Event(
                        t=tc,
                        type="kick",
                        actors=[],
                        position_mm=(float(xy_c[0]), float(xy_c[1])),
                        confidence=min(v / 1000, 1.0),
                        meta={"velocity_mm_s": float(v)},
                    )
                    events.append(ev)
                    current_events.append(ev)

            # Gol: balón en ROI portería
            if ball_mm is not None:
                for side in ("left", "right"):
                    if is_in_goal_roi(ball_mm, side):
                        last_event_id += 1
                        ev = Event(
                            t=t_now,
                            type="goal",
                            actors=[],
                            position_mm=(float(ball_mm[0]), float(ball_mm[1])),
                            confidence=1.0,
                            meta={"side": side},
                        )
                        events.append(ev)
                        current_events.append(ev)

            # Posesión
            pos = None
            if ball_mm is not None and robots_mm_this_frame:
                pos = closest_robot_possession(
                    ball_mm, robots_mm_this_frame, teams_this_frame
                )
                # Retención: si mismo robot por > T segundos
                if pos.track_id is not None and pos.distance_mm < POSSESSION_RADIUS_MM:
                    if (
                        ball_retention_robot == pos.track_id
                        and ball_retention_start is not None
                    ):
                        elapsed = t_now - ball_retention_start
                        if elapsed > 1.5:
                            last_event_id += 1
                            ev = Event(
                                t=t_now,
                                type="retention",
                                actors=[pos.track_id],
                                position_mm=(float(ball_mm[0]), float(ball_mm[1])),
                                confidence=min(elapsed / 3.0, 1.0),
                                meta={"duration_s": elapsed, "team": pos.team},
                            )
                            events.append(ev)
                            current_events.append(ev)
                            ball_retention_start = None  # reset
                    else:
                        ball_retention_robot = pos.track_id
                        ball_retention_start = t_now
                else:
                    ball_retention_robot = None
                    ball_retention_start = None

            # No progress (ventana 5s)
            if len(ball_positions_mm) >= int(5 / (args.stride / meta.fps)):
                window = np.array(
                    [
                        p[1]
                        for p in ball_positions_mm[-int(5 / (args.stride / meta.fps)) :]
                    ]
                )
                if is_no_progress(window, dt_s=args.stride / meta.fps):
                    # solo emitir 1 cada 2 segundos
                    if (
                        not events
                        or events[-1].type != "no_progress"
                        or t_now - events[-1].t > 2.0
                    ):
                        last_event_id += 1
                        ev = Event(
                            t=t_now,
                            type="no_progress",
                            actors=[],
                            position_mm=(
                                float(window.mean(axis=0)[0]),
                                float(window.mean(axis=0)[1]),
                            ),
                            confidence=0.8,
                            meta={},
                        )
                        events.append(ev)
                        current_events.append(ev)

            # Robot dañado (ventana 60s)
            for tid, history in robot_positions_mm.items():
                if len(history) < int(DAMAGED_TIME_S / (args.stride / meta.fps)):
                    continue
                xy_hist = np.array(
                    [
                        h[1]
                        for h in history[
                            -int(DAMAGED_TIME_S / (args.stride / meta.fps)) :
                        ]
                    ]
                )
                diffs = np.linalg.norm(np.diff(xy_hist, axis=0), axis=1) / (
                    args.stride / meta.fps
                )
                if is_damaged_robot(diffs, dt_s=args.stride / meta.fps):
                    if not events or not (
                        events[-1].type == "damaged"
                        and tid in events[-1].actors
                        and t_now - events[-1].t < 30
                    ):
                        last_event_id += 1
                        ev = Event(
                            t=t_now,
                            type="damaged",
                            actors=[tid],
                            position_mm=(float(xy_hist[-1][0]), float(xy_hist[-1][1])),
                            confidence=0.7,
                            meta={},
                        )
                        events.append(ev)
                        current_events.append(ev)

            # 6. Anotar frame
            annotated = frame.copy()
            overlay_polygon(annotated, corners, color=(255, 255, 0))
            for tr in tracks:
                draw_track(
                    annotated,
                    tr.bbox_xyxy,
                    tr.track_id,
                    teams_this_frame.get(tr.track_id),
                    tr.confidence,
                )
            draw_ball(annotated, ball_state)
            draw_possession_info(annotated, pos)
            # Banner de eventos recientes (últimos 0.6 s)
            recent = [e for e in events if t_now - e.t < 0.6]
            draw_event_banner(annotated, recent)
            writer.write(annotated)

            record["frames"].append(
                {
                    "frame_idx": idx,
                    "t_s": t_now,
                    "ball": {
                        "found": bool(ball_state.found),
                        "cx": ball_state.cx,
                        "cy": ball_state.cy,
                        "world_mm": ball_mm.tolist() if ball_mm is not None else None,
                        "source": ball_source or ball_state.source,
                        "confidence": ball_state.confidence,
                    },
                    "robots": tracks_record,
                    "possession": (
                        {
                            "track_id": pos.track_id,
                            "team": pos.team,
                            "distance_mm": pos.distance_mm,
                        }
                        if pos and pos.track_id is not None
                        else None
                    ),
                }
            )

            n_processed += 1
            if n_processed % 10 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  frame {idx:5d}  t={t_now:5.1f}s  robots={len(tracks)}  "
                    f"events_total={len(events)}  "
                    f"({n_processed}/{max_frames_to_read // args.stride} a {n_processed / elapsed:.2f} fps)"
                )
    finally:
        writer.close()

    pipeline_time = time.time() - t_start

    # Guardar JSONs
    with open(args.out / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    events_json = [
        {
            "t": e.t,
            "type": e.type,
            "actors": e.actors,
            "position_mm": list(e.position_mm),
            "confidence": e.confidence,
            "meta": e.meta,
        }
        for e in events
    ]
    with open(args.out / "events.json", "w", encoding="utf-8") as f:
        json.dump(events_json, f, indent=2, ensure_ascii=False)

    # Visualizaciones
    print("\nGenerando visualizaciones...")
    all_robot_positions = np.array(
        [xy for tid, hist in robot_positions_mm.items() for _, xy in hist]
    )
    all_ball_positions = np.array([xy for _, xy in ball_positions_mm])
    if len(all_robot_positions) > 0:
        cv2.imwrite(
            str(args.out / "heatmap_robots.png"), render_heatmap(all_robot_positions)
        )
    if len(all_ball_positions) > 0:
        cv2.imwrite(
            str(args.out / "heatmap_ball.png"), render_heatmap(all_ball_positions)
        )

    trails_traj = {
        tid: np.array([xy for _, xy in hist])
        for tid, hist in robot_positions_mm.items()
    }
    cv2.imwrite(
        str(args.out / "trails.png"),
        render_trails(
            trails_traj,
            ball_trajectory=all_ball_positions if len(all_ball_positions) else None,
        ),
    )

    if robot_positions_mm:
        last_robots = {tid: hist[-1][1] for tid, hist in robot_positions_mm.items()}
        # Asignar team por mayoría histórica
        last_teams = {}
        for tid, hist in robot_team_history.items():
            non_none = [t for t in hist if t]
            if non_none:
                from collections import Counter

                last_teams[tid] = Counter(non_none).most_common(1)[0][0]
            else:
                last_teams[tid] = None
        last_ball = ball_positions_mm[-1][1] if ball_positions_mm else None
        cv2.imwrite(
            str(args.out / "voronoi_final.png"),
            render_voronoi(last_robots, last_teams, ball_mm=last_ball),
        )

    # Resumen
    from collections import Counter

    event_counts = Counter(e.type for e in events)
    summary = {
        "frames_processed": n_processed,
        "pipeline_time_s": pipeline_time,
        "effective_fps": n_processed / pipeline_time if pipeline_time > 0 else 0,
        "events_total": len(events),
        "events_by_type": dict(event_counts),
        "tracks_seen": len(robot_positions_mm),
        "ball_positions_world": len(ball_positions_mm),
    }
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs en: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
