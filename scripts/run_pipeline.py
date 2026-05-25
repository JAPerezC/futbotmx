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
    detect_collisions,
    detect_kick,
    detect_pass_or_interception,
    is_damaged_robot,
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
from src.metrics.stats import MatchStats
from src.tracking.ball import BallTracker
from src.tracking.reid import AdaptiveTeamClassifier, _dominant_feature  # noqa: F401
from src.tracking.robots import RobotTracker
from src.utils.calib import (
    compute_homography,
    project_points,
)
from src.utils.io import VideoWriter, probe, read_frames
from src.viz.dashboard import render_dashboard
from src.viz.heatmap import render_heatmap, render_heatmap_by_team
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
    "pass": (200, 100, 255),
    "interception": (255, 100, 200),
    "collision": (0, 0, 255),
    "possession": (255, 255, 255),
}


def draw_stats_banner(image, stats: MatchStats, t_s: float, duration_s: float):
    """Banner persistente arriba con score + posesión + tiempo.

    Sirve como narrativa visual del partido en cualquier frame (§ 3.5.2).
    Tamaño escalado para video portrait 1360x1808 — visible a distancia.
    """
    h, w = image.shape[:2]
    bh = 140
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.line(image, (0, bh), (w, bh), (255, 215, 0), 3)
    # Score grande
    score = f"A {stats.score_a} - {stats.score_b} B"
    cv2.putText(
        image,
        score,
        (20, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    # Posesión
    pa, pb = stats.possession_pct_a, stats.possession_pct_b
    pos_text = f"pos  A {pa:.0f}%   B {pb:.0f}%"
    cv2.putText(
        image,
        pos_text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    # Tiempo
    time_text = f"t = {t_s:5.1f} / {duration_s:5.1f}s"
    (tw, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(
        image,
        time_text,
        (w - tw - 20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 215, 0),
        2,
        cv2.LINE_AA,
    )
    # Conteo eventos compacto, alineado a la derecha
    counts = stats.event_counts
    summary_parts = [
        f"{n}:{counts[n]}"
        for n in (
            "goal",
            "kick",
            "pass",
            "interception",
            "retention",
            "collision",
            "no_progress",
        )
        if counts.get(n, 0) > 0
    ]
    if summary_parts:
        evt_text = " | ".join(summary_parts)
        (ew, _), _ = cv2.getTextSize(evt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(
            image,
            evt_text,
            (w - ew - 20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )


def overlay_polygon(image, corners, color=(0, 200, 255)):
    """Dibuja el cuadrilátero de las 4 esquinas, clippeado al frame.

    Evita el efecto "triángulo estirado" cuando una esquina está fuera
    del frame: cv2.clipLine recorta cada segmento al rectángulo visible.
    """
    if corners is None or len(corners) < 4:
        return
    h, w = image.shape[:2]
    pts = corners.astype(np.int32)
    rect = (0, 0, w, h)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        visible, q1, q2 = cv2.clipLine(rect, p1, p2)
        if visible:
            cv2.line(image, q1, q2, color, 2, cv2.LINE_AA)
    # Marca las esquinas que SÍ están dentro del frame
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (int(x), int(y)), 8, color, -1)
            cv2.circle(image, (int(x), int(y)), 8, (0, 0, 0), 2)


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
    """Banner INFERIOR con eventos recientes (últimos 0.6s).

    Movido a la parte inferior para no tapar el banner persistente de stats.
    Sin overlay translúcido pesado: solo franja delgada + texto a color.
    """
    if not recent_events:
        return
    h, w = image.shape[:2]
    banner_h = 70
    y0 = h - banner_h
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    cv2.line(image, (0, y0), (w, y0), (255, 215, 0), 3)
    x = 20
    for ev in recent_events:
        color = EVENT_COLOR.get(ev.type, (255, 255, 255))
        text = f"{ev.type.upper()} {ev.t:.1f}s"
        cv2.putText(
            image,
            text,
            (x, y0 + 48),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )
        x += int(cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)[0][0]) + 40
        if x > w - 100:
            break


def draw_possession_info(image, pos, has_event_banner: bool = False):
    if pos is None or pos.track_id is None:
        return
    h, w = image.shape[:2]
    text = (
        f"posesion id={pos.track_id} team={pos.team or '?'} d={pos.distance_mm:.0f}mm"
    )
    y = h - 90 if has_event_banner else h - 30
    cv2.putText(
        image,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# -------- calibración --------


def calibrate_homography(video_path, max_frames_to_try=60):
    """Calibra esquinas REALES del campo (líneas blancas) + porterías por color.

    Usa `detect_field_geometry` v2: Hough sobre líneas blancas dentro del
    fieltro verde + fallback a convex hull. Devuelve también los bbox de
    las porterías AMARILLA y AZUL detectadas, que se usan para detectar
    goles reales (balón dentro del bbox de portería).

    Returns:
        (H, corners, goals_img, idx)
            H: 3x3 homografía imagen → mundo (mm)
            corners: (4,2) esquinas en orden TL,TR,BR,BL
            goals_img: lista de GoalDetection (0..2)
            idx: índice del frame de calibración
    """
    from src.utils.field_detect_v2 import detect_field_geometry

    print("  buscando frame para calibrar (líneas blancas + porterías)...")
    tried = 0
    best_geom = None
    best_idx = 0
    best_inside = -1
    last_frame = None
    for idx, frame in read_frames(video_path, stride=5):
        tried += 1
        last_frame = frame
        h_img, w_img = frame.shape[:2]
        geom = detect_field_geometry(frame)
        if geom.success:
            inside = sum(
                1 for x, y in geom.corners_img if 0 <= x < w_img and 0 <= y < h_img
            )
            method = geom.debug.get("method", "?")
            goal_colors = [g.color for g in geom.goals]
            # Privilegiar frames con MÁS porterías detectadas, luego más esquinas.
            score = len(goal_colors) * 10 + inside
            if score > best_inside:
                best_inside = score
                best_geom = geom
                best_idx = idx
                # Si tenemos 2 porterías y 4 esquinas, parar pronto.
                if len(goal_colors) == 2 and inside == 4:
                    print(
                        f"  OK [{method}] esquinas={inside}/4 goals={goal_colors} "
                        f"frame idx={idx} (óptimo)"
                    )
                    H = compute_homography(geom.corners_img)
                    return H, geom.corners_img, geom.goals, idx
        if tried >= max_frames_to_try:
            break

    if best_geom is not None:
        method = best_geom.debug.get("method", "?")
        inside = sum(
            1
            for x, y in best_geom.corners_img
            if 0 <= x < last_frame.shape[1] and 0 <= y < last_frame.shape[0]
        )
        goal_colors = [g.color for g in best_geom.goals]
        print(
            f"  OK [{method}] esquinas={inside}/4 goals={goal_colors} frame idx={best_idx}"
        )
        H = compute_homography(best_geom.corners_img)
        return H, best_geom.corners_img, best_geom.goals, best_idx

    print("  WARN: no se pudo calibrar automáticamente; usando esquinas por defecto")
    h, w = last_frame.shape[:2]
    fallback = np.array(
        [
            [w * 0.1, h * 0.2],
            [w * 0.9, h * 0.2],
            [w * 0.95, h * 0.9],
            [w * 0.05, h * 0.9],
        ],
        dtype=np.float64,
    )
    return compute_homography(fallback), fallback, [], 0


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
    p.add_argument(
        "--recalib-every",
        type=int,
        default=60,
        help=(
            "recalibrar porterías por color cada N frames procesados; 0 desactiva. "
            "Compensa el drift del bbox cuando la cámara se mueve (defecto 60)."
        ),
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
    H, corners, goals_img, _ = calibrate_homography(args.video)
    goals_by_color: dict[str, np.ndarray] = {g.color: g.bbox_xyxy for g in goals_img}

    # Cargar SAM 3
    print(f"Cargando SAM 3.1 ({args.model})...")
    t0 = time.time()
    processor, model = load_model(args.model, device=args.device)
    print(f"  cargado en {time.time() - t0:.1f}s")

    robot_tracker = RobotTracker(min_hits=1, det_thresh=0.2, iou_threshold=0.3)
    ball_tracker = BallTracker(dt=args.stride / meta.fps, max_missing_frames=20)
    # Classifier v2: warmup 30 frames, recompute online cada 15, ventana de
    # votación 20, peso de saturación 0.5. Resuelve colapso 100%/0%.
    team_clf = AdaptiveTeamClassifier(
        warmup_frames=30,
        recompute_every=15,
        vote_window=20,
        hue_separation_min=20,
        sat_weight=0.5,
    )
    match_stats = MatchStats()

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
    # Para pase/intercepción
    prev_owner_track: int | None = None
    prev_owner_team: str | None = None
    prev_ball_xy_owner_loss: np.ndarray | None = None
    # Para deduplicar colisiones
    last_collision_t: dict[tuple[int, int], float] = {}

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
        # Recalibración online de porterías: cada N frames procesados,
        # re-detectamos las porterías por color y actualizamos goals_by_color.
        # Resuelve el drift cuando la cámara se mueve (el bbox calibrado en
        # frame 0 deja de coincidir con la portería real).
        recalib_every_n = args.recalib_every
        from src.utils.field_detect_v2 import detect_goals_by_color

        for idx, frame in read_frames(args.video, stride=args.stride):
            if idx >= max_frames_to_read:
                break
            t_now = idx / meta.fps

            # 0. Recalibración online (no recalculamos esquinas, solo porterías)
            if (
                recalib_every_n > 0
                and n_processed > 0
                and n_processed % recalib_every_n == 0
            ):
                new_goals = detect_goals_by_color(frame)
                if new_goals:
                    goals_by_color = {g.color: g.bbox_xyxy for g in new_goals}

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
                feat = _dominant_feature(frame, tr.bbox_xyxy)
                team_clf.observe(tr.track_id, feat)
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
                        "team_hue": feat[0] if feat else None,
                        "team_sat": feat[1] if feat else None,
                        "confidence": tr.confidence,
                    }
                )
            team_clf.end_frame()

            ball_mm = None
            if ball_state.found:
                xy_img = np.array([[ball_state.cx, ball_state.cy]])
                ball_mm = project_points(xy_img, H)[0]
                ball_positions_mm.append((t_now, ball_mm))
                match_stats.update_ball_position(ball_mm, t_now)

            # Acumular posiciones de robots en MatchStats
            for tr in tracks:
                w_xy = robots_mm_this_frame.get(tr.track_id)
                if w_xy is not None:
                    match_stats.update_robot_position(
                        tr.track_id, w_xy, teams_this_frame.get(tr.track_id), t_now
                    )

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
                    match_stats.register_event("kick")

            # Gol: balón DENTRO del bbox de una portería REAL detectada por color.
            # Mucho más confiable que el ROI virtual en mundo mm: usa la
            # portería visible (amarilla/azul) detectada en la calibración.
            if ball_state.found and goals_by_color:
                ball_px = np.array([ball_state.cx, ball_state.cy], dtype=np.float64)
                for goal_color, bbox in goals_by_color.items():
                    x1, y1, x2, y2 = bbox
                    if x1 <= ball_px[0] <= x2 and y1 <= ball_px[1] <= y2:
                        # Anti-spam: 1 gol cada 3 segundos
                        if (
                            events
                            and events[-1].type == "goal"
                            and t_now - events[-1].t < 3.0
                        ):
                            break
                        last_event_id += 1
                        scoring_team = prev_owner_team
                        ev = Event(
                            t=t_now,
                            type="goal",
                            actors=[prev_owner_track] if prev_owner_track else [],
                            position_mm=(
                                float(ball_mm[0]) if ball_mm is not None else 0.0,
                                float(ball_mm[1]) if ball_mm is not None else 0.0,
                            ),
                            confidence=1.0,
                            meta={
                                "goal_color": goal_color,
                                "scoring_team": scoring_team,
                                "ball_px": (float(ball_px[0]), float(ball_px[1])),
                            },
                        )
                        events.append(ev)
                        current_events.append(ev)
                        match_stats.register_event("goal", team=scoring_team)
                        break

            # Colisiones entre robots (anti-spam 0.5s por par)
            for a, b, d in detect_collisions(robots_mm_this_frame):
                key = (a, b)
                last_t = last_collision_t.get(key, -1e9)
                if t_now - last_t > 0.5:
                    last_collision_t[key] = t_now
                    last_event_id += 1
                    ev = Event(
                        t=t_now,
                        type="collision",
                        actors=[a, b],
                        position_mm=(
                            float(
                                (
                                    robots_mm_this_frame[a][0]
                                    + robots_mm_this_frame[b][0]
                                )
                                / 2
                            ),
                            float(
                                (
                                    robots_mm_this_frame[a][1]
                                    + robots_mm_this_frame[b][1]
                                )
                                / 2
                            ),
                        ),
                        confidence=1.0 - d / 50.0,
                        meta={"distance_mm": d},
                    )
                    events.append(ev)
                    current_events.append(ev)
                    match_stats.register_event("collision")

            # Posesión
            pos = None
            current_owner_track = None
            current_owner_team = None
            if ball_mm is not None and robots_mm_this_frame:
                pos = closest_robot_possession(
                    ball_mm, robots_mm_this_frame, teams_this_frame
                )
                if pos.track_id is not None and pos.distance_mm < POSSESSION_RADIUS_MM:
                    current_owner_track = pos.track_id
                    current_owner_team = pos.team
                # Acumular tiempo de posesión
                match_stats.update_possession(
                    current_owner_team, args.stride / meta.fps
                )

                # Retención: si mismo robot por > T segundos
                if current_owner_track is not None:
                    if (
                        ball_retention_robot == current_owner_track
                        and ball_retention_start is not None
                    ):
                        elapsed = t_now - ball_retention_start
                        if elapsed > 1.5:
                            last_event_id += 1
                            ev = Event(
                                t=t_now,
                                type="retention",
                                actors=[current_owner_track],
                                position_mm=(float(ball_mm[0]), float(ball_mm[1])),
                                confidence=min(elapsed / 3.0, 1.0),
                                meta={
                                    "duration_s": elapsed,
                                    "team": current_owner_team,
                                },
                            )
                            events.append(ev)
                            current_events.append(ev)
                            match_stats.register_event("retention")
                            ball_retention_start = None
                    else:
                        ball_retention_robot = current_owner_track
                        ball_retention_start = t_now
                else:
                    ball_retention_robot = None
                    ball_retention_start = None

            # Pase / Intercepción al detectar cambio de poseedor
            if (
                current_owner_track is not None
                and prev_owner_track is not None
                and current_owner_track != prev_owner_track
            ):
                kind = detect_pass_or_interception(
                    prev_owner_track,
                    prev_owner_team,
                    current_owner_track,
                    current_owner_team,
                    prev_ball_xy_owner_loss,
                    ball_mm,
                )
                if kind in ("pass", "interception"):
                    last_event_id += 1
                    ev = Event(
                        t=t_now,
                        type=kind,
                        actors=[prev_owner_track, current_owner_track],
                        position_mm=(float(ball_mm[0]), float(ball_mm[1])),
                        confidence=0.85,
                        meta={
                            "from_team": prev_owner_team,
                            "to_team": current_owner_team,
                        },
                    )
                    events.append(ev)
                    current_events.append(ev)
                    match_stats.register_event(kind)

            # Actualizar memoria de poseedor anterior
            if current_owner_track is not None:
                if current_owner_track != prev_owner_track:
                    prev_owner_track = current_owner_track
                    prev_owner_team = current_owner_team
                    prev_ball_xy_owner_loss = (
                        ball_mm.copy() if ball_mm is not None else None
                    )
                else:
                    # mismo dueño, actualizar última posición conocida
                    if ball_mm is not None:
                        prev_ball_xy_owner_loss = ball_mm.copy()

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
                        match_stats.register_event("no_progress")

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
                        match_stats.register_event("damaged")

            match_stats.end_frame(t_now)

            # 6. Anotar frame
            annotated = frame.copy()
            # Cuadrilátero del campo (esquinas reales por líneas blancas)
            overlay_polygon(annotated, corners, color=(0, 255, 0))
            # Porterías reales detectadas por color
            for goal_color, bbox in goals_by_color.items():
                x1, y1, x2, y2 = [int(v) for v in bbox]
                box_color = (0, 255, 255) if goal_color == "yellow" else (255, 100, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 4)
                cv2.putText(
                    annotated,
                    f"GOAL {goal_color.upper()}",
                    (x1, max(40, y1 - 12)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )
            for tr in tracks:
                draw_track(
                    annotated,
                    tr.bbox_xyxy,
                    tr.track_id,
                    teams_this_frame.get(tr.track_id),
                    tr.confidence,
                )
            draw_ball(annotated, ball_state)
            recent = [e for e in events if t_now - e.t < 0.6]
            draw_possession_info(annotated, pos, has_event_banner=bool(recent))
            # Banner inferior con eventos recientes (PRIMERO, no pisa stats)
            if recent:
                draw_event_banner(annotated, recent)
            # Banner persistente arriba (score + posesión + tiempo) — SIEMPRE al final
            draw_stats_banner(annotated, match_stats, t_now, duration)
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

    # Heatmaps por equipo
    team_heatmaps = render_heatmap_by_team(dict(match_stats.positions_by_team_mm))
    for team, img_hm in team_heatmaps.items():
        cv2.imwrite(str(args.out / f"heatmap_team_{team}.png"), img_hm)

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

    # Resumen ampliado (incluye stats agregadas + run metadata)
    summary = {
        "video": args.video.name,
        "duration_s": duration,
        "stride": args.stride,
        "fps_in": meta.fps,
        "fps_out": fps_out,
        "frames_processed": n_processed,
        "pipeline_time_s": pipeline_time,
        "effective_fps": n_processed / pipeline_time if pipeline_time > 0 else 0,
        "events_total": len(events),
        **match_stats.to_dict(),
    }
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Dashboard HTML
    try:
        dashboard_path = render_dashboard(
            summary,
            events_json,
            record["frames"],
            output_path=args.out / "dashboard.html",
            video_name=args.video.name,
        )
        print(f"  dashboard: {dashboard_path}")
    except Exception as e:
        print(f"  WARN: dashboard falló: {e}")

    # Minimap top-down animado (vista cenital sintetizada con la homografía)
    try:
        import subprocess

        mm_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "render_minimap_video.py"),
            "--run",
            str(args.out),
        ]
        mm_res = subprocess.run(mm_cmd, capture_output=True, text=True)
        if mm_res.returncode == 0:
            print(f"  minimap: {args.out / 'minimap.mp4'}")
        else:
            print(f"  WARN minimap: {mm_res.stderr[-200:]}")
    except Exception as e:
        print(f"  WARN minimap falló: {e}")

    print("\n=== RESUMEN ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs en: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
