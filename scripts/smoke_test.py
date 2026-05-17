"""Smoke test reproducible — valida toda la instalación en ~30 segundos.

Verifica que cada componente del pipeline funciona end-to-end sobre un
frame de prueba. Imprime versiones, GPU disponible, métricas por etapa
y exit code 0 si todo pasa.

Cumple § 3.2.2 (reproducibilidad) y sirve como CI manual antes de
demos críticos.

Uso:
    python scripts/smoke_test.py

Salida esperada:
    [✓] Python / torch / transformers / GPU
    [✓] SAM 3.1 carga
    [✓] Inferencia ball + robots
    [✓] HSV ball detector
    [✓] OC-SORT
    [✓] Kalman ball
    [✓] AdaptiveTeamClassifier
    [✓] Homografía
    [✓] Eventos rule-based
    [✓] MatchStats
    [✓] Visualizaciones (heatmap/trails/voronoi)
    OK
"""

from __future__ import annotations

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


def stage(name: str):
    print(f"[ ] {name}...", end="", flush=True)
    return time.time()


def ok(t0: float):
    print(f"\r[✓] {time.time() - t0:6.2f}s  ", end="")


def fail(msg: str):
    print(f"\r[✗] {msg}")
    sys.exit(1)


def main() -> int:
    # 1. Versiones
    t = stage("Versiones")
    import torch
    import transformers
    import supervision
    import boxmot
    import scipy

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "supervision": supervision.__version__,
        "boxmot": boxmot.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            if torch.cuda.is_available()
            else None
        ),
    }
    ok(t)
    print(versions)

    # 2. Frame de prueba (sintético si no existe el real)
    frame_path = ROOT / "data" / "processed" / "sample_frames" / "frame_24s.jpg"
    if frame_path.exists():
        frame = cv2.imread(str(frame_path))
    else:
        frame = np.full((720, 1280, 3), (60, 140, 50), dtype=np.uint8)
        cv2.circle(frame, (800, 360), 14, (40, 130, 255), -1)

    # 3. SAM 3.1
    t = stage("SAM 3.1 load")
    from src.segmentation.sam3 import load_model, segment_with_text
    from src.segmentation.prompts import BALL, PROMPT_ALL_ROBOTS

    proc, mdl = load_model(half_precision=True)
    ok(t)

    t = stage("SAM 3.1 inference")
    seg = segment_with_text(frame, [BALL, PROMPT_ALL_ROBOTS], proc, mdl, threshold=0.2)
    ok(t)
    print(f"ball={len(seg[BALL])} robots={len(seg[PROMPT_ALL_ROBOTS])}")

    # 4. HSV ball fallback
    t = stage("HSV ball")
    from src.segmentation.baselines import find_ball_centroid

    det = find_ball_centroid(frame)
    ok(t)
    print(f"found={det.found} cx={det.cx:.0f} cy={det.cy:.0f}")

    # 5. OC-SORT
    t = stage("OC-SORT")
    from src.tracking.robots import RobotTracker

    rt = RobotTracker(min_hits=1)
    dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=float)
    out = rt.update(dets, frame)
    ok(t)
    print(f"tracks={len(out)}")

    # 6. Kalman ball
    t = stage("Kalman ball")
    from src.tracking.ball import BallTracker

    bt = BallTracker(dt=1 / 30)
    state = bt.update(np.array([100.0, 200.0]))
    ok(t)
    print(f"init source={state.source}")

    # 7. AdaptiveTeamClassifier
    t = stage("AdaptiveTeamClassifier")
    from src.tracking.reid import AdaptiveTeamClassifier

    clf = AdaptiveTeamClassifier(warmup_frames=2)
    for _ in range(3):
        clf.observe(1, 145)
        clf.observe(2, 60)
        clf.end_frame()
    a, b = clf.assign(1), clf.assign(2)
    ok(t)
    print(f"team(1)={a} team(2)={b}")

    # 8. Homografía
    t = stage("Homografía")
    from src.utils.calib import compute_homography, project_points

    corners = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=float)
    H = compute_homography(corners)
    pt = project_points(np.array([[300.0, 250.0]]), H)
    ok(t)
    print(f"center → {pt[0]}")

    # 9. Detección de field corners
    t = stage("Field corners")
    from src.utils.field_detect import detect_field_corners

    res = detect_field_corners(frame)
    ok(t)
    print(f"success={res.success}")

    # 10. Eventos
    t = stage("Eventos rule-based")
    from src.events.rules import detect_collisions, detect_kick

    v = detect_kick(np.array([0, 0]), np.array([100, 0]), 1 / 30)
    cols = detect_collisions({1: np.array([0.0, 0.0]), 2: np.array([20.0, 0.0])})
    ok(t)
    print(f"kick_v={v:.0f}mm/s collisions={len(cols)}")

    # 11. MatchStats
    t = stage("MatchStats")
    from src.metrics.stats import MatchStats

    s = MatchStats()
    s.register_event("goal", team="A")
    s.update_possession("A", 1.0)
    s.update_possession("B", 0.5)
    ok(t)
    print(f"score A={s.score_a} pct_A={s.possession_pct_a:.0f}%")

    # 12. Visualizaciones
    t = stage("Visualizaciones")
    from src.viz.heatmap import render_heatmap
    from src.viz.trails import render_trails
    from src.viz.voronoi import render_voronoi

    pos = np.array([[1000.0, 800.0], [1100.0, 850.0]])
    hm = render_heatmap(pos)
    trails = render_trails({1: pos}, ball_trajectory=pos)
    vor = render_voronoi({1: pos[0], 2: pos[1]}, {1: "A", 2: "B"})
    ok(t)
    print(f"hm={hm.shape} trails={trails.shape} voronoi={vor.shape}")

    print("\n✅ SMOKE TEST OK — todos los componentes funcionan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
