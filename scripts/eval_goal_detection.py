"""Evaluación honesta de detección de goles vs ground truth manual.

Compara los goles detectados automáticamente (events.json) contra los goles
reales anotados por humano (manual_goals.yaml). Reporta Precision/Recall/F1
con tolerancia temporal configurable.

El detector rule-based suele disparar varias veces seguidas cuando el balón
entra (frames consecutivos con balón dentro de la portería). Para evitar
inflar artificialmente los falsos positivos, fusionamos detecciones consecutivas
dentro de --merge-window segundos en un único evento.

El matching es greedy 1-a-1: cada gol real se asigna a la detección más cercana
dentro de --tolerance segundos. Sin reuso (un detector sólo cuenta una vez).

Uso:
    python scripts/eval_goal_detection.py \\
        --runs-dir data/processed/runs \\
        --ground-truth tests/data/manual_goals.yaml \\
        --output reports/goal_validation.md \\
        --merge-window 3.0 \\
        --tolerance 2.0

    # Generar también detections_reference.yaml para acelerar la anotación humana:
    python scripts/eval_goal_detection.py --emit-detections-reference \\
        --runs-dir data/processed/runs \\
        --reference-out tests/data/detections_reference.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("eval_goal_detection")


@dataclass
class GoalDetection:
    """Detección agrupada (puede representar varios frames consecutivos)."""

    t_start: float
    t_end: float
    goal_color: str | None
    scoring_team: str | None
    n_raw_detections: int

    @property
    def t_center(self) -> float:
        return (self.t_start + self.t_end) / 2


@dataclass
class GoalGT:
    """Gol real anotado por humano."""

    t: float
    goal_color: str | None
    notes: str = ""


def load_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    return json.loads(events_path.read_text(encoding="utf-8"))


def extract_goal_detections(
    events: list[dict], merge_window: float
) -> list[GoalDetection]:
    """Filtra eventos type=goal y fusiona consecutivos dentro de merge_window."""
    raw_goals = sorted(
        (e for e in events if e.get("type") == "goal"), key=lambda e: e["t"]
    )
    if not raw_goals:
        return []

    clusters: list[list[dict]] = [[raw_goals[0]]]
    for ev in raw_goals[1:]:
        if ev["t"] - clusters[-1][-1]["t"] <= merge_window:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])

    result: list[GoalDetection] = []
    for cluster in clusters:
        meta_first = cluster[0].get("meta", {}) or {}
        # Si hay scoring_team consistente entre frames, lo conservamos
        teams = {
            (e.get("meta", {}) or {}).get("scoring_team")
            for e in cluster
            if (e.get("meta", {}) or {}).get("scoring_team")
        }
        result.append(
            GoalDetection(
                t_start=cluster[0]["t"],
                t_end=cluster[-1]["t"],
                goal_color=meta_first.get("goal_color"),
                scoring_team=next(iter(teams)) if len(teams) == 1 else None,
                n_raw_detections=len(cluster),
            )
        )
    return result


def match_greedy(
    gt: list[GoalGT], detections: list[GoalDetection], tolerance: float
) -> tuple[list[tuple[GoalGT, GoalDetection]], list[GoalGT], list[GoalDetection]]:
    """Matching greedy 1-a-1 por proximidad temporal.

    Retorna (matched_pairs, false_negatives, false_positives).
    """
    available = list(detections)
    matched: list[tuple[GoalGT, GoalDetection]] = []
    unmatched_gt: list[GoalGT] = []

    for g in sorted(gt, key=lambda x: x.t):
        # buscar la detección más cercana dentro de tolerancia
        candidates = [
            (abs(d.t_center - g.t), d)
            for d in available
            if abs(d.t_center - g.t) <= tolerance
        ]
        if not candidates:
            unmatched_gt.append(g)
            continue
        candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]
        matched.append((g, best))
        available.remove(best)

    return matched, unmatched_gt, available


def metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate(
    runs_dir: Path,
    ground_truth: dict[str, list[dict]],
    merge_window: float,
    tolerance: float,
) -> dict:
    """Evalúa cada video listado en ground_truth contra su events.json."""
    per_video: list[dict] = []
    tp_total = fp_total = fn_total = 0

    for video_name, gt_raw in ground_truth.items():
        run_dir = runs_dir / video_name
        events_path = run_dir / "events.json"
        events = load_events(events_path)
        detections = extract_goal_detections(events, merge_window)

        gt = [
            GoalGT(
                t=float(g["t"]),
                goal_color=g.get("goal_color"),
                notes=g.get("notes", ""),
            )
            for g in (gt_raw or [])
        ]
        matched, fn, fp = match_greedy(gt, detections, tolerance)
        tp = len(matched)
        p, r, f1 = metrics(tp, len(fp), len(fn))

        per_video.append(
            {
                "video": video_name,
                "events_path_exists": events_path.exists(),
                "gt_count": len(gt),
                "detections_merged": len(detections),
                "tp": tp,
                "fp": len(fp),
                "fn": len(fn),
                "precision": p,
                "recall": r,
                "f1": f1,
                "matches": [
                    {"gt_t": g.t, "det_t": d.t_center, "delta_s": d.t_center - g.t}
                    for g, d in matched
                ],
                "false_positives": [
                    {
                        "t_start": d.t_start,
                        "t_end": d.t_end,
                        "n_raw": d.n_raw_detections,
                    }
                    for d in fp
                ],
                "false_negatives": [{"t": g.t, "notes": g.notes} for g in fn],
            }
        )
        tp_total += tp
        fp_total += len(fp)
        fn_total += len(fn)

    p_agg, r_agg, f1_agg = metrics(tp_total, fp_total, fn_total)
    return {
        "params": {"merge_window_s": merge_window, "tolerance_s": tolerance},
        "per_video": per_video,
        "aggregate": {
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": p_agg,
            "recall": r_agg,
            "f1": f1_agg,
        },
    }


def render_markdown(report: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Validación manual de detección de goles\n")
    lines.append(
        f"> Parámetros: merge_window = {report['params']['merge_window_s']} s, "
        f"tolerance = {report['params']['tolerance_s']} s.\n"
    )
    lines.append("## Por video\n")
    lines.append(
        "| Video | GT real | Detectados (fusionados) | TP | FP | FN | P | R | F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in report["per_video"]:
        nota = "" if v["events_path_exists"] else " ⚠ sin events.json"
        lines.append(
            f"| {v['video']}{nota} | {v['gt_count']} | {v['detections_merged']} | "
            f"{v['tp']} | {v['fp']} | {v['fn']} | "
            f"{v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |"
        )

    agg = report["aggregate"]
    lines.append("\n## Agregado global\n")
    lines.append("| TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {agg['tp']} | {agg['fp']} | {agg['fn']} | "
        f"{agg['precision']:.3f} | {agg['recall']:.3f} | {agg['f1']:.3f} |"
    )

    lines.append("\n## Limitaciones declaradas\n")
    lines.append(
        "- Ground truth anotado por un humano del equipo con precisión ±2 s; no "
        "doble-anotación."
    )
    lines.append(
        "- Tolerancia de matching configurable (default ±2 s) — el reglamento no "
        "fija criterio temporal."
    )
    lines.append(
        f"- Fusión de detecciones consecutivas dentro de "
        f"{report['params']['merge_window_s']} s para evitar inflar FP por "
        "sobre-detección del rule-based."
    )
    lines.append(
        "- Métrica reportada como evidencia honesta; reemplazaría sn-trackeval "
        "HOTA/IDF1/MOTA (rechazado por falta de GT real de tracking)."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def emit_detections_reference(
    runs_dir: Path, out_path: Path, merge_window: float
) -> None:
    """Genera un YAML con detecciones fusionadas para acelerar la anotación humana."""
    reference: dict[str, list[dict]] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        events_path = run_dir / "events.json"
        if not events_path.exists():
            continue
        events = load_events(events_path)
        detections = extract_goal_detections(events, merge_window)
        reference[run_dir.name] = [
            {
                "t_center_s": round(d.t_center, 2),
                "t_start_s": round(d.t_start, 2),
                "t_end_s": round(d.t_end, 2),
                "n_raw_detections": d.n_raw_detections,
                "goal_color": d.goal_color,
                "scoring_team": d.scoring_team,
            }
            for d in detections
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Detecciones de gol agrupadas (autogenerado).\n"
        "# Úsalo como guía para revisar el video y llenar manual_goals.yaml.\n\n"
        + yaml.safe_dump(reference, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-dir", type=Path, default=ROOT / "data" / "processed" / "runs"
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "tests" / "data" / "manual_goals.yaml",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "goal_validation.md"
    )
    p.add_argument("--merge-window", type=float, default=3.0)
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument(
        "--emit-detections-reference",
        action="store_true",
        help="En lugar de evaluar, escribe el YAML de detecciones fusionadas.",
    )
    p.add_argument(
        "--reference-out",
        type=Path,
        default=ROOT / "tests" / "data" / "detections_reference.yaml",
    )
    args = p.parse_args()

    if args.emit_detections_reference:
        emit_detections_reference(args.runs_dir, args.reference_out, args.merge_window)
        logger.info("Referencia de detecciones escrita en %s", args.reference_out)
        return 0

    if not args.ground_truth.exists():
        logger.error("No existe ground truth: %s", args.ground_truth)
        return 1

    gt_data = yaml.safe_load(args.ground_truth.read_text(encoding="utf-8")) or {}
    report = evaluate(args.runs_dir, gt_data, args.merge_window, args.tolerance)
    render_markdown(report, args.output)
    logger.info("Reporte escrito en %s", args.output)
    agg = report["aggregate"]
    logger.info(
        "Agregado: TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f",
        agg["tp"],
        agg["fp"],
        agg["fn"],
        agg["precision"],
        agg["recall"],
        agg["f1"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
