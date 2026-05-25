"""Narrativa automática del partido en lenguaje natural (español).

Lee `events.json` + `summary.json` + `tracks.json` de un run del
pipeline y produce un comentario cronológico estructurado:

    commentary.md    — markdown con secciones (resumen ejecutivo,
                       cronología minuto a minuto, estadísticas finales)
    commentary.txt   — texto plano (útil para reel/subtítulos)

Implementación: plantillas estructuradas en español, sin requerir un
LLM externo. Determinista, rápido, idéntico para el mismo run. Si en
el futuro se quiere mejorar la prosa, esto sirve como prompt
estructurado para Claude/GPT en un script separado.

Uso:
    python scripts/render_commentary.py --run data/processed/runs/IMG_9821
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EVENT_VERBS = {
    "kick": "remate",
    "pass": "pase",
    "interception": "intercepción",
    "retention": "retención prolongada",
    "no_progress": "fase sin progreso",
    "collision": "choque entre robots",
    "damaged": "robot inmóvil prolongado",
    "goal": "GOL",
}

EVENT_DESCRIPTIONS = {
    "kick": (
        "el balón cambia bruscamente de velocidad (más de 0.5 m/s) — "
        "señal de un golpe directo"
    ),
    "pass": (
        "el balón cambia de poseedor dentro del mismo equipo tras "
        "recorrer al menos 30 cm"
    ),
    "interception": ("el balón cambia de poseedor hacia un robot del equipo rival"),
    "retention": (
        "un mismo robot conserva el balón más de 1.5 s a menos de 90 mm "
        "(infracción según AutoRefs RoboCup SSL)"
    ),
    "no_progress": ("el balón se mueve menos de 50 mm durante 5 segundos consecutivos"),
    "collision": "dos robots quedan a menos de 50 mm de distancia",
    "damaged": (
        "un robot permanece prácticamente inmóvil (menos de 20 mm/s) "
        "durante más de 60 segundos"
    ),
    "goal": ("el balón entra dentro del recuadro de la portería detectada por color"),
}


def _fmt_time(t: float) -> str:
    """Convierte segundos a 'mm:ss.s'."""
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:04.1f}"


def _event_phrase(ev: dict) -> str:
    """Frase corta describiendo un evento individual."""
    t = float(ev.get("t", 0))
    etype = ev.get("type", "?")
    meta = ev.get("meta") or {}
    actors = ev.get("actors") or []
    base = f"**{_fmt_time(t)}** — {EVENT_VERBS.get(etype, etype).upper()}"
    if etype == "goal":
        team = meta.get("scoring_team") or "?"
        color = meta.get("goal_color", "?")
        suffix = f" del equipo **{team}** sobre la portería {color}"
        if actors:
            suffix += f" (último poseedor: track {actors[0]})"
        return base + suffix
    if etype == "kick":
        v = meta.get("velocity_mm_s", 0)
        return base + f" — velocidad {v:.0f} mm/s"
    if etype in ("pass", "interception"):
        f_team = meta.get("from_team") or "?"
        t_team = meta.get("to_team") or "?"
        if actors and len(actors) >= 2:
            return base + (
                f" — del track {actors[0]} (equipo {f_team}) al track "
                f"{actors[1]} (equipo {t_team})"
            )
        return base + f" — entre equipos {f_team} → {t_team}"
    if etype == "retention":
        dur = meta.get("duration_s", 0)
        team = meta.get("team") or "?"
        actor = actors[0] if actors else "?"
        return base + (
            f" — track {actor} (equipo {team}) retiene el balón {dur:.1f} segundos"
        )
    if etype == "collision":
        if len(actors) >= 2:
            return base + f" — tracks {actors[0]} y {actors[1]}"
        d = meta.get("distance_mm", 0)
        return base + f" — distancia {d:.0f} mm"
    if etype == "no_progress":
        return base + " — el balón apenas se desplazó en 5 s"
    if etype == "damaged":
        actor = actors[0] if actors else "?"
        return base + f" — track {actor}"
    return base


def _summarize_team(label: str, summary: dict) -> str:
    pa = summary.get("possession_pct", {}).get(label, 0)
    goals = summary.get("score", {}).get(label, 0)
    robots = summary.get("robots", {})
    team_robots = [tid for tid, r in robots.items() if r.get("team") == label]
    gol_palabra = "gol" if goals == 1 else "goles"
    track_palabra = "track único" if len(team_robots) == 1 else "tracks únicos"
    return (
        f"- Equipo **{label}**: {goals} {gol_palabra} · "
        f"{pa:.0f}% de posesión · {len(team_robots)} {track_palabra} "
        "identificados"
    )


def _ball_stats(summary: dict) -> str:
    ball = summary.get("ball", {})
    dist_m = ball.get("distance_mm", 0) / 1000
    vmax_ms = ball.get("max_speed_mm_s", 0) / 1000
    vavg_ms = ball.get("avg_speed_mm_s", 0) / 1000
    # La cancha mide 2.19 m × 1.58 m. Distancias > 50 m o velocidades >
    # 5 m/s en 60 s son artefacto de saltos por oclusión del balón, no
    # movimiento físico real.
    note = ""
    if dist_m > 50 or vmax_ms > 5:
        note = (
            " *(las cifras incluyen artefactos por saltos del tracking "
            "cuando el balón se ocluye; el valor físico real es menor)*"
        )
    return (
        f"- Balón: distancia recorrida {dist_m:.1f} m · "
        f"velocidad máxima {vmax_ms:.2f} m/s · "
        f"velocidad promedio {vavg_ms:.2f} m/s" + note
    )


def _top_robots(summary: dict, n: int = 5) -> list[str]:
    robots = summary.get("robots", {})
    items = sorted(
        robots.items(),
        key=lambda kv: kv[1].get("distance_mm", 0),
        reverse=True,
    )[:n]
    return [
        f"  - track {tid} (equipo {r.get('team') or '?'}): "
        f"distancia {r.get('distance_mm', 0) / 1000:.2f} m · "
        f"vel. máx. {r.get('max_speed_mm_s', 0) / 1000:.2f} m/s"
        for tid, r in items
    ]


def build_commentary(run_dir: Path) -> tuple[str, str]:
    """Devuelve (markdown, plain_text) del comentario completo."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))

    video = summary.get("video", "?")
    dur = summary.get("duration_s", 0)
    sa = summary.get("score", {}).get("A", 0)
    sb = summary.get("score", {}).get("B", 0)
    n_events = summary.get("events_total", len(events))
    counts = summary.get("events_by_type", {})

    # Construir cronología
    important = [
        e
        for e in events
        if e["type"]
        in ("goal", "pass", "interception", "retention", "collision", "damaged")
    ]
    important.sort(key=lambda e: e["t"])

    # Markdown
    md = []
    md.append(f"# Crónica del partido — `{video}`\n")
    md.append(
        "> Generado automáticamente por el pipeline de **AJOLOTES FC** "
        "a partir de `events.json` y `summary.json`.\n"
    )
    md.append("## Resumen ejecutivo\n")
    md.append(
        f"Partido de **{dur:.1f} segundos** analizado con SAM 3.1 "
        f"fine-tuned (LoRA, mIoU 0.912 sobre nuestro dataset). "
        f"El pipeline detectó **{n_events} eventos** "
        f"distribuidos en {len(counts)} categorías.\n"
    )
    md.append(f"**Marcador final: A {sa} — {sb} B**\n")
    md.append(_summarize_team("A", summary))
    md.append(_summarize_team("B", summary))
    md.append("\n" + _ball_stats(summary))
    md.append("\n### Top 5 robots por distancia recorrida\n")
    md.extend(_top_robots(summary, 5))

    md.append("\n## Cronología destacada\n")
    if not important:
        md.append("_(No se detectaron eventos destacados.)_\n")
    else:
        for ev in important[:40]:
            md.append(_event_phrase(ev))
    if len(important) > 40:
        md.append(f"\n_... y {len(important) - 40} eventos destacados más._\n")

    md.append("\n## Desglose por tipo de evento\n")
    md.append("| Evento | Conteo | Definición operacional |")
    md.append("|---|---|---|")
    for etype, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        desc = EVENT_DESCRIPTIONS.get(etype, "—")
        md.append(f"| {EVENT_VERBS.get(etype, etype)} | {count} | {desc} |")

    md.append("\n## Limitaciones honestas\n")
    posa = summary.get("possession_pct", {}).get("A", 0)
    posb = summary.get("possession_pct", {}).get("B", 0)
    if (posa > 95) or (posb > 95):
        md.append(
            "- La estimación de posesión está fuertemente sesgada hacia "
            "un equipo, probablemente porque la cámara de espectador "
            "enfoca a un solo lado del campo durante la mayoría del clip. "
            "El sistema sólo puede atribuir posesión a los robots que "
            "efectivamente ve.\n"
        )
    if counts.get("goal", 0) > 6:
        md.append(
            "- El conteo de goles parece elevado. Puede deberse a drift "
            "del bbox de portería al moverse la cámara (el flag "
            "`--recalib-every` mitiga pero no elimina este efecto en "
            "tomas muy oblicuas).\n"
        )
    if counts.get("kick", 0) > 100:
        md.append(
            "- El detector de remate usa un umbral fijo de velocidad "
            "(0.5 m/s en un frame); en clips con motion blur frecuente "
            "se sobre-detecta. Es esperado y documentado.\n"
        )

    md.append("\n---\n")
    md.append(
        "_Las definiciones operacionales de cada evento están en "
        "`src/events/rules.py` y siguen el espíritu de los AutoRefs de "
        "RoboCup SSL._\n"
    )
    markdown = "\n".join(md)

    # Versión texto plano (para reel/subtítulos)
    lines = [
        f"Crónica del partido — {video}",
        f"Marcador final: A {sa} - {sb} B",
        f"Duración: {dur:.1f} s",
        f"Eventos: {n_events}",
        f"Posesión: A {posa:.0f}% / B {posb:.0f}%",
        "",
        "Momentos destacados:",
    ]
    for ev in important[:15]:
        phrase = _event_phrase(ev).replace("**", "")
        lines.append(f"  {phrase}")
    plain = "\n".join(lines)
    return markdown, plain


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()

    if not (args.run / "events.json").exists():
        print(f"ERROR: {args.run / 'events.json'} no existe")
        return 1

    md, plain = build_commentary(args.run)
    (args.run / "commentary.md").write_text(md, encoding="utf-8")
    (args.run / "commentary.txt").write_text(plain, encoding="utf-8")
    print(f"OK commentary.md ({len(md)} chars)")
    print(f"OK commentary.txt ({len(plain)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
