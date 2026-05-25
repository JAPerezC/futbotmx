"""Capa LLM sobre commentary.md: prosa narrativa estilo radio deportivo.

Toma el commentary determinista generado por scripts/render_commentary.py y lo
refina con Claude Sonnet 4.6 para producir:

    commentary_pro.md             — crónica narrativa lista para publicación
    commentary_pro_subtitles.txt  — 10-12 subtítulos cortos para el reel

Diseño:
- Prompt cacheable: el system prompt va con cache_control=ephemeral para que
  ejecuciones sobre varios videos en sesión amorticen el costo del prefix.
- Output forzado a JSON estructurado para parseo robusto.
- Fallback explícito si ANTHROPIC_API_KEY no está presente: copia commentary.md
  con un encabezado advertencia, manteniendo el pipeline operacional.

Uso:
    python scripts/render_commentary_llm.py --run data/processed/runs/IMG_9821
    python scripts/render_commentary_llm.py --run ... --model claude-sonnet-4-6
    python scripts/render_commentary_llm.py --run ... --no-llm  # forzar fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("render_commentary_llm")

SYSTEM_PROMPT = """Eres un narrador deportivo profesional en español mexicano \
neutro, con cadencia de radio futbolera. Tu tarea es refinar la crónica \
determinista de un partido de fútbol robótico (Copa FutBotMX 2026) para que \
suene natural y atractiva sin perder rigor.

REGLAS ESTRICTAS:
1. NO inventes eventos, marcadores ni nombres de equipos: usa SOLO los hechos \
que te da el usuario.
2. Mantén la duración real del partido y el marcador exacto que se reporte.
3. Cuando una métrica esté marcada como artefacto del tracking, contextualiza \
sin omitirla ("lectura con ruido del seguimiento", "valor con interferencia").
4. Tono: cálido y medido. Está prohibido "amigos", "aficionados", "¡goooool!" \
exagerado, "qué partidazo", "fenomenal". Usa giros sobrios: "se filtra", "no \
perdona", "qué jugada", "buena visión", "se queda en el intento".
5. La cronología debe ser prosa fluida, no listas con guiones. Los timestamps \
van inline entre paréntesis: "(00:28)".
6. Acentos correctos siempre: año, niño, México, posición, recibió, hacia.
7. Formato Markdown: usa # como título y ## como subtítulos. NO uses tablas.

OUTPUT OBLIGATORIO: un único JSON válido con estas dos llaves:
{
  "cronica_md": "<markdown narrativo de 250-350 palabras>",
  "subtitulos_reel": [
    {"t": "mm:ss", "text": "subtítulo corto, máx 60 caracteres"},
    ...
  ]
}
Devuelve EXCLUSIVAMENTE el JSON, sin prefacios ni cierres."""


def build_user_prompt(commentary_md: str, summary: dict, video_name: str) -> str:
    score = summary.get("score", {})
    poss = summary.get("possession_pct", {})
    ball = summary.get("ball", {})
    events = summary.get("events_by_type", {})
    artifacts = summary.get("tracking_artifacts", {})
    duration = summary.get("duration_s", 0)

    datos = [
        f"Video: {video_name}",
        f"Duración: {duration:.1f} s",
        f"Marcador: A {score.get('A', 0)} — B {score.get('B', 0)}",
        f"Posesión: A {poss.get('A', 0):.0f}% · B {poss.get('B', 0):.0f}%",
        f"Balón: distancia {ball.get('distance_mm', 0) / 1000:.1f} m · "
        f"velocidad media {ball.get('avg_speed_mm_s', 0) / 1000:.2f} m/s · "
        f"velocidad máxima {ball.get('max_speed_mm_s', 0) / 1000:.2f} m/s",
        f"Eventos detectados: {events}",
        f"Saltos de tracking descartados (artefactos filtrados): "
        f"balón={artifacts.get('n_jumps_discarded_ball', 0)}, "
        f"robots={artifacts.get('n_jumps_discarded_robots', 0)}",
    ]

    return (
        "Datos cuantitativos del partido analizado:\n"
        + "\n".join(f"- {d}" for d in datos)
        + "\n\n"
        + "CRÓNICA DETERMINISTA (base, NO copies literal — refina y conviértela "
        "en narrativa):\n\n"
        + commentary_md
        + "\n\n"
        + "Recuerda: solo el JSON especificado, nada más."
    )


def call_llm(api_key: str, model: str, system: str, user: str) -> str:
    """Llamada síncrona a Claude con prompt caching del system."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.3,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    # Loggear el uso de tokens para auditoría de costo
    usage = getattr(resp, "usage", None)
    if usage:
        logger.info(
            "Tokens: input=%s output=%s cache_create=%s cache_read=%s",
            getattr(usage, "input_tokens", "?"),
            getattr(usage, "output_tokens", "?"),
            getattr(usage, "cache_creation_input_tokens", "?"),
            getattr(usage, "cache_read_input_tokens", "?"),
        )
    return resp.content[0].text


def parse_json_response(text: str) -> dict:
    """Extrae el primer JSON balanceado del response del LLM."""
    # Buscar { ... } balanceado (acepta posibles ```json fences alrededor).
    text = text.strip()
    # Remover bloques de código si los hay
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Encontrar el primer { y el último } balanceado
    start = text.find("{")
    if start < 0:
        raise ValueError("No se encontró JSON en la respuesta del LLM")
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("JSON sin balancear en la respuesta del LLM")
    blob = text[start : end + 1]
    return json.loads(blob)


def write_outputs(parsed: dict, run_dir: Path, model_name: str) -> tuple[Path, Path]:
    pro_md = run_dir / "commentary_pro.md"
    subs_txt = run_dir / "commentary_pro_subtitles.txt"

    header = (
        "<!-- Generado por scripts/render_commentary_llm.py "
        f"usando {model_name}. Refina commentary.md determinista. -->\n\n"
    )
    pro_md.write_text(header + parsed["cronica_md"].strip() + "\n", encoding="utf-8")

    lines = []
    for s in parsed.get("subtitulos_reel", []):
        t = s.get("t", "??:??")
        txt = s.get("text", "").strip()
        if txt:
            lines.append(f"[{t}] {txt}")
    subs_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pro_md, subs_txt


def write_fallback(commentary_md: str, run_dir: Path, reason: str) -> tuple[Path, Path]:
    pro_md = run_dir / "commentary_pro.md"
    subs_txt = run_dir / "commentary_pro_subtitles.txt"

    warning = (
        "> **Aviso**: no se ejecutó la capa LLM "
        f"({reason}). Mostrando crónica determinista como respaldo.\n"
        "> Para activar la prosa narrativa, configura `ANTHROPIC_API_KEY` "
        "en tu `.env` y vuelve a correr `scripts/render_commentary_llm.py`.\n\n"
    )
    pro_md.write_text(warning + commentary_md, encoding="utf-8")
    subs_txt.write_text(
        "# Sin subtítulos generados por LLM. Usa commentary.txt determinista.\n",
        encoding="utf-8",
    )
    return pro_md, subs_txt


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Carpeta del run con commentary.md, summary.json, events.json.",
    )
    p.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Modelo Anthropic (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Forzar fallback determinista (útil para tests sin red).",
    )
    args = p.parse_args()

    run_dir = args.run
    commentary_path = run_dir / "commentary.md"
    summary_path = run_dir / "summary.json"

    if not commentary_path.exists():
        logger.error("Falta commentary.md en %s", run_dir)
        return 1
    if not summary_path.exists():
        logger.error("Falta summary.json en %s", run_dir)
        return 1

    commentary_md = commentary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_name = summary.get("video", run_dir.name)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.no_llm or not api_key:
        reason = "--no-llm flag" if args.no_llm else "ANTHROPIC_API_KEY ausente"
        logger.warning("Fallback activo: %s", reason)
        pro_md, subs_txt = write_fallback(commentary_md, run_dir, reason)
        logger.info("Output (fallback): %s · %s", pro_md, subs_txt)
        return 0

    user_prompt = build_user_prompt(commentary_md, summary, video_name)
    try:
        raw = call_llm(api_key, args.model, SYSTEM_PROMPT, user_prompt)
        parsed = parse_json_response(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falla LLM (%s). Activando fallback determinista.", exc)
        pro_md, subs_txt = write_fallback(commentary_md, run_dir, f"error: {exc}")
        return 0

    if "cronica_md" not in parsed:
        logger.error("Respuesta del LLM sin 'cronica_md'. Fallback.")
        pro_md, subs_txt = write_fallback(
            commentary_md, run_dir, "respuesta del LLM mal formada"
        )
        return 0

    pro_md, subs_txt = write_outputs(parsed, run_dir, args.model)
    logger.info("Output: %s · %s", pro_md, subs_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
