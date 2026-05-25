"""Compone el video demo master vertical 1080x1920 (≤2 min) para la convocatoria.

Cumple § 3.5.3: "video de máximo 2 minutos que muestre la vista original junto
al resultado segmentado, indicadores visuales de segmentación, tracking,
visualizaciones, y breve explicación".

Formato vertical 1080x1920 (Reels-ready): el mismo MP4 sirve como entregable
para la convocatoria y como reel ≥30 s de Instagram (§ 3.5.3, descalificatorio).

Estructura del master (115 s totales):
    1. Intro          ( 8 s): título y subtítulos sobre fondo negro
    2. Segmentación   (22 s): extracto annotated.mp4
    3. Tracking       (20 s): extracto annotated_with_minimap.mp4
    4. Highlights     (30 s): 4 goles detectados, ~7.5 s cada uno
    5. Visualizaciones(15 s): heatmaps por equipo + trails (estáticas)
    6. Outro          (20 s): métricas + URL del repo

Genera segmentos parciales en data/processed/runs/IMG_9821/_demo_tmp/ y los
concatena al final con el demuxer concat de ffmpeg.

Uso:
    python scripts/build_master_demo.py
    python scripts/build_master_demo.py --run-dir data/processed/runs/IMG_9821 \\
        --out reports/demo_master_v1.mp4 --keep-tmp
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WIDTH = 1080
HEIGHT = 1920
FPS = 30
# Fuente Segoe UI (instalada por defecto en Windows). El escape \: es para
# que ffmpeg no interprete el dos puntos como separador de filtro.
FONT_REG = "C\\:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C\\:/Windows/Fonts/segoeuib.ttf"

logger = logging.getLogger("build_master_demo")


@dataclass
class Segment:
    name: str
    duration_s: float
    out_path: Path


def run_ffmpeg(args: list[str], description: str) -> None:
    logger.info("ffmpeg: %s", description)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló en: {description}")


def escape_drawtext(text: str) -> str:
    """Escapa caracteres especiales para drawtext: ', :, %, \\."""
    return (
        text.replace("\\", "\\\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
    )


def drawtext_filter(
    text: str,
    *,
    y: int | str,
    font: str = FONT_REG,
    fontsize: int = 56,
    fontcolor: str = "white",
    box: bool = True,
    box_alpha: float = 0.5,
    x: str = "(w-text_w)/2",
) -> str:
    parts = [
        f"drawtext=fontfile='{font}'",
        f"text='{escape_drawtext(text)}'",
        f"x={x}",
        f"y={y}",
        f"fontsize={fontsize}",
        f"fontcolor={fontcolor}",
        "expansion=none",  # desactiva strftime/{var} y permite % literal
    ]
    if box:
        parts.append("box=1")
        parts.append(f"boxcolor=black@{box_alpha}")
        parts.append("boxborderw=24")
    return ":".join(parts)


def make_intro(out: Path) -> None:
    """8 s de fondo negro con título y subtítulos."""
    filters = ",".join(
        [
            drawtext_filter(
                "AJOLOTES FC", y=600, font=FONT_BOLD, fontsize=128, box=False
            ),
            drawtext_filter("Copa FutBotMX 2026", y=820, fontsize=64, box=False),
            drawtext_filter(
                "Categoría Profesional",
                y=920,
                fontsize=56,
                box=False,
                fontcolor="0xFFD700",
            ),
            drawtext_filter(
                "Análisis con SAM 3.1 + LoRA", y=1100, fontsize=48, box=False
            ),
            drawtext_filter(
                "OC-SORT · Kalman · Homografía", y=1180, fontsize=44, box=False
            ),
        ]
    )
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={WIDTH}x{HEIGHT}:d=8:r={FPS}",
            "-vf",
            filters,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            str(out),
        ],
        "intro 8 s",
    )


def make_video_segment(
    input_video: Path,
    out: Path,
    *,
    start_s: float,
    duration_s: float,
    label_top: str,
    label_bottom: str,
    label_top_size: int = 56,
    label_bottom_size: int = 44,
) -> None:
    """Recorta del video fuente, escala vertical, añade labels arriba/abajo."""
    # Escalar a 1080 wide manteniendo aspect, padear vertical a 1080x1920 con
    # padding centrado.
    scale_pad = f"scale={WIDTH}:-2,pad={WIDTH}:{HEIGHT}:0:({HEIGHT}-ih)/2:black"
    label_filters = ",".join(
        [
            drawtext_filter(label_top, y=80, font=FONT_BOLD, fontsize=label_top_size),
            drawtext_filter(
                label_bottom,
                y=HEIGHT - 80 - label_bottom_size,
                fontsize=label_bottom_size,
            ),
        ]
    )
    full_filter = f"{scale_pad},{label_filters}"
    run_ffmpeg(
        [
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(input_video),
            "-t",
            f"{duration_s:.3f}",
            "-vf",
            full_filter,
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out),
        ],
        f"segmento de {input_video.name} {start_s:.1f}-{start_s + duration_s:.1f}s",
    )


def make_highlights_concat(
    highlight_paths: list[Path], out: Path, *, total_duration_s: float
) -> None:
    """Concatena los 4 highlights y aplica el formato vertical con label."""
    if not highlight_paths:
        raise ValueError("No hay highlights disponibles")
    per_clip = total_duration_s / len(highlight_paths)
    # Generar un MP4 temporal por clip (recortado y formateado), luego concatenar.
    tmp_files: list[Path] = []
    label_top = "Goles detectados automáticamente"
    label_bottom = "Detección rule-based del balón en portería"
    for i, clip in enumerate(highlight_paths):
        tmp = out.parent / f"_hl_{i}.mp4"
        tmp_files.append(tmp)
        make_video_segment(
            clip,
            tmp,
            start_s=0,
            duration_s=per_clip,
            label_top=label_top,
            label_bottom=label_bottom,
        )

    list_file = out.parent / "_hl_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in tmp_files),
        encoding="utf-8",
    )
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out),
        ],
        "concatenar 4 highlights",
    )
    for f in [*tmp_files, list_file]:
        f.unlink(missing_ok=True)


def make_static_images_segment(
    image_paths: list[Path], out: Path, *, duration_per_image_s: float
) -> None:
    """Concatena varias imágenes con cada una mostrada N segundos.

    Cada imagen se escala a vertical con label arriba y abajo.
    """
    tmp_files: list[Path] = []
    label_top = "Heatmaps y trayectorias por equipo"
    captions = [
        "Heatmap equipo A (posiciones acumuladas)",
        "Heatmap equipo B (posiciones acumuladas)",
        "Trayectorias completas (trails)",
    ]
    for i, img in enumerate(image_paths):
        if not img.exists():
            logger.warning("Imagen no disponible, se omite: %s", img)
            continue
        tmp = out.parent / f"_img_{i}.mp4"
        tmp_files.append(tmp)
        scale_pad = f"scale={WIDTH}:-2,pad={WIDTH}:{HEIGHT}:0:({HEIGHT}-ih)/2:black"
        labels = ",".join(
            [
                drawtext_filter(label_top, y=80, font=FONT_BOLD, fontsize=56),
                drawtext_filter(
                    captions[i] if i < len(captions) else "",
                    y=HEIGHT - 124,
                    fontsize=44,
                ),
            ]
        )
        run_ffmpeg(
            [
                "-loop",
                "1",
                "-t",
                f"{duration_per_image_s:.3f}",
                "-i",
                str(img),
                "-vf",
                f"{scale_pad},{labels}",
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(tmp),
            ],
            f"imagen estática {img.name}",
        )

    if not tmp_files:
        raise RuntimeError("Ninguna imagen disponible para el segmento")

    list_file = out.parent / "_img_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in tmp_files),
        encoding="utf-8",
    )
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out),
        ],
        "concatenar imágenes estáticas",
    )
    for f in [*tmp_files, list_file]:
        f.unlink(missing_ok=True)


def make_outro(out: Path) -> None:
    """20 s de fondo negro con métricas finales y URL del repo."""
    filters = ",".join(
        [
            drawtext_filter(
                "AJOLOTES FC", y=200, font=FONT_BOLD, fontsize=112, box=False
            ),
            drawtext_filter(
                "Resultados", y=380, fontsize=56, box=False, fontcolor="0xFFD700"
            ),
            drawtext_filter("LoRA fine-tuning SAM 3.1", y=560, fontsize=48, box=False),
            drawtext_filter(
                "mIoU 0.046 → 0.912 (+1882%)",
                y=640,
                fontsize=52,
                font=FONT_BOLD,
                box=False,
            ),
            drawtext_filter(
                "4/4 líneas innovación § 3.7.3", y=780, fontsize=44, box=False
            ),
            drawtext_filter("95 tests automatizados", y=860, fontsize=44, box=False),
            drawtext_filter(
                "Pipeline 100% reproducible", y=940, fontsize=44, box=False
            ),
            drawtext_filter(
                "Visualizaciones", y=1100, fontsize=44, box=False, fontcolor="0xFFD700"
            ),
            drawtext_filter(
                "Dashboard · Minimap · Crónica", y=1180, fontsize=44, box=False
            ),
            drawtext_filter(
                "Heatmaps · Trails · Voronoi", y=1240, fontsize=44, box=False
            ),
            drawtext_filter("Repositorio público", y=1500, fontsize=40, box=False),
            drawtext_filter(
                "github.com/JAPerezC/futbotmx-ajolotesfc",
                y=1560,
                fontsize=40,
                font=FONT_BOLD,
                box=False,
                fontcolor="0x00BFFF",
            ),
            drawtext_filter(
                "Licencia MIT · SAM 3.1 by Meta AI", y=1720, fontsize=36, box=False
            ),
        ]
    )
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={WIDTH}x{HEIGHT}:d=20:r={FPS}",
            "-vf",
            filters,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            str(out),
        ],
        "outro 20 s",
    )


def concat_segments(segments: list[Path], out: Path) -> None:
    """Concatena todos los segmentos finales en el master.

    Re-encodifica para garantizar timestamps y parámetros consistentes (los
    segmentos vienen de fuentes distintas con frame rates posibles distintos).
    """
    list_file = out.parent / "_master_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in segments),
        encoding="utf-8",
    )
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vf",
            f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos,setsar=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(out),
        ],
        "concatenar master final",
    )
    list_file.unlink(missing_ok=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "runs" / "IMG_9821",
    )
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "demo_master_v1.mp4")
    p.add_argument(
        "--keep-tmp",
        action="store_true",
        help="No borrar la carpeta de segmentos temporales (útil para debug)",
    )
    args = p.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        logger.error("No existe el run-dir: %s", run_dir)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = run_dir / "_demo_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    annotated = run_dir / "annotated.mp4"
    annotated_minimap = run_dir / "annotated_with_minimap.mp4"
    highlights_dir = run_dir / "highlights"
    heatmap_a = run_dir / "heatmap_team_A.png"
    heatmap_b = run_dir / "heatmap_team_B.png"
    trails = run_dir / "trails.png"

    if not annotated.exists():
        logger.error("Falta %s", annotated)
        return 1

    segments: list[Path] = []

    intro = tmp_dir / "01_intro.mp4"
    make_intro(intro)
    segments.append(intro)

    seg2 = tmp_dir / "02_segmentation.mp4"
    make_video_segment(
        annotated,
        seg2,
        start_s=0,
        duration_s=22,
        label_top="SAM 3.1 + LoRA · mIoU 0.912",
        label_bottom="Segmentación de robots, campo y balón",
    )
    segments.append(seg2)

    seg3 = tmp_dir / "03_tracking.mp4"
    source3 = annotated_minimap if annotated_minimap.exists() else annotated
    make_video_segment(
        source3,
        seg3,
        start_s=22,
        duration_s=20,
        label_top="Tracking OC-SORT + Kalman · Top-down",
        label_bottom="Vista cenital sintetizada con homografía",
    )
    segments.append(seg3)

    seg4 = tmp_dir / "04_highlights.mp4"
    highlight_paths = (
        sorted(highlights_dir.glob("goal_*.mp4")) if highlights_dir.exists() else []
    )
    # Filtrar highlights por separación temporal mínima de 4 s para evitar
    # mostrar el mismo gol repetido cuando el detector dispara varias veces
    # consecutivas con el balón atascado dentro de la portería.
    selected_highlights: list[Path] = []
    last_t = -1.0e9
    for hl in highlight_paths:
        m = re.search(r"_t(\d+\.?\d*)_", hl.name)
        if not m:
            continue
        t = float(m.group(1))
        if t - last_t >= 4.0:
            selected_highlights.append(hl)
            last_t = t
        if len(selected_highlights) >= 4:
            break
    if selected_highlights:
        make_highlights_concat(selected_highlights, seg4, total_duration_s=30)
        segments.append(seg4)
    else:
        logger.warning("Sin highlights espaciados, se salta el segmento de goles")

    seg5 = tmp_dir / "05_visualizations.mp4"
    images = [heatmap_a, heatmap_b, trails]
    if any(img.exists() for img in images):
        make_static_images_segment(images, seg5, duration_per_image_s=5.0)
        segments.append(seg5)
    else:
        logger.warning("Sin heatmaps/trails, se salta el segmento de visualizaciones")

    outro = tmp_dir / "06_outro.mp4"
    make_outro(outro)
    segments.append(outro)

    concat_segments(segments, args.out)

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Master listo en: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
