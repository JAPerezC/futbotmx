# AJOLOTES FC — análisis CV de fútbol robótico con SAM 3.1

Sistema de visión por computadora para análisis automático de partidos de
fútbol robótico, desarrollado por el equipo **AJOLOTES FC** para la
**Copa FutBotMX 2026 — Capítulo Visión por Computadora**
(Secihti · Meta · CENTRO).

> Equipo: **AJOLOTES FC**
> Categoría: **Profesional**
> Modelo base obligatorio: **SAM 3 / SAM 3.1 (Meta)**

## TL;DR

Pipeline end-to-end que toma video crudo de un partido de fútbol robótico
y produce:

- **Segmentación** SAM 3.1 (Meta) de campo, balón y robots — con
  half-precision en GPU (RTX 5080, ~2.5× speedup, mismas detecciones).
- **Tracking** OC-SORT (robots) + Kalman 2D (balón) con IDs persistentes.
- **Re-ID adaptativo** por color de bandera (k-means HSV sin hardcoding).
- **Homografía 4-puntos** auto-detectada → coordenadas mundo (mm).
- **Eventos del reglamento**: gol, kick, pase, intercepción, retención,
  colisión, no_progress, robot dañado (8 tipos detectables).
- **Estadísticas en tiempo real**: score A-B, % posesión por equipo,
  distancia y velocidad de cada robot y del balón.
- **Visualizaciones**: video anotado con banner persistente + heatmaps
  por equipo + trails top-down + Voronoi + **dashboard HTML interactivo**.

**Reel Instagram**: pendiente de publicación, link aquí antes del cierre.

## Arquitectura

```
video crudo (1080p o portrait HEVC, 30/60 fps)
        │
        ▼
[ingest]  → frames RGB (cv2)
        │
        ▼
[calibrate]  → homografía 4-puntos (esquinas detectadas auto via HSV)
        │     imagen ↔ mundo (mm), campo 2190x1580 mm
        ▼
[segment SAM 3.1]  → máscaras de campo, balón, robots
        │            inferencia HF, ~0.5-1.5 s por prompt @ 1080p (RTX 5080)
        ▼
[track]  → OC-SORT robots + Kalman 2D balón (con HSV fallback)
        │   re-ID por color de bandera (HSV cascada)
        ▼
[events rule-based]  → kick, gol, retención, no progress, dañado
        │              umbrales del reglamento FutBotMX + AutoRefs SSL
        ▼
[viz]  → annotated.mp4 + heatmap + trails + voronoi + JSON
```

Detalles en [`docs/architecture.md`](docs/architecture.md) y justificación
de cada decisión con literatura 2024-2026 en
[`docs/literature-review.md`](docs/literature-review.md).

> **¿Nuevo en visión por computadora?** Lee la
> [documentación completa explicada desde cero](docs/documentacion-completa.md)
> ([PDF](docs/documentacion-completa.pdf)): explica todos los conceptos
> básicos (píxeles, HSV, segmentación, tracking, homografía, SAM 3, LoRA)
> y recorre el pipeline paso a paso. Incluye glosario.

## Innovación sobre SAM 3 (requisito Profesional § 3.7.3)

Cubrimos **las cuatro** líneas oficiales:

1. **Prompts y contexto**: validamos empíricamente que prompts simples
   (`"soccer robot"`, score 0.94) superan a prompts elaborados
   (`"small mobile soccer robot with a colored flag"`, score 0.34).
   Diseño de prompts especializado al dominio en
   [`src/segmentation/prompts.py`](src/segmentation/prompts.py).
2. **Integración con trackers**: cascada SAM 3.1 → OC-SORT (robots) +
   Kalman 2D (balón) + clasificador HSV adaptativo con votación
   temporal (identidad de equipo).
3. **Post-procesamiento geométrico**: detección de esquinas reales por
   líneas blancas Hough con fallback a convex hull (v2); porterías
   detectadas por color (amarilla/azul) como bbox real para ROI de gol
   (no virtual). Homografía 4-puntos imagen ↔ mundo (mm) → velocidad,
   posesión por proximidad, eventos rule-based al estilo AutoRefs SSL.
4. **Fine-tuning LoRA** sobre SAM 3.1: rank=8, target q/k/v/o_proj de
   vision_encoder + mask_decoder (3.88M params, 0.46% del modelo).
   Dataset pseudo-supervisado de 524 máscaras curadas (score ≥ 0.6 +
   filtros geométricos) sobre 15 videos del Drive oficial. 15 epochs
   × 80 samples en **16.8 min** sobre RTX 5080. Flujo en
   [`docs/lora-finetuning.md`](docs/lora-finetuning.md).

   **Resultados cuantitativos** (validación rigurosa, 60 samples val
   por split por video, sin leakage):

   | Métrica | SAM 3.1 base | LoRA fine-tuned | Mejora |
   |---|---|---|---|
   | **mIoU global** | 0.046 | **0.895** | **+1844%** |
   | mIoU robots | 0.049 | **0.934** | +1796% |
   | mIoU balón | 0.036 | **0.776** | +2050% |

   Reporte completo en
   `data/processed/lora_checkpoints/validation_lora_best_iou0.879_ep15.json`.
   Reproducible con
   `python scripts/validate_lora.py --ckpt <path>`.

Estado del arte 2026 cruzado vía survey externo + bibliografía
([`refs.bib`](refs.bib)) con foco en `teamaware_sam_2025`,
`sam3_lora_sompote`, `pnlcalib_2024`, `handheld_football_mot_2025`.

## Resultados

| Métrica | Valor | Comentario |
|---------|-------|------------|
| Tests unitarios | **83/83 ✅** | cobertura por módulo, pasa con `pytest tests/` |
| Smoke test reproducible | ~13 s | `scripts/smoke_test.py` valida 12 componentes |
| Carga modelo SAM 3.1 | 13 s primera vez, 5 s con cache | fp16 reduce VRAM a 1.67 GB |
| Inferencia SAM 3.1 (portrait 1808p, 2 prompts) | ~1 s por frame | RTX 5080 + fp16 |
| Pipeline efectivo | **~1.06 fps** con stride 5 (~2.5× vs fp32) | 100 s → ~10 min |
| Detección balón | conf 0.47-1.00, fallback HSV | robusto a oclusiones cortas |
| Tracking robots | IDs persistentes + re-ID HSV adaptativo | bajos ID-switches en clips ≤ 30 s |
| Eventos detectados (video-1054 110 s) | 1 gol, 61 kicks, 9 no_progress | reglamento § 4.4 |
| Eventos detectados (video-988 29 s) | 1 retención, 169 kicks, 3 no_progress | falta § 4.4.1 detectada |

Outputs por run en `data/processed/runs/<videoname>/`:

| Archivo | Contenido |
|---------|-----------|
| `annotated.mp4` | Video original con bboxes + balón + banner persistente (score/posesión/tiempo) + banners de eventos |
| `dashboard.html` | **Dashboard interactivo plotly** standalone — abrir en cualquier browser |
| `tracks.json` | Trayectorias por frame en imagen + mundo (mm) |
| `events.json` | Lista de eventos con timestamp, tipo, posición mm, metadata |
| `summary.json` | Métricas agregadas: score, posesión %, distancia/velocidad balón y por robot, conteos |
| `heatmap_ball.png` · `heatmap_robots.png` | Densidad top-down con Gaussian blur |
| `heatmap_team_A.png` · `heatmap_team_B.png` | Densidad **por equipo** (colormap por team) |
| `trails.png` | Trayectorias completas top-down con porterías amarilla/azul |
| `voronoi_final.png` | Control de espacio en último frame, coloreado por equipo |

El video lado a lado para el jurado (≤ 2 min) se genera con
`scripts/make_side_by_side.py`. Para reproducir velocidad real con
frame doubling: usar ffmpeg `[1:v]fps=fps=60` filter.

## Hardware verificado

- NVIDIA GeForce RTX 5080 Laptop, 16 GB VRAM, driver 591.86, CUDA 13.1.
- Python 3.12.10, torch 2.11.0+cu128, transformers 5.8.1.
- Windows 11 Home.

LoRA rank 8 (12 GB VRAM) o rank 16 (16 GB) son ambos viables en este
hardware. Inferencia SAM 3.1 usa ~8 GB de VRAM.

## Instalación

```bash
# 1. Clonar
git clone git@github.com:JAPerezC/futbotmx-ajolotesfc.git
cd futbotmx-ajolotesfc

# 2. Entorno (Python 3.12 obligatorio para SAM 3)
py -3.12 -m venv .venv

# 3. Dependencias base
.venv/Scripts/pip install -r requirements.txt

# 4. PyTorch con CUDA (separado por el índice externo)
.venv/Scripts/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 5. Autenticación HuggingFace (modelo SAM 3 es gated)
#    Crear token en https://huggingface.co/settings/tokens (Read)
.venv/Scripts/hf auth login --token <TU_TOKEN>

# 6. Aceptar términos en https://huggingface.co/facebook/sam3

# 7. Si tu red usa SSL inspection (Norton/Kaspersky/etc.), truststore
#    ya está en requirements.txt — Python usará el cert store del SO.

# 8. Verificar
.venv/Scripts/python -m pytest tests/ -q
# Smoke test reproducible: confirma versiones + GPU + cada componente
.venv/Scripts/python scripts/smoke_test.py
```

## Reproducir el demo

```bash
# Procesar un video del dataset oficial
.venv/Scripts/python scripts/run_pipeline.py \
    --video data/raw/drive_samples/video-977.mov \
    --stride 3

# Generar video lado a lado para el jurado (≤2 min)
.venv/Scripts/python scripts/make_side_by_side.py \
    --original data/raw/drive_samples/video-977.mov \
    --annotated data/processed/runs/video-977/annotated.mp4 \
    --out reports/demo_video-977.mp4
```

Outputs en `data/processed/runs/<videoname>/`:
- `annotated.mp4` — video con bboxes, balón, banners de eventos
- `tracks.json` — trayectorias en imagen y mundo (mm)
- `events.json` — eventos detectados con timestamp
- `heatmap_robots.png`, `heatmap_ball.png` — densidad de actividad
- `trails.png` — trayectorias top-down completas
- `voronoi_final.png` — control de espacio
- `summary.json` — métricas agregadas

## Dataset

Videos oficiales en Google Drive (123 clips cortos vertical HEVC 60 fps):
[carpeta oficial](https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD).

Para bajar localmente:

```bash
.venv/Scripts/python -c "
from src.utils.network import enable_system_ssl; enable_system_ssl()
import gdown
gdown.download_folder(
    'https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD',
    output='data/raw/drive_samples',
)
"
```

**Importante** (`docs/dataset-inspection.md`): los videos del Drive son
**portrait** (~1360×1808 px), **HEVC**, **60 fps**, **clips cortos
3-14 s** (no partidos completos). Cada archivo `video-XXX_singular_display.mov`
representa una jugada individual.

## Estructura del repo

```
futbotmx/
├── src/
│   ├── segmentation/    SAM 3.1 wrapper, prompts, baseline HSV ball
│   ├── tracking/        OC-SORT robots, Kalman ball, HSV re-ID
│   ├── events/          rule-based events (estilo AutoRefs SSL)
│   ├── viz/             heatmap, trails, voronoi
│   └── utils/           homografía, IO video, network (SSL)
├── scripts/             run_pipeline, make_side_by_side, sam3_smoke_test, etc.
├── tests/               pytest, 50+ tests verdes
├── docs/                arquitectura, literatura, compliance, dataset, plan
├── data/                gitignored (videos, frames, runs)
├── requirements.txt     stack pinneado y verificado
└── refs.bib             22 referencias bibliográficas
```

## Atribución de dependencias (§ 3.6)

| Dependencia | Rol | Licencia |
|-------------|-----|----------|
| [SAM 3 / SAM 3.1](https://github.com/facebookresearch/sam3) (Meta) | Segmentación base (obligatorio convocatoria) | Meta SAM License — **no MIT** |
| [transformers](https://github.com/huggingface/transformers) (HF) | API SAM 3 | Apache 2.0 |
| [BoxMOT](https://github.com/mikel-brostrom/boxmot) | OC-SORT/ByteTrack tracking | AGPL-3.0 |
| [supervision](https://github.com/roboflow/supervision) (Roboflow) | Utilidades de detection/mask | MIT |
| [opencv-python](https://github.com/opencv/opencv-python) | IO video + homografía + Kalman | Apache 2.0 |
| [truststore](https://github.com/sethmlarson/truststore) | SSL via cert store del SO | MIT |
| [gdown](https://github.com/wkentaro/gdown) | Descarga del Drive oficial | MIT |
| [scipy](https://scipy.org/), [numpy](https://numpy.org/), [pillow](https://python-pillow.org/) | Computación | BSD |

Toda dependencia se respeta según su licencia. Ver
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) para texto completo.

## Licencia del proyecto

Código bajo **MIT** (ver [`LICENSE`](LICENSE)). El uso de SAM 3 se rige por
la licencia propia de Meta — los pesos NO se vendorizan en este repo; se
descargan al instalar.

## Equipo y créditos

**AJOLOTES FC** — registrado para la **Categoría Profesional**:

| Integrante | Rol |
|---|---|
| Macias Sobrino, María Fernanda | Integrante |
| De Unanue Tiscareño, Adolfo Javier | Integrante |
| Cisneros Villarán, Stephany | Integrante |
| Pérez Castellanos, Jorge Alejandro | Integrante |

Los perfiles públicos de cada integrante (GitHub / LinkedIn) se proporcionan
en el formulario oficial de registro (§ 3.3 de la convocatoria); los datos
personales (edad, INE, etc.) se entregan únicamente a la Secihti conforme
al aviso de privacidad LFPDPPP (§ 3.12) y no se publican aquí.

Asistencia con LLMs autorizados en la convocatoria (§ 3.1.3): Claude
(Anthropic) para documentación, depuración y exploración de literatura.
La totalidad del diseño técnico, decisiones de arquitectura y validación
empírica son responsabilidad y autoría del equipo.

## Recursos oficiales

- Convocatoria Secihti: https://secihti.mx/futbotmx/
- Formulario registro: https://forms.cloud.microsoft/r/m8cwt7D7i0
- Reglas torneo: https://secihti.mx/wp-content/uploads/2026/01/Reglas_Copa_FutBotMX_v3_2026-01-21.pdf
- SAM 3 repo: https://github.com/facebookresearch/sam3
- Paper SAM 3: https://arxiv.org/abs/2511.16719
- Contacto: futbotmx@secihti.mx
- Sede final: UPIITA-IPN, CDMX, 24-26 jun 2026

## Calendario

- **2026-05-22 23:59**: cierre de registro (formulario + URL repo).
- **2026-06-19 23:59**: deadline entregable GitHub (repo congelado).
- 2026-06-24 → 26: Copa FutBotMX presencial + premiación.
