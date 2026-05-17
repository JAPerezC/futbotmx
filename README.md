# futbotmx — análisis CV de fútbol robótico con SAM 3.1

Sistema de visión por computadora para análisis automático de partidos de
fútbol robótico, desarrollado para la **Copa FutBotMX 2026 — Capítulo
Visión por Computadora** (Secihti · Meta · CENTRO).

> Categoría: **Profesional** · Modelo base obligatorio: **SAM 3 / SAM 3.1 (Meta)**

## TL;DR

Pipeline end-to-end que toma video crudo de un partido de fútbol robótico
y produce: segmentación SAM 3.1 de campo/balón/robots, tracking
multi-objeto con IDs persistentes, rectificación a vista top-down,
detección rule-based de eventos (kick, gol, retención, falta de
progreso, robot dañado) y visualizaciones (heatmap, trails, Voronoi).

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

## Innovación sobre SAM 3 (requisito Profesional § 3.7.3)

Cubrimos **tres** de las cuatro líneas oficiales:

1. **Prompts y contexto**: validamos empíricamente que prompts simples
   (`"soccer robot"`, score 0.94) superan a prompts elaborados
   (`"small mobile soccer robot with a colored flag"`, score 0.34).
   Diseño de prompts especializado al dominio en
   [`src/segmentation/prompts.py`](src/segmentation/prompts.py).
2. **Integración con trackers**: cascada SAM 3.1 → OC-SORT (robots) +
   Kalman 2D (balón) + clasificador HSV (identidad de equipo).
3. **Post-procesamiento geométrico**: homografía clásica con líneas
   blancas conocidas → coordenadas mundo (mm) → cálculo de velocidad,
   posesión por proximidad, detección rule-based de eventos al estilo
   de los AutoRefs de RoboCup SSL.

**Fine-tuning LoRA** (4ª línea): infraestructura lista (rank 8 sobre
encoder+decoder, ~12 GB VRAM), pendiente de anotar 200-300 frames del
dataset oficial. Documentado en `docs/literature-review.md` § 2.

## Resultados

| Métrica | Valor | Comentario |
|---------|-------|------------|
| Tests unitarios | **38/38 + 13/13 = 51/51 ✅** | (38 baselines + 13 trackers) |
| Carga modelo SAM 3.1 | 60 s primera vez, 5 s con cache | Drop-in para SAM 3.1 |
| Inferencia SAM 3.1 (1080p) | 0.5-1.5 s por prompt | RTX 5080 Laptop, 16 GB |
| Pipeline efectivo | ~0.25 fps con 2 prompts | Procesa 96 s en ~6 min |
| Detección balón (HSV+SAM 3) | confianza 0.87-1.00 | Robusto a oclusiones cortas |
| Tracking robots | IDs persistentes entre frames | OC-SORT estable |

Capturas y videos en `data/processed/runs/<video>/`. El video lado a
lado para el jurado (≤2 min) se genera con
`scripts/make_side_by_side.py`.

## Hardware verificado

- NVIDIA GeForce RTX 5080 Laptop, 16 GB VRAM, driver 591.86, CUDA 13.1.
- Python 3.12.10, torch 2.11.0+cu128, transformers 5.8.1.
- Windows 11 Home.

LoRA rank 8 (12 GB VRAM) o rank 16 (16 GB) son ambos viables en este
hardware. Inferencia SAM 3.1 usa ~8 GB de VRAM.

## Instalación

```bash
# 1. Clonar
git clone git@github.com:JAPerezC/futbotmx.git
cd futbotmx

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

- **Equipo**: pendiente de definir (2-4 personas, perfiles en formulario de registro).
- **Mentor / asesor**: por confirmar.
- Asistencia con LLMs autorizados en la convocatoria (§ 3.1.3): Claude (Anthropic) para
  documentación, debugging y exploración de literatura.

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
